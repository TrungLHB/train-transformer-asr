"""Training loop for ASR models (CTC and Transformer decoder)."""

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from ..models.asr_model import CTCASRModel, TransformerASRModel
from ..data.dataset import CharTokenizer
from .metrics import compute_cer, compute_wer

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Number of ref/hyp sample pairs to print after each validation pass
_NUM_SAMPLES_TO_SHOW = 3


# ---------------------------------------------------------------------------
# Learning-rate schedule: Transformer warmup (Vaswani et al., 2017)
# ---------------------------------------------------------------------------

def get_transformer_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int,
) -> LambdaLR:
    """Noam learning rate schedule with linear warmup and inverse sqrt decay."""

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return d_model ** (-0.5) * min(
            step ** (-0.5), step * warmup_steps ** (-1.5)
        )

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vram_gb() -> Optional[float]:
    """Return current GPU memory allocated in GB, or None if not on CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return None


def _fmt_vram() -> str:
    v = _vram_gb()
    return f"{v:.2f} GB" if v is not None else "N/A"


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer for CTC and Transformer ASR models.

    Args:
        model:       An :class:`~src.models.asr_model.CTCASRModel` or
                     :class:`~src.models.asr_model.TransformerASRModel`.
        tokenizer:   A :class:`~src.data.dataset.CharTokenizer`.\
        cfg:         OmegaConf DictConfig (see ``configs/base_config.yaml``).
        train_loader: Training DataLoader.
        valid_loader: Validation DataLoader.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CharTokenizer,
        cfg,
        train_loader,
        valid_loader,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.is_ctc = isinstance(model, CTCASRModel)

        tcfg = cfg.training
        self.device = torch.device(
            cfg.hardware.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tcfg.learning_rate,
            weight_decay=tcfg.weight_decay,
        )
        self.scheduler = get_transformer_lr_scheduler(
            self.optimizer,
            cfg.encoder.d_model,
            tcfg.warmup_steps,
        )
        self.scaler = GradScaler(enabled=cfg.hardware.fp16 and self.device.type == "cuda")

        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tcfg.log_dir)

        # ── Weights & Biases ──────────────────────────────────────────────
        wcfg = getattr(cfg, "wandb", None)
        self._wandb = wcfg is not None and getattr(wcfg, "enabled", False) and _WANDB_AVAILABLE
        if self._wandb:
            import omegaconf
            wandb.init(
                project=wcfg.project,
                entity=(wcfg.entity if wcfg.entity else None),
                name=(wcfg.run_name if wcfg.run_name else None),
                tags=list(wcfg.tags) if wcfg.tags else [],
                config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            )
            wandb.watch(self.model, log="gradients", log_freq=tcfg.log_interval)
            logger.info("  W&B run: %s", wandb.run.url)
        elif wcfg is not None and getattr(wcfg, "enabled", False):
            logger.warning("  wandb not installed – W&B logging disabled. Run: pip install wandb")

        self.global_step = 0
        self.best_valid_wer = float("inf")
        self.early_stop_counter = 0

        # History for end-of-training summary
        self._history: list[dict] = []

        # Loss functions
        if self.is_ctc:
            self.ctc_loss_fn = nn.CTCLoss(
                blank=tokenizer.blank_id,
                reduction="mean",
                zero_infinity=True,
            )
        else:
            # Label smoothing cross-entropy
            self.ce_loss_fn = nn.CrossEntropyLoss(
                ignore_index=tokenizer.blank_id,
                label_smoothing=0.1,
            )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self, batch) -> tuple[float, float]:
        """Returns (loss, grad_norm)."""
        features, feature_lens, tokens, token_lens = batch
        features = features.to(self.device)
        feature_lens = feature_lens.to(self.device)
        tokens = tokens.to(self.device)
        token_lens = token_lens.to(self.device)

        self.optimizer.zero_grad()

        with autocast(enabled=self.cfg.hardware.fp16 and self.device.type == "cuda"):
            if self.is_ctc:
                log_probs, enc_lens = self.model(features, feature_lens)
                loss = self.ctc_loss_fn(log_probs, tokens, enc_lens, token_lens)
            else:
                sos = torch.full(
                    (tokens.size(0), 1), self.tokenizer.sos_id,
                    dtype=torch.long, device=self.device
                )
                decoder_input = torch.cat([sos, tokens[:, :-1]], dim=1)
                target_lens_in = token_lens

                logits = self.model(features, decoder_input, feature_lens, target_lens_in)
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = tokens.reshape(-1)
                loss = self.ce_loss_fn(logits_flat, targets_flat)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.training.clip_grad_norm
        ).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item(), grad_norm

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_refs, all_hyps = [], []

        for batch in self.valid_loader:
            features, feature_lens, tokens, token_lens = batch
            features = features.to(self.device)
            feature_lens = feature_lens.to(self.device)
            tokens = tokens.to(self.device)
            token_lens = token_lens.to(self.device)

            if self.is_ctc:
                log_probs, enc_lens = self.model(features, feature_lens)
                loss = self.ctc_loss_fn(log_probs, tokens, enc_lens, token_lens)
                total_loss += loss.item()
                hyp_ids_list = self.model.decode(features, feature_lens, beam_size=10)
            else:
                sos = torch.full(
                    (tokens.size(0), 1), self.tokenizer.sos_id,
                    dtype=torch.long, device=self.device
                )
                decoder_input = torch.cat([sos, tokens[:, :-1]], dim=1)
                logits = self.model(features, decoder_input, feature_lens, token_lens)
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = tokens.reshape(-1)
                loss = self.ce_loss_fn(logits_flat, targets_flat)
                total_loss += loss.item()
                hyp_ids_list = self.model.decode(features, feature_lens, beam_size=5)

            for b in range(tokens.size(0)):
                ref_ids = tokens[b, : token_lens[b]].tolist()
                ref_text = self.tokenizer.decode(ref_ids)
                hyp_text = self.tokenizer.decode(hyp_ids_list[b])
                all_refs.append(ref_text)
                all_hyps.append(hyp_text)

        n = max(len(self.valid_loader), 1)
        avg_loss = total_loss / n
        word_err = compute_wer(all_refs, all_hyps)
        char_err = compute_cer(all_refs, all_hyps)

        self.model.train()
        return {
            "loss": avg_loss,
            "wer": word_err,
            "cer": char_err,
            "refs": all_refs,
            "hyps": all_hyps,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        tcfg = self.cfg.training
        model_type = "CTC" if self.is_ctc else "Transformer"
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # ── Startup banner ────────────────────────────────────────────────
        logger.info(_separator("═"))
        logger.info("  ASR Training  │  decoder=%s  │  params=%s  │  device=%s",
                    model_type, f"{n_params:,}", self.device)
        logger.info("  Train batches: %d  │  Valid batches: %d  │  Max epochs: %d",
                    len(self.train_loader), len(self.valid_loader), tcfg.max_epochs)
        logger.info("  Effective batch size: %d  │  FP16: %s  │  VRAM: %s",
                    tcfg.batch_size * tcfg.grad_accumulation_steps,
                    self.cfg.hardware.fp16, _fmt_vram())
        logger.info(_separator("═"))

        self.model.train()

        for epoch in range(1, tcfg.max_epochs + 1):
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            t0 = time.time()
            num_steps = len(self.train_loader)

            # ── Epoch header ──────────────────────────────────────────────
            logger.info(_separator())
            logger.info("  Epoch %d / %d", epoch, tcfg.max_epochs)
            logger.info(_separator())

            for step, batch in enumerate(self.train_loader, 1):
                loss, grad_norm = self._train_step(batch)
                epoch_loss += loss
                epoch_grad_norm += grad_norm
                self.global_step += 1

                if self.global_step % tcfg.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed_step = time.time() - t0
                    steps_per_sec = step / elapsed_step
                    eta_sec = (num_steps - step) / max(steps_per_sec, 1e-6)

                    logger.info(
                        "  step %4d/%d │ loss=%.4f │ grad_norm=%.3f │ "
                        "lr=%.2e │ %.1f steps/s │ ETA %ds │ VRAM %s",
                        step, num_steps, loss, grad_norm,
                        lr, steps_per_sec, int(eta_sec), _fmt_vram(),
                    )

                    self.writer.add_scalar("train/loss", loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)

                    if getattr(self, "_wandb", False):
                        wandb.log({
                            "train/loss": loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/vram_gb": _vram_gb() or 0.0,
                        }, step=self.global_step)

            avg_epoch_loss = epoch_loss / max(num_steps, 1)
            avg_grad_norm = epoch_grad_norm / max(num_steps, 1)
            elapsed = time.time() - t0

            logger.info(
                "  ─ Epoch %d done │ avg_loss=%.4f │ avg_grad_norm=%.3f │ %.1fs",
                epoch, avg_epoch_loss, avg_grad_norm, elapsed,
            )

            if getattr(self, "_wandb", False):
                wandb.log({
                    "epoch/train_loss": avg_epoch_loss,
                    "epoch/avg_grad_norm": avg_grad_norm,
                    "epoch/elapsed_s": elapsed,
                    "epoch": epoch,
                }, step=self.global_step)

            # ── Validation ────────────────────────────────────────────────
            if epoch % tcfg.eval_interval == 0:
                logger.info("  Evaluating …")
                t_val = time.time()
                metrics = self.evaluate()
                val_elapsed = time.time() - t_val

                wer_val = metrics["wer"]
                cer_val = metrics["cer"]
                val_loss = metrics["loss"]
                wer_delta = wer_val - self.best_valid_wer  # negative = improvement

                self.writer.add_scalar("valid/loss", val_loss, epoch)
                self.writer.add_scalar("valid/wer", wer_val, epoch)
                self.writer.add_scalar("valid/cer", cer_val, epoch)

                if getattr(self, "_wandb", False):
                    wandb.log({
                        "valid/loss": val_loss,
                        "valid/wer": wer_val,
                        "valid/cer": cer_val,
                        "epoch": epoch,
                    }, step=self.global_step)

                logger.info(_separator("·"))
                logger.info(
                    "  Validation │ loss=%.4f │ WER=%.4f (%s) │ CER=%.4f │ %.1fs",
                    val_loss, wer_val,
                    f"{wer_delta:+.4f}" if self.best_valid_wer < float("inf") else "first",
                    cer_val, val_elapsed,
                )

                # ── Sample predictions ────────────────────────────────────
                logger.info("  Sample predictions:")
                refs, hyps = metrics["refs"], metrics["hyps"]
                
                if getattr(self, "_wandb", False):
                    cols = ["ref", "hyp"]
                    rows = [[refs[i] or "", hyps[i] or ""] for i in range(min(_NUM_SAMPLES_TO_SHOW, len(refs)))]
                    wandb.log(
                        {"valid/sample_predictions": wandb.Table(columns=cols, data=rows)},
                        step=self.global_step,
                    )
                    
                for i in range(min(_NUM_SAMPLES_TO_SHOW, len(refs))):
                    logger.info("    [%d] REF: %s", i + 1, refs[i] or "<empty>")
                    logger.info("        HYP: %s", hyps[i] or "<empty>")
                logger.info(_separator("·"))

                # ── Checkpoint & early stopping ───────────────────────────
                ckpt_dir = tcfg.checkpoint_dir
                if wer_val < self.best_valid_wer:
                    self.best_valid_wer = wer_val
                    self.early_stop_counter = 0
                    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "wer": wer_val,
                            "cer": cer_val,
                        },
                        ckpt_path,
                    )
                    logger.info("  ✓ New best WER=%.4f  →  saved to %s", wer_val, ckpt_path)
                    
                    if getattr(self, "_wandb", False):
                        wandb.run.summary["best_wer"] = wer_val
                        wandb.run.summary["best_cer"] = cer_val
                        wandb.run.summary["best_epoch"] = epoch
                        if getattr(self.cfg.wandb, "log_model", False):
                            artifact = wandb.Artifact(
                                name=f"best_model-{wandb.run.id}",
                                type="model",
                                metadata={"epoch": epoch, "wer": wer_val, "cer": cer_val},
                            )
                            artifact.add_file(ckpt_path)
                            wandb.log_artifact(artifact)
                else:
                    self.early_stop_counter += 1
                    patience_left = tcfg.early_stopping_patience - self.early_stop_counter
                    logger.info(
                        "  ✗ No improvement  (patience %d/%d, %d left)",
                        self.early_stop_counter, tcfg.early_stopping_patience, patience_left,
                    )
                    if not tcfg.save_best_only:
                        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
                        torch.save(
                            {
                                "epoch": epoch,
                                "global_step": self.global_step,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "wer": wer_val,
                                "cer": cer_val,
                            },
                            ckpt_path,
                        )

                    if self.early_stop_counter >= tcfg.early_stopping_patience:
                        logger.info(
                            "  Early stopping triggered at epoch %d "
                            "(no improvement for %d epochs).",
                            epoch, tcfg.early_stopping_patience,
                        )
                        self._history.append({
                            "epoch": epoch, "train_loss": avg_epoch_loss,
                            "val_loss": val_loss, "wer": wer_val, "cer": cer_val,
                        })
                        break

                self._history.append({
                    "epoch": epoch, "train_loss": avg_epoch_loss,
                    "val_loss": val_loss, "wer": wer_val, "cer": cer_val,
                })
            else:
                self._history.append({
                    "epoch": epoch, "train_loss": avg_epoch_loss,
                    "val_loss": None, "wer": None, "cer": None,
                })

        # ── End-of-training summary ───────────────────────────────────────
        self.writer.close()
        logger.info(_separator("═"))
        logger.info("  Training complete  │  Best WER: %.4f", self.best_valid_wer)
        logger.info(_separator())

        eval_rows = [r for r in self._history if r["wer"] is not None]
        if eval_rows:
            logger.info("  %-6s  %-12s  %-12s  %-10s  %-10s", "Epoch", "Train Loss", "Val Loss", "WER", "CER")
            logger.info("  " + "─" * 56)
            for r in eval_rows:
                logger.info(
                    "  %-6d  %-12.4f  %-12.4f  %-10.4f  %-10.4f",
                    r["epoch"], r["train_loss"], r["val_loss"], r["wer"], r["cer"],
                )
        logger.info(_separator("═"))
        
        if getattr(self, "_wandb", False):
            wandb.finish()

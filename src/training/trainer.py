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

from ..models.asr_model import CTCASRModel, TransformerASRModel
from ..data.dataset import CharTokenizer
from .metrics import compute_cer, compute_wer


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
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer for CTC and Transformer ASR models.

    Args:
        model:       An :class:`~src.models.asr_model.CTCASRModel` or
                     :class:`~src.models.asr_model.TransformerASRModel`.
        tokenizer:   A :class:`~src.data.dataset.CharTokenizer`.
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

        self.global_step = 0
        self.best_valid_wer = float("inf")
        self.early_stop_counter = 0

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

    def _train_step(self, batch) -> float:
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
                # targets for teacher forcing: prepend <sos>
                sos = torch.full(
                    (tokens.size(0), 1), self.tokenizer.sos_id,
                    dtype=torch.long, device=self.device
                )
                decoder_input = torch.cat([sos, tokens[:, :-1]], dim=1)
                target_lens_in = token_lens  # includes <sos>

                logits = self.model(features, decoder_input, feature_lens, target_lens_in)
                # Shift: compare logits at t with tokens at t+1
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = tokens.reshape(-1)
                loss = self.ce_loss_fn(logits_flat, targets_flat)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.training.clip_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

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
        return {"loss": avg_loss, "wer": word_err, "cer": char_err}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        tcfg = self.cfg.training
        self.model.train()

        for epoch in range(1, tcfg.max_epochs + 1):
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(self.train_loader, 1):
                loss = self._train_step(batch)
                epoch_loss += loss
                self.global_step += 1

                if self.global_step % tcfg.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/loss", loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

            avg_epoch_loss = epoch_loss / max(len(self.train_loader), 1)
            elapsed = time.time() - t0

            if epoch % tcfg.eval_interval == 0:
                metrics = self.evaluate()
                wer_val = metrics["wer"]
                cer_val = metrics["cer"]
                val_loss = metrics["loss"]

                self.writer.add_scalar("valid/loss", val_loss, epoch)
                self.writer.add_scalar("valid/wer", wer_val, epoch)
                self.writer.add_scalar("valid/cer", cer_val, epoch)

                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={avg_epoch_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"WER={wer_val:.4f} | "
                    f"CER={cer_val:.4f} | "
                    f"elapsed={elapsed:.1f}s"
                )

                # Save best checkpoint
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
                    print(f"  ✓ Best checkpoint saved to {ckpt_path}")
                else:
                    self.early_stop_counter += 1
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
                        print(
                            f"Early stopping triggered after {epoch} epochs "
                            f"(no improvement for {tcfg.early_stopping_patience} epochs)."
                        )
                        break
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={avg_epoch_loss:.4f} | "
                    f"elapsed={elapsed:.1f}s"
                )

        self.writer.close()
        print("Training complete. Best WER:", self.best_valid_wer)

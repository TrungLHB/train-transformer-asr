"""
train.py – Main training entry-point with Weights & Biases experiment monitoring.

Usage
-----
    python train.py              # use config.yaml defaults
    python train.py training.learning_rate=5e-4 wandb.entity=my_org
    python train.py --sweep      # run a W&B hyperparameter sweep agent
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from jiwer import cer, wer
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from data import build_dataloaders
from model import TransformerASR


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Learning-rate schedule ────────────────────────────────────────────────────

def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Noam / Transformer learning-rate schedule."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


# ── Validation ────────────────────────────────────────────────────────────────

def evaluate(model, val_loader, tokenizer, device, cfg, epoch: int) -> dict:
    """Run one validation pass and return a metrics dict."""
    model.eval()
    total_loss = total_ce = total_ctc = 0.0
    n_batches = 0

    all_refs, all_hyps_attn, all_hyps_ctc = [], [], []
    audio_samples: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            with autocast(enabled=cfg.training.fp16):
                out = model(features, feature_lengths, labels, label_lengths)

            total_loss += out["loss"].item()
            total_ce += out["ce_loss"].item()
            total_ctc += out["ctc_loss"].item()
            n_batches += 1

            # Greedy decode (transformer attention decoder)
            hyps_attn = model.decode_greedy(features, feature_lengths)
            hyps_ctc = model.decode_ctc_greedy(features, feature_lengths)

            for ref, hyp_a, hyp_c in zip(batch["texts"], hyps_attn, hyps_ctc):
                all_refs.append(ref)
                all_hyps_attn.append(tokenizer.decode(hyp_a, skip_special_tokens=True))
                all_hyps_ctc.append(tokenizer.decode(hyp_c, skip_special_tokens=True))

            # Collect audio samples for the first batch only
            if len(audio_samples) < cfg.wandb.log_audio_samples:
                for i in range(min(cfg.wandb.log_audio_samples - len(audio_samples), len(batch["texts"]))):
                    audio_samples.append(
                        {
                            "audio": batch["audios"][i].numpy(),
                            "reference": batch["texts"][i],
                            "hypothesis_attn": all_hyps_attn[-(len(batch["texts"]) - i)],
                            "hypothesis_ctc": all_hyps_ctc[-(len(batch["texts"]) - i)],
                        }
                    )

    metrics = {
        "val/loss": total_loss / n_batches,
        "val/ce_loss": total_ce / n_batches,
        "val/ctc_loss": total_ctc / n_batches,
        "val/wer_attn": wer(all_refs, all_hyps_attn),
        "val/cer_attn": cer(all_refs, all_hyps_attn),
        "val/wer_ctc": wer(all_refs, all_hyps_ctc),
        "val/cer_ctc": cer(all_refs, all_hyps_ctc),
        "epoch": epoch,
    }

    # Log audio samples to W&B
    if audio_samples:
        sample_rate = cfg.data.sample_rate
        audio_table = wandb.Table(columns=["audio", "reference", "hypothesis_attn", "hypothesis_ctc"])
        for s in audio_samples:
            audio_table.add_data(
                wandb.Audio(s["audio"], sample_rate=sample_rate),
                s["reference"],
                s["hypothesis_attn"],
                s["hypothesis_ctc"],
            )
        metrics["val/audio_samples"] = audio_table

    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg):
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── W&B initialisation ────────────────────────────────────────────────────
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.data.language_code, "transformer", "ctc"],
    )
    # Allow W&B sweep agent to override config values
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = _deep_merge(cfg_dict, dict(wandb.config))
    cfg = OmegaConf.create(cfg_dict)

    # ── Tokenizer & feature extractor (Whisper multilingual as backbone) ──────
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-base", language=cfg.data.language_code, task="transcribe"
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id
    sos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg, tokenizer, feature_extractor)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransformerASR(
        vocab_size=vocab_size,
        n_mels=cfg.features.n_mels,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        ctc_weight=cfg.model.ctc_weight,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"model/n_parameters": n_params}, step=0)

    if cfg.wandb.watch_model:
        wandb.watch(model, log="all", log_freq=cfg.wandb.watch_log_freq)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.training.fp16)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_wer = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")
        ):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            with autocast(enabled=cfg.training.fp16):
                out = model(features, feature_lengths, labels, label_lengths)
                loss = out["loss"] / cfg.training.grad_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += out["loss"].item()

            if (batch_idx + 1) % cfg.training.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

                # Update learning rate (Noam schedule)
                lr = get_lr(global_step + 1, cfg.model.d_model, cfg.training.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # ── W&B: per-step training metrics ────────────────────────────
                if global_step % cfg.wandb.log_interval == 0:
                    wandb.log(
                        {
                            "train/loss": out["loss"].item(),
                            "train/ce_loss": out["ce_loss"].item(),
                            "train/ctc_loss": out["ctc_loss"].item(),
                            "train/grad_norm": grad_norm.item(),
                            "train/learning_rate": lr,
                        },
                        step=global_step,
                    )

        # ── End of epoch: validation ──────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, tokenizer, device, cfg, epoch)
        val_metrics["train/epoch_loss"] = epoch_loss / len(train_loader)
        wandb.log(val_metrics, step=global_step)

        print(
            f"Epoch {epoch:3d} | loss={val_metrics['train/epoch_loss']:.4f} "
            f"| WER(attn)={val_metrics['val/wer_attn']:.4f} "
            f"| WER(ctc)={val_metrics['val/wer_ctc']:.4f}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt_path = output_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "val_wer_attn": val_metrics["val/wer_attn"],
            },
            ckpt_path,
        )

        # Upload checkpoint as a W&B Artifact
        if cfg.wandb.save_artifact:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                metadata={
                    "epoch": epoch,
                    "val_wer_attn": val_metrics["val/wer_attn"],
                    "val_wer_ctc": val_metrics["val/wer_ctc"],
                },
            )
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact)

        # Keep best model
        if val_metrics["val/wer_attn"] < best_wer:
            best_wer = val_metrics["val/wer_attn"]
            best_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            wandb.summary["best_wer_attn"] = best_wer
            wandb.summary["best_epoch"] = epoch

    wandb.finish()


# ── W&B Sweep configuration ───────────────────────────────────────────────────

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/wer_attn", "goal": "minimize"},
    "parameters": {
        "training.learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "model.d_model": {"values": [128, 256, 512]},
        "model.n_heads": {"values": [4, 8]},
        "model.num_encoder_layers": {"values": [4, 6, 8]},
        "model.ctc_weight": {"distribution": "uniform", "min": 0.1, "max": 0.5},
        "training.dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
    },
}


def run_sweep(cfg):
    """Initialise a W&B sweep and start one agent."""
    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
    )
    wandb.agent(sweep_id, function=lambda: train(cfg), count=20)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, creating nested keys as needed."""
    result = dict(base)
    for key, value in override.items():
        keys = key.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


# ── Entry-point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Transformer ASR with W&B monitoring")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-notation overrides, e.g. training.learning_rate=1e-4")
    parser.add_argument("--sweep", action="store_true", help="Launch a W&B hyperparameter sweep")
    args = parser.parse_args()

    cfg = OmegaConf.load("config.yaml")
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    if args.sweep:
        run_sweep(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()

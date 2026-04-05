#!/usr/bin/env python3
"""Train an ASR model with a CTC decoder.

Usage::

    python scripts/train_ctc.py \\
        --config configs/ctc_config.yaml \\
        [--language cy] \\
        [--checkpoint_dir ./checkpoints/ctc]

The script loads a HuggingFace ASR dataset (default: Google FLEURS), builds a
character-level tokeniser, and trains a Conformer encoder with a CTC
decoder.  Training progress is logged to TensorBoard.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf

from src.data.dataset import CharTokenizer, build_dataloaders
from src.models.asr_model import CTCASRModel
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTC ASR model")
    parser.add_argument(
        "--config",
        default="configs/ctc_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--language", default=None, help="Override dataset language code")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--log_dir", default=None, help="Override TensorBoard log directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override max training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--vocab_file", default=None, help="Path to pre-built vocab file")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    base_cfg = OmegaConf.load("configs/base_config.yaml")
    model_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, model_cfg)

    if args.language:
        cfg.data.language = args.language
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        cfg.training.log_dir = args.log_dir
    if args.epochs:
        cfg.training.max_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.training.seed)

    # Build tokeniser
    vocab_path = args.vocab_file or os.path.join(
        cfg.training.checkpoint_dir, "vocab.txt"
    )
    if os.path.exists(vocab_path):
        logger.info("Loading tokenizer from %s", vocab_path)
        tokenizer = CharTokenizer.load(vocab_path)
    else:
        logger.info(
            "Vocab file not found – building from %s dataset (lang=%s).",
            cfg.data.dataset_name,
            cfg.data.language,
        )
        from datasets import load_dataset
        raw = load_dataset(
            cfg.data.dataset_name,
            cfg.data.language,
            cache_dir=cfg.data.cache_dir,
            trust_remote_code=True,
        )
        transcript_field = getattr(cfg.data, "transcript_field", "sentence")
        all_texts = []
        for split in [cfg.data.train_split, cfg.data.valid_split]:
            all_texts.extend(raw[split][transcript_field])
        tokenizer = CharTokenizer().build_from_texts(all_texts)
        tokenizer.save(vocab_path)
        logger.info("Tokenizer saved to %s  (vocab_size=%d)", vocab_path, tokenizer.vocab_size)

    # Build dataloaders
    logger.info("Building dataloaders …")
    train_loader, valid_loader, _ = build_dataloaders(cfg, tokenizer)
    logger.info("Train batches: %d | Valid batches: %d", len(train_loader), len(valid_loader))

    # Build model
    enc_cfg = cfg.encoder
    dec_cfg = cfg.decoder
    model = CTCASRModel(
        vocab_size=tokenizer.vocab_size,
        n_mels=cfg.audio.n_mels,
        d_model=enc_cfg.d_model,
        num_heads=enc_cfg.num_heads,
        num_encoder_layers=enc_cfg.num_layers,
        ff_dim=enc_cfg.ff_dim,
        dropout=enc_cfg.dropout,
        conv_kernel_size=enc_cfg.conv_kernel_size,
        encoder_type=enc_cfg.encoder_type,
        blank_id=tokenizer.blank_id,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: CTC ASR  |  parameters: %s", f"{n_params:,}")

    # Train
    trainer = Trainer(model, tokenizer, cfg, train_loader, valid_loader)
    trainer.train()


if __name__ == "__main__":
    main()

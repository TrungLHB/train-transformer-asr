#!/usr/bin/env python3
"""Evaluate a trained ASR model on the test set.

Usage::

    python scripts/evaluate.py \\
        --checkpoint checkpoints/transformer/best_model.pt \\
        --config configs/transformer_config.yaml \\
        --vocab_file checkpoints/transformer/vocab.txt \\
        [--language cy] \\
        [--beam_size 5] \\
        [--output results.txt]

Prints WER and CER and optionally writes per-utterance predictions.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf

from src.data.dataset import ASRDataset, CharTokenizer
from src.data.preprocessing import LogMelFilterBank, collate_fn
from src.models.asr_model import CTCASRModel, TransformerASRModel, build_model
from src.training.metrics import compute_cer, compute_wer
from src.utils.logger import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ASR model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config used during training",
    )
    parser.add_argument(
        "--vocab_file", required=True, help="Path to vocabulary file (vocab.txt)"
    )
    parser.add_argument("--language", default=None, help="Override dataset language code")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on")
    parser.add_argument("--beam_size", type=int, default=None, help="Override beam size")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument(
        "--output", default=None, help="Optional file to write per-utterance results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_cfg = OmegaConf.load("configs/base_config.yaml")
    model_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, model_cfg)

    if args.language:
        cfg.data.language = args.language

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokeniser
    tokenizer = CharTokenizer.load(args.vocab_file)
    logger.info("Loaded tokenizer: vocab_size=%d", tokenizer.vocab_size)

    # Build model
    model = build_model(cfg, vocab_size=tokenizer.vocab_size)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint from %s  (epoch %d, WER=%.4f)",
        args.checkpoint,
        ckpt.get("epoch", -1),
        ckpt.get("wer", float("nan")),
    )

    # Build test dataloader
    from datasets import load_dataset

    raw = load_dataset(
        cfg.data.dataset_name,
        cfg.data.language,
        cache_dir=cfg.data.cache_dir,
        trust_remote_code=True,
    )
    feat_ext = LogMelFilterBank(
        sample_rate=cfg.audio.sample_rate,
        n_mels=cfg.audio.n_mels,
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        win_length=cfg.audio.win_length,
        f_min=cfg.audio.f_min,
        f_max=cfg.audio.f_max,
    )
    test_ds = ASRDataset(
        hf_dataset=raw[args.split],
        tokenizer=tokenizer,
        feature_extractor=feat_ext,
        augment=False,
        max_duration=cfg.audio.max_duration,
        min_duration=cfg.audio.min_duration,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info("Test set: %d utterances  |  %d batches", len(test_ds), len(test_loader))

    is_ctc = isinstance(model, CTCASRModel)
    beam_size = args.beam_size or (10 if is_ctc else 5)

    all_refs, all_hyps, all_transcripts = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            features, feature_lens, tokens, token_lens = batch
            features = features.to(device)
            feature_lens = feature_lens.to(device)
            tokens = tokens.to(device)
            token_lens = token_lens.to(device)

            hyp_ids_list = model.decode(features, feature_lens, beam_size=beam_size)

            for b in range(tokens.size(0)):
                ref_ids = tokens[b, : token_lens[b]].tolist()
                ref_text = tokenizer.decode(ref_ids)
                hyp_text = tokenizer.decode(hyp_ids_list[b])
                all_refs.append(ref_text)
                all_hyps.append(hyp_text)
                all_transcripts.append((ref_text, hyp_text))

    wer_score = compute_wer(all_refs, all_hyps)
    cer_score = compute_cer(all_refs, all_hyps)

    print(f"\n{'=' * 50}")
    print(f"  Split : {args.split}")
    print(f"  WER   : {wer_score * 100:.2f} %")
    print(f"  CER   : {cer_score * 100:.2f} %")
    print(f"{'=' * 50}\n")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"WER: {wer_score * 100:.2f}%\nCER: {cer_score * 100:.2f}%\n\n")
            for i, (ref, hyp) in enumerate(all_transcripts):
                f.write(f"[{i}] REF: {ref}\n")
                f.write(f"[{i}] HYP: {hyp}\n\n")
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()

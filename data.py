"""Dataset loading and preprocessing utilities for ASR training."""

from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class ASRDataset(Dataset):
    """
    Wraps a Hugging Face `datasets.Dataset` split for ASR training.

    Expected columns: ``audio`` (dict with ``array`` and ``sampling_rate``) and ``sentence``.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        feature_extractor,
        max_audio_length_sec: float = 15.0,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_samples = int(max_audio_length_sec * feature_extractor.sampling_rate)
        # Filter long audio in advance to speed up data loading
        self.data = hf_dataset.filter(
            lambda ex: len(ex["audio"]["array"]) <= self.max_samples,
            desc="Filtering long audio",
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        example = self.data[idx]
        audio_array = torch.tensor(example["audio"]["array"], dtype=torch.float32)
        sr = example["audio"]["sampling_rate"]

        # Resample if necessary
        target_sr = self.feature_extractor.sampling_rate
        if sr != target_sr:
            audio_array = T.Resample(orig_freq=sr, new_freq=target_sr)(audio_array)

        # Extract log-mel features: (n_mels, time)
        features = self.feature_extractor(
            audio_array.numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
        ).input_features.squeeze(0)

        # Tokenise transcript
        text = example["sentence"].strip()
        token_ids = self.tokenizer(text).input_ids

        return {
            "features": features,               # (n_mels, time)
            "feature_length": features.shape[-1],
            "labels": torch.tensor(token_ids, dtype=torch.long),
            "label_length": len(token_ids),
            "text": text,
            "audio": audio_array,               # kept for W&B audio logging
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad a list of samples to the same length within a batch."""
    features = pad_sequence(
        [s["features"].T for s in batch], batch_first=True, padding_value=0.0
    )
    # (batch, time, n_mels) → (batch, n_mels, time)
    features = features.permute(0, 2, 1)
    feature_lengths = torch.tensor([s["feature_length"] for s in batch], dtype=torch.long)

    labels = pad_sequence([s["labels"] for s in batch], batch_first=True, padding_value=0)
    label_lengths = torch.tensor([s["label_length"] for s in batch], dtype=torch.long)

    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "labels": labels,
        "label_lengths": label_lengths,
        "texts": [s["text"] for s in batch],
        "audios": [s["audio"] for s in batch],
    }


def build_dataloaders(
    cfg,
    tokenizer,
    feature_extractor,
) -> tuple[DataLoader, DataLoader]:
    """Load the Hugging Face dataset and return train/val DataLoaders."""
    from datasets import load_dataset  # imported lazily to keep top-level imports clean

    raw = load_dataset(
        cfg.data.dataset_name,
        cfg.data.language_code,
        trust_remote_code=True,
    )

    train_ds = ASRDataset(
        raw[cfg.data.train_split],
        tokenizer,
        feature_extractor,
        max_audio_length_sec=cfg.data.max_audio_length_sec,
    )
    val_ds = ASRDataset(
        raw[cfg.data.val_split],
        tokenizer,
        feature_extractor,
        max_audio_length_sec=cfg.data.max_audio_length_sec,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader

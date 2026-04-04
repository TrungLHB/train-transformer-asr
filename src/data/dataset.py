"""ASR dataset loader supporting Hugging Face datasets (e.g. Common Voice)."""

import os
import re
import unicodedata
from typing import Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .preprocessing import LogMelFilterBank, SpecAugment, collate_fn


# ---------------------------------------------------------------------------
# Character-level tokeniser
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Simple character-level tokeniser.

    Special tokens:
        <blank>  (id 0)  – CTC blank
        <eos>    (id 1)  – end of sequence
        <sos>    (id 2)  – start of sequence
        <unk>    (id 3)  – unknown character
    """

    BLANK = "<blank>"
    EOS = "<eos>"
    SOS = "<sos>"
    UNK = "<unk>"
    SPECIAL_TOKENS = [BLANK, EOS, SOS, UNK]

    def __init__(self, vocab: Optional[List[str]] = None) -> None:
        if vocab is None:
            vocab = []
        self.vocab: List[str] = self.SPECIAL_TOKENS + [
            t for t in vocab if t not in self.SPECIAL_TOKENS
        ]
        self._token2id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}

    @property
    def blank_id(self) -> int:
        return self._token2id[self.BLANK]

    @property
    def eos_id(self) -> int:
        return self._token2id[self.EOS]

    @property
    def sos_id(self) -> int:
        return self._token2id[self.SOS]

    @property
    def unk_id(self) -> int:
        return self._token2id[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self._token2id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        chars = []
        for i in ids:
            token = self.vocab[i] if i < len(self.vocab) else self.UNK
            if remove_special and token in self.SPECIAL_TOKENS:
                continue
            chars.append(token)
        return "".join(chars)

    def build_from_texts(self, texts: List[str]) -> "CharTokenizer":
        """Build vocabulary from a list of transcripts."""
        chars = sorted({ch for text in texts for ch in text})
        self.vocab = self.SPECIAL_TOKENS + [
            c for c in chars if c not in self.SPECIAL_TOKENS
        ]
        self._token2id = {t: i for i, t in enumerate(self.vocab)}
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(token + "\n")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            vocab = [line.rstrip("\n") for line in f]
        tokenizer = cls.__new__(cls)
        tokenizer.vocab = vocab
        tokenizer._token2id = {t: i for i, t in enumerate(vocab)}
        return tokenizer


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_transcript(text: str) -> str:
    """Lower-case, strip accents and remove non-alphabetic/space chars."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s']", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ASRDataset(Dataset):
    """PyTorch Dataset wrapping a Hugging Face dataset split.

    Expects each example to have at minimum:
        - ``audio``   : dict with ``array`` (numpy) and ``sampling_rate``.
        - ``sentence``: raw transcript string.

    Args:
        hf_dataset: A Hugging Face ``datasets.Dataset`` object.
        tokenizer: A :class:`CharTokenizer` instance.
        feature_extractor: :class:`~src.data.preprocessing.LogMelFilterBank`.
        augment: If ``True`` apply SpecAugment during ``__getitem__``.
        spec_augment: Optional :class:`~src.data.preprocessing.SpecAugment`.
        max_duration: Clips longer than this (seconds) are skipped.
        min_duration: Clips shorter than this (seconds) are skipped.
        normalize_text: Apply transcript normalisation.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer: CharTokenizer,
        feature_extractor: LogMelFilterBank,
        augment: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        normalize_text: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.spec_augment = spec_augment
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize_text = normalize_text

        # Filter by duration up-front so __len__ is accurate
        sample_rate = 16000  # expected after resampling
        self.data = [
            ex
            for ex in hf_dataset
            if self.min_duration
            <= len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
            <= self.max_duration
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        example = self.data[idx]
        audio_info = example["audio"]

        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)
        sr = audio_info["sampling_rate"]

        # Resample to 16 kHz if necessary
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16000
            )
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        # Extract features: (n_mels, T)
        features = self.feature_extractor(waveform.unsqueeze(0)).squeeze(0)

        # SpecAugment
        if self.augment and self.spec_augment is not None:
            features = self.spec_augment(features.unsqueeze(0)).squeeze(0)

        # Tokenise transcript
        transcript = example.get("sentence", "")
        if self.normalize_text:
            transcript = normalize_transcript(transcript)
        token_ids = self.tokenizer.encode(transcript)

        return {"features": features, "token_ids": token_ids, "transcript": transcript}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, tokenizer: CharTokenizer):
    """Build train / validation / test dataloaders from a Hydra/OmegaConf config.

    Args:
        cfg: OmegaConf DictConfig (see ``configs/base_config.yaml``).
        tokenizer: Fitted :class:`CharTokenizer`.

    Returns:
        train_loader, valid_loader, test_loader
    """
    from datasets import load_dataset  # heavy import – defer until needed

    audio_cfg = cfg.audio
    feat_ext = LogMelFilterBank(
        sample_rate=audio_cfg.sample_rate,
        n_mels=audio_cfg.n_mels,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        win_length=audio_cfg.win_length,
        f_min=audio_cfg.f_min,
        f_max=audio_cfg.f_max,
    )
    aug = SpecAugment(
        freq_mask_param=cfg.spec_augment.freq_mask_param,
        time_mask_param=cfg.spec_augment.time_mask_param,
        num_freq_masks=cfg.spec_augment.num_freq_masks,
        num_time_masks=cfg.spec_augment.num_time_masks,
    )

    data_cfg = cfg.data
    dataset = load_dataset(
        data_cfg.dataset_name,
        data_cfg.language,
        cache_dir=data_cfg.cache_dir,
        trust_remote_code=True,
    )

    def _make(split, augment):
        return ASRDataset(
            hf_dataset=dataset[split],
            tokenizer=tokenizer,
            feature_extractor=feat_ext,
            augment=augment,
            spec_augment=aug if augment else None,
            max_duration=audio_cfg.max_duration,
            min_duration=audio_cfg.min_duration,
        )

    train_ds = _make(data_cfg.train_split, augment=True)
    valid_ds = _make(data_cfg.valid_split, augment=False)
    test_ds = _make(data_cfg.test_split, augment=False)

    loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader

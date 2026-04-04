"""Audio feature extraction and SpecAugment data augmentation."""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class LogMelFilterBank(nn.Module):
    """Compute log-mel filterbank features from raw waveform.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of mel filterbank channels.
        n_fft: Size of FFT window.
        hop_length: Hop length between frames in samples.
        win_length: Window length in samples.
        f_min: Minimum frequency for mel filterbank.
        f_max: Maximum frequency for mel filterbank.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 80.0,
        f_max: float = 7600.0,
    ) -> None:
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, time) or (time,) float tensor.

        Returns:
            features: (batch, n_mels, frames) log-mel features.
        """
        mel = self.mel_spectrogram(waveform)  # (..., n_mels, frames)
        log_mel = torch.log(mel + 1e-9)
        return log_mel


class SpecAugment(nn.Module):
    """SpecAugment: time and frequency masking for ASR data augmentation.

    Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition", Interspeech 2019.

    Args:
        freq_mask_param: Maximum width of each frequency mask (F).
        time_mask_param: Maximum width of each time mask (T).
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, n_mels, frames) or (n_mels, frames) tensor.

        Returns:
            Augmented features with the same shape.
        """
        for _ in range(self.num_freq_masks):
            features = self.freq_mask(features)
        for _ in range(self.num_time_masks):
            features = self.time_mask(features)
        return features


def collate_fn(
    batch: List[dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (features, token_ids) pairs to equal length.

    Args:
        batch: List of dicts with keys "features" (n_mels, T) and
               "token_ids" (list[int]).

    Returns:
        features:      (B, n_mels, T_max) padded feature tensor.
        feature_lens:  (B,) integer tensor of original frame counts.
        tokens:        (B, U_max) padded token id tensor.
        token_lens:    (B,) integer tensor of original token counts.
    """
    features = [item["features"] for item in batch]
    token_ids = [torch.tensor(item["token_ids"], dtype=torch.long) for item in batch]

    feature_lens = torch.tensor([f.shape[-1] for f in features], dtype=torch.long)
    token_lens = torch.tensor([t.shape[0] for t in token_ids], dtype=torch.long)

    # Pad features along time dimension (last dim)
    max_feat_len = feature_lens.max().item()
    n_mels = features[0].shape[0]
    padded_features = torch.zeros(len(features), n_mels, max_feat_len)
    for i, f in enumerate(features):
        padded_features[i, :, : f.shape[-1]] = f

    # Pad token sequences
    max_tok_len = token_lens.max().item()
    padded_tokens = torch.zeros(len(token_ids), max_tok_len, dtype=torch.long)
    for i, t in enumerate(token_ids):
        padded_tokens[i, : t.shape[0]] = t

    return padded_features, feature_lens, padded_tokens, token_lens

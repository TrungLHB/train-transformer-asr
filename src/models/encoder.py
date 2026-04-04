"""Conformer / Transformer encoder for ASR.

The encoder maps a sequence of log-mel filterbank frames to a sequence of
high-level representations.  Two variants are supported:

* ``"transformer"`` – standard Transformer encoder with positional encoding.
* ``"conformer"``  – Conformer encoder that combines self-attention with
  depth-wise convolution for better local/global modelling.

Reference (Conformer):
    Gulati et al., "Conformer: Convolution-augmented Transformer for Speech
    Recognition", Interspeech 2020.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Conformer building blocks
# ---------------------------------------------------------------------------

class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    Args:
        d_model: Model dimension.
        kernel_size: Depth-wise convolution kernel size (must be odd).
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model)
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)          # (B, d_model, T)
        x = self.pointwise_conv1(x)    # (B, 2*d_model, T)
        x = self.glu(x)                # (B, d_model, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)       # (B, T, d_model)


class FeedForwardModule(nn.Module):
    """Macaron-style feed-forward module (half residual weight)."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerBlock(nn.Module):
    """Single Conformer encoder block.

    Structure: FF → MHSA → Conv → FF → LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_dim, dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            src_key_padding_mask: (batch, seq_len) boolean mask; True = pad.
        Returns:
            x: (batch, seq_len, d_model)
        """
        x = self.ff1(x)

        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.self_attn_dropout(x)
        x = residual + x

        x = x + self.conv(x)
        x = self.ff2(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Standard Transformer encoder block
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        return residual + x


# ---------------------------------------------------------------------------
# Top-level Encoder
# ---------------------------------------------------------------------------

class ASREncoder(nn.Module):
    """Acoustic encoder: conv sub-sampling + stack of encoder blocks.

    Args:
        n_mels: Number of mel filterbank channels.
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
        conv_kernel_size: Conformer depth-wise conv kernel size.
        encoder_type: ``"conformer"`` or ``"transformer"``.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        encoder_type: str = "conformer",
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type

        # Conv sub-sampling: (B, 1, n_mels, T) -> (B, d_model, T//4)
        self.conv_subsample = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After 2× strided conv on mel axis: out_mels = ceil(n_mels / 4)
        subsampled_mels = math.ceil(n_mels / 4)
        self.input_proj = nn.Linear(256 * subsampled_mels, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        if encoder_type == "conformer":
            self.layers = nn.ModuleList(
                [
                    ConformerBlock(d_model, num_heads, ff_dim, conv_kernel_size, dropout)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)
                    for _ in range(num_layers)
                ]
            )

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features:     (B, n_mels, T) float tensor.
            feature_lens: (B,) integer lengths before padding (optional).

        Returns:
            enc_out:   (B, T', d_model) encoded representations.
            enc_lens:  (B,) integer lengths after sub-sampling.
        """
        # Add channel dim: (B, 1, n_mels, T)
        x = features.unsqueeze(1)
        x = self.conv_subsample(x)          # (B, 256, n_mels', T')
        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * M)  # (B, T', C*M)
        x = self.input_proj(x)              # (B, T', d_model)
        x = self.pos_enc(x)

        # Compute padding mask
        if feature_lens is not None:
            enc_lens = (feature_lens / 4).ceil().long()
            max_len = x.size(1)
            padding_mask = (
                torch.arange(max_len, device=x.device).unsqueeze(0) >= enc_lens.unsqueeze(1)
            )  # True = pad position
        else:
            enc_lens = torch.full((B,), x.size(1), dtype=torch.long, device=x.device)
            padding_mask = None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        return x, enc_lens

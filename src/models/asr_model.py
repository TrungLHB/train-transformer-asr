"""Combined ASR model: shared encoder with Transformer or CTC decoder."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import ASREncoder
from .ctc_decoder import CTCDecoder
from .transformer_decoder import TransformerDecoder


class CTCASRModel(nn.Module):
    """End-to-end CTC ASR model.

    Architecture: Conformer/Transformer encoder → CTC linear projection.

    Args:
        vocab_size:       Output vocabulary size.
        n_mels:           Number of mel filterbank channels.
        d_model:          Encoder/model dimension.
        num_heads:        Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        ff_dim:           Feed-forward hidden dimension.
        dropout:          Dropout probability.
        conv_kernel_size: Conformer depth-wise conv kernel size.
        encoder_type:     ``"conformer"`` or ``"transformer"``.
        blank_id:         CTC blank token id.
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        encoder_type: str = "conformer",
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = ASREncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            encoder_type=encoder_type,
        )
        self.ctc_decoder = CTCDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            blank_id=blank_id,
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features:     (B, n_mels, T) log-mel features.
            feature_lens: (B,) frame lengths before padding.

        Returns:
            log_probs: (T', B, vocab_size) CTC log-probabilities.
            enc_lens:  (B,) encoder output lengths.
        """
        enc_out, enc_lens = self.encoder(features, feature_lens)
        log_probs = self.ctc_decoder(enc_out)
        return log_probs, enc_lens

    def decode(
        self,
        features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        beam_size: int = 10,
    ):
        """Run encoder + CTC prefix beam search decoding.

        Args:
            features:     (B, n_mels, T) log-mel features.
            feature_lens: (B,) frame lengths before padding.
            beam_size:    Number of beams.

        Returns:
            List of B lists of predicted token ids.
        """
        enc_out, enc_lens = self.encoder(features, feature_lens)
        return self.ctc_decoder.prefix_beam_search(enc_out, enc_lens, beam_size)


class TransformerASRModel(nn.Module):
    """End-to-end attention-based sequence-to-sequence ASR model.

    Architecture: Conformer/Transformer encoder → Transformer decoder.

    Args:
        vocab_size:           Output vocabulary size.
        n_mels:               Number of mel filterbank channels.
        d_model:              Encoder/decoder model dimension.
        num_heads:            Number of attention heads.
        num_encoder_layers:   Number of encoder layers.
        num_decoder_layers:   Number of decoder layers.
        ff_dim:               Feed-forward hidden dimension.
        dropout:              Dropout probability.
        conv_kernel_size:     Conformer depth-wise conv kernel size.
        encoder_type:         ``"conformer"`` or ``"transformer"``.
        sos_id:               Start-of-sequence token id.
        eos_id:               End-of-sequence token id.
        max_target_length:    Maximum decoding length.
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        encoder_type: str = "conformer",
        sos_id: int = 2,
        eos_id: int = 1,
        max_target_length: int = 200,
    ) -> None:
        super().__init__()
        self.encoder = ASREncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            encoder_type=encoder_type,
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            sos_id=sos_id,
            eos_id=eos_id,
            max_target_length=max_target_length,
        )

    def _make_memory_padding_mask(
        self, enc_lens: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        return (
            torch.arange(max_len, device=enc_lens.device).unsqueeze(0)
            >= enc_lens.unsqueeze(1)
        )

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        target_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        Args:
            features:     (B, n_mels, T) log-mel features.
            targets:      (B, U) token ids with leading <sos>.
            feature_lens: (B,) frame lengths before padding.
            target_lens:  (B,) target lengths (including <sos>).

        Returns:
            logits: (B, U, vocab_size)
        """
        enc_out, enc_lens = self.encoder(features, feature_lens)
        mem_mask = self._make_memory_padding_mask(enc_lens, enc_out.size(1))
        return self.decoder(targets, enc_out, target_lens, mem_mask)

    def decode(
        self,
        features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        length_penalty: float = 1.0,
    ):
        """Encode features and run beam search decoding.

        Args:
            features:     (B, n_mels, T) log-mel features (B=1 supported).
            feature_lens: (B,) frame lengths before padding.
            beam_size:    Number of beams.
            length_penalty: Beam search length penalty exponent.

        Returns:
            List of B lists of predicted token ids.
        """
        enc_out, enc_lens = self.encoder(features, feature_lens)
        results = []
        for b in range(enc_out.size(0)):
            mem = enc_out[b : b + 1]
            elen = enc_lens[b : b + 1]
            mem_mask = self._make_memory_padding_mask(elen, mem.size(1))
            ids = self.decoder.beam_search(
                mem, mem_mask, beam_size=beam_size, length_penalty=length_penalty
            )
            results.append(ids)
        return results


def build_model(cfg, vocab_size: int) -> nn.Module:
    """Instantiate the correct model from a Hydra/OmegaConf config.

    Args:
        cfg:        OmegaConf DictConfig (see ``configs/``).
        vocab_size: Target vocabulary size.

    Returns:
        An :class:`CTCASRModel` or :class:`TransformerASRModel` instance.
    """
    enc = cfg.encoder
    dec = cfg.decoder

    if dec.type == "ctc":
        return CTCASRModel(
            vocab_size=vocab_size,
            n_mels=cfg.audio.n_mels,
            d_model=enc.d_model,
            num_heads=enc.num_heads,
            num_encoder_layers=enc.num_layers,
            ff_dim=enc.ff_dim,
            dropout=enc.dropout,
            conv_kernel_size=enc.conv_kernel_size,
            encoder_type=enc.encoder_type,
            blank_id=dec.blank_id,
        )
    elif dec.type == "transformer":
        return TransformerASRModel(
            vocab_size=vocab_size,
            n_mels=cfg.audio.n_mels,
            d_model=enc.d_model,
            num_heads=enc.num_heads,
            num_encoder_layers=enc.num_layers,
            num_decoder_layers=dec.num_layers,
            ff_dim=enc.ff_dim,
            dropout=enc.dropout,
            conv_kernel_size=enc.conv_kernel_size,
            encoder_type=enc.encoder_type,
            sos_id=dec.sos_id,
            eos_id=dec.eos_id,
            max_target_length=dec.max_target_length,
        )
    else:
        raise ValueError(f"Unknown decoder type: {dec.type!r}")

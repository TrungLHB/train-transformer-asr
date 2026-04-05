"""Attention-based sequence-to-sequence (Transformer) decoder for ASR.

The decoder attends over encoder outputs to generate token sequences
autoregressively, using teacher forcing during training and beam search
during inference.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PositionalEncoding


class TransformerDecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block with masked self-attention and
    cross-attention over encoder outputs."""

    def __init__(
        self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:                     (B, U, d_model) decoder input.
            memory:                  (B, T, d_model) encoder output.
            tgt_mask:                (U, U) causal mask.
            tgt_key_padding_mask:    (B, U) padding mask for decoder input.
            memory_key_padding_mask: (B, T) padding mask for encoder output.

        Returns:
            (B, U, d_model)
        """
        residual = tgt
        x = self.norm1(tgt)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, memory, memory,
                                key_padding_mask=memory_key_padding_mask)
        x = self.dropout2(x)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        return residual + x


class TransformerDecoder(nn.Module):
    """Full autoregressive Transformer decoder.

    Args:
        vocab_size: Target vocabulary size.
        d_model:    Model dimension (must match encoder).
        num_heads:  Number of attention heads.
        num_layers: Number of decoder blocks.
        ff_dim:     Feed-forward hidden dimension.
        dropout:    Dropout probability.
        sos_id:     Start-of-sequence token id.
        eos_id:     End-of-sequence token id.
        max_target_length: Maximum decoding steps.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        sos_id: int = 2,
        eos_id: int = 1,
        max_target_length: int = 200,
    ) -> None:
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_target_length = max_target_length
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        targets: torch.Tensor,
        memory: torch.Tensor,
        target_lens: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        Args:
            targets:   (B, U) integer token ids (including leading <sos>).
            memory:    (B, T, d_model) encoder output.
            target_lens: (B,) lengths for target padding mask.
            memory_key_padding_mask: (B, T) bool mask for encoder output.

        Returns:
            logits: (B, U, vocab_size)
        """
        U = targets.size(1)
        causal_mask = self._make_causal_mask(U, targets.device)

        tgt_key_padding_mask = None
        if target_lens is not None:
            tgt_key_padding_mask = (
                torch.arange(U, device=targets.device).unsqueeze(0) >= target_lens.unsqueeze(1)
            )

        x = self.embed(targets) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        x = self.final_norm(x)
        return self.output_proj(x)  # (B, U, vocab_size)

    @torch.no_grad()
    def beam_search(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        length_penalty: float = 1.0,
    ) -> List[List[int]]:
        """Greedy / beam search decoding (batch_size=1 for simplicity).

        Args:
            memory:   (1, T, d_model) encoder output.
            memory_key_padding_mask: (1, T) bool mask.
            beam_size: Number of beams.
            length_penalty: Exponent applied to sequence length.

        Returns:
            best_sequence: List of predicted token ids (excluding <sos>/<eos>).
        """
        device = memory.device
        B = memory.size(0)
        assert B == 1, "beam_search currently supports batch_size=1 only"

        # Each beam: (score, [token_ids], done)
        beams = [(0.0, [self.sos_id], False)]

        for _ in range(self.max_target_length):
            if all(done for _, _, done in beams):
                break

            candidates = []
            for score, ids, done in beams:
                if done:
                    candidates.append((score, ids, True))
                    continue

                tgt = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = self.forward(tgt, memory,
                                      memory_key_padding_mask=memory_key_padding_mask)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (1, vocab)
                topk_scores, topk_ids = log_probs[0].topk(beam_size)

                for s, tok in zip(topk_scores.tolist(), topk_ids.tolist()):
                    new_ids = ids + [tok]
                    new_score = score + s
                    is_done = tok == self.eos_id
                    candidates.append((new_score, new_ids, is_done))

            # Keep top-k beams; apply length penalty for finished beams
            candidates.sort(
                key=lambda c: c[0] / (len(c[1]) ** length_penalty), reverse=True
            )
            beams = candidates[:beam_size]

        best_ids = beams[0][1][1:]  # strip <sos>
        if best_ids and best_ids[-1] == self.eos_id:
            best_ids = best_ids[:-1]
        return best_ids

    @torch.no_grad()
    def batched_greedy_decode(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Fast batched greedy decoding without beam branching."""
        B = memory.size(0)
        device = memory.device
        
        # Start with <sos> for all sequences in batch
        tgt = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_target_length):
            logits = self.forward(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
            next_toks = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_toks], dim=1)

            finished |= (next_toks.squeeze(1) == self.eos_id)
            if finished.all():
                break

        results = []
        for b in range(B):
            row_ids = tgt[b, 1:].tolist()  # strip <sos>
            if self.eos_id in row_ids:
                row_ids = row_ids[:row_ids.index(self.eos_id)]
            results.append(row_ids)
            
        return results

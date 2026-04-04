"""CTC (Connectionist Temporal Classification) decoder for ASR.

The CTC decoder is a simple linear projection from encoder output to
vocabulary logits, trained with the CTC loss.  At inference time, prefix
beam search is used to decode the best hypothesis.

Reference:
    Graves et al., "Connectionist Temporal Classification: Labelling
    Unsegmented Sequence Data with Recurrent Neural Networks", ICML 2006.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCDecoder(nn.Module):
    """CTC decoder: linear projection from encoder hidden states to vocab.

    Args:
        vocab_size: Size of the output vocabulary (includes blank token).
        d_model:    Model dimension (must match encoder output dimension).
        blank_id:   Token id for the CTC blank symbol (default: 0).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.blank_id = blank_id
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_out: (B, T, d_model) encoder output.

        Returns:
            log_probs: (T, B, vocab_size) log-probabilities required by
                       ``torch.nn.CTCLoss``.
        """
        logits = self.output_proj(enc_out)          # (B, T, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)   # (B, T, vocab_size)
        return log_probs.permute(1, 0, 2)           # (T, B, vocab_size)

    @torch.no_grad()
    def greedy_decode(
        self,
        enc_out: torch.Tensor,
        enc_lens: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Greedy CTC decoding (collapse repeats, remove blanks).

        Args:
            enc_out:  (B, T, d_model) encoder output.
            enc_lens: (B,) valid frame lengths.

        Returns:
            decoded: List of B lists of token ids.
        """
        logits = self.output_proj(enc_out)           # (B, T, vocab)
        best_ids = logits.argmax(dim=-1)             # (B, T)

        results = []
        for b in range(best_ids.size(0)):
            length = enc_lens[b].item() if enc_lens is not None else best_ids.size(1)
            ids = best_ids[b, :length].tolist()

            # Collapse repeats and remove blank
            collapsed = []
            prev = None
            for tok in ids:
                if tok != prev:
                    if tok != self.blank_id:
                        collapsed.append(tok)
                    prev = tok
            results.append(collapsed)
        return results

    @torch.no_grad()
    def prefix_beam_search(
        self,
        enc_out: torch.Tensor,
        enc_lens: Optional[torch.Tensor] = None,
        beam_size: int = 10,
    ) -> List[List[int]]:
        """CTC prefix beam search decoding.

        Implements the standard CTC prefix beam search algorithm that
        maintains a dictionary of (prefix, (p_b, p_nb)) pairs, where
        p_b is the probability of the prefix ending in blank and p_nb
        is the probability ending in a non-blank symbol.

        Args:
            enc_out:  (B, T, d_model) encoder output.
            enc_lens: (B,) valid frame lengths.
            beam_size: Number of beams to keep.

        Returns:
            decoded: List of B lists of token ids (best hypothesis each).
        """
        B, T_max, _ = enc_out.shape
        logits = self.output_proj(enc_out)                    # (B, T, vocab)
        log_probs = F.log_softmax(logits, dim=-1)             # (B, T, vocab)
        probs = log_probs.exp()                               # (B, T, vocab)
        results = []

        NEG_INF = float("-inf")

        for b in range(B):
            length = int(enc_lens[b].item()) if enc_lens is not None else T_max
            frame_probs = probs[b, :length, :].cpu()          # (length, vocab)

            # beams: dict mapping prefix tuple -> (p_blank, p_nonblank)
            beams = {(): (1.0, 0.0)}

            for t in range(length):
                p = frame_probs[t]                            # (vocab,)
                new_beams: dict = {}

                for prefix, (p_b, p_nb) in beams.items():
                    p_total = p_b + p_nb

                    # Extend with blank
                    key = prefix
                    pb_new, pnb_new = new_beams.get(key, (0.0, 0.0))
                    pb_new += p_total * p[self.blank_id].item()
                    new_beams[key] = (pb_new, pnb_new)

                    # Extend with each non-blank token
                    for tok in range(p.size(0)):
                        if tok == self.blank_id:
                            continue
                        tok_prob = p[tok].item()
                        new_prefix = prefix + (tok,)

                        pb_new2, pnb_new2 = new_beams.get(new_prefix, (0.0, 0.0))
                        if prefix and prefix[-1] == tok:
                            # Same token: only blank can "separate" them
                            pnb_new2 += p_b * tok_prob
                        else:
                            pnb_new2 += p_total * tok_prob
                        new_beams[new_prefix] = (pb_new2, pnb_new2)

                # Prune to top-k beams
                beams = dict(
                    sorted(
                        new_beams.items(),
                        key=lambda x: x[1][0] + x[1][1],
                        reverse=True,
                    )[:beam_size]
                )

            best_prefix = max(beams, key=lambda k: sum(beams[k]))
            results.append(list(best_prefix))

        return results

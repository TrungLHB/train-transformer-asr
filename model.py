"""ASR model: Transformer encoder-decoder with optional CTC auxiliary loss."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AudioEncoder(nn.Module):
    """Two-layer Conv sub-sampling followed by a Transformer encoder."""

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        # Sub-sampling: reduce time dimension by 4×
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        conv_out_dim = 32 * math.ceil(n_mels / 4)
        self.linear = nn.Linear(conv_out_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Log-mel spectrogram (batch, n_mels, time)
            lengths: Original (unpadded) time lengths (batch,)
        Returns:
            encoder_out: (batch, time', d_model)
            out_lengths: (batch,) lengths after sub-sampling
        """
        # (batch, 1, n_mels, time)
        x = x.unsqueeze(1)
        x = self.conv(x)
        # (batch, channels, freq, time') → (batch, time', channels * freq)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.linear(x)
        # Positional encoding expects (seq, batch, d_model); batch_first=True transformer is fine
        out_lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
        out_lengths = torch.div(out_lengths - 1, 2, rounding_mode="floor") + 1

        key_padding_mask = self._make_key_padding_mask(out_lengths, t)
        encoder_out = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        return encoder_out, out_lengths

    @staticmethod
    def _make_key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """True where position is a padding token (ignored by attention)."""
        batch = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch, -1)
        return mask >= lengths.unsqueeze(1)


class TransformerASR(nn.Module):
    """
    Transformer encoder-decoder ASR model.

    Supports two decoding strategies:
    - Autoregressive (transformer decoder cross-attending encoder output)
    - CTC (linear projection of encoder output, used as auxiliary or standalone loss)
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 256,
        n_heads: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        ctc_weight: float = 0.3,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        blank_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.ctc_weight = ctc_weight
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.blank_id = blank_id

        self.encoder = AudioEncoder(
            n_mels=n_mels,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # CTC head
        self.ctc_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            audio: (batch, n_mels, time)
            audio_lengths: (batch,)
            targets: (batch, tgt_len) token ids, including sos/eos
            target_lengths: (batch,) unpadded lengths
        Returns:
            Dictionary with keys: loss, ce_loss, ctc_loss, logits
        """
        encoder_out, enc_lengths = self.encoder(audio, audio_lengths)

        # ── Cross-entropy (autoregressive decoder) loss ──────────────────────
        tgt_input = targets[:, :-1]   # drop last token (eos)
        tgt_output = targets[:, 1:]   # drop first token (sos)

        tgt_emb = self.embedding(tgt_input) * math.sqrt(self.d_model)
        tgt_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=audio.device)
        tgt_key_padding = tgt_input == self.pad_id

        enc_key_padding = AudioEncoder._make_key_padding_mask(enc_lengths, encoder_out.size(1))

        decoder_out = self.transformer_decoder(
            tgt_emb,
            encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=enc_key_padding,
        )
        logits = self.output_projection(decoder_out)  # (batch, tgt_len-1, vocab)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_id,
        )

        # ── CTC loss ─────────────────────────────────────────────────────────
        ctc_log_probs = F.log_softmax(self.ctc_projection(encoder_out), dim=-1)
        # CTC expects (time, batch, vocab)
        ctc_log_probs = ctc_log_probs.permute(1, 0, 2)
        # Strip sos/eos from targets for CTC
        ctc_targets = tgt_output  # targets without sos; CTC uses non-padded lengths
        ctc_target_lengths = target_lengths - 1  # subtract sos
        ctc_loss = F.ctc_loss(
            ctc_log_probs,
            ctc_targets,
            enc_lengths,
            ctc_target_lengths,
            blank=self.blank_id,
            zero_infinity=True,
        )

        loss = (1 - self.ctc_weight) * ce_loss + self.ctc_weight * ctc_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "ctc_loss": ctc_loss,
            "logits": logits,
        }

    @torch.no_grad()
    def decode_greedy(self, audio: torch.Tensor, audio_lengths: torch.Tensor, max_len: int = 200) -> list[list[int]]:
        """Greedy autoregressive decoding."""
        encoder_out, enc_lengths = self.encoder(audio, audio_lengths)
        batch = audio.size(0)
        device = audio.device

        ys = torch.full((batch, 1), self.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.embedding(ys) * math.sqrt(self.d_model)
            tgt_len = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            enc_key_padding = AudioEncoder._make_key_padding_mask(enc_lengths, encoder_out.size(1))
            out = self.transformer_decoder(tgt_emb, encoder_out, tgt_mask=tgt_mask, memory_key_padding_mask=enc_key_padding)
            next_token = self.output_projection(out[:, -1, :]).argmax(-1)  # (batch,)
            finished |= next_token == self.eos_id
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        sequences = []
        for i in range(batch):
            tokens = ys[i, 1:].tolist()  # strip sos
            if self.eos_id in tokens:
                tokens = tokens[: tokens.index(self.eos_id)]
            sequences.append(tokens)
        return sequences

    @torch.no_grad()
    def decode_ctc_greedy(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> list[list[int]]:
        """CTC greedy decoding (best-path, collapse repeats, remove blank)."""
        encoder_out, _ = self.encoder(audio, audio_lengths)
        log_probs = F.log_softmax(self.ctc_projection(encoder_out), dim=-1)
        best_paths = log_probs.argmax(-1)  # (batch, time)

        sequences = []
        for path in best_paths.tolist():
            tokens = []
            prev = self.blank_id
            for t in path:
                if t != self.blank_id and t != prev:
                    tokens.append(t)
                prev = t
            sequences.append(tokens)
        return sequences

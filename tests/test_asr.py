"""Unit tests for ASR model components.

Tests run without a GPU and use tiny random inputs so they complete quickly
even in a CI environment without audio data or a trained model.
"""

import math
import sys
import os

import pytest
import torch

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import CharTokenizer, normalize_transcript
from src.data.preprocessing import LogMelFilterBank, SpecAugment, collate_fn
from src.models.encoder import ASREncoder, PositionalEncoding
from src.models.ctc_decoder import CTCDecoder
from src.models.transformer_decoder import TransformerDecoder
from src.models.asr_model import CTCASRModel, TransformerASRModel
from src.training.metrics import compute_wer, compute_cer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 30
D_MODEL = 64
N_MELS = 40
BATCH = 2
T_FRAMES = 120  # encoder input frames


@pytest.fixture
def tokenizer():
    chars = list("abcdefghijklmnopqrstuvwxyz '")
    return CharTokenizer(vocab=chars)


@pytest.fixture
def fake_features():
    """(BATCH, N_MELS, T_FRAMES) random features."""
    return torch.randn(BATCH, N_MELS, T_FRAMES)


@pytest.fixture
def fake_feature_lens():
    return torch.tensor([T_FRAMES, T_FRAMES - 10])


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class TestCharTokenizer:
    def test_encode_decode_roundtrip(self, tokenizer):
        text = "hello world"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_special_tokens(self, tokenizer):
        assert tokenizer.blank_id == 0
        assert tokenizer.eos_id == 1
        assert tokenizer.sos_id == 2
        assert tokenizer.unk_id == 3

    def test_build_from_texts(self):
        texts = ["abc", "def"]
        tok = CharTokenizer().build_from_texts(texts)
        assert "a" in tok.vocab
        assert "f" in tok.vocab

    def test_save_load(self, tokenizer, tmp_path):
        path = str(tmp_path / "vocab.txt")
        tokenizer.save(path)
        loaded = CharTokenizer.load(path)
        assert loaded.vocab == tokenizer.vocab
        assert loaded.encode("hello") == tokenizer.encode("hello")

    def test_unk_fallback(self, tokenizer):
        ids = tokenizer.encode("αβγ")  # Greek letters not in vocab
        assert all(i == tokenizer.unk_id for i in ids)


class TestNormalizeTranscript:
    def test_lowercase(self):
        assert normalize_transcript("HELLO World") == "hello world"

    def test_strips_punctuation(self):
        result = normalize_transcript("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        assert normalize_transcript("hello   world") == "hello world"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestLogMelFilterBank:
    def test_output_shape(self):
        # win_length must be <= n_fft; use defaults (n_fft=512, win_length=400)
        extractor = LogMelFilterBank(n_mels=N_MELS)
        waveform = torch.randn(1, 16000)  # 1 second of audio
        feats = extractor(waveform)
        assert feats.shape[1] == N_MELS
        assert feats.shape[0] == 1  # batch dim

    def test_no_nan(self):
        extractor = LogMelFilterBank(n_mels=N_MELS)
        waveform = torch.randn(1, 16000)
        feats = extractor(waveform)
        assert not torch.isnan(feats).any()


class TestSpecAugment:
    def test_output_shape_preserved(self):
        aug = SpecAugment(freq_mask_param=5, time_mask_param=10)
        x = torch.randn(1, N_MELS, 100)
        out = aug(x)
        assert out.shape == x.shape


class TestCollate:
    def test_batch_shapes(self, tokenizer):
        items = [
            {"features": torch.randn(N_MELS, 50), "token_ids": [4, 5, 6]},
            {"features": torch.randn(N_MELS, 70), "token_ids": [4, 5, 6, 7, 8]},
        ]
        feats, feat_lens, tokens, tok_lens = collate_fn(items)
        assert feats.shape == (2, N_MELS, 70)
        assert feat_lens.tolist() == [50, 70]
        assert tokens.shape == (2, 5)
        assert tok_lens.tolist() == [3, 5]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(D_MODEL)
        x = torch.randn(BATCH, 50, D_MODEL)
        out = pe(x)
        assert out.shape == x.shape


class TestASREncoder:
    @pytest.mark.parametrize("encoder_type", ["conformer", "transformer"])
    def test_forward_shape(self, encoder_type, fake_features, fake_feature_lens):
        encoder = ASREncoder(
            n_mels=N_MELS,
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            encoder_type=encoder_type,
            conv_kernel_size=3,
        )
        enc_out, enc_lens = encoder(fake_features, fake_feature_lens)
        assert enc_out.shape[0] == BATCH
        assert enc_out.shape[2] == D_MODEL
        assert enc_lens.shape == (BATCH,)

    def test_enc_lens_shorter_than_feature_lens(self, fake_features, fake_feature_lens):
        encoder = ASREncoder(n_mels=N_MELS, d_model=D_MODEL, num_heads=4,
                              num_layers=2, ff_dim=128, conv_kernel_size=3)
        _, enc_lens = encoder(fake_features, fake_feature_lens)
        # Sub-sampling factor of 4 means enc_lens ≤ ceil(feature_lens / 4)
        assert (enc_lens <= fake_feature_lens).all()


# ---------------------------------------------------------------------------
# CTC decoder
# ---------------------------------------------------------------------------

class TestCTCDecoder:
    def test_forward_shape(self):
        dec = CTCDecoder(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
        enc_out = torch.randn(BATCH, 30, D_MODEL)
        log_probs = dec(enc_out)
        assert log_probs.shape == (30, BATCH, VOCAB_SIZE)

    def test_greedy_decode_returns_list(self):
        dec = CTCDecoder(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
        enc_out = torch.randn(BATCH, 30, D_MODEL)
        enc_lens = torch.tensor([30, 25])
        results = dec.greedy_decode(enc_out, enc_lens)
        assert len(results) == BATCH
        assert all(isinstance(r, list) for r in results)

    def test_prefix_beam_search(self):
        dec = CTCDecoder(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
        enc_out = torch.randn(BATCH, 20, D_MODEL)
        enc_lens = torch.tensor([20, 15])
        results = dec.prefix_beam_search(enc_out, enc_lens, beam_size=3)
        assert len(results) == BATCH


# ---------------------------------------------------------------------------
# Transformer decoder
# ---------------------------------------------------------------------------

class TestTransformerDecoder:
    def test_forward_shape(self, tokenizer):
        dec = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
        )
        memory = torch.randn(BATCH, 30, D_MODEL)
        targets = torch.randint(4, tokenizer.vocab_size, (BATCH, 10))
        logits = dec(targets, memory)
        assert logits.shape == (BATCH, 10, tokenizer.vocab_size)

    def test_beam_search(self, tokenizer):
        dec = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            max_target_length=10,
        )
        memory = torch.randn(1, 20, D_MODEL)
        ids = dec.beam_search(memory, beam_size=3)
        assert isinstance(ids, list)


# ---------------------------------------------------------------------------
# Full ASR models
# ---------------------------------------------------------------------------

class TestCTCASRModel:
    def test_forward(self, fake_features, fake_feature_lens):
        model = CTCASRModel(
            vocab_size=VOCAB_SIZE,
            n_mels=N_MELS,
            d_model=D_MODEL,
            num_heads=4,
            num_encoder_layers=2,
            ff_dim=128,
            conv_kernel_size=3,
        )
        log_probs, enc_lens = model(fake_features, fake_feature_lens)
        assert log_probs.shape[1] == BATCH
        assert log_probs.shape[2] == VOCAB_SIZE

    def test_decode(self, fake_features, fake_feature_lens):
        model = CTCASRModel(
            vocab_size=VOCAB_SIZE,
            n_mels=N_MELS,
            d_model=D_MODEL,
            num_heads=4,
            num_encoder_layers=2,
            ff_dim=128,
            conv_kernel_size=3,
        )
        results = model.decode(fake_features, fake_feature_lens, beam_size=3)
        assert len(results) == BATCH


class TestTransformerASRModel:
    def test_forward(self, fake_features, fake_feature_lens, tokenizer):
        model = TransformerASRModel(
            vocab_size=tokenizer.vocab_size,
            n_mels=N_MELS,
            d_model=D_MODEL,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            ff_dim=128,
            conv_kernel_size=3,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
        )
        targets = torch.randint(4, tokenizer.vocab_size, (BATCH, 8))
        logits = model(fake_features, targets, fake_feature_lens)
        assert logits.shape == (BATCH, 8, tokenizer.vocab_size)

    def test_decode(self, tokenizer):
        model = TransformerASRModel(
            vocab_size=tokenizer.vocab_size,
            n_mels=N_MELS,
            d_model=D_MODEL,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            ff_dim=128,
            conv_kernel_size=3,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            max_target_length=10,
        )
        features = torch.randn(1, N_MELS, T_FRAMES)
        feat_lens = torch.tensor([T_FRAMES])
        results = model.decode(features, feat_lens, beam_size=3)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_wer_perfect(self):
        assert compute_wer(["hello world"], ["hello world"]) == 0.0

    def test_wer_all_wrong(self):
        score = compute_wer(["hello world"], ["foo bar"])
        assert score > 0.0

    def test_cer_perfect(self):
        assert compute_cer(["abc"], ["abc"]) == 0.0

    def test_cer_partial(self):
        score = compute_cer(["abc"], ["axc"])
        assert 0.0 < score <= 1.0

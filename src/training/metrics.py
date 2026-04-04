"""WER / CER metrics for ASR evaluation."""

from typing import List

from jiwer import wer, cer


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Compute Word Error Rate.

    Args:
        references: List of ground-truth transcripts.
        hypotheses: List of decoded transcripts.

    Returns:
        WER as a float in [0, 1].
    """
    return wer(references, hypotheses)


def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """Compute Character Error Rate.

    Args:
        references: List of ground-truth transcripts.
        hypotheses: List of decoded transcripts.

    Returns:
        CER as a float in [0, 1].
    """
    return cer(references, hypotheses)

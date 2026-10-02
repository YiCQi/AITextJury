"""End-to-end tests for the local-LM detector path.

Marked `ml` — they need torch + transformers and are gated behind
`pytest --runml` (they do real model loads/forwards; the very first run
also downloads gpt2, which is slow on CPU). Without --runml or without
torch these are skipped with a clear reason.
"""
import asyncio

import pytest

from zeroaibench.detectors.base import AnalysisContext
from zeroaibench.detectors.lm_common import HUB, tokenize_for_heatmap
from zeroaibench.segmenter import segment_text

pytestmark = pytest.mark.ml

AI_TEXT = (
    "In the rapidly evolving landscape of modern technology, it is important "
    "to note that artificial intelligence has fundamentally transformed the "
    "way we work. Moreover, it is worth mentioning that these systems "
    "continue to improve. It is worth noting that the implications are vast."
)


def _ctx(text: str) -> AnalysisContext:
    return AnalysisContext(
        text=text,
        segmentation=segment_text(text),
        providers=None,
        settings={},
    )


def test_tokenize_for_heatmap_offsets_are_absolute():
    tt = tokenize_for_heatmap("gpt2", AI_TEXT,
                              _ctx(AI_TEXT).segmentation.sentences, 512)
    assert tt.input_ids, "tokenizer returned no tokens"
    start, end = tt.offsets[0]
    assert AI_TEXT[start:end].strip(), "offsets must be absolute char spans"
    assert "gpt2" in tt.model_id


def test_lm_perplexity_end_to_end():
    from zeroaibench.detectors.lm_perplexity import LMPerplexityDetector

    outcome = asyncio.run(LMPerplexityDetector().analyze(_ctx(AI_TEXT)))
    assert outcome.raw_score is not None
    assert outcome.raw_direction == "lower_is_ai"
    assert outcome.signals["perplexity"] > 0
    assert outcome.signals["tokens_scored"] > 0


def test_binoculars_signals_carry_model_metadata():
    """Binoculars tags signals with *model names* — the wire schema must
    accept non-numeric SignalValue (regression guard for the 422 we hit
    when fast_detect_gpt emitted {"model": "gpt2"})."""
    from zeroaibench.detectors.binoculars import BinocularsDetector

    outcome = asyncio.run(BinocularsDetector().analyze(_ctx(AI_TEXT)))
    assert outcome.raw_score is not None
    assert outcome.raw_direction == "lower_is_ai"
    assert isinstance(outcome.signals["performer_model"], str)
    assert isinstance(outcome.signals["observer_model"], str)


def test_signal_value_schema_accepts_mixed_types():
    from zeroaibench.schemas import DetectorResult

    res = DetectorResult(
        detector_id="t", name="t", family="stat",
        raw_score=4.5, raw_direction="lower_is_ai",
        signals={"perplexity": 13.25, "model": "gpt2",
                 "reference_used": True, "note": None})
    dumped = res.model_dump(mode="json")
    assert dumped["signals"]["model"] == "gpt2"
    assert dumped["signals"]["reference_used"] is True

"""Example third-party plugin — demonstrates the AITextJury plugin contract.

A plugin is any Python file exposing:

    register(registry)  ── calls registry.register(MyDetector())

Place plugins either in `plugins/` (repo-root, bundled examples) or in
`data/plugins/` (user directory, gitignored) — both are scanned at startup.

This one ("LengthRhythm") is intentionally simple: it measures only how
*mechanically even* the paragraph lengths are. That is by design — plugins
should be narrow, explainable units of evidence rather than black boxes.
"""
from __future__ import annotations

import math

from aitextjury.detectors.base import AnalysisContext, BaseDetector, \
    DetectorError, RawOutcome, RawSegment
from aitextjury.schemas import Availability


class LengthRhythmDetector(BaseDetector):
    id = "length_rhythm"
    name = "Length Rhythm (plugin example)"
    family = "plugin"
    description = (
        "Example plugin: how rhythmically even the paragraph lengths are. "
        "Models that answer with template prose tend to emit blocks of "
        "very similar size; humans lurch. Deliberately single-feature.")
    link = "https://github.com/YiCQi/AITextJury/blob/main/docs/DETECTOR_API.md"
    requires: list[str] = []
    default_enabled = False
    DEFAULT_BANDS = (0.55, 0.18)

    def raw_direction(self) -> str:
        return "higher_is_ai"

    def availability(self, ctx: AnalysisContext | None = None) -> Availability:
        return Availability(ok=True)

    async def analyze(self, ctx: AnalysisContext) -> RawOutcome:
        paras = ctx.segmentation.paragraphs
        if len(paras) < 2:
            raise DetectorError("need ≥2 paragraphs")
        lens = [len(p.text.split()) or 1 for p in paras]
        mean = sum(lens) / len(lens)
        var = sum((x - mean) ** 2 for x in lens) / (len(lens) - 1)
        cv = math.sqrt(var) / mean
        # evenness = 1 - cv mapped through a soft curve; even -> high
        raw = 1.0 / (1.0 + math.exp((cv - 0.30) * 8))
        seg = [RawSegment(segment_id=p.id, value=raw) for p in paras]
        return RawOutcome(
            raw_score=raw,
            raw_direction="higher_is_ai",
            signals={"para_len_mean": mean, "para_len_cv": cv},
            segment_scores=seg,
            evidence=[],
            model="length-rhythm-v1",
            reference_note="Bundled plugin example — an invitation, not a "
                           "verdict. Use it to learn the detector API.")


def register(registry) -> None:
    registry.register(LengthRhythmDetector())

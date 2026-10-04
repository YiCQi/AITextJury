"""Detector base classes — the heart of the unified Detector API.

Any detector — built-in or third-party plugin — subclasses `BaseDetector`,
reports its `availability()`, and implements `analyze()` returning a
`RawOutcome`. Normalization / verdict / confidence are applied *centrally*
in `finalize()` using either a fitted calibration (see calibration.py) or the
detector's documented default bands, so that every score displayed anywhere is
comparable P(AI)-style probability.

Design notes:
* Detectors must never hard-fail the whole run: they raise `DetectorError`
  and the engine turns that into a per-detector result with an error message.
* Heavy sync work (torch) should run via `asyncio.to_thread`.
* Raw score carries an explicit direction documented by the detector author.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Availability, CalibrationInfo, DetectorResult, \
    EvidenceItem, SegmentScore, Verdict


class DetectorError(RuntimeError):
    """Raised by detectors when they cannot produce a result."""


@dataclass
class RawSegment:
    """A detector-native score for a segment (sentence or paragraph)."""
    segment_id: str
    value: float
    extras: dict = field(default_factory=dict)


@dataclass
class RawOutcome:
    """What a detector returns before standardization.

    `signals` is a free-form bag: mostly numeric statistics, but metadata
    entries (which model scored the text...) are legal too — see
    schemas.SignalValue. Cached values round-trip through JSON as-is.
    """
    raw_score: float | None
    raw_direction: str        # "higher_is_ai" | "lower_is_ai"
    signals: dict[str, Any] = field(default_factory=dict)
    segment_scores: list[RawSegment] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    model: str | None = None
    reference_note: str | None = None

    def to_cache(self) -> dict:
        return {
            "raw_score": self.raw_score,
            "raw_direction": self.raw_direction,
            "signals": self.signals,
            "segment_scores": [
                {"segment_id": s.segment_id, "value": s.value, "extras": s.extras}
                for s in self.segment_scores
            ],
            "evidence": [e.model_dump() for e in self.evidence],
            "model": self.model,
            "reference_note": self.reference_note,
        }

    @classmethod
    def from_cache(cls, d: dict) -> "RawOutcome":
        return cls(
            raw_score=d.get("raw_score"),
            raw_direction=d.get("raw_direction", "higher_is_ai"),
            signals=d.get("signals", {}),
            segment_scores=[
                RawSegment(s["segment_id"], s["value"], s.get("extras", {}))
                for s in d.get("segment_scores", [])
            ],
            evidence=[EvidenceItem(**e) for e in d.get("evidence", [])],
            model=d.get("model"),
            reference_note=d.get("reference_note"),
        )


class CalibrationFitter(abc.ABC):
    """Interface the shared ml/detector calibration store provides."""

    def get(self, detector_id: str) -> dict | None: ...


class AnalysisContext:
    """Everything a detector may need, passed in by the engine."""

    def __init__(self, *, text: str, segmentation, providers,
                 settings: dict, get_detector_settings: dict = None,
                 calibration: CalibrationFitter | None = None,
                 data_dir=None, language: str = "en"):
        self.text = text
        self.segmentation = segmentation
        self.providers = providers
        self.settings = settings          # global detect settings
        self._detector_settings = get_detector_settings or {}
        self.calibration = calibration
        self.data_dir = data_dir
        self.language = language

    def det_settings(self, detector_id: str) -> dict:
        return dict(self._detector_settings.get(detector_id, {}))


class BaseDetector(abc.ABC):
    # --- identity -----------------------------------------------------------
    id: str = "base"
    name: str = "Base"
    family: str = "plugin"
    description: str = ""
    link: str | None = None            # paper / repo
    requires: list[str] = []           # pip packages (informational)
    default_enabled: bool = True
    heavy: bool = False                # slower / downloads models

    # Default normalization bands used until a calibration fit exists.
    #   mid: raw value separating human|AI
    #   slope: sigmoid temperature
    # Smaller |slope| -> sharper transitions.
    DEFAULT_BANDS: tuple[float, float] = (0.5, 0.2)

    # ------------------------------------------------------------- abstract --

    @abc.abstractmethod
    def availability(self, ctx: AnalysisContext | None = None) -> Availability:
        """Cheap check: installed deps, configured keys, etc."""

    @abc.abstractmethod
    async def analyze(self, ctx: AnalysisContext) -> RawOutcome:
        """Run the detector. Raise DetectorError on recoverable failure."""

    # ------------------------------------------------------------- helpers --

    def bands_help(self) -> str:
        mid, slope = self.DEFAULT_BANDS
        return (f"uncalibrated default bands: mid={mid}, slope={slope} "
                f"— run /api/calibration to fit on a labeled set")

    # ---------------------------------------------------------- finalize ----

    def _map_raw(self, raw: float, fit: dict | None) -> float:
        """raw -> normalized P(AI)-style probability in [0,1]."""
        import math
        if fit and fit.get("ok"):
            mu, sd = fit["mu"], max(fit["sd"], 1e-6)
            w, b = fit["w"], fit["b"]
            z = (raw - mu) / sd
            return 1.0 / (1.0 + math.exp(-(w * z + b)))
        mid, slope = self.DEFAULT_BANDS
        if self.raw_direction() == "higher_is_ai":
            return 1.0 / (1.0 + math.exp(-(raw - mid) / slope))
        return 1.0 / (1.0 + math.exp((raw - mid) / slope))

    def raw_direction(self) -> str:
        return "higher_is_ai"

    def threshold(self, fit: dict | None) -> float:
        if fit and fit.get("ok") and fit.get("threshold") is not None:
            return float(fit["threshold"])
        return 0.5

    def finalize(self, outcome: RawOutcome, ctx: AnalysisContext,
                 segmentation, runtime_ms: int = 0,
                 error: str | None = None) -> DetectorResult:
        fit = None
        calib_info = CalibrationInfo()
        if ctx.calibration is not None:
            store = ctx.calibration.get(self.id)
            if store:
                fit = store
                m = store.get("metrics", {})
                calib_info = CalibrationInfo(
                    status="calibrated",
                    dataset=store.get("dataset", ""),
                    n_samples=store.get("n", 0),
                    auc=m.get("auc"), ece=m.get("ece"),
                    brier=m.get("brier"), accuracy=m.get("acc"),
                    threshold=store.get("threshold", 0.5),
                    fitted_at=store.get("when"),
                )
            else:
                calib_info = CalibrationInfo(status="uncalibrated",
                                             threshold=0.5)
        score = None
        verdict = None
        confidence = None
        thr = self.threshold(fit if fit else None) if fit else 0.5
        seg_scores: list[SegmentScore] = []
        if not fit:
            thr = 0.5

        if outcome is not None and outcome.raw_score is not None:
            score = self._map_raw(outcome.raw_score, fit)
            if score >= thr + 0.05:
                verdict = Verdict.likely_ai
            elif score <= thr - 0.05:
                verdict = Verdict.likely_human
            else:
                verdict = Verdict.uncertain
            confidence = min(1.0, 0.5 + 1.6 * abs(score - thr))

            seg_by_id = {s.id: s for s in segmentation.all()}
            for raw_seg in outcome.segment_scores:
                seg = seg_by_id.get(raw_seg.segment_id)
                if seg is None:
                    continue
                seg_scores.append(SegmentScore(
                    id=seg.id, kind=seg.kind, text=seg.text,
                    start=seg.start, end=seg.end, parent_id=seg.parent_id,
                    score=self._map_raw(raw_seg.value, fit),
                    raw_score=raw_seg.value,
                    extras=raw_seg.extras,
                ))

        return DetectorResult(
            detector_id=self.id,
            name=self.name,
            family=self.family,
            link=self.link,
            reference_note=outcome.reference_note if outcome else None,
            score=score,
            raw_score=outcome.raw_score if outcome else None,
            raw_direction=self.raw_direction(),
            verdict=verdict,
            confidence=confidence,
            threshold=thr,
            signals=outcome.signals if outcome else {},
            segment_scores=seg_scores,
            evidence=outcome.evidence if outcome else [],
            calibration=calib_info,
            runtime_ms=runtime_ms,
            model=outcome.model if outcome else None,
            error=error,
        )

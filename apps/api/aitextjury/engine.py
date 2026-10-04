"""Engine — orchestrates detector runs, consensus and caching.

Per run:
1. segment the text once (paragraphs + sentences with offsets),
2. resolve detector list (requested ⊂ available, default = default_enabled),
3. execute detectors concurrently with per-kind timeouts,
4. standardize results (`finalize`) applying calibration fits,
5. compute the consensus meta-result (weighted mean + agreement +
   cross-detector sentence heatmap),
6. persist the report into history.

Caching: results are content-addressed per (detector, text, effective
settings); raw outcomes are cached so re-analyzing identical text is free
(including BYOK LLM-Judge calls that would re-bill your key).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone

from . import config as cfg
from .calibration import CalibrationStore
from .detectors.base import AnalysisContext, BaseDetector, DetectorError, \
    RawOutcome
from .detectors import DetectorRegistry
from .providers.manager import ProviderManager
from .schemas import (AnalyzeReport, ConsensusEntry, ConsensusResult,
                      EvidenceItem, HistoryEntry, SegmentScore, TextStats,
                      Verdict)
from .segmenter import Segment, segment_text
from .utils import detect_language, word_tokens


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cache_key(detector: BaseDetector, text: str, settings: dict) -> str:
    payload = json.dumps(
        {"id": detector.id, "settings": settings, "text": text},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class Engine:
    def __init__(self, registry: DetectorRegistry,
                 providers: ProviderManager,
                 calibration: CalibrationStore | None = None,
                 settings: dict | None = None,
                 history_store=None):
        self.registry = registry
        self.providers = providers
        self.calibration = calibration or CalibrationStore()
        self.settings = (settings or {}).get("detect", {}) or {}
        self.detector_settings = (settings or {}).get("detectors", {}) or {}
        self.history = history_store

    # ------------------------------------------------------------- settings

    def effective_settings(self) -> dict:
        merged = dict(cfg.DEFAULT_SETTINGS["detect"])
        merged.update(self.settings or {})
        return merged

    def detector_eff(self, det_id: str) -> dict:
        base = dict(cfg.DEFAULT_SETTINGS["detectors"].get(det_id, {}))
        base.update(self.detector_settings.get(det_id, {}) or {})
        return base

    # ------------------------------------------------------------- history --

    def _load_settings_file(self) -> None:
        pass  # settings injected upstream (main.py keeps single source)

    # -------------------------------------------------------------- public --

    def available_detectors(self) -> list[dict]:
        """Registry snapshot + live availability (for the picker UI)."""
        out = []
        for det in self.registry.ordered():
            if det.id == "llm_judge":
                av = det.availability(_ProviderProbe(self.providers))
            else:
                av = det.availability(None)
            base = {
                "id": det.id,
                "name": det.name,
                "family": det.family,
                "description": det.description,
                "link": det.link,
                "requires": det.requires,
                "default_enabled": det.default_enabled,
                "heavy": det.heavy,
                "available": av.ok,
                "reason": av.reason,
                "hints": av.hints,
                "uncalibrated": self.calibration.get(det.id) is None,
                "bands_help": det.bands_help(),
            }
            out.append(base)
        return out

    async def analyze(self, text: str, detector_ids: list[str] | None = None,
                      force_refresh: bool = False) -> AnalyzeReport:
        t0 = time.monotonic()
        text = text.strip()
        if not text:
            raise ValueError("empty text")

        settings = self.effective_settings()
        if len(text) > settings.get("max_text_chars", 60000):
            raise ValueError("text too long — split it up")

        segmentation = segment_text(text)
        lang, cjk = detect_language(text)
        n_words = len(word_tokens(text, lang))

        ctx = AnalysisContext(
            text=text, segmentation=segmentation,
            providers=self.providers, settings=settings,
            get_detector_settings={
                det_id: self.detector_eff(det_id)
                for det_id in self.registry.ids()
            },
            calibration=self.calibration, language=lang)

        # resolve detector set
        if detector_ids:
            chosen: list[BaseDetector] = []
            for det_id in detector_ids:
                d = self.registry.get(det_id)
                if d is None:
                    raise ValueError(f"unknown detector: {det_id}")
                chosen.append(d)
        else:
            chosen = [d for d in self.registry.ordered() if d.default_enabled]
        if not chosen:
            raise ValueError("no detectors selected")

        # run all detectors concurrently
        jobs = [self._run_one(det, ctx, force_refresh) for det in chosen]
        raw_results = await asyncio.gather(*jobs)

        results = []
        for det, (outcome, runtime_ms, error) in zip(chosen, raw_results):
            results.append(det.finalize(outcome, ctx, segmentation,
                                        runtime_ms, error))

        consensus = self._consensus(results, segmentation)

        report = AnalyzeReport(
            id=uuid.uuid4().hex[:12],
            created_at=_now_iso(),
            title=(text[:60].replace("\n", " ") + ("…" if len(text) > 60 else "")),
            stats=TextStats(
                chars=len(text), words=n_words,
                sentences=len(segmentation.sentences),
                paragraphs=len(segmentation.paragraphs),
                language=lang, cjk_ratio=cjk),
            text=text,
            paragraphs=[SegmentScore(
                id=p.id, kind="paragraph", text=p.text,
                start=p.start, end=p.end, parent_id=None)
                for p in segmentation.paragraphs],
            sentences=[SegmentScore(
                id=s.id, kind="sentence", text=s.text,
                start=s.start, end=s.end, parent_id=s.parent_id)
                for s in segmentation.sentences],
            results=results,
            consensus=consensus,
            options={"detectors": [d.id for d in chosen],
                     "force_refresh": force_refresh},
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
        if self.history is not None:
            try:
                self.history.save(self._to_history_entry(report))
            except Exception:
                pass
        return report

    # ------------------------------------------------------------ one det --

    def _timeout_for(self, det: BaseDetector) -> float:
        settings = self.effective_settings()
        return float(settings.get(
            "timeout_llm" if det.family == "byok_llm" else "timeout_local",
            120))

    async def _run_one(self, det: BaseDetector, ctx: AnalysisContext,
                       force_refresh: bool):
        t0 = time.monotonic()
        timeout = self._timeout_for(det)
        settings = self.effective_settings()
        det_settings = ctx.det_settings(det.id)

        # cache lookup -------------------------------------------------
        outcome: RawOutcome | None = None
        error: str | None = None
        use_cache = settings.get("cache_enabled", True)
        key = cache_key(det, ctx.text, det_settings or {})
        if use_cache and not force_refresh:
            cached = self._cache_read(det.id, key)
            if cached is not None:
                outcome = RawOutcome.from_cache(cached)

        if outcome is None and error is None:
            try:
                outcome = await asyncio.wait_for(
                    det.analyze(ctx), timeout=timeout)
                if use_cache:
                    self._cache_write(det.id, key, outcome)
            except DetectorError as e:
                error = str(e)
            except asyncio.TimeoutError:
                error = f"timed out after {timeout:.0f}s"
            except Exception as e:  # pragma: no cover
                error = f"{type(e).__name__}: {e}"
        return outcome, int((time.monotonic() - t0) * 1000), error

    # -------------------------------------------------------------- cache --

    def _cache_dir(self):
        return cfg.subdir("cache")

    def _cache_read(self, det_id: str, key: str) -> dict | None:
        p = self._cache_dir() / f"{det_id}_{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_write(self, det_id: str, key: str, outcome: RawOutcome) -> None:
        try:
            p = self._cache_dir() / f"{det_id}_{key}.json"
            p.write_text(json.dumps(outcome.to_cache(), ensure_ascii=False),
                         encoding="utf-8")
        except Exception:
            pass

    # ---------------------------------------------------------- consensus --

    def _consensus(self, results, segmentation) -> ConsensusResult:
        ok = [r for r in results if r.score is not None]
        notes: list[str] = []
        contributors: list[ConsensusEntry] = []
        score = 0.5
        verdict = Verdict.uncertain
        agreement = 0.0

        uncalibrated = [r.name for r in ok if r.calibration.status != "calibrated"]
        if uncalibrated:
            notes.append("uncalibrated detectors (default bands): "
                         + ", ".join(uncalibrated))
        failed = [f"{r.name} ({r.error})" for r in results if r.error]
        if failed:
            notes.append("failed detectors: " + "; ".join(failed))
        if len(ok) == 1:
            notes.append("only one detector produced a score — broaden the "
                         "panel for real coverage")

        if ok:
            weights = []
            for r in ok:
                acc = r.calibration.accuracy
                w = 1.0 if acc is None else max(0.25, min(2.0, acc ** 2 + 0.15))
                weights.append(w)
                contributors.append(ConsensusEntry(
                    detector_id=r.detector_id, score=r.score,
                    verdict=r.verdict or Verdict.uncertain,
                    weight=round(w, 3),
                    calibrated=r.calibration.status == "calibrated"))
            wsum = sum(weights)
            score = sum(r.score * w for r, w in zip(ok, weights)) / wsum
            # pairwise verdict agreement
            same, total = 0, 0
            for i in range(len(ok)):
                for j in range(i + 1, len(ok)):
                    total += 1
                    v1 = _band(ok[i])
                    v2 = _band(ok[j])
                    if v1 == v2:
                        same += 1
            agreement = (same / total) if total else 1.0
            spread = max(r.score for r in ok) - min(r.score for r in ok)
            if len(ok) >= 2 and spread > 0.45 and agreement < 0.5:
                notes.append("high disagreement across detectors — inspect "
                             "the evidence before forming an opinion")
            spans = [r.score for r in ok]
            mu = sum(spans) / len(spans)
            std = math.sqrt(sum((s - mu) ** 2 for s in spans) / len(spans))
            if score >= 0.55 and std < 0.15 and len(ok) >= 2:
                verdict = Verdict.likely_ai
            elif score <= 0.45 and std < 0.15 and len(ok) >= 2:
                verdict = Verdict.likely_human
            elif score >= 0.65 or score <= 0.35:
                verdict = Verdict.likely_ai if score >= 0.65 else Verdict.likely_human
            else:
                verdict = Verdict.uncertain

        # cross-detector sentence heatmap ------------------------------
        seg_scores: list[SegmentScore] = []
        per_det_segments: dict[str, dict[str, float]] = {}
        for r in ok:
            for s in r.segment_scores:
                if s.score is not None:
                    per_det_segments.setdefault(r.detector_id, {})[s.id] = s.score
        if len(per_det_segments) >= 1:
            for sent in segmentation.sentences:
                vals: list[float] = []
                cover = 0
                for det_id, m in per_det_segments.items():
                    if sent.id in m:
                        vals.append(m[sent.id])
                        cover += 1
                if vals:
                    seg_scores.append(SegmentScore(
                        id=sent.id, kind="sentence", text=sent.text,
                        start=sent.start, end=sent.end, parent_id=sent.parent_id,
                        score=sum(vals) / len(vals),
                        extras={"detector_coverage": cover}))
            if len(per_det_segments) == 1:
                notes.append("single-detector heatmap — enable more "
                             "sentence-scoring detectors (LLM/stylometry) for "
                             "cross-validated highlighting")

        return ConsensusResult(
            score=round(score, 4), verdict=verdict, agreement=round(agreement, 3),
            contributors=contributors, segment_scores=seg_scores,
            notes=notes)

    # ------------------------------------------------------------- history

    def _to_history_entry(self, report: AnalyzeReport) -> HistoryEntry:
        return HistoryEntry(
            id=report.id, created_at=report.created_at, title=report.title,
            stats=report.stats,
            consensus_score=report.consensus.score,
            consensus_verdict=report.consensus.verdict,
            detectors_used=[r.detector_id for r in report.results])


class _ProviderProbe:
    """Minimal provider view for availability() probes."""

    def __init__(self, pm: ProviderManager):
        self._pm = pm

    def build_all_enabled(self):
        return self._pm.build_all_enabled()


def _band(r) -> str:
    if r.score is None:
        return "unknown"
    if r.score >= r.threshold + 0.1:
        return "ai"
    if r.score <= r.threshold - 0.1:
        return "human"
    return "uncertain"

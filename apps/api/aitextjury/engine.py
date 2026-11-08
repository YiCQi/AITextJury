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
from .calibration import CalibrationStore, apply_fit, safe_filename
from .detectors.base import ABSTAIN_PREFIX, AnalysisContext, BaseDetector, \
    DetectorError, RawOutcome
from .detectors import DetectorRegistry
from .detectors.llm_judge import LLMJudgeDetector
from .evidence import build_fusion, conformal_p
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
        self._fusions: dict[tuple, object] = {}    # (ids, store version) -> Fusion

    # ------------------------------------------------------------- settings

    def effective_settings(self) -> dict:
        merged = dict(cfg.DEFAULT_SETTINGS["detect"])
        merged.update(self.settings or {})
        return merged

    def detector_eff(self, det_id: str) -> dict:
        # "llm_judge@deepseek" inherits everything set for "llm_judge"
        base = self.detector_eff(det_id.split("@")[0]) if "@" in det_id else {}
        base.update(cfg.DEFAULT_SETTINGS["detectors"].get(det_id, {}))
        base.update(self.detector_settings.get(det_id, {}) or {})
        return base

    # -------------------------------------------------------------- jurors --

    def jurors(self) -> list[BaseDetector]:
        """Registry detectors, with the LLM Judge expanded per provider.

        Every enabled provider becomes its own juror (`llm_judge@<id>`), so
        several APIs judge side by side, each calibrated and weighted on its
        own. With no provider configured the unbound `llm_judge` stays as an
        (unavailable) placeholder that explains how to add one.
        """
        out: list[BaseDetector] = []
        for det in self.registry.ordered():
            if det.id != LLMJudgeDetector.id:
                out.append(det)
                continue
            judges = [LLMJudgeDetector(p["id"], p.get("note") or p["id"])
                      for p in self.providers.enabled()]
            out.extend(judges or [det])
        return out

    def resolve(self, detector_ids: list[str]) -> list[BaseDetector]:
        """Ids -> jurors. Bare `llm_judge` means the whole LLM jury."""
        by_id = {d.id: d for d in self.jurors()}
        chosen: list[BaseDetector] = []
        for det_id in detector_ids:
            if det_id == LLMJudgeDetector.id:
                group = [d for d in by_id.values() if d.family == "byok_llm"]
            elif det_id in by_id:
                group = [by_id[det_id]]
            elif det_id.startswith(LLMJudgeDetector.id + "@"):
                group = [LLMJudgeDetector(det_id.split("@", 1)[1])]
            else:
                raise ValueError(f"unknown detector: {det_id}")
            chosen.extend(d for d in group if d not in chosen)
        return chosen

    def juror_weight(self, det: BaseDetector, cal) -> float:
        """Consensus weight: measured skill (or the juror's prior) x the user's
        `weight` multiplier from detector settings (0 mutes a juror)."""
        try:
            mult = float(self.detector_eff(det.id).get("weight", 1.0))
        except (TypeError, ValueError):
            mult = 1.0
        mult = max(0.0, min(5.0, mult))
        return consensus_weight(cal, det.uncalibrated_weight) * mult

    # ------------------------------------------------------------- history --

    def _load_settings_file(self) -> None:
        pass  # settings injected upstream (main.py keeps single source)

    # -------------------------------------------------------------- public --

    def available_detectors(self) -> list[dict]:
        """Registry snapshot + live availability (for the picker UI)."""
        out = []
        for det in self.jurors():
            if det.family == "byok_llm":
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
                "recommended": det.recommended,
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
            chosen = self.resolve(detector_ids)
        else:
            chosen = [d for d in self.jurors() if d.default_enabled]
        if not chosen:
            raise ValueError("no detectors selected")
        ctx._detector_settings.update(
            {d.id: self.detector_eff(d.id) for d in chosen})

        # run all detectors concurrently
        jobs = [self._run_one(det, ctx, force_refresh) for det in chosen]
        raw_results = await asyncio.gather(*jobs)

        results = []
        for det, (outcome, runtime_ms, error) in zip(chosen, raw_results):
            results.append(det.finalize(outcome, ctx, segmentation,
                                        runtime_ms, error))
        self._attach_references(results, lang)

        consensus = self._consensus(results, segmentation,
                                    {d.id: d for d in chosen},
                                    evidence_units(text))

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
        # checked before the cache so stale out-of-language scores never leak
        reason = det.abstain_reason(ctx)
        if reason:
            return None, int((time.monotonic() - t0) * 1000), ABSTAIN_PREFIX + reason
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
        p = self._cache_dir() / f"{safe_filename(det_id)}_{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_write(self, det_id: str, key: str, outcome: RawOutcome) -> None:
        try:
            p = self._cache_dir() / f"{safe_filename(det_id)}_{key}.json"
            p.write_text(json.dumps(outcome.to_cache(), ensure_ascii=False),
                         encoding="utf-8")
        except Exception:
            pass

    # ---------------------------------------------------------- consensus --

    # ------------------------------------------------------- statistics --

    def _attach_references(self, results, language: str) -> None:
        """Place each detector's score among the human calibration texts
        (same language when there are enough of them)."""
        store = self.calibration.scores
        loaded: dict[str, dict] = {}
        for r in results:
            fit = self.calibration.get(r.detector_id)
            if r.raw_score is None or not fit:
                continue
            ds = fit.get("dataset", "")
            data = loaded.setdefault(ds, store.load(ds))
            col = data["scores"].get(r.detector_id)
            if not col:
                continue
            humans = [(k, raw) for k, raw in col.items()
                      if data["samples"].get(k, {}).get("label") == 0]
            same = [raw for k, raw in humans
                    if data["samples"][k].get("language") == language]
            ref_lang = language if len(same) >= 20 else "all"
            raws = same if ref_lang != "all" else [raw for _, raw in humans]
            if len(raws) < 10:
                continue
            ref = [apply_fit(fit, x) for x in raws]
            p = apply_fit(fit, r.raw_score)
            r.reference = {
                "dataset": ds, "language": ref_lang, "n_human": len(ref),
                "more_ai_than": sum(1 for x in ref if x < p) / len(ref),
                "p_human": round(conformal_p(ref, p, upper=True), 4)}

    def _fusion_for(self, det_ids: list[str]):
        store = self.calibration.scores
        key = (tuple(det_ids), store.version())
        if key not in self._fusions:
            self._fusions[key] = build_fusion(store, det_ids)
        return self._fusions[key]

    def _consensus(self, results, segmentation,
                   dets: dict[str, BaseDetector] | None = None,
                   units: float | None = None) -> ConsensusResult:
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
        abstained = [f"{r.name} ({r.error[len(ABSTAIN_PREFIX):]})" for r in results
                     if r.error and r.error.startswith(ABSTAIN_PREFIX)]
        if abstained:
            notes.append("abstained (out of their depth on this text): "
                         + "; ".join(abstained))
        failed = [f"{r.name} ({r.error})" for r in results
                  if r.error and not r.error.startswith(ABSTAIN_PREFIX)]
        if failed:
            notes.append("failed detectors: " + "; ".join(failed))
        if len(ok) == 1:
            notes.append("only one detector produced a score — broaden the "
                         "panel for real coverage")

        if ok:
            weights = []
            for r in ok:
                det = (dets or {}).get(r.detector_id)
                w = (self.juror_weight(det, r.calibration) if det is not None
                     else consensus_weight(r.calibration))
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

        # statistical fusion: replaces the heuristic vote when per-sample
        # calibration scores exist for the detectors that ran -------------
        method, evidence = "weighted", None
        if ok:
            stacked = self._stacked(ok, dets or {})
            if stacked is not None:
                score, verdict, evidence, coefs = stacked
                method = "stacked"
                for c in contributors:
                    c.weight = round(coefs.get(c.detector_id, 0.0), 3)
                if evidence["excluded"]:
                    notes.append("not in the statistical fusion (no calibration "
                                 "scores yet - recalibrate): "
                                 + ", ".join(evidence["excluded"]))
            elif any(r.calibration.status == "calibrated" for r in ok):
                notes.append("statistical fusion unavailable - recalibrate on a "
                             "labeled set so per-sample scores are stored")

        min_units = float(self.effective_settings().get("min_words", 40))
        if units is not None and units < min_units:
            verdict = Verdict.insufficient
            notes.insert(0, f"too little text for a verdict (~{units:.0f} words; "
                            f"need {min_units:.0f}) - scores are shown, but "
                            "they are too noisy to rely on")

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
            notes=notes, method=method, evidence=evidence)

    def _stacked(self, ok, dets: dict[str, BaseDetector]):
        """Fused P(AI) + interval + conformal verdict, or None."""
        store = self.calibration.scores
        have = set()
        for ds in store.datasets():
            have |= set(store.load(ds)["scores"])
        members, excluded = [], []
        for r in ok:
            det = dets.get(r.detector_id)
            muted = det is not None and self.juror_weight(det, None) == 0
            if muted or r.raw_score is None:
                continue
            (members if r.detector_id in have else excluded).append(r)
        if not members:
            return None
        ids = sorted(r.detector_id for r in members)
        fusion = self._fusion_for(ids)
        if fusion is None:
            return None
        raw = {r.detector_id: r.raw_score for r in members}
        out = fusion.score([raw[d] for d in fusion.det_ids])
        alpha = float(self.effective_settings().get("fpr_target", 0.05))
        if out["p_human"] <= alpha:
            verdict = Verdict.likely_ai
        elif out["p_ai"] <= alpha:
            verdict = Verdict.likely_human
        else:
            verdict = Verdict.uncertain
        tpr = fusion.cv_tpr.get(0.01 if alpha <= 0.01 else 0.05)
        evidence = {
            "interval": [round(v, 4) for v in out["interval"]],
            "p_human": round(out["p_human"], 4), "p_ai": round(out["p_ai"], 4),
            "alpha": alpha, "dataset": fusion.dataset, "n_train": fusion.n,
            "n_human": int(len(fusion.human_ref)), "n_ai": int(len(fusion.ai_ref)),
            "cv_auroc": round(fusion.cv_auc, 4) if fusion.cv_auc is not None else None,
            "cv_tpr": round(tpr["tpr"], 4) if tpr else None,
            "cv_fpr": round(tpr["fpr"], 4) if tpr else None,
            "members": fusion.det_ids,
            "excluded": [r.name for r in excluded],
        }
        coefs = {d: float(c) for d, c in zip(fusion.det_ids, fusion.w[1:])}
        return out["p"], verdict, evidence, coefs

    # ------------------------------------------------------------- history

    def _to_history_entry(self, report: AnalyzeReport) -> HistoryEntry:
        return HistoryEntry(
            id=report.id, created_at=report.created_at, title=report.title,
            stats=report.stats,
            consensus_score=report.consensus.score,
            consensus_verdict=report.consensus.verdict,
            detectors_used=[r.detector_id for r in report.results])


UNCALIBRATED_WEIGHT = 0.5
WEIGHT_SHRINK_N = 30       # samples at which the measured skill counts half


def consensus_weight(cal, prior: float = UNCALIBRATED_WEIGHT) -> float:
    """A juror's vote weight from its calibration record.

    Skill is the Gini coefficient 2·AUC−1 (0 = coin flip, 1 = perfect,
    threshold-free), falling back to 2·acc−1 when AUC is missing. Mapped to
    [0.25, 2.0], then shrunk toward the juror's prior by sample size so a
    fit on a handful of texts can't dominate. An uncalibrated juror gets its
    prior: 0.5 by default (below any calibrated juror with real skill, above
    a chance one); LLM judges start at 1.0 (`uncalibrated_weight`).
    """
    if cal is None or cal.status != "calibrated":
        return prior
    if cal.auc is not None:
        skill = 2.0 * cal.auc - 1.0
    elif cal.accuracy is not None:
        skill = 2.0 * cal.accuracy - 1.0
    else:
        return prior
    measured = 0.25 + 1.75 * max(0.0, min(1.0, skill))
    n = max(0, cal.n_samples or 0)
    k = n / (n + WEIGHT_SHRINK_N)
    return (1.0 - k) * prior + k * measured


class _ProviderProbe:
    """Minimal provider view for availability() probes."""

    def __init__(self, pm: ProviderManager):
        self._pm = pm

    def build_all_enabled(self):
        return self._pm.build_all_enabled()

    def build(self, pid: str):
        return self._pm.build(pid)


def evidence_units(text: str) -> float:
    """Length in 'words', counting two CJK characters as one word."""
    import re
    cjk = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
    latin = len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text))
    return latin + cjk / 2


def _band(r) -> str:
    if r.score is None:
        return "unknown"
    if r.score >= r.threshold + 0.1:
        return "ai"
    if r.score <= r.threshold - 0.1:
        return "human"
    return "uncertain"

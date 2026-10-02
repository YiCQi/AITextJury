"""Calibration — turning raw detector statistics into comparable probabilities.

A detector's raw score (nats of surprisal, cross-agreement, stylometric pull)
means little until mapped to a P(AI)-style probability with a decision
threshold. This module:

1. runs a detector over a *labeled* corpus (built-in demo set or user jsonl),
2. fits a 1-D logistic map (pure Python — no numpy/scipy dependency),
3. picks the threshold that maximizes accuracy on the set,
4. computes honesty metrics: AUC, ECE (expected calibration error),
   Brier score — and reports them in the UI,
5. stores the fit under data/calibration/<detector>.json.

The philosophy: every score **must** carry its calibration status. An
uncalibrated number is displayed as such, never silently dressed up as a
trustworthy probability.

Labeled-set format (JSONL, in data/bench/):
    {"text": "...", "label": 1, "source": "gpt-4o / 2026 essay", "language": "zh"}
    {"text": "...", "label": 0, "source": "human forum post"}
`language` is optional — omit it and it's auto-detected per sample.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from . import config as cfg

BUILTIN_PACKAGED = Path(__file__).parent / "bench" / "demo.jsonl"


# ------------------------------------------------------------------ corpus --

@dataclass
class Corpus:
    name: str
    samples: list[dict]      # {"text": str, "label": int, "source": str}

    @property
    def n(self) -> int:
        return len(self.samples)

    def label_counts(self) -> dict[str, int]:
        ai = sum(1 for s in self.samples if s["label"] == 1)
        return {"ai": ai, "human": len(self.samples) - ai}


def load_builtins() -> list[Corpus]:
    out: list[Corpus] = []
    if BUILTIN_PACKAGED.exists():
        samples = _read_jsonl(BUILTIN_PACKAGED)
        if samples:
            out.append(Corpus("demo", samples))
    return out


def load_user_corpora() -> list[Corpus]:
    out: list[Corpus] = []
    bench_dir = cfg.data_dir() / "bench"
    if bench_dir.is_dir():
        for f in sorted(bench_dir.glob("*.jsonl")):
            samples = _read_jsonl(f)
            if samples:
                out.append(Corpus(f.stem, samples))
    return out


def all_corpora() -> dict[str, Corpus]:
    corpora: dict[str, Corpus] = {}
    for c in load_builtins() + load_user_corpora():
        corpora[c.name] = c
    return corpora


def _read_jsonl(path: Path) -> list[dict]:
    samples = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "text" in obj and \
                    int(obj.get("label", -1)) in (0, 1):
                samples.append({
                    "text": str(obj["text"]),
                    "label": int(obj["label"]),
                    "source": str(obj.get("source", "")),
                    # optional explicit language ("zh"/"en"/…); when absent
                    # the calibrator auto-detects per sample at run time.
                    "language": str(obj.get("language", "")),
                })
    except Exception:
        return []
    return samples


# --------------------------------------------------------------------- fit --

def logistic_fit(raws: list[float], labels: list[int],
                 epochs: int = 600, lr: float = 0.5) -> dict:
    """1-feature standardized logistic regression by gradient descent."""
    n = len(raws)
    mu = sum(raws) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in raws) / n) or 1.0
    zs = [(x - mu) / sd for x in raws]
    w, b = 0.0, 0.0
    for _ in range(epochs):
        gw, gb = 0.0, 0.0
        for z, y in zip(zs, labels):
            p = 1.0 / (1.0 + math.exp(-(w * z + b)))
            e = p - y
            gw += e * z
            gb += e
        w -= lr * gw / n
        b -= lr * gb / n
    return {"ok": True, "w": w, "b": b, "mu": mu, "sd": sd}


def apply_fit(fit: dict, raw: float) -> float:
    z = (raw - fit["mu"]) / max(fit["sd"], 1e-9)
    return 1.0 / (1.0 + math.exp(-(fit["w"] * z + fit["b"])))


def metrics(raws: list[float], labels: list[int], fit: dict) -> dict:
    ps = [apply_fit(fit, r) for r in raws]
    # accuracy-maximizing threshold over a grid
    best_thr, best_acc = 0.5, -1.0
    for t in [i * 0.05 for i in range(1, 20)]:
        acc = sum(1 for p, y in zip(ps, labels)
                  if (p >= t) == (y == 1)) / len(ps)
        if acc > best_acc:
            best_acc, best_thr = acc, t
    # AUC (Mann-Whitney U, ties = 0.5)
    pos = [p for p, y in zip(ps, labels) if y == 1]
    neg = [p for p, y in zip(ps, labels) if y == 0]
    if pos and neg:
        wins = 0.0
        for a in pos:
            for b_ in neg:
                wins += 1.0 if a > b_ else (0.5 if a == b_ else 0.0)
        auc = wins / (len(pos) * len(neg))
    else:
        auc = None
    # ECE over 10 equal-width bins
    ece = 0.0
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        bucket = [(p, y) for p, y in zip(ps, labels) if lo <= p < hi or
                  (i == 9 and p <= 1.0)]
        if bucket:
            w = len(bucket) / len(ps)
            avg = sum(p for p, _ in bucket) / len(bucket)
            frac = sum(1 for _, y in bucket if y == 1) / len(bucket)
            ece += w * abs(avg - frac)
    brier = sum((p - y) ** 2 for p, y in zip(ps, labels)) / len(ps)
    return {"auc": auc, "ece": ece, "brier": brier,
            "acc": best_acc, "threshold": best_thr}


# ---------------------------------------------------------------- store --

class CalibrationStore:
    """Directory-backed storage; doubles as the engine's fit provider."""

    def __init__(self, root: Path | None = None):
        self.root = root or cfg.subdir("calibration")

    def _path(self, detector_id: str) -> Path:
        safe = detector_id.replace("/", "_")
        return self.root / f"{safe}.json"

    def get(self, detector_id: str) -> dict | None:
        p = self._path(detector_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, detector_id: str, fit: dict) -> None:
        fit = dict(fit)
        fit["when"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._path(detector_id).write_text(
            json.dumps(fit, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def summary(self) -> dict[str, dict]:
        out = {}
        for f in sorted(self.root.glob("*.json")):
            try:
                out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
        return out


async def calibrate_detector(detector, corpus: Corpus, *,
                             max_chars: int = 4000) -> dict:
    """Run one detector over a corpus, fit + metrics, store, return summary."""
    import asyncio

    from .detectors.base import AnalysisContext, DetectorError
    from .providers.manager import ProviderManager
    from .segmenter import segment_text
    from .utils import detect_language

    raws, labels = [], []
    errors: list[str] = []
    # Calibration uses a minimal context: providers only needed by llm_judge
    providers = ProviderManager()
    det_settings = {}

    class _Fits:
        def get(self, _):
            return None  # meaningful: calibration must NOT use existing fits

    for i, sample in enumerate(corpus.samples):
        text = sample["text"][:max_chars]
        seg = segment_text(text)
        lang = sample.get("language") or detect_language(text)[0]
        ctx = AnalysisContext(
            text=text, segmentation=seg, providers=providers,
            settings={}, get_detector_settings=det_settings,
            calibration=_Fits(), language=lang)
        try:
            outcome = await detector.analyze(ctx)
            if outcome.raw_score is None:
                continue
            raws.append(float(outcome.raw_score))
            labels.append(sample["label"])
        except DetectorError as e:
            errors.append(str(e))
        except Exception as e:  # defensive: one bad sample must not kill the run
            errors.append(f"{type(e).__name__}: {e}")

    if len(raws) < 4 or len(set(labels)) < 2:
        return {"ok": False,
                "reason": (f"insufficient usable samples ({len(raws)} usable, "
                           "need ≥4 with both labels present)"),
                "errors": errors[:5]}

    fit = logistic_fit(raws, labels)
    m = metrics(raws, labels, fit)
    fit.update({
        "n": len(raws),
        "dataset": corpus.name,
        **{k: v for k, v in m.items() if k != "threshold"},
        "threshold": m["threshold"],
    })
    return {"ok": True, "fit": fit, "errors": errors[:5]}


def fit_to_public(fit: dict) -> dict:
    """Compact shape echoed to the UI."""
    return {
        "status": "calibrated" if fit.get("ok") else "failed",
        "dataset": fit.get("dataset", ""),
        "n": fit.get("n", 0),
        "auc": fit.get("auc"),
        "ece": fit.get("ece"),
        "brier": fit.get("brier"),
        "accuracy": fit.get("acc"),
        "threshold": fit.get("threshold", 0.5),
        "when": fit.get("when", ""),
        "reason": fit.get("reason"),
    }

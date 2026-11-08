"""Statistical evidence layer — the part a bare "ask an LLM" can never give.

Every calibration run stores each detector's raw score per labeled sample
(`data/calibration/scores/<dataset>.json`). From those rows this module
derives statements the verdict can be held to:

* **stacked fusion** — a ridge-regularised logistic regression over the
  detectors that actually scored this text, trained on the calibration
  rows they share (instead of a hand-made weighting formula);
* **uncertainty** — the fusion is refit on bootstrap resamples; the spread
  of its predictions is the reported 90 % interval;
* **error-rate control** — conformal p-values against out-of-fold scores of
  the *human* (resp. *AI*) calibration texts: "AI" is only claimed when
  p_human <= alpha, so on exchangeable human text at most ~alpha are
  flagged;
* **evaluation** — AUROC with bootstrap CI, TPR at fixed FPR, per language.

numpy is the only dependency; everything is small (n <= a few hundred
rows, d <= ~10 features), so fits take milliseconds and results are cached.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIN_ROWS = 20          # fewer shared calibration rows -> no fusion
MIN_PER_CLASS = 8
BOOTSTRAPS = 200
CV_FOLDS = 5


def sample_key(text: str) -> str:
    """Stable id of a calibration text (same truncation as the run)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ store --

class ScoreStore:
    """Per-sample raw detector scores from calibration runs, one file per
    dataset: {"samples": {key: {label, language}}, "scores": {det: {key: raw}}}.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, dataset: str) -> Path:
        return self.root / f"{dataset}.json"

    def load(self, dataset: str) -> dict:
        p = self._path(dataset)
        if not p.exists():
            return {"samples": {}, "scores": {}}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"samples": {}, "scores": {}}

    def put(self, dataset: str, detector_id: str, records: list[dict]) -> None:
        """Replace one detector's rows: records = [{key, raw, label, language}]."""
        if not records:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        data = self.load(dataset)
        for r in records:
            data["samples"][r["key"]] = {"label": int(r["label"]),
                                         "language": r.get("language", "")}
        data["scores"][detector_id] = {r["key"]: float(r["raw"]) for r in records}
        self._path(dataset).write_text(json.dumps(data), encoding="utf-8")

    def datasets(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def version(self) -> float:
        """Changes whenever any score file changes (cache invalidation)."""
        if not self.root.is_dir():
            return 0.0
        return max((p.stat().st_mtime for p in self.root.glob("*.json")),
                   default=0.0)

    def matrix(self, dataset: str, det_ids: list[str],
               language: str | None = None):
        """Rows where every detector in det_ids has a score -> (X, y, langs)."""
        data = self.load(dataset)
        cols = [data["scores"].get(d) for d in det_ids]
        if any(c is None for c in cols):
            return None
        keys = [k for k in data["samples"]
                if all(k in c for c in cols)
                and (language is None or data["samples"][k]["language"] == language)]
        if not keys:
            return None
        X = np.array([[c[k] for c in cols] for k in keys], dtype=float)
        y = np.array([data["samples"][k]["label"] for k in keys], dtype=int)
        langs = [data["samples"][k]["language"] for k in keys]
        return X, y, langs


# ------------------------------------------------------------- statistics --

def auroc(scores, labels) -> float | None:
    """Mann-Whitney AUROC (ties count half); None without both classes."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = _rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ranks over ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auroc_ci(scores, labels, level: float = 0.95, b: int = 1000,
             seed: int = 0) -> tuple[float, float] | None:
    """Stratified bootstrap percentile interval for AUROC."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        p = rng.choice(pos, len(pos))
        n = rng.choice(neg, len(neg))
        vals.append(auroc(np.concatenate([p, n]),
                          np.r_[np.ones(len(p)), np.zeros(len(n))]))
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def tpr_at_fpr(scores, labels, fpr: float) -> dict | None:
    """Detection rate when at most `fpr` of human texts may be flagged.

    The threshold is the lowest score that keeps the empirical human false
    positive rate <= fpr. With n human samples the finest achievable FPR is
    1/n — reported as `resolution` so a "1 %" on 50 humans isn't oversold.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    human, ai = np.sort(s[y == 0]), s[y == 1]
    if len(human) == 0 or len(ai) == 0:
        return None
    allowed = int(math.floor(fpr * len(human)))     # humans allowed above thr
    thr = human[len(human) - allowed - 1] if allowed < len(human) else -np.inf
    return {"tpr": float(np.mean(ai > thr)),
            "fpr": float(np.mean(human > thr)),
            "threshold": float(thr), "n_human": int(len(human)),
            "resolution": 1.0 / len(human)}


def conformal_p(reference, value: float, upper: bool = True) -> float:
    """(1 + #{ref at least as extreme}) / (n + 1).

    upper=True: how unusually HIGH `value` is among reference scores (vs
    human texts: small p = more AI-like than nearly all humans).
    """
    ref = np.asarray(reference, float)
    k = int(np.sum(ref >= value)) if upper else int(np.sum(ref <= value))
    return (1 + k) / (len(ref) + 1)


def ece(probs, labels, bins: int = 10) -> float:
    p = np.asarray(probs, float)
    y = np.asarray(labels, int)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b_ in range(bins):
        m = idx == b_
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


# ---------------------------------------------------------------- logistic --

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
               iters: int = 30, w0: np.ndarray | None = None) -> np.ndarray:
    """Ridge logistic regression by Newton/IRLS. X already standardised;
    returns [intercept, coefs...]. The intercept is not penalised."""
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1]) if w0 is None else w0.copy()
    reg = np.eye(Xb.shape[1]) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) + reg @ w
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _sigmoid(np.c_[np.ones(len(X)), X] @ w)


def _folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(f, int) for f in folds if f]


def oof_predictions(X: np.ndarray, y: np.ndarray, k: int = CV_FOLDS,
                    seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities: every row scored by a model that never saw it."""
    out = np.empty(len(y))
    for test in _folds(y, k, seed):
        train = np.setdiff1d(np.arange(len(y)), test)
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        w = fit_logreg((X[train] - mu) / sd, y[train])
        out[test] = predict(w, (X[test] - mu) / sd)
    return out


# ------------------------------------------------------------------ fusion --

@dataclass
class Fusion:
    det_ids: list[str]
    dataset: str
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    boot: np.ndarray                     # (B, d+1) refits on resamples
    human_ref: np.ndarray                # out-of-fold P(AI) of human rows
    ai_ref: np.ndarray                   # out-of-fold P(AI) of AI rows
    cv_auc: float | None
    cv_tpr: dict = field(default_factory=dict)   # fpr -> tpr_at_fpr dict
    n: int = 0

    def score(self, raws: list[float], level: float = 0.90) -> dict:
        x = (np.asarray(raws, float)[None, :] - self.mu) / self.sd
        p = float(predict(self.w, x)[0])
        bs = predict_many(self.boot, x)
        lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
        return {"p": p, "interval": [float(lo), float(hi)],
                "p_human": conformal_p(self.human_ref, p, upper=True),
                "p_ai": conformal_p(self.ai_ref, p, upper=False)}


def predict_many(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _sigmoid(W[:, 0] + W[:, 1:] @ x[0])


def build_fusion(store: ScoreStore, det_ids: list[str],
                 seed: int = 0) -> Fusion | None:
    """Fit the stacked model on the dataset where these detectors share the
    most labeled rows; None when there isn't enough evidence to do it right."""
    best = None
    for ds in store.datasets():
        m = store.matrix(ds, det_ids)
        if m is not None and (best is None or len(m[1]) > len(best[1][1])):
            best = (ds, m)
    if best is None:
        return None
    ds, (X, y, _) = best
    if len(y) < MIN_ROWS or min(np.sum(y == 0), np.sum(y == 1)) < MIN_PER_CLASS:
        return None
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = fit_logreg(Z, y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(fit_logreg(Z[idx], y[idx], w0=w, iters=12))
    oof = oof_predictions(X, y, seed=seed)
    return Fusion(
        det_ids=list(det_ids), dataset=ds, mu=mu, sd=sd, w=w,
        boot=np.array(boot), human_ref=oof[y == 0], ai_ref=oof[y == 1],
        cv_auc=auroc(oof, y),
        cv_tpr={f: tpr_at_fpr(oof, y, f) for f in (0.01, 0.05)},
        n=int(len(y)))


# -------------------------------------------------------------- evaluation --

def evaluate(store: ScoreStore, dataset: str, fits: dict[str, dict],
             names: dict[str, str] | None = None) -> dict:
    """Per-detector and fused metrics on one dataset, overall and per language.

    Single detectors are scored by rank (AUROC / TPR@FPR need no fit; the
    direction comes from the calibration fit). Fusion rows are cross-validated
    (out-of-fold), so they're not flattered by being tested on training data.
    """
    data = store.load(dataset)
    names = names or {}
    langs = sorted({s["language"] for s in data["samples"].values() if s["language"]})
    rows = []
    for det_id, col in data["scores"].items():
        fit = fits.get(det_id) or {}
        sign = -1.0 if fit.get("w", 1.0) < 0 else 1.0
        for lang in [None] + langs:
            keys = [k for k in col if lang is None
                    or data["samples"].get(k, {}).get("language") == lang]
            if not keys:
                continue
            s = sign * np.array([col[k] for k in keys])
            y = np.array([data["samples"][k]["label"] for k in keys])
            row = _metrics_row(det_id, names.get(det_id, det_id), lang, s, y)
            if row and fit and lang is None:
                from .calibration import apply_fit
                row["ece"] = ece([apply_fit(fit, col[k]) for k in keys], y)
            if row:
                rows.append(row)
    # fused jury per language: every detector that covers that language
    for lang in langs:
        covering = [d for d, col in data["scores"].items()
                    if _coverage(data, col, lang) >= 0.8]
        m = store.matrix(dataset, covering, lang) if covering else None
        if m is None:
            continue
        X, y, _ = m
        if len(y) < MIN_ROWS or min(np.sum(y == 0), np.sum(y == 1)) < MIN_PER_CLASS:
            continue
        row = _metrics_row("fusion", f"Jury fusion ({len(covering)} detectors, CV)",
                           lang, oof_predictions(X, y), y)
        if row:
            row["members"] = covering
            rows.append(row)
    return {"dataset": dataset, "n_samples": len(data["samples"]),
            "languages": langs, "rows": rows}


def _coverage(data: dict, col: dict, lang: str) -> float:
    keys = [k for k, s in data["samples"].items() if s["language"] == lang]
    return sum(1 for k in keys if k in col) / len(keys) if keys else 0.0


def _metrics_row(det_id, name, lang, s, y) -> dict | None:
    a = auroc(s, y)
    if a is None:
        return None
    ci = auroc_ci(s, y, b=500)
    t1, t5 = tpr_at_fpr(s, y, 0.01), tpr_at_fpr(s, y, 0.05)
    return {"detector_id": det_id, "name": name, "language": lang or "all",
            "n": int(len(y)), "n_human": int(np.sum(y == 0)),
            "auroc": a, "auroc_ci": list(ci) if ci else None,
            "tpr_at_1fpr": t1["tpr"] if t1 else None,
            "tpr_at_5fpr": t5["tpr"] if t5 else None,
            "fpr_resolution": t1["resolution"] if t1 else None}

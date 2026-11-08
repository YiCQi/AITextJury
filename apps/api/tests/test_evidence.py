import asyncio

import numpy as np

from aitextjury.calibration import CalibrationStore, all_corpora, calibrate_detector
from aitextjury.detectors import build_default_registry
from aitextjury.engine import Engine, evidence_units
from aitextjury.evidence import (ScoreStore, auroc, auroc_ci, build_fusion,
                                 conformal_p, fit_logreg, predict, tpr_at_fpr)
from aitextjury.providers.manager import ProviderManager
from aitextjury.schemas import Verdict


def test_auroc_basics():
    assert auroc([3, 4, 5], [1, 1, 1]) is None
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    lo, hi = auroc_ci(np.r_[np.arange(30) + 10, np.arange(30)], np.r_[np.ones(30), np.zeros(30)])
    assert 0.5 < lo <= hi <= 1.0


def test_tpr_at_fpr_respects_the_human_budget():
    human, ai = np.arange(100.0), np.arange(50.0, 150.0)
    s, y = np.r_[human, ai], np.r_[np.zeros(100), np.ones(100)]
    r = tpr_at_fpr(s, y, 0.05)
    assert r["fpr"] <= 0.05 and r["tpr"] == 0.55 and r["resolution"] == 0.01


def test_conformal_p():
    ref = np.arange(99.0)                       # 99 human scores 0..98
    assert conformal_p(ref, 1000.0) == 1 / 100  # above every human -> minimal p
    assert conformal_p(ref, -1.0) == 1.0
    assert conformal_p(ref, -1.0, upper=False) == 1 / 100


def test_logreg_separates():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-1, 1, (50, 2)), rng.normal(1, 1, (50, 2))]
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    w = fit_logreg(X, y)
    assert auroc(predict(w, X), y) > 0.85


def _synthetic_store(tmp_path):
    rng = np.random.default_rng(1)
    store = ScoreStore(tmp_path / "scores")
    good, noise = [], []
    for i in range(80):
        label = i % 2
        key = f"s{i}"
        good.append({"key": key, "raw": rng.normal(2.0 * label, 0.7), "label": label, "language": "en"})
        noise.append({"key": key, "raw": rng.normal(0, 1), "label": label, "language": "en"})
    store.put("bench", "good", good)
    store.put("bench", "noise", noise)
    return store


def test_fusion_interval_and_pvalues(tmp_path):
    store = _synthetic_store(tmp_path)
    fusion = build_fusion(store, ["good", "noise"])
    assert fusion is not None and fusion.n == 80 and fusion.cv_auc > 0.8
    ai_like = fusion.score([4.0, 0.0])
    human_like = fusion.score([-2.0, 0.0])
    lo, hi = ai_like["interval"]
    assert lo <= ai_like["p"] <= hi
    assert ai_like["p_human"] < 0.05 < human_like["p_human"]
    assert human_like["p_ai"] < 0.05
    assert build_fusion(store, ["good", "missing"]) is None   # no shared rows


def test_evidence_units_counts_cjk_as_half_words():
    assert evidence_units("one two three") == 3
    assert evidence_units("今天天气很好") == 3


def _engine(tmp_path):
    return Engine(registry=build_default_registry(),
                  providers=ProviderManager(path=tmp_path / "providers.json"),
                  calibration=CalibrationStore(root=tmp_path / "calib"),
                  settings={"detect": {"cache_enabled": False}, "detectors": {}})


def test_stacked_consensus_after_real_calibration(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "calib").mkdir()
    det = engine.registry.get("stylometry")
    corpus = all_corpora()["hc3"]
    fit = asyncio.run(calibrate_detector(det, corpus, providers=engine.providers))
    assert fit["ok"] and len(fit["records"]) == fit["fit"]["n"]
    engine.calibration.put("stylometry", fit["fit"])
    engine.calibration.scores.put("hc3", "stylometry", fit["records"])

    text = next(s["text"] for s in corpus.samples if s["label"] == 1 and s["language"] == "en")
    report = asyncio.run(engine.analyze(text, ["stylometry"]))
    c = report.consensus
    assert c.method == "stacked"
    ev = c.evidence
    assert ev["members"] == ["stylometry"] and ev["n_train"] == fit["fit"]["n"]
    assert ev["interval"][0] <= c.score <= ev["interval"][1]
    if c.verdict == Verdict.likely_ai:
        assert ev["p_human"] <= ev["alpha"]
    ref = report.results[0].reference
    assert ref and ref["n_human"] >= 20 and 0.0 <= ref["more_ai_than"] <= 1.0

    short = asyncio.run(engine.analyze("Just a few words here.", ["stylometry"]))
    assert short.consensus.verdict == Verdict.insufficient


def test_bench_texts_are_unique():
    """Duplicates would leak across CV folds and inflate n."""
    texts = [s["text"] for s in all_corpora()["hc3"].samples]
    assert len(texts) == len(set(texts))

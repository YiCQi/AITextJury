import asyncio

import pytest

from aitextjury.calibration import CalibrationStore
from aitextjury.detectors import build_default_registry
from aitextjury.engine import Engine
from aitextjury.providers.manager import ProviderManager


def make_engine(tmp_path):
    registry = build_default_registry()
    providers = ProviderManager()
    calib = CalibrationStore(root=tmp_path / "calib")
    settings = {
        "detect": {"cache_enabled": True, "timeout_local": 30,
                   "timeout_llm": 30, "max_history": 10},
        "detectors": {},
    }
    return Engine(registry=registry, providers=providers,
                  calibration=calib, settings=settings)


def test_analyze_stylometry_only(tmp_path):
    engine = make_engine(tmp_path)
    text = ("In today's fast-paced world, it is important to note that "
            "seamless integration is crucial. Furthermore, leveraging "
            "robust technology unlocks our full potential.")
    report = asyncio.run(engine.analyze(text, ["stylometry"]))
    assert report.consensus.score >= 0.0
    assert report.results[0].detector_id == "stylometry"
    assert report.results[0].score is not None
    assert report.consensus.contributors
    assert report.stats.sentences >= 1


def test_analyze_unknown_detector_raises(tmp_path):
    engine = make_engine(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(engine.analyze("some text", ["nope"]))


def test_analyze_empty_text_raises(tmp_path):
    engine = make_engine(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(engine.analyze("   \n "))


def test_error_detector_not_fatal(tmp_path):
    engine = make_engine(tmp_path)
    # llm_judge has no provider -> must surface as error, not crash
    report = asyncio.run(engine.analyze(
        "Some text to analyze. It has multiple sentences, which is good.",
        ["stylometry", "llm_judge"]))
    by_id = {r.detector_id: r for r in report.results}
    assert by_id["stylometry"].error is None
    assert by_id["llm_judge"].error is not None
    assert "provider" in (by_id["llm_judge"].error or "").lower()


def test_cache_hit_second_run(tmp_path):
    engine = make_engine(tmp_path)
    text = "Cache me please. This text has exactly enough length, apparently yes."
    r1 = asyncio.run(engine.analyze(text, ["stylometry"]))
    r2 = asyncio.run(engine.analyze(text, ["stylometry"]))
    assert r1.results[0].score is not None
    assert r2.results[0].score == r1.results[0].score
    force = asyncio.run(engine.analyze(text, ["stylometry"], force_refresh=True))
    assert force.results[0].score == r1.results[0].score  # deterministic


def test_consensus_weights_and_notes(tmp_path):
    engine = make_engine(tmp_path)
    text = ("Furthermore, it is important to note that smaller paragraphs "
            "help readability. Moreover, lists are pivotal.\n\n"
            "Second paragraph. Matters a tiny bit.")
    report = asyncio.run(engine.analyze(text, ["stylometry"]))
    c = report.consensus
    assert 0.0 <= c.score <= 1.0
    assert 0.0 <= c.agreement <= 1.0
    assert c.verdict is not None
    # single detector -> we expect a coverage note
    assert any("single" in n or "only one" in n
               for n in c.notes)


def test_history_saved_when_store_attached(tmp_path):
    from aitextjury.history import HistoryStore
    engine = make_engine(tmp_path)
    engine.history = HistoryStore(root=tmp_path / "hist", max_entries=5)
    text = "History should record this run. Second sentence right here."
    report = asyncio.run(engine.analyze(text, ["stylometry"]))
    entries = engine.history.list()
    assert any(e.id == report.id for e in entries)


ZH_TEXT = ("今天天气很好，我们一起去公园散步。公园里有很多人在锻炼身体。"
           "\n\n后来下雨了，我们只好回家，路上还买了两杯热奶茶。")


def test_consensus_weight_rewards_measured_skill():
    from aitextjury.engine import UNCALIBRATED_WEIGHT, consensus_weight
    from aitextjury.schemas import CalibrationInfo
    uncal = consensus_weight(CalibrationInfo())
    good = consensus_weight(CalibrationInfo(status="calibrated", auc=0.95, n_samples=200))
    chance = consensus_weight(CalibrationInfo(status="calibrated", auc=0.5, n_samples=200))
    tiny = consensus_weight(CalibrationInfo(status="calibrated", auc=0.95, n_samples=4))
    assert uncal == UNCALIBRATED_WEIGHT
    assert good > uncal > chance          # skill beats unknown beats coin flip
    assert uncal < tiny < good            # few samples -> shrunk toward neutral
    acc_only = consensus_weight(CalibrationInfo(status="calibrated", accuracy=0.9, n_samples=200))
    assert acc_only > uncal


def test_finalize_reads_top_level_fit_metrics(tmp_path):
    engine = make_engine(tmp_path)
    engine.calibration.root.mkdir(parents=True, exist_ok=True)
    engine.calibration.put("stylometry", {
        "ok": True, "w": 1.0, "b": 0.0, "mu": 0.0, "sd": 1.0, "n": 200,
        "dataset": "hc3", "auc": 0.9, "acc": 0.85, "ece": 0.1, "brier": 0.15,
        "threshold": 0.5})
    report = asyncio.run(engine.analyze(
        "Furthermore, it is important to note this. Moreover, it is pivotal.",
        ["stylometry"]))
    cal = report.results[0].calibration
    assert cal.status == "calibrated" and cal.auc == 0.9 and cal.accuracy == 0.85
    assert report.consensus.contributors[0].weight > 0.5


def test_gpt2_detectors_abstain_on_chinese(tmp_path):
    engine = make_engine(tmp_path)
    ids = ["stylometry", "lm_perplexity", "fast_detect_gpt", "binoculars"]
    report = asyncio.run(engine.analyze(ZH_TEXT, ids))
    by_id = {r.detector_id: r for r in report.results}
    for det_id in ids[1:]:
        assert by_id[det_id].score is None
        assert by_id[det_id].error.startswith("abstained: ")
    assert by_id["stylometry"].score is not None
    assert [c.detector_id for c in report.consensus.contributors] == ["stylometry"]
    notes = " ".join(report.consensus.notes)
    assert "abstained" in notes and "failed detectors" not in notes


def test_language_gate_override_and_multilingual_model():
    from aitextjury.detectors.lm_common import english_only_abstain, is_english_only

    class Ctx:
        language = "zh"
    assert is_english_only("gpt2") and is_english_only("openai-community/gpt2-medium")
    assert not is_english_only("Qwen/Qwen2.5-0.5B")
    assert english_only_abstain(Ctx(), ["gpt2"], {})
    assert english_only_abstain(Ctx(), ["gpt2"], {"allow_any_language": True}) is None
    assert english_only_abstain(Ctx(), ["Qwen/Qwen2.5-0.5B"], {}) is None
    Ctx.language = "en"
    assert english_only_abstain(Ctx(), ["gpt2"], {}) is None


def test_builtin_hc3_bench_is_balanced_bilingual():
    from aitextjury.calibration import all_corpora
    hc3 = all_corpora()["hc3"]
    assert hc3.n == 200 and hc3.label_counts() == {"ai": 100, "human": 100}
    for lang in ("en", "zh"):
        part = [s for s in hc3.samples if s["language"] == lang]
        assert len(part) == 100 and sum(s["label"] for s in part) == 50
    import json
    from pathlib import Path
    import aitextjury
    raw = Path(aitextjury.__file__).parent / "bench" / "hc3.jsonl"
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines()]
    assert all(r.get("license") for r in rows)  # per-sample attribution

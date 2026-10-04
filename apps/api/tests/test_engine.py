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

import asyncio

from zeroaibench.calibration import (CalibrationStore, all_corpora,
                                     apply_fit, logistic_fit, metrics)
from zeroaibench.detectors.base import AnalysisContext
from zeroaibench.detectors.stylometry import StylometryDetector
from zeroaibench.segmenter import segment_text

AI_TEXT = (
    "In today's fast-paced digital landscape, businesses must leverage "
    "cutting-edge technology. Furthermore, it is important to note that "
    "seamless integration delivers robust results. Moreover, organizations "
    "that harness the power of data unlock their full potential.\n\n"
    "Firstly, leaders should foster innovation. Secondly, they must remain "
    "pivotal. In conclusion, the future belongs to those who adapt. "
    "Ultimately, this journey is a testament to progress."
)

HUMAN_TEXT = (
    "So the heater died again. Third time this winter!\n\n"
    "Called the landlord. He said the guy would come Tuesday. Nobody showed. "
    "Called Thursday — \"oh, you weren't home\". I WAS home. Sitting right "
    "next to the radiator in a blanket burrito, watching my own breath "
    "indoors (not cheap, by the way, this apartment).\n\n"
    "New heater works now. Hisses a bit. Whatever. At least I can feel my "
    "fingers while typing this. 21 days, full rent, zero apologies."
)


def make_ctx(text, calibration=None, providers=None):
    return AnalysisContext(
        text=text, segmentation=segment_text(text),
        providers=providers, settings={}, get_detector_settings={},
        calibration=calibration)


async def run_det(det, ctx):
    return await det.analyze(ctx)


def test_logistic_fit_separates_two_clusters():
    raws = [1.0] * 10 + [9.0] * 10
    labels = [0] * 10 + [1] * 10
    fit = logistic_fit(raws, labels)
    assert apply_fit(fit, 1.0) < 0.2
    assert apply_fit(fit, 9.0) > 0.8


def test_metrics_shapes():
    raws = [1.0, 0.5, 1.2, 8.0, 9.0, 8.5]
    labels = [0, 0, 0, 1, 1, 1]
    fit = logistic_fit(raws, labels)
    m = metrics(raws, labels, fit)
    assert m["auc"] == 1.0
    assert 0.0 <= m["ece"] <= 1.0
    assert m["acc"] == 1.0
    assert 0.0 <= m["threshold"] <= 1.0


def test_store_roundtrip(tmp_path):
    store = CalibrationStore(root=tmp_path)
    assert store.get("stylometry") is None
    fit = logistic_fit([2.5], [1])
    fit.update({"n": 1, "dataset": "demo", "threshold": 0.55})
    store.put("stylometry", fit)
    got = store.get("stylometry")
    assert got["dataset"] == "demo"
    assert got["when"]
    assert "stylometry" in store.summary()


def test_demo_corpus_has_both_labels():
    corpora = all_corpora()
    assert "demo" in corpora
    counts = corpora["demo"].label_counts()
    assert counts["ai"] >= 8 and counts["human"] >= 8
    for s in corpora["demo"].samples:
        assert s["text"].strip() and s["label"] in (0, 1)


def test_stylometry_direction_ai_higher_than_human():
    det = StylometryDetector()
    ai = asyncio.run(run_det(det, make_ctx(AI_TEXT)))
    human = asyncio.run(run_det(det, make_ctx(HUMAN_TEXT)))
    assert ai.raw_direction == "higher_is_ai"
    assert ai.raw_score > human.raw_score + 0.15
    assert ai.segment_scores, "stylometry must produce sentence scores"
    assert "tell_phrase_hits" in ai.signals


def test_stylometry_availability_always_ok():
    det = StylometryDetector()
    assert det.availability(None).ok is True


def test_stylometry_evidence_lists_reasons():
    det = StylometryDetector()
    raw = asyncio.run(run_det(det, make_ctx(AI_TEXT)))
    assert raw.evidence, "expected evidence explanations for AI-ish text"
    titles = " | ".join(e.title for e in raw.evidence)
    assert "AI" in titles or "leans" in titles


def test_finalize_applies_calibration_fit(tmp_path):
    det = StylometryDetector()

    class FixedCalib:
        payload = {"ok": True, "w": 2.0, "b": -1.0, "mu": 0.4,
                   "sd": 0.1, "threshold": 0.7}

        def get(self, detector_id):
            return self.payload

    ctx = make_ctx("Hello world. This is a test sentence for calibration.",
                   calibration=FixedCalib())
    raw = asyncio.run(run_det(det, ctx))
    result = det.finalize(raw, ctx, ctx.segmentation)
    assert result.score is not None and 0.0 <= result.score <= 1.0
    assert result.calibration.status == "calibrated"
    assert result.threshold == 0.7
    assert result.verdict is not None


def test_finalize_default_bands_without_fit():
    det = StylometryDetector()
    ctx = make_ctx(HUMAN_TEXT)
    raw = asyncio.run(run_det(det, ctx))
    result = det.finalize(raw, ctx, ctx.segmentation)
    assert result.calibration.status in ("uncalibrated",)
    assert result.threshold == 0.5
    assert result.score < 0.5  # human-ish text should map below mid

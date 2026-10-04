"""HuggingFace sequence-classifier detector — bring your own fine-tuned model.

BYOM (bring your own model): point it at any text-classification model on
the Hub known to distinguish AI-generated from human text — e.g. fine-tuned
RoBERTa/DeBERTa detectors trained on HC3 or similar corpora.

The auto label-mapper accepts common label conventions:
* {"LABEL_1"|"fake"|"ai"|"machine"|"generated"|"chatgpt"|"llm"} -> AI class
* everything else ({LABEL_0, "human", "real", ...}) -> human class

Long texts are scored per-sentence (cheap forward passes) and aggregated.
Sentence probabilities power the heatmap directly.
"""
from __future__ import annotations

import asyncio

from ..schemas import Availability, EvidenceItem
from .base import AnalysisContext, BaseDetector, DetectorError, RawOutcome, \
    RawSegment
from .lm_common import ml_status, ml_requirement_hint

AI_LABEL_HINTS = ("fake", "ai", "machine", "generated", "chatgpt", "llm",
                  "artificial", "synthetic", "robot")
DEFAULT_MODEL = ""   # must be set explicitly — no surprise downloads


class HFClassifierDetector(BaseDetector):
    id = "hf_classifier"
    name = "HF Classifier (BYOM)"
    family = "classifier"
    description = (
        "Any HuggingFace text-classification model you choose (e.g. HC3-style "
        "RoBERTa detectors). Model id set via detector settings; nothing is "
        "downloaded until you configure one.")
    link = "https://huggingface.co/models?pipeline_tag=text-classification&search=chatgpt"
    requires = ["torch", "transformers"]
    default_enabled = False
    heavy = True
    DEFAULT_BANDS = (0.5, 0.2)

    def raw_direction(self) -> str:
        return "higher_is_ai"

    def availability(self, ctx: AnalysisContext | None = None) -> Availability:
        ok, err = ml_status()
        if not ok:
            return Availability(
                ok=False, reason=f"missing optional ML deps: {err}",
                hints=[ml_requirement_hint()])
        settings = (ctx.det_settings(self.id) if ctx else {}) or {}
        if not settings.get("model"):
            return Availability(
                ok=False,
                reason="no model configured — set detector setting "
                        "`hf_classifier.model` to a Hub id",
                hints=['example: model = "Hello-SimpleAI/chatgpt-detector-roberta"'])
        return Availability(ok=True)

    async def analyze(self, ctx: AnalysisContext) -> RawOutcome:
        try:
            return await asyncio.to_thread(self._analyze_sync, ctx)
        except DetectorError:
            raise
        except Exception as e:
            raise DetectorError(f"{type(e).__name__}: {e}") from e

    # ------------------------------------------------------------ sync path --

    def _analyze_sync(self, ctx: AnalysisContext) -> RawOutcome:
        settings = ctx.det_settings(self.id)
        model_id = settings.get("model")
        if not model_id:
            raise DetectorError("hf_classifier.model not configured")

        try:
            from transformers import pipeline
        except Exception as e:
            raise DetectorError(f"transformers unavailable: {e}")

        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1

        try:
            clf = pipeline("text-classification", model=model_id,
                           device=device, top_k=2)
        except Exception as e:
            raise DetectorError(
                f"failed to load '{model_id}': {e}. Check the model id and "
                "that it is a text-classification model.")

        sentences = ctx.segmentation.sentences
        # the model knows its max length; batches of sentences, 512 safer
        results: list[float] = []
        ai_label = None
        errors = 0
        for s in sentences:
            text = s.text.strip()
            if not text:
                results.append(0.5)
                continue
            try:
                out = clf(text[:4000], truncation=True)[0]
                if isinstance(out, list):   # top_k=2 -> list of dicts
                    ai_label = ai_label or self._pick_ai_label(out)
                    entry = next((e for e in out
                                  if e["label"].lower() == ai_label), None)
                    if entry:
                        results.append(float(entry["score"]))
                    else:
                        results.append(0.5)
                else:
                    ai_label = ai_label or "unknown"
                    results.append(self._scalar_p_ai(out))
            except Exception:
                errors += 1
                results.append(0.5)
        if errors == len(sentences):
            raise DetectorError("classifier failed on every sentence")

        p_ai_mean = sum(results) / max(1, len(results))
        ranked = sorted(zip(sentences, results), key=lambda t: -t[1])[:5]

        signals = {
            "p_ai_mean": p_ai_mean,
            "sentences": len(results),
            "failed_sentences": errors,
            "model": model_id,
        }
        evidence = [
            EvidenceItem(
                title="classifier probability",
                detail=(f"{model_id}: mean P(AI)={p_ai_mean:.2f} across "
                        f"{len(results)} sentences "
                        f"(AI class: '{ai_label or 'auto'})."),
                severity="high" if p_ai_mean > 0.8 else
                         ("warn" if p_ai_mean > 0.6 else "info")),
        ]
        if ranked:
            s_top, p_top = ranked[0]
            evidence.append(EvidenceItem(
                title="most suspicious sentence",
                detail=f"P={p_top:.2f} — “{s_top.text.strip()[:100]}”",
                severity="warn"))
        seg_scores = [RawSegment(segment_id=s.id, value=p)
                      for s, p in zip(sentences, results)]
        return RawOutcome(
            raw_score=p_ai_mean,
            raw_direction="higher_is_ai",
            signals=signals,
            segment_scores=seg_scores,
            evidence=evidence,
            model=model_id,
            reference_note=("Fine-tuned classifiers transfer poorly across "
                            "domains and models; calibrate on your own data."))

    # ------------------------------------------------------------ helpers --

    def _pick_ai_label(self, entries) -> str | None:
        labs = [str(e.get("label", "")).lower() for e in entries]
        if len(labs) != 2:
            return None
        for lab in labs:
            stem = lab.replace("label_", "")
            if any(h in stem for h in AI_LABEL_HINTS):
                return lab
        # LABEL_1 assumed AI, LABEL_0 human (common convention)
        if labs == ["label_0", "label_1"]:
            return "label_1"
        return None

    def _scalar_p_ai(self, out: dict) -> float:
        lab = str(out.get("label", "")).lower()
        stem = lab.replace("label_", "")
        if any(h in stem for h in AI_LABEL_HINTS) or lab == "label_1":
            return float(out.get("score", 0.5))
        return 1.0 - float(out.get("score", 0.5))

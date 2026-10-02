"""LLM Perplexity detector — the simplest LM-based evidence.

Runs a small causal LM locally (default `gpt2`, configurable) and reports
mean per-token surprisal of the text. Machine-fluent text is repeatedly more
predictable to these models than almost anything a human types — but fluency
isn't proof. Per-sentence surprisal feeds the heatmap, so users can see
*which* sentences the model finds predictable.

Raw score: mean NLL per token (bits stay in nats; reported as signals too).
Direction: lower = more AI-like. Normalization uses a fitted calibration or
default bands; both are exposed in the UI so nobody mistakes a heuristic for
a guarantee.

Model notes: `gpt2`'s BPE covers any byte sequence (CJK texts tokenize into
byte-fallback fragments) — cross-language perplexity signals are *weaker*
for non-English text. Point `model` at a multilingual small LM (e.g.
Qwen2.5-0.5B) to sharpen non-English analysis.
"""
from __future__ import annotations

import asyncio

from ..schemas import Availability, EvidenceItem
from .base import AnalysisContext, BaseDetector, DetectorError, RawOutcome, \
    RawSegment
from .lm_common import (HUB, aggregate_to_sentences, build_chunks, \
    forward_stats, ml_status, ml_requirement_hint, tokenize_for_heatmap)

CHUNK_TOKENS = 512
OVERLAP = 128
DEFAULT_MODEL = "gpt2"


class LMPerplexityDetector(BaseDetector):
    id = "lm_perplexity"
    name = "LLM Perplexity"
    family = "local_lm"
    description = (
        "Mean per-token surprisal under a locally-run causal LM "
        f"(default {DEFAULT_MODEL}); per-sentence surprisal powers the "
        "heatmap. Requires torch + transformers.")
    link = "https://huggingface.co/openai-community/gpt2"
    requires = ["torch", "transformers"]
    default_enabled = False
    heavy = True
    DEFAULT_BANDS = (3.0, 1.2)   # mid nats; lower -> AI (lower_is_ai)

    def raw_direction(self) -> str:
        return "lower_is_ai"

    def availability(self, ctx: AnalysisContext | None = None) -> Availability:
        ok, err = ml_status()
        if not ok:
            return Availability(
                ok=False, reason=f"missing optional ML deps: {err}",
                hints=[ml_requirement_hint()])
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
        model_id = settings.get("model", DEFAULT_MODEL)
        device = settings.get("device", "auto")
        model = HUB.causal(model_id, device if device != "auto" else None)

        tt = tokenize_for_heatmap(model_id, ctx.text,
                                  ctx.segmentation.sentences, CHUNK_TOKENS)
        chunks = build_chunks(tt, overlap=OVERLAP)
        if not chunks:
            raise DetectorError("text produced no tokens")

        token_values: list[float | None] = [None] * len(tt.input_ids)
        nll_sum = 0.0
        n_tok = 0
        for ch in chunks:
            if len(ch.token_ids) < 2:
                continue
            _, target_logp, _ = forward_stats(model, ch.token_ids)
            vals = (-target_logp).tolist()          # (len-1,) nats
            # vals[k] scores token at global position ch.token_pos[k+1]
            for k, v in enumerate(vals):
                g = ch.token_pos[k + 1]
                if token_values[g] is None:
                    token_values[g] = v
                nll_sum += v
                n_tok += 1
        if n_tok == 0:
            raise DetectorError("no scoreable tokens")

        mean_nll = nll_sum / n_tok
        sent_agg = aggregate_to_sentences(tt, token_values)
        top_sents = sorted(sent_agg.items(), key=lambda kv: kv[1][0])[:5]

        signals = {
            "mean_token_nll": mean_nll,
            "perplexity": 2.718281828 ** mean_nll,   # e^NLL (nats)
            "tokens_scored": n_tok,
            "n_chunks": len(chunks),
        }
        evidence = [
            EvidenceItem(
                title="mean surprisal",
                detail=(f"average per-token NLL = {mean_nll:.2f} nats "
                        f"(perplexity ≈ {signals['perplexity']:.1f}). "
                        "Fluent machine prose usually sits well below "
                        "human-written text on the same topic; this is a "
                        "soft signal, not a proof."),
                severity="warn" if mean_nll < 2.2 else "info"),
        ]
        if top_sents:
            sid, (v, n) = top_sents[0]
            seg = next((s for s in ctx.segmentation.all() if s.id == sid), None)
            snippet = (seg.text.strip()[:90] + "…") if seg else ""
            evidence.append(EvidenceItem(
                title="most predictable sentences",
                detail=(f"lowest-surprisal sentence (NLL {v:.2f}): “{snippet}”"),
                severity="info"))

        seg_scores = [RawSegment(segment_id=sid, value=v,
                                 extras={"tokens": n})
                      for sid, (v, n) in sent_agg.items()]
        return RawOutcome(
            raw_score=mean_nll,
            raw_direction="lower_is_ai",
            signals=signals,
            segment_scores=seg_scores,
            evidence=evidence,
            model=f"{model_id} (local)",
            reference_note=("Un-adversarial surprisal baseline; combine with "
                            "cross-model detectors rather than trusting alone."))

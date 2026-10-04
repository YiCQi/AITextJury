"""Binoculars-style cross-model agreement detector.

Reference: Hans et al., *Spotting AI's Touch: Identifying LLM-Generated Text
Through Binoculars* — https://arxiv.org/abs/2401.12070 —
official repo: https://github.com/ahans30/binoculars-detector

The paper's idea: score how strongly a *performer* model's next-token
beliefs are mirrored by an *observer* model, relativized by how surprising
the actual text is to the observer:

    Binoculars := Perplexity / Cross-Perplexity

We implement the cross-quantity in its vectorised closed form (log space) —
documented here because this port is faithful-in-spirit, deliberately
transparent, and **not bit-equal** to the official implementation:

    b_t = surprisal_O(t) + crossH(P_t → O_t)
        = (−log O_t(x_t))  +  Σ_w P_t(w) · (−log O_t(w))

    raw = mean_t b_t        direction: LOWER = more AI-like

AI-composed text: the two models agree with each other and with the actual
tokens — both terms shrink. Human text: the observer is surprised, and the
performer's beliefs diverge from the observer — both terms grow.

The original paper pairs Falcon-7B/Falcon-7B-instruct and reports a
threshold of 0.901; with the tiny default pair here (performer=gpt2,
observer=gpt2-medium) you MUST re-fit bands on a labeled set (see
/api/calibration) before trusting the score. Signals expose every component
(surprisal, cross entropy, geometric ratio) for full transparency.

Model pair guidance: performance tracks pair quality — original-style pairs
(base-vs-instruct of the same family) work best; larger pairs = slower.
"""
from __future__ import annotations

import asyncio
import math

from ..schemas import Availability, EvidenceItem
from .base import AnalysisContext, BaseDetector, DetectorError, RawOutcome, \
    RawSegment
from .lm_common import (HUB, aggregate_to_sentences, build_chunks,
                        cross_ent_terms, forward_stats, ml_status,
                        ml_requirement_hint, tokenize_for_heatmap)

CHUNK_TOKENS = 512
DEFAULT_PERFORMER = "gpt2"
DEFAULT_OBSERVER = "gpt2-medium"


class BinocularsDetector(BaseDetector):
    id = "binoculars"
    name = "Binoculars"
    family = "local_lm"
    description = (
        f"Cross-model agreement (performer {DEFAULT_PERFORMER} vs observer "
        f"{DEFAULT_OBSERVER}, configurable). Combines observer surprisal with "
        "performer→observer cross entropy into one lower-is-AI score "
        "— the heart of the Binoculars idea. Requires torch + transformers.")
    link = "https://arxiv.org/abs/2401.12070"
    requires = ["torch", "transformers"]
    default_enabled = False
    heavy = True
    DEFAULT_BANDS = (4.5, 2.2)   # rough gpt2-pair guess; run calibration!

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
        performer_id = settings.get("performer_model", DEFAULT_PERFORMER)
        observer_id = settings.get("observer_model", DEFAULT_OBSERVER)
        device = settings.get("device", "auto")

        # tokenizer shared: pair models must use compatible vocab; enforce same
        # tokenizer family by tokenizing with the performer's tokenizer.
        P = HUB.causal(performer_id, device if device != "auto" else None)
        O = HUB.causal(observer_id, device if device != "auto" else None) \
            if observer_id != performer_id else P

        tt = tokenize_for_heatmap(performer_id, ctx.text,
                                  ctx.segmentation.sentences, CHUNK_TOKENS)
        chunks = build_chunks(tt, overlap=128)
        if not chunks:
            raise DetectorError("text produced no tokens")

        token_values: list[float | None] = [None] * len(tt.input_ids)
        sur_sum = 0.0
        cross_sum = 0.0
        counted = 0

        for ch in chunks:
            ids = ch.token_ids
            if len(ids) < 2:
                continue
            logp_P, _, _ = forward_stats(P, ids)
            logp_O, tgt_O, _ = forward_stats(O, ids)
            surprisal = (-tgt_O).tolist()                     # (T-1,)
            crossH = cross_ent_terms(logp_O, logp_P).tolist()   # (T-1,)
            for k in range(len(surprisal)):
                g = ch.token_pos[k + 1]
                b = surprisal[k] + crossH[k]
                if token_values[g] is None:
                    token_values[g] = b
                sur_sum += surprisal[k]
                cross_sum += crossH[k]
                counted += 1

        if counted == 0:
            raise DetectorError("no scoreable tokens")

        surprisal_mean = sur_sum / counted
        cross_mean = cross_sum / counted
        raw = surprisal_mean + cross_mean
        sent_agg = aggregate_to_sentences(tt, token_values)

        signals = {
            "surprisal_obs": surprisal_mean,
            "cross_perplexity": cross_mean,
            "binoculars_raw": raw,
            "geometric_ratio_pp_over_cp": math.exp(surprisal_mean - cross_mean),
            "tokens_scored": counted,
            "performer_model": performer_id,
            "observer_model": observer_id,
        }
        evidence = [
            EvidenceItem(
                title="observer surprisal",
                detail=(f"mean NLL of actual tokens under the observer = "
                        f"{surprisal_mean:.2f} nats — how surprising the text "
                        "is to a plain LM."),
                severity="warn" if surprisal_mean < 2.0 else "info"),
            EvidenceItem(
                title="performer→observer cross entropy",
                detail=(f"mean crossH = {cross_mean:.2f} nats — how much the "
                        "two models' next-token beliefs diverge. Machine text "
                        "contracts both numbers; human text inflates them."),
                severity="warn" if cross_mean < 2.0 else "info"),
            EvidenceItem(
                title="combined score",
                detail=(f"surprisal + crossH = {raw:.2f} (lower = more "
                        "AI-like). Calibrate on a labeled set before "
                        "trusting the default bands."),
                severity="warn" if raw < 3.5 else "info"),
        ]
        seg_scores = [RawSegment(segment_id=sid, value=v,
                                 extras={"tokens": n})
                      for sid, (v, n) in sent_agg.items()]
        return RawOutcome(
            raw_score=raw,
            raw_direction="lower_is_ai",
            signals=signals,
            segment_scores=seg_scores,
            evidence=evidence,
            model=f"{performer_id} ⊕ {observer_id} (local)",
            reference_note=("Binoculars (Hans et al. 2024), closed-form "
                            "variant: surprisal_O + crossH(P→O)."))

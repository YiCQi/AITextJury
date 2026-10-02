"""Fast-DetectGPT-style curvature detector (conditional probability curvature).

Reference: Bao et al., *Fast-DetectGPT: Efficient Zero-shot Detection of
Machine-Generated Text via Conditional Probability Curvature*, ICLR 2024.
https://arxiv.org/abs/2310.05130 — official repo: https://github.com/baoguangsheng/fast-detect-gpt

Implementation here uses the paper's *independent-perturbation* form, which
avoids autoregressive resampling and lets a single extra forward pass per
perturbation cover the whole chunk:

    curvature(x) = NLL_M(x) − (1/K) Σ_k NLL_M(x̃ᵏ)

where M is the detection model and x̃ᵏ is a "locally plausible, globally
incoherent" sibling of the text built by independently sampling each token
position t from the reference model R(· | x_<t) (the final token stays as an
anchor, per the paper). Machine-composed text sits at a high-probability
peak of M: disturbing it raises its NLL sharply. Human text does not sit at
such a peak, so the contrast is small. Direction: higher = more AI-like.

This is a faithful-but-not-bit-equal port: the official code conditions
resampling on perturbed prefixes in a windowed scheme. We ship the clean
independent form (equivalent in expectation, several times faster), and
document the difference. Per-position values v_t = nll_true(t) −

───────────────────────────────────────────────
sentence attribution: nll_perturbed(t), averaged onto sentences for the heatmap.
"""
from __future__ import annotations

import asyncio

from ..schemas import Availability, EvidenceItem
from .base import AnalysisContext, BaseDetector, DetectorError, RawOutcome, \
    RawSegment
from .lm_common import (HUB, aggregate_to_sentences, build_chunks, \
    forward_stats, ml_status, ml_requirement_hint, tokenize_for_heatmap)

CHUNK_TOKENS = 512
DEFAULT_K_SAMPLES = 6
DEFAULT_MODEL = "gpt2"       # detection model (M)
DEFAULT_REFERENCE = "gpt2"   # reference model (R)


class FastDetectGPTDetector(BaseDetector):
    id = "fast_detect_gpt"
    name = "Fast-DetectGPT"
    family = "local_lm"
    description = (
        "Conditional probability curvature via contrastive perturbation "
        f"(detection model {DEFAULT_MODEL}, reference {DEFAULT_REFERENCE}). "
        "AI text sits at a high-probability peak of the local LM; perturbing "
        "tokens raises its NLL measurably more than for human text. "
        "Requires torch + transformers.")
    link = "https://arxiv.org/abs/2310.05130"
    requires = ["torch", "transformers"]
    default_enabled = False
    heavy = True
    DEFAULT_BANDS = (0.9, 0.7)   # raw curvature; higher -> AI

    def raw_direction(self) -> str:
        return "higher_is_ai"

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
        import torch

        settings = ctx.det_settings(self.id)
        model_id = settings.get("model", DEFAULT_MODEL)
        ref_id = settings.get("reference_model", DEFAULT_MODEL) or model_id
        k_samples = int(settings.get("k_samples", DEFAULT_K_SAMPLES))
        device = settings.get("device", "auto")
        M = HUB.causal(model_id, device if device != "auto" else None)
        R = HUB.causal(ref_id, device if device != "auto" else None) \
            if ref_id != model_id else M

        tt = tokenize_for_heatmap(model_id, ctx.text,
                                  ctx.segmentation.sentences, CHUNK_TOKENS)
        chunks = build_chunks(tt, overlap=128)
        if not chunks:
            raise DetectorError("text produced no tokens")

        token_values: list[float | None] = [None] * len(tt.input_ids)
        nll_true_sum = 0.0
        nll_pert_sum = 0.0
        counted = 0
        pert_consumed = 0

        for ch in chunks:
            ids = ch.token_ids
            T = len(ids)
            if T < 4:
                continue
            # 1) NLL of the true text under M (teacher-forced)
            logp_M_rows, tgt_logp_M, _ = forward_stats(M, ids)
            nll_true = (-tgt_logp_M).tolist()          # (T-1,) for tokens 1..T-1

            # 2) per-position sampler rows from the reference model R
            if R is M:
                logp_R_rows = logp_M_rows
            else:
                logp_R_rows, _, _ = forward_stats(R, ids)
            probs_R = logp_R_rows.exp()

            # 3) build K perturbed siblings; final token (T-1) stays anchored
            pert_nll_rows: list[list[float]] = []
            for _ in range(k_samples):
                sampled = torch.multinomial(probs_R, 1).view(-1)  # (T-1,)
                pert = list(ids)
                for idx in range(1, T - 1):        # keep position 0 & anchor
                    pert[idx] = int(sampled[idx - 1])
                _, tgt_logp_p, _ = forward_stats(M, pert)
                pert_nll = (-tgt_logp_p).tolist()
                # perturbed position idx is scored by row idx-1
                pert_nll_rows.append(pert_nll)
                pert_consumed += 1

            # 4) per-position contrast v_t = nll_true(t) − mean_k nll_pert(t)
            for k_token in range(1, T):
                g = ch.token_pos[k_token]
                true_v = nll_true[k_token - 1]
                pert_mean = sum(row[k_token - 1] for row in pert_nll_rows) / len(pert_nll_rows)
                v = true_v - pert_mean
                if token_values[g] is None:
                    token_values[g] = v
                nll_true_sum += true_v
                nll_pert_sum += pert_mean
                counted += 1

        if counted == 0:
            raise DetectorError("no scoreable tokens")

        raw = (nll_true_sum - nll_pert_sum) / counted
        sent_agg = aggregate_to_sentences(tt, token_values)
        signals = {
            "nll_true": nll_true_sum / counted,
            "nll_perturbed": nll_pert_sum / counted,
            "curvature": raw,
            "tokens_scored": counted,
            "k_samples": k_samples,
            "model": model_id,
            "reference_model": ref_id,
        }
        evidence = [
            EvidenceItem(
                title="curvature",
                detail=(f"NLL(true)−NLL(perturbed) = {raw:+.2f} nats on average "
                        f"(true {signals['nll_true']:.2f} vs perturbed "
                        f"{signals['nll_perturbed']:.2f}). Machine-composed text "
                        "defends its probability peak against token-selective "
                        "perturbation far better than human text."),
                severity="high" if raw > 0.9 else ("warn" if raw > 0.2 else "info")),
            EvidenceItem(
                title="method note",
                detail=("Independent-perturbation form of the CPC score; "
                        "the official implementation differs in sampling "
                        "context — see detector reference_note / docs."),
                severity="info"),
        ]
        seg_scores = [RawSegment(segment_id=sid, value=v,
                                 extras={"tokens": n})
                      for sid, (v, n) in sent_agg.items()]
        return RawOutcome(
            raw_score=raw,
            raw_direction="higher_is_ai",
            signals=signals,
            segment_scores=seg_scores,
            evidence=evidence,
            model=f"{model_id} vs {ref_id} (local)",
            reference_note=("Fast-DetectGPT (Bao et al., ICLR 2024) — "
                            "independent-perturbation implementation"))

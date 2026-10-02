"""LLM Judge — a BYOK remote LLM as a detector with structured, auditable output.

The judge receives the text as numbered paragraphs plus a strict JSON schema
and returns: verdict, confidence, per-paragraph suspicion with reasons, and
a summary. The workbench maps the paragraph flags back onto sentence
segments so the judge's reasoning shows up in the heatmap.

Anti-hallucination guardrails:
* JSON is extracted with balanced-brace scanning (``` fences tolerated);
* parse failure -> DetectorError with the raw tail attached (never a fake 0.5);
* verdict mapping is documented here and mirrored in the UI.

Selecting the provider: detector settings `provider_id` (e.g. "openai",
"deepseek:fast", "ollama", "gemini") — configured in the Providers tab;
if absent, the first enabled provider is used. `model` overrides the
provider default.

Cost note: the judge prompt is capped at ~9k chars of text. Enable caching
(default on) so _re-analyzing identical text never re-bills your key.
"""
from __future__ import annotations

import json
import re

from ..schemas import Availability, EvidenceItem
from .base import AnalysisContext, BaseDetector, DetectorError, RawOutcome, \
    RawSegment

MAX_JUDGE_CHARS = 9000

SYSTEM_PROMPT = """You are the LLM-Judge detector inside ZeroAIBench, an open
workbench that collects *evidence* about whether text is AI-generated.
You will see a numbered list of paragraphs from one document. Assess whether
the writing looks machine-generated (LLM chat completions style) or
human-written, then answer with STRICT JSON only — no prose outside JSON:

{
  "verdict": "human" | "ai" | "mixed",
  "confidence": 0-100,
  "summary": "1-3 sentences: what tipped your judgment",
  "signals": {"feature": value from -100..100 (positive = AI-ish) e.g. {"uniformity": 40, "lexical_sameness": 55}},
  "flagged_paragraphs": [
    {"index": <paragraph number>,
     "likelihood": "high"|"medium"|"low",
     "reason": "short why"}
  ]
}

Rules:
- Judge register, fluency, template structure and vagueness — not factual
  correctness, and NOT the topic.
- Human text may be clean writing; AI text may be stylistically casual.
  Weigh evidence conservatively; use "mixed" when genuinely torn.
- confidence = how sure you are about your verdict.
- Flag only the most suspicious paragraphs (or none for clean human text).
- Answer JSON in the SAME LANGUAGE the text is written in."""

SCORE_SEVERITY = {"high": (0.90, "high"), "medium": (0.70, "warn"),
                  "low": (0.55, "info")}


class LLMJudgeDetector(BaseDetector):
    id = "llm_judge"
    name = "LLM Judge"
    family = "byok_llm"
    description = (
        "Your own LLM (OpenAI / Gemini / DeepSeek / OpenRouter / Ollama / any "
        "OpenAI-compatible API — BYOK) judges the text with structured "
        "verdict, per-paragraph suspicion flags and reasoning.")
    link = "https://github.com/zeroaibench/zeroaibench/blob/main/docs/BYOK.md"
    requires: list[str] = []
    default_enabled = False
    heavy = False
    DEFAULT_BANDS = (0.5, 0.18)   # the raw is already a probability

    def raw_direction(self) -> str:
        return "higher_is_ai"

    # ----------------------------------------------------------- provider --

    def _pick_provider(self, ctx: AnalysisContext):
        settings = ctx.det_settings(self.id)
        pid = settings.get("provider_id", "")
        if pid:
            prov = ctx.providers.build(pid)
            if prov is None:
                raise DetectorError(f"provider '{pid}' not configured")
            return prov, settings.get("model", "")
        enabled = ctx.providers.build_all_enabled()
        if not enabled:
            raise DetectorError(
                "no BYOK provider configured — add one in Providers (a local "
                "Ollama needs no key)")
        return enabled[0], settings.get("model", "")

    def availability(self, ctx: AnalysisContext | None = None) -> Availability:
        if ctx is None:
            return Availability(ok=True, reason="needs any enabled provider")
        # accept both a full AnalysisContext and lightweight probe objects
        providers = getattr(ctx, "providers", None) or ctx
        if not providers or not providers.build_all_enabled():
            return Availability(
                ok=False,
                reason="no provider configured (add one under Providers — e.g. "
                       "local Ollama without any API key)",
                hints=["PUT /api/keys to add a provider"])
        return Availability(ok=True)

    # ------------------------------------------------------------- analyze --

    async def analyze(self, ctx: AnalysisContext) -> RawOutcome:
        try:
            return await self._analyze_async(ctx)
        except DetectorError:
            raise
        except Exception as e:
            raise DetectorError(f"{type(e).__name__}: {e}") from e

    async def _analyze_async(self, ctx: AnalysisContext) -> RawOutcome:
        prov, model_override = self._pick_provider(ctx)
        paras = ctx.segmentation.paragraphs
        body = self._build_body(ctx, paras)
        user_msg = {"role": "user", "content": body}
        raw = await prov.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, user_msg],
            model=model_override, temperature=0.0, max_tokens=1500,
            json_mode=True,
            timeout=float(ctx.settings.get("timeout_llm", 120)),
        )
        data = self._parse_json(raw)

        verdict = str(data.get("verdict", "")).lower().strip()
        conf = self._num(data.get("confidence"), 50.0)
        conf = max(0.0, min(100.0, conf))
        if verdict not in ("human", "ai", "mixed"):
            verdict = "mixed"
        # verdict/confidence -> P(AI) raw
        if verdict == "ai":
            p_ai = conf / 100.0
        elif verdict == "human":
            p_ai = 1.0 - conf / 100.0
        else:  # mixed pulls suspicious paragraphs toward ~0.65 max
            p_ai = 0.35 + 0.35 * (conf / 100.0)

        summary = str(data.get("summary", ""))[:600]
        signals_in = data.get("signals") or {}
        signals: dict[str, float] = {}
        if isinstance(signals_in, dict):
            for k, v in list(signals_in.items())[:8]:
                try:
                    signals[str(k)] = float(v) / 100.0
                except Exception:
                    continue
        signals["judge_p_ai"] = p_ai
        signals["judge_confidence"] = conf / 100.0

        # ---------------------------------------------------------- evidence --
        evidence = [EvidenceItem(
            title=f"judge verdict: {verdict} ({conf:.0f}%)",
            detail=summary or "(no summary provided)",
            severity="high" if verdict == "ai" and conf >= 70 else
                     ("warn" if verdict == "mixed" or (verdict == "ai" and conf >= 50) else "info"))]

        flagged = data.get("flagged_paragraphs") or []
        para_by_index = {p.index: p for p in paras}
        seg_scores: list[RawSegment] = []
        flagged_norm: dict[str, float] = {}
        for item in flagged[:12]:
            if not isinstance(item, dict):
                continue
            idx = int(round(self._num(item.get("index"), -1)))
            para = para_by_index.get(idx)
            if para is None:
                continue
            likelihood = str(item.get("likelihood", "medium")).lower()
            if likelihood not in SCORE_SEVERITY:
                likelihood = "medium"
            value, sev = SCORE_SEVERITY[likelihood]
            reason = str(item.get("reason", ""))[:300]
            evidence.append(EvidenceItem(
                title=f"paragraph {idx} flagged ({likelihood})",
                detail=reason or "(no reason given)",
                severity=sev))
            # propagate to this paragraph's sentences
            flagged_norm[para.id] = value
            for s in ctx.segmentation.sentences:
                if s.parent_id == para.id:
                    seg_scores.append(RawSegment(
                        segment_id=s.id, value=value,
                        extras={"reason": reason, "likelihood": likelihood}))

        # unflagged paragraphs -> pull toward human side proportionally
        n_unflagged = len([p for p in paras if p.id not in flagged_norm])
        if not flagged:
            # judge explicitly found nothing suspicious
            for p in paras:
                for s in ctx.segmentation.sentences:
                    if s.parent_id == p.id:
                        seg_scores.append(RawSegment(
                            segment_id=s.id, value=1.0 - conf / 100.0,
                            extras={"reason": "no suspicious paragraph flagged"}))
        return RawOutcome(
            raw_score=p_ai,
            raw_direction="higher_is_ai",
            signals=signals,
            segment_scores=seg_scores,
            evidence=evidence,
            model=f"{prov.id}:{prov.pick_model(model_override)}",
            reference_note=(
                "LLM-as-judge evidence; provider/model/config shown in result "
                "metadata. Judge opinions vary by model, prompt and language."))

    # ------------------------------------------------------------- helpers --

    def _build_body(self, ctx, paras) -> str:
        parts = ["Is the following text AI-generated? Paragraphs:", ""]
        budget = MAX_JUDGE_CHARS
        for p in paras:
            snippet = p.text.strip()
            if budget <= 0:
                parts.append(f"[{p.index}] (skipped — input longer than judge budget)")
                continue
            if len(snippet) > budget:
                snippet = snippet[:budget]
            budget -= len(snippet) + 8
            snippet = re.sub(r"\s+", " ", snippet)
            parts.append(f"[{p.index}] {snippet}")
        parts.append("")
        parts.append("Respond with the strict JSON schema only.")
        return "\n".join(parts)

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        # strip ``` fences if the model used them
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw, flags=0).strip()
        try:
            return json.loads(raw)
        except Exception:
            pass
        # balanced-brace scan: first {...} block
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = raw[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        start = -1
        raise DetectorError(
            f"judge returned non-JSON output (len={len(raw)}); tail: "
            f"{raw[-160:]!r}")

    @staticmethod
    def _num(x, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

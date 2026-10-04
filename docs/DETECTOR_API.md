# The Detector API — write your own detector

A detector is any Python class subclassing
`aitextjury.detectors.base.BaseDetector` plus a `register(registry)` hook
function. Everything else — scheduling, timeouts, caching, normalization,
verdicts, confidence, calibration, consensus, the UI — is handled by the
platform. You implement exactly one thing: **turning text into evidence**.

## Where plugins live

* `plugins/*.py` — bundled examples, versioned with the repo
* `data/plugins/*.py` — your own, git-ignored, auto-discovered at startup

Files starting with `_` are ignored. Load errors are collected and reported
(`GET /api/meta` → `plugin_errors`) without breaking the workbench.

## The contract

```python
from aitextjury.detectors.base import (
    AnalysisContext, BaseDetector, DetectorError, RawOutcome, RawSegment)
from aitextjury.schemas import Availability, EvidenceItem


class MyStylometricThing(BaseDetector):
    # --- identity ------------------------------------------------------
    id = "my_thing"                    # unique, stable
    name = "My Thing"
    family = "plugin"                  # stylometry|local_lm|classifier|byok_llm|plugin|meta
    description = "One honest sentence about what it measures."
    link = "https://arxiv.org/…"        # paper/repo, shown as "method ↗"
    requires = []                      # pip packages, informational
    default_enabled = False            # True = auto-selected for every run
    heavy = True                       # downloads models / >10 s typical

    # Default bands used until calibrated:  (mid, slope)
    # score = sigmoid((raw - mid) / slope)   for "higher_is_ai"
    # score = sigmoid((mid - raw) / slope)   for "lower_is_ai"
    DEFAULT_BANDS = (0.5, 0.2)

    # --- availability ----------------------------------------------------
    def availability(self, ctx: AnalysisContext | None = None) -> Availability:
        # cheap check: deps present? provider configured? model set?
        return Availability(ok=True)  # or Availability(ok=False, reason="…", hints=[…])

    # --- the actual metric ------------------------------------------------
    def raw_direction(self) -> str:
        return "higher_is_ai"          # or "lower_is_ai"

    async def analyze(self, ctx: AnalysisContext) -> RawOutcome:
        text = ctx.text
        sentences = ctx.segmentation.sentences   # has .id .text .start .end .parent_id
        settings = ctx.det_settings(self.id)      # merged user settings
        if something_missing:
            raise DetectorError("explain what the user must do")  # → result.error

        seg_scores = [RawSegment(segment_id=s.id, value=…,
                                 extras={"why": str}) for s in sentences]
        return RawOutcome(
            raw_score=…,                    # float | None — the statistic
            raw_direction="higher_is_ai",
            signals={"my_stat": …},        # shown raw in the card
            segment_scores=seg_scores,      # feeds the heatmap
            evidence=[EvidenceItem(
                title="why this leans AI",
                detail="plain-language explanation shown to humans",
                severity="info"|"warn"|"high")],
            model="my-model v1",           # shown in the card header
            reference_note="short method note / caveats")


def register(registry):                 # ← called at startup
    registry.register(MyStylometricThing())
```

### AnalysisContext fields

| field | what it gives you |
|---|---|
| `ctx.text` | full input text |
| `ctx.segmentation` | `.paragraphs`, `.sentences` (id/text/start/end/parent_id) |
| `ctx.providers` | `ProviderManager` — build BYOK LLM providers (see llm_judge.py) |
| `ctx.det_settings(id)` | per-detector settings merged: defaults < `data/settings.json` |
| `ctx.calibration` | fit store — **don't read your own fit** (used by finalize) |
| `ctx.settings` | global detect settings (timeouts, cache flags… |
| `ctx.language` | `"en" | "zh" | "mixed"` hint |

## Rules of the house

1. **One score, honestly mapped.** `raw_score` must be a single float whose
   direction you declare. Interpretable, documented stats beat kitchen sinks.
2. **Explain yourself.** `evidence` items are the product. A user seeing
   "burstiness 0.4 → uniform rhythm" can reason; a user seeing "0.72" cannot.
3. **Segment when possible.** Sentence- or paragraph-level values make your
   detector appear in the heatmap (which is where trust is built). If you
   can only score globally, return an empty `segment_scores`.
4. **Fail loudly, die never.** Raise `DetectorError("human-readable")` for
   recoverable problems; import-guard optional deps and report them in
   `availability()` rather than at import time.
5. **Don't over-fit.** `DEFAULT_BANDS` are a documented prior, clearly
   labeled "uncalibrated" by the UI until `/api/calibration/run` fits your
   real mapping on a labeled set.
6. **Declare models.** Set `model=` on the outcome (shows provenance in
   history/reports). Never hard-download multi-GB models without a setting
   and an `availability()` hint.
7. **BYOK discipline.** Remote calls only via `ctx.providers`. Never invent
   env vars or store keys yourself.

## Local-LM helper utilities (`aitextjury.detectors.lm_common`)

If you build a torch/transformers detector, reuse:

* `HUB` — process-wide model/tokenizer cache (shares `gpt2` across detectors)
* `tokenize_for_heatmap(model_id, text, sentences, max_ctx)` — whole-text
  tokenization with char offsets for sentence attribution
* `build_chunks(tt, overlap)` — context-window chunking aligned at sentence
  boundaries
* `forward_stats(model, ids)` — teacher-forced log-prob rows + target logp
  + entropy per position
* `cross_ent_terms(logp_detector, logp_reference)` — H(R→M) per position
* `aggregate_to_sentences(tt, values)` — per-token values → sentence means

See `detectors/lm_perplexity.py` (~100 lines) for the minimal pattern and
`fast_detect_gpt.py` / `binoculars.py` for pair-model patterns.

## Testing your plugin

```bash
python -m aitextjury.cli detectors           # availability + errors
python -m aitextjury.cli analyze t.txt -d my_thing
python -m aitextjury.cli calibrate -d my_thing --dataset demo
```

The workbench treats your plugin exactly like a core detector: caching,
calibration, consensus, history, heatmap, UI cards.

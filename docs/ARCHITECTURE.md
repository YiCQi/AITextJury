# Architecture

## Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│ apps/web  (React + Vite + TS)                                        │
│   Workbench: text input · detector panel · consensus card            │
│   Heatmap (sentence-level, per-detector or consensus layer)          │
│   History · Calibration · Providers(BYOK) · Methods                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ /api/*
┌──────────────────────────────▼───────────────────────────────────────┐
│ apps/api  zeroaibench (FastAPI)                                      │
│                                                                      │
│  main.py    routes + World singleton (settings/registry/providers)  │
│  engine.py  orchestration: segment once → run detectors in parallel   │
│             with per-kind timeouts → finalize → consensus → history    │
│  registry   DetectorRegistry: core detectors + plugin auto-discovery  │
│  detectors/ BaseDetector subclasses (the Detector API implementors)  │
│  providers/ BYOK adapters: openai_compatible | gemini (pure httpx)   │
│  calibration.py  logistic fits, threshold search, AUC/ECE/Brier       │
│  HistoryStore / Engine cache: JSON files under data/                  │
└───────────────────────────────────────────────────────────────────────┘
```

## The run pipeline

1. **Segmentation** (`segmenter.py`): paragraphs (blank-line) and sentences
   (punctuation, CJK-aware: 。！？；…) with absolute char offsets. Every
   detector and the UI speak the same offsets.
2. **Context**: `AnalysisContext(text, segmentation, providers, settings,
   calibration)` — detectors may read text, segmentation, their settings
   and (BYOK ones) the provider manager. They may not write global state.
3. **Execution**: `engine.analyze` resolves the detector set, then runs all
   detectors concurrently (`asyncio.gather`). Slow sync work (torch) runs in
   threads via `asyncio.to_thread`; each detector gets a timeout by family
   (`timeout_local` / `timeout_llm`). Failures become per-detector results
   with an `error` message — never a broken run.
4. **Raw outcomes → standardized results**: detectors return `RawOutcome`
   (raw score + direction + signals + per-segment raw values + evidence).
   `BaseDetector.finalize` maps raw → P(AI)-style probabilities using the
   detector's stored calibration fit (or documented `DEFAULT_BANDS`), picks
   the verdict against the calibrated threshold, attaches calibration
   metrics, and converts per-segment raw values into `SegmentScore`s.
5. **Consensus** (meta detector): weighted mean of scores (weights from
   calibration accuracy), pairwise verdict agreement, cross-detector
   sentence-average heatmap, explicit notes (uncalibrated detectors, single
   coverage, high disagreement…).
6. **History & cache**: reports to `data/history/<id>.json` (+ manifest,
   capped); raw outcomes cached content-addressed under `data/cache/`
   (keys never cached, BYOK judge calls not re-billed on re-runs).

## Why "finalize" is centralized

Normalization must be consistent across detectors for comparison and
consensus, and must update when calibration improves without re-running
detectors (cache stores raw outcomes — normalized views are recomputed).

`_map_raw`: fitted logistic `σ(w·(raw−μ)/σ + b)` when a fit exists (the fit
learns the direction from labels, so no direction flags are needed), else
documented default bands `σ(±(raw − mid)/slope)` with the direction the
detector declares.

## Score semantics

* `score`: normalized P(AI)-style probability in [0,1] — higher = more
  AI-like, everywhere in the system.
* `raw_score` + `raw_direction`: the detector-native statistic and how to
  read it (e.g. Binoculars curves lower-is-AI; Fast-DetectGPT higher-is-AI).
* `verdict`: `likely_ai` / `likely_human` / `uncertain` — the near-threshold
  band deliberately resolves to *uncertain*.
* `confidence`: heuristic distance-from-threshold; displayed with the
  threshold, never instead of it.

## Calibration store

`calibration.py` runs a detector over a labeled corpus (packaged demo set
from `bench/demo.jsonl`, or user sets in `data/bench/*.jsonl`), fits 1-D
logistic regression in pure Python (standardized z + GD), selects an
accuracy-maximizing threshold over a grid, and computes AUC (Mann-Whitney,
ties 0.5), ECE (10 equal-width bins), Brier. Fits persist under
`data/calibration/<detector>.json` and feed both `finalize` and consensus
weights.

## BYOK provider layer

Two adapters, zero vendor SDKs (pure `httpx`):

* `openai_compatible` — `/chat/completions` with Bearer key: covers OpenAI,
  DeepSeek, OpenRouter, Groq, Ollama's OpenAI shim, vLLM, LM Studio…
* `gemini` — `generateContent` REST with the key in the URL, system
  instruction + JSON mime support.

Key resolution: provider's stored `api_key` → template `env_key` env var.
`providers.json` is the only storage; API responses mask keys
(`sk-12…4f2a`). `llm_judge` uses the provider from its detector settings
(`provider_id`, default = first enabled) with model override.

## Frontend notes

The heatmap depends on report-embedded segmentation (`paragraphs`,
`sentences` in the AnalyzeReport payload), so history-reloaded runs render
identically to live runs. Layers: consensus mean or any detector's own
sentence scores; hover reveals every detector's value for the sentence plus
judge reasons.

## Extension points

* detectors: subclass `BaseDetector` (see DETECTOR_API.md) — plugin dirs
  `plugins/` and `data/plugins/` auto-load `register(registry)` files.
* providers: add a kind in `providers/__init__.py` (implement `chat`,
  `health`, `list_models`).
* corpora: drop labeled JSONL in `data/bench/`.
* engine caches & timeouts: `data/settings.json` (via UI Settings or
  `PUT /api/settings`).

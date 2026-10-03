# ZeroAIBench Roadmap

The project's [north star](../README.md): become the **VirusTotal for
AI-generated text** — a neutral, open workbench where many detection
methods run side by side on the same text, and where *anyone* can publish a
detector plugin and have it openly compared.

This document separates **what exists today** from what's next so the
roadmap never drifts into vaporware. Everything on the right side of each
line is *not* implemented yet.

## Where we are — v0.1.0

Already in the box:

* Unified Detector API with honest score semantics (raw score + direction,
  calibration-aware mapping, `uncertain` band instead of fake confidence)
* Six built-in detectors: Stylometry, LM-perplexity (GPT-2), Fast-DetectGPT,
  Binoculars, BYOM Hugging-Face classifier, BYOK LLM-Judge
* One example third-party plugin (`length_rhythm`) proving the plugin
  surface works outside the core package
* Consensus aggregation (calibration-weighted + agreement reporting),
  paragraph/sentence evidence, sentence heatmaps
* Calibration engine with AUC / ECE / Brier / accuracy-max threshold on
  user-supervised corpora; a 24-sample demo corpus (explicitly *a demo*)
* BYOK providers (OpenAI-shaped + Gemini), local-first key storage
* Web workbench, history, calibration and providers UIs; REST API; Docker

## Phase 1 — breadth of detection (plugin pull)

*   **More zero-config classifiers.** Ship curated BYOM recipes
    (Hello-SimpleAI text detectors, RoBERTa-base detector family, Binoculars
    variant newer than GPT-2/GPT-2-medium) as documented `model:` overrides
    rather than new code — the `hf_classifier` and Binoculars detectors
    already accept arbitrary model ids via `settings.detectors`.
*   **GLTR-style token-rank coloring.** A "why" view richer than heatmaps:
    color each token by its rank under a local LM (top-k green / mid orange /
    tail red). The token-score plumbing (`lm_common.tokenize_for_heatmap`)
    already returns offsets; this is a detector + UI feature, no new core.
*   **N-gram / retrieval boundary checks.** Detect memorized training or
    prompt scaffolding leakage via local n-gram fingerprinting. Keep it
    honest: never report "this equals ChatGPT output" without an actual
    matched source.
*   **Watermark detection hooks.** Optional support for watermark schemes
    (e.g. Kirchenbauer-style green-list statistics) so users who control
    the generator can verify *their own* outputs.

## Phase 2 — the comparison platform

*   **Plugin registry.** `POST /api/plugins/install?url=…` against a curated
    GitHub list; signed submissions with per-plugin report cards. A
    plugin's code stays in a sandboxed subprocess by default.
*   **Public benchmark, twice-honest.** Versioned Chinese + English
    evaluation corpora with documented provenance (which generator, when,
    temperature, domain — mixed news/academic/casual), plus adversarial
    sections (paraphrased AI text, human-edited AI text, AI-edited human
    text) where we *expect* detectors to fail. ECE per detector, not just
    AUC: an overconfident detector is worse than a humble one.
*   **Fair run harness.** A `bench` CLI that runs every registered detector
    over a corpus with identical timing, seeds, and budgets, producing
    comparison tables. Never merge native and API detectors into one metric
    without labeling which is which.
*   **Leaderboard with receipts.** Every public claim links to the exact
    corpus, code version, and per-sample judgments — reproducible runs or
    it didn't happen.

## Phase 3 — make the "why" sharper

*   **Per-claim evidence panels.** Beyond "this paragraph looks AI":
    which sentence, which statistical signal (burstiness? uniform sentence
    length? stock phrases? low surprisal?), with the concrete numbers.
*   **Cross-judge caching.** Store judge verdicts keyed by (provider,
    model-version, text-hash) so re-analyses never re-bill; surfaced in the
    report as `from_cache: true`.
*   **Per-detector model pinning in the UI.** Today Binoculars/Judge models
    are set in `settings.detectors`; expose them as first-class UI fields
    with "available on this machine" checks.
*   **Language-specific calibration curves.** Separate fits for zh/en
    instead of one pooled curve — corpus samples can already declare
    `language` explicitly (with per-sample auto-detection as fallback);
    the remaining work is storing and picking per-language fits.

## Phase 4 — surface area

*   **Portable Windows zip (the honest "exe").** Bundle python-embed
    (~20 MB) with the repo + first-run script, published as a GitHub Release
    artifact: download → unzip → double-click, no Python install at all.
    A true single-file PyInstaller .exe remains possible-but-not-preferred:
    bundling torch makes it 2 GB+, startup unpacks 1–3 min every run, and
    antivirus false-positives are routine — the portable zip gets the same
    UX with none of those costs.
*   **CLI-first workflows.** `zeroaibench analyze file.txt --json` and
    batch-mode calibration for CI pipelines (nightly eval corpora).
*   **Browser extension** (post content from any page into local
    workbench) and a share-card export (text + heatmap + per-detector
    verdicts, no narrative spin).
*   **Federation-lite.** An API-only mode where another ZeroAIBench node
    can contribute detector results without sharing the raw text —
    digest-keyed, opt-in.

## Non-goals (written down so they stay non-goals)

*   **No single "AI %" as the product.** The one-number AI-rate is the
    failure mode of the current tool ecosystem; ZeroAIBench always surfaces
    per-detector scores, calibration quality, and disagreement.
*   **No academic-authorship policing features.** No integrate-with-LMS,
    no "flag this student" UX, no auto-reporting. False positives at
    paragraph level are asymmetrical weapons; the UI keeps the `uncertain`
    band loud and the consensus in view.
*   **No proprietary detectors.** If a method can't be run or inspected
    locally, it can be *linked* as an external result in a report, but it
    will never be called a detection result inside the workbench.
*   **No training-era arms race.** We will not chase every new paraphrase;
    we'll document failure modes honestly instead (see README
    "Honest limitations").

## Contributing a detector

The plugin contract is deliberately small: subclass `BaseDetector`,
implement `analyze() -> RawOutcome`, register via a `register(registry)`
function in a Python file under `plugins/` or `data/plugins/`. See
[DETECTOR_API.md](DETECTOR_API.md) — including the house rules (one honest
score, explain in evidence, declare your models, don't over-fit, fail
loudly but never crash the run).

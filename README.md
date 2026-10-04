# ZeroAIBench

**An open workbench for AI-generated text detection. Evidence, not verdicts.**

ZeroAIBench is not "yet another AI detector" that prints one unreliable
percentage. It is a **platform** where you paste text, run *many* independent
detection methods side by side through one unified **Detector API**, and
look at the underlying evidence — per-sentence heatmaps, surprisal,
cross-model agreement, stylometric fingerprint, calibration quality — before
forming any opinion. Think of it as a VirusTotal-style workbench for
AI-generated text: anyone can write a detector plugin, plug it in, and
compare methods openly.

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZeroAIBench Workbench                        │
│  text ─▶ segmenter ─▶ detectors (parallel, cached) ─▶ consensus │
└───────────┬──────────────────────────────────────────────────────┘
            │
   ┌────────┴────────────────────────────────────────────┐
   │ Detector API  (every detector returns the same shape)│
   └────────┬────────────────────────────────────────────┘
        ┌────┴─────┬─────────────┬────────────┬─────────────┐
   stylometry  local-LM      classifier   LLM-Judge      your
   (stats)   perplexity /   (BYOM, HF)   (BYOK: OpenAI, plugins
             Fast-DetectGPT /          Gemini, DeepSeek, (plugins/)
             Binoculars                Ollama, anything)
```

## Quickstart

Requires [Python 3.10+](https://www.python.org/downloads/) and Node 18+.

```bash
git clone https://github.com/YiCQi/ZeroAIBench.git
cd ZeroAIBench

# Windows (PowerShell)
scripts\setup.ps1 -Ml     # one-time: private venv + all deps incl. torch
scripts\dev.ps1           # workbench at http://localhost:5173, API at :8000

# Linux / macOS
./scripts/setup.sh --with-ml
./scripts/dev.sh
```

* **Why the flag**: without `-Ml` / `--with-ml` you get a light install (no
  torch) and the four LM-based detectors stay `unavailable` — each shows the
  fix command on its panel card. Re-running setup with the flag is safe
  (the venv is reused). GPT-2-family weights (~0.5–2 GB) download on first
  analysis; if HuggingFace is blocked, set `HF_ENDPOINT=https://hf-mirror.com`
  first. After that everything runs offline.
* **Docker** (no Python/Node needed, torch included):
  `docker compose up` → http://localhost:8000
* Got `ModuleNotFoundError: fastapi`? You ran Python outside the venv — use
  the dev script, or activate `apps/api/.venv` first.

## Using the workbench

Paste text in the **Workbench** tab, tick detectors, hit **Analyze**, then
read the page top-to-bottom:

1. **Consensus** — the calibration-weighted vote across detectors. When
   detectors disagree, the notes say who said what; disagreement is
   information, not a bug.
2. **Detector cards** — every `score` is normalized so **higher = more
   AI**, always. Raw values keep their native direction and each card
   states it (e.g. *Binoculars: raw 6.4, lower = AI*).
3. **Heatmap** — which *parts* of the text look AI, per sentence/paragraph.
4. **Evidence chips** — the quotable numbers behind the verdict:
   perplexity, burstiness, stock-phrase hits…

The other tabs:

* **Providers** — paste any OpenAI-compatible or Gemini key (DeepSeek,
  OpenRouter, local Ollama — even keyless) to enable the **LLM Judge**.
  Keys stay in the local `data/providers.json`, masked on screen.
* **Calibration** — drop labeled samples into `data/bench/*.jsonl`
  (`{"text": ..., "label": 1}` AI / `0` human) and hit **Recalibrate**:
  each detector gets an honest AUC / accuracy, and consensus weights
  follow measured quality.
* **Methodology** — what each detector measures and what it is blind to.

There is also a CLI with the same engine:

```bash
python -m zeroaibench.cli analyze article.txt -d stylometry -o report.json
python -m zeroaibench.cli calibrate -d stylometry --dataset demo
```

> Privacy: everything except the LLM Judge runs **on your machine** — text,
> history and keys never leave it. Point the Judge at a local Ollama/vLLM
> and ZeroAIBench is fully offline.

## The detector panel

| detector | type | needs | idea |
|---|---|---|---|
| **Stylometry** | local stats | nothing | burstiness, repetition profile, connective boilerplate, LLM-register "tells" (EN+ZH) |
| **LLM Perplexity** | local LM | torch+transformers | mean per-token surprisal under a small causal LM (default gpt2, configurable) |
| **Fast-DetectGPT** | local LM | torch+transformers | conditional probability curvature via contrastive perturbation ([Bao et al., ICLR'24](https://arxiv.org/abs/2310.05130)) — documented variant |
| **Binoculars** | local LM pair | torch+transformers | performer/observer cross-model agreement ([Hans et al. 2024](https://arxiv.org/abs/2401.12070)) — documented closed form |
| **HF Classifier** | BYOM | torch+transformers | any HuggingFace text-classification model you choose |
| **LLM Judge** | BYOK | a provider key or local Ollama | your LLM judges with a strict-JSON protocol, flags paragraphs, explains |
| **Plugins** | community | anything | e.g. the bundled `length_rhythm` example (30 lines) |

Every result shows normalized score, raw statistic (+ direction), verdict,
threshold, signals, evidence, and **calibration status** — or an honest
error message.

## BYOK — your keys, your machine

OpenAI, Gemini, DeepSeek, OpenRouter, Groq, keyless local Ollama, or any
OpenAI-compatible endpoint (vLLM, LM Studio, …). Keys live only in the
local `data/providers.json`, are sent only to the endpoint you configure,
never echoed back unmasked, and fall back to the provider's env var
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) when empty. Details:
[docs/BYOK.md](docs/BYOK.md).

## Calibration — the honesty engine

Out of the box, detectors run on documented **default bands** and are
visibly flagged `uncalibrated`. Fit them on labeled data (UI button or
`POST /api/calibration/run`) and each detector reports **AUC / accuracy /
ECE / Brier** while consensus weights detectors by measured accuracy. The
bundled demo set is a *demo* — calibrate on data from your own domain.

## Plugins — anyone can add a detector

```python
# data/plugins/my_detector.py  (or plugins/ for bundled examples)
from zeroaibench.detectors.base import BaseDetector, RawOutcome
from zeroaibench.schemas import Availability

class MyDetector(BaseDetector):
    id, name, family, description, DEFAULT_BANDS = ...  # see docs/DETECTOR_API.md

    def availability(self, ctx=None):
        return Availability(ok=True)

    async def analyze(self, ctx):
        ...                      # ctx.text, ctx.segmentation, ctx.providers…
        return RawOutcome(raw_score=..., raw_direction="higher_is_ai",
                          signals={...}, segment_scores=[...],
                          evidence=[EvidenceItem(title=…, detail=…)])

def register(registry):
    registry.register(MyDetector())
```

Restart the API — your detector appears in the UI with calibration, caching
and consensus treated identically to built-ins. Full contract:
[docs/DETECTOR_API.md](docs/DETECTOR_API.md).

## Repository layout

```
apps/api/zeroaibench/     FastAPI backend, detector registry, engine,
                          calibration, BYOK providers, CLI, bench corpus
apps/web/                 React + Vite + TypeScript workbench UI
plugins/                  bundled example plugins
data/                     runtime state (history, keys, cache, fits) — local-only
docs/                     architecture, detector API, BYOK, roadmap
tests are under apps/api/tests
```

## Honest limitations

* **Adversarial text defeats detectors.** Rewritten, paraphrased or
  human-edited AI text and heavily-polished human text genuinely overlap.
  ZeroAIBench surfaces evidence and lets humans decide — it must not be
  used as proof, or to accuse students/authors.
* Local-LM detectors default to small English-centric models (`gpt2`); for
  Chinese/other languages point them at multilingual models (e.g.
  `Qwen2.5-0.5B`) via detector settings.
* Scores are probabilities only as far as their calibration says (that's
  why calibration status is displayed everywhere).

## Related open-source work

Star counts checked 2026-10-03. Most of the ecosystem slots right in:

* [baoguangsheng/fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt) (434★) — the Fast-DetectGPT paper (ICLR'24), reference for our implementation. Likewise [ahans30/Binoculars](https://github.com/ahans30/Binoculars) (421★, ICML'24).
* [Hello-SimpleAI](https://huggingface.co/Hello-SimpleAI) `chatgpt-detector-roberta` / `-long` — drop-in BYOM ids for the HF Classifier.
* [YuchuanTian/AIGC_text_detector](https://github.com/YuchuanTian/AIGC_text_detector) (472★) — MPU multiscale detection (ICLR'24 spotlight).
* [lynote-ai/ai-text-detector](https://github.com/lynote-ai/ai-text-detector) (445★) — local, cautious, explainable; kindred philosophy. Also [Jihuai-wpy/SeqXGPT](https://github.com/Jihuai-wpy/SeqXGPT) (101★, sentence-level like our heatmap) and [ai-detected/ai-content-detectors](https://github.com/ai-detected/ai-content-detectors) (166★, awesome-list).
* [liamdugan/raid](https://github.com/liamdugan/raid) (217★) — big adversarial benchmark, a natural corpus for the Calibration tab; [martiansideofthemoon/ai-detection-paraphrases](https://github.com/martiansideofthemoon/ai-detection-paraphrases) (205★) — why paraphrase attacks make single scores unreliable, i.e. why this workbench shows disagreement.

## License

MIT — see [LICENSE](LICENSE). Detector methods belong to their authors and
papers, linked from each detector card and the docs.

## More docs

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit
* [docs/DETECTOR_API.md](docs/DETECTOR_API.md) — write your own detector
* [docs/BYOK.md](docs/BYOK.md) — provider configuration details
* [docs/ROADMAP.md](docs/ROADMAP.md) — where this is heading

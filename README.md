# ZeroAIBench

**An open workbench for AI-generated text detection. Evidence, not verdicts.**

ZeroAIBench is not "yet another AI detector" that prints one unreliable
percentage. It is a **platform** where you paste text, run *many* independent
detection methods side by side through one unified **Detector API**, and look
at the underlying evidence — per-sentence heatmaps, surprisal, cross-model
agreement, stylometric fingerprint, calibration quality — before forming any
opinion. Think of it as a VirusTotal-style workbench for AI-generated text:
anyone can write a detector plugin, plug it in, and compare methods openly.

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

## Why

* Single-number detectors are opaque and easy to trust wrong. ZeroAIBench
  shows **why** a text looks AI-generated: which sentences, which signals,
  with how much confidence — and how well-calibrated that confidence is.
* Different methods fail differently. Combining independent evidence —
  stylometry surprisal, model-agreement, judge-reasoning — is more robust
  than any single score, and *disagreement itself is surfaced as a signal*.
* Detection research moves fast. Rather than freezing one method, the
  Detector API makes methods swappable: built-in ones today, community
  plugins tomorrow, publicly comparable forever.

## Quickstart

Clone, one command to set up, one command to run. The setup script creates an
**isolated virtualenv** (`apps/api/.venv`) — your system/conda Python is never
touched, and no pre-existing environment is assumed or required:

```bash
git clone https://github.com/YiCQi/ZeroAIBench.git
cd ZeroAIBench

# Windows (PowerShell)
scripts\setup.ps1         # one-time: venv + backend deps + web deps (~1 min)
scripts\dev.ps1           # start API + UI, opens http://localhost:5173

# Linux / macOS
./scripts/setup.sh
./scripts/dev.sh
```

That's it for the fast path — stylometry and plugin detectors are live out of
the box; add `-Ml` (Windows) / `--with-ml` (Linux/macOS) to setup for the
native-LM detectors (~2 GB: torch + GPT-2 family, then fully offline).

<details>
<summary><b>What the scripts do</b> — the manual equivalent, if you prefer</summary>

```bash
# Backend (Python >= 3.10) — in a fresh venv of your own:
cd apps/api
pip install -e .                # fastapi + uvicorn + httpx + pydantic
pip install -e ".[ml]"         # optional: torch + transformers
python -m zeroaibench          # API on http://127.0.0.1:8000

# Frontend (Node >= 18):
cd apps/web
npm install
npm run dev                    # http://localhost:5173 (proxies /api -> 8000)
```
</details>

**Docker** (no Python/Node needed at all; image builds the frontend for you
and serves it on :8000):

```bash
docker compose up               # http://localhost:8000  (UI + API together)
```

Set `SLIM: "1"` in `docker-compose.yml` `build.args` for a ~200 MB image
without the torch/transformers stack (Stylometry + plugins + BYOK Judge
still work). Model caches and your provider keys live in the named volume.

**CLI** (same engine, scriptable):

```bash
python -m zeroaibench.cli detectors
python -m zeroaibench.cli analyze article.txt -d stylometry,lm_perplexity -o report.json
python -m zeroaibench.cli calibrate -d stylometry --dataset demo
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Python 3.10+ not found` from setup | Install Python 3.10+ from python.org (tick *Add to PATH*), or `winget install Python.Python.3.12`, or skip Python entirely with `docker compose up` |
| `ModuleNotFoundError: fastapi` etc. | You ran `python -m zeroaibench` outside the venv. Use `scripts\dev.ps1`, or activate first: `apps\api\.venv\Scripts\activate` (Windows) / `source apps/api/.venv/bin/activate` (macOS/Linux) |
| pip hangs / times out | Setup auto-retries via the Tsinghua mirror; to do it manually: `pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| npm install fails | `npm config set registry https://registry.npmmirror.com`, rerun setup |
| GPT-2 model download (after `-Ml`) is blocked | `$env:HF_ENDPOINT = "https://hf-mirror.com"` (Windows) or `export HF_ENDPOINT=https://hf-mirror.com` (bash) before analyzing |
| Port 8000 / 5173 already in use | Stop the other process, or `uvicorn zeroaibench.main:app --port 8001` + the Vite proxy line in `apps/web/vite.config.ts` |

### 快速开始（中文）

```bash
git clone https://github.com/YiCQi/ZeroAIBench.git && cd ZeroAIBench
scripts\setup.ps1        # Windows：一次性建好隔离环境（不碰你的 conda/系统 Python）
scripts\dev.ps1          # 启动后端+前端，自动打开 http://localhost:5173
# Linux/macOS 同理：./scripts/setup.sh && ./scripts/dev.sh
# 或者一条命令（自带前端构建）：docker compose up → http://localhost:8000
```

粘贴文章 → 勾选检测器（Stylometry 零依赖即可用；本地 LM 检测器需要
`pip install -e ".[ml]"`；LLM Judge 在 Providers 页接入你自己的
OpenAI / Gemini / DeepSeek / OpenRouter / Ollama / 任何 OpenAI 兼容端点）→
Analyze。你会得到：每个检测器的分数 + 原始统计量 + 句子级 heatmap +
证据解释 + 校准状态（AUC/ECE）+ 跨检测器共识与分歧提示。

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
threshold, signals, evidence items, per-sentence scores, runtime, and
**calibration status** (or an honest error message).

## BYOK — your keys, your machine

Providers are configured in the UI (Providers tab) or `data/providers.json`:
OpenAI, Gemini, DeepSeek, OpenRouter, Groq, **local Ollama (no key)**, or any
OpenAI-compatible endpoint (vLLM, LM Studio, …). Keys live only in that local
file, are sent **only** to the endpoint you configure, and are never echoed
back unmasked. Empty key ⇒ falls back to the provider's env var
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, …). See [docs/BYOK.md](docs/BYOK.md).

## Calibration — the honesty engine

Out of the box detectors run on documented **default bands** and are visibly
flagged `uncalibrated`. Drop labeled samples into `data/bench/*.jsonl`
(`{"text": "...", "label": 1}` for AI / `0` for human) or use the built-in
demo set, then hit **Recalibrate** (UI or
`POST /api/calibration/run`). The backend fits a logistic map, chooses a
max-accuracy threshold, and reports **AUC / accuracy / ECE / Brier** per
detector. Consensus weights detectors by measured accuracy. The bundled
demo set is a *demo* — calibrate on data from your own domain for real use.

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
plugins/                 bundled example plugins
data/                    runtime state (history, keys, cache, fits) — local-only
docs/                    architecture, detector API, BYOK, roadmap
tests are under apps/api/tests
```

## Honest limitations

* **Adversarial text defeats detectors.** Rewritten, paraphrased or
  human-edited AI text and heavily-polished human text genuinely overlap.
  ZeroAIBench surfaces evidence and let humans decide — it must not be used
  as proof, or to accuse students/authors.
* Local-LM detectors default to small English-centric models (`gpt2`); for
  Chinese/other languages point them at multilingual models (e.g.
  `Qwen2.5-0.5B`) via detector settings.
* The bundled demo calibration set is small and hand-written for demo
  purposes. It is not a benchmark.
* Scores are probabilities only as far as their calibration says (that's why
  calibration status is displayed everywhere).

## License

MIT — see [LICENSE](LICENSE). Detector methods belong to their authors and
papers, linked from each detector card and the docs.

## More docs

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit
* [docs/DETECTOR_API.md](docs/DETECTOR_API.md) — write your own detector
* [docs/BYOK.md](docs/BYOK.md) — provider configuration details
* [docs/ROADMAP.md](docs/ROADMAP.md) — where this is heading

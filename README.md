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

## Quickstart — two doors, same house

### 🖱️ No-code (Windows): download → double-click

You never touch a terminal. Nothing is installed into your system.

1. On the GitHub repo page click **Code → Download ZIP**, unzip anywhere
   (or `git clone` if you have git).
2. Inside the folder, double-click **`安装环境.bat`** — once, ever (~1–3 min:
   it creates a private environment; your existing Python/conda is untouched).
3. From now on, double-click **`启动工作台.bat`** any time → your browser
   opens **http://localhost:8000**.
   - The black window that appears **is** the tool: keep it open while using,
     close it to stop.
   - For Binoculars / Fast-DetectGPT / LLM-Perplexity, one extra line in a
     terminal once — see *Native LM detectors* below.

Requirements: Windows + [Python 3.10+](https://www.python.org/downloads/)
(tick **"Add Python to PATH"**). That's all — the built-in web UI ships
inside the package, so **Node.js is not needed**.

### ⌨️ Developers / Linux / macOS: clone + scripts

```bash
git clone https://github.com/YiCQi/ZeroAIBench.git
cd ZeroAIBench

# Windows (PowerShell)
scripts\setup.ps1         # one-time: venv + backend deps + web deps (~1 min)
scripts\dev.ps1           # Vite dev UI on :5173 with hot-reload (needs Node 18+)

# Linux / macOS
./scripts/setup.sh
./scripts/dev.sh
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

### Native LM detectors (optional, one command)

LLM-Perplexity / Fast-DetectGPT / Binoculars / HF-Classifier need torch.
Enable them (once):

```powershell
# Windows
scripts\setup.ps1 -Ml          # + ~2 GB: torch + transformers
```
```bash
# Linux / macOS
./scripts/setup.sh --with-ml
```

First analysis then auto-downloads the GPT-2 family (~2 GB) and caches it
— fully offline after that. Behind a restricted network, set
`HF_ENDPOINT=https://hf-mirror.com` first (see Troubleshooting).

### Using the workbench (the part no README usually tells you)

The **Workbench tab** is 90% of your time: paste the text → tick detectors →
**Analyze**. Then read the page top-to-bottom in this order:

1. **Consensus** — the calibration-weighted vote across detectors, with an
   explicit agreement level. If detectors disagree, the notes say *who said
   what* — disagreement is information, not a bug.
2. **Per-detector cards** — every `score` is normalized so **higher = more
   AI**, always (`uncertain` is a verdict, not a failure). Raw values shown
   under evidence can point the other way; each card states its direction,
   e.g. *Binoculars: raw 6.4, lower = AI*.
3. **Heatmap** — red ≈ AI-looking, blue ≈ human-lenient, per sentence or
   paragraph. Answers "**which part** looks AI."
4. **Evidence chips** — the numbers behind the verdict: perplexity 2.9,
   burstiness 0.08, 11 stock-phrase hits… Answers "**why**." These are the
   bits you can quote to a human.

**Providers tab**: paste any OpenAI-compatible or Gemini API key (DeepSeek,
OpenRouter, or a local Ollama — even keyless) to enable the **LLM Judge**
detector. Keys stay in local `data/providers.json`, masked on screen.

**Calibration tab**: one click fits honest score curves on a labeled corpus
(try the bundled 24-sample `demo` first) — afterwards cards show their AUC /
accuracy, and consensus weights follow calibrated quality automatically.
Feed your own `data/bench/*.jsonl` whenever you have labeled text.

**Methodology tab**: what each detector measures, what it's blind to.

> Privacy: all detectors except the LLM Judge run **on your machine**; text,
> history, keys never leave it. Turn off the Judge (or point it at local
> Ollama/vLLM) and ZeroAIBench is fully offline.

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

**零基础用户（Windows，全程不碰命令行）：**

1. 项目主页点 **Code → Download ZIP**，解压到任意位置
2. 双击 **`安装环境.bat`**（只需一次，约 1–3 分钟，自动创建隔离环境，不碰你电脑上的任何其他 Python）
3. 以后每次用：双击 **`启动工作台.bat`** → 浏览器自动打开 http://localhost:8000
   - 弹出的黑窗口就是程序本体：使用期间**保持开着**，关掉窗口就是退出
4. 想加 Binoculars / Fast-DetectGPT / LLM-Perplexity？在终端里跑一次
   `scripts\setup.ps1 -Ml`（见上面 *Native LM detectors*）

（前提：装好 Python 3.10+，安装时勾选 **"Add Python to PATH"**。不需要 Node。）

**开发者 / Linux / macOS：**

```bash
git clone https://github.com/YiCQi/ZeroAIBench.git && cd ZeroAIBench
scripts\setup.ps1        # Windows：一次性建好隔离环境（不碰你的 conda/系统 Python）
scripts\dev.ps1          # 启动后端+Vite 热重载前端（:5173，改前端代码时用）
# Linux/macOS 同理：./scripts/setup.sh && ./scripts/dev.sh
# 或者一条命令（自带前端构建）：docker compose up → http://localhost:8000
# 日常自用其实只需：cd apps\api && python -m zeroaibench → :8000（UI 已打进包里）
```

粘贴文章 → 勾选检测器（Stylometry 零依赖即可用；本地 LM 检测器需要
`pip install -e ".[ml]"`；LLM Judge 在 Providers 页接入你自己的
OpenAI / Gemini / DeepSeek / OpenRouter / Ollama / 任何 OpenAI 兼容端点）→
Analyze。你会得到：每个检测器的分数 + 原始统计量 + 句子级 heatmap +
证据解释 + 校准状态（AUC/ECE）+ 跨检测器共识与分歧提示。

### 关于 exe 版本（为什么不打包单个 .exe）

技术上可行（PyInstaller），但体验反而更差：torch 打进去约 2 GB+，
onefile 模式**每次启动都需要解压 1–3 分钟**，杀毒软件误报频发。所以：
- **现在的"绿色版"就是答案**——ZIP 下载 + 双击两个 .bat，已经是
  "不用装任何东西"的体验（只依赖 Python）；
- 下一步见 [ROADMAP](docs/ROADMAP.md)：将用嵌入式 Python（免装
  Python，约 20 MB）打包 portable zip，做到**真·零依赖**单压缩包。

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

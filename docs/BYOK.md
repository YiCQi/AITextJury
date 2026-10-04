# BYOK — Bring Your Own Key

AITextJury's LLM-Judge detector (and anything else that needs a large
model — future detectors, summarizers) calls **your** provider with **your**
key. There are no vendor SDKs in this project — provider adapters are plain
[`httpx`](https://www.python-httpx.org/) — and there is exactly one place a
key can be stored: `data/providers.json` on your own machine.

```
AITextJury ──(your key, httpx)──► provider you configured
                                          OpenAI / Gemini / DeepSeek /
                                          OpenRouter / Groq / Ollama /
                                          vLLM / LM Studio / one-api / …
```

## The two adapter kinds

| `kind`              | Speaks                                            | Covers |
|---------------------|---------------------------------------------------|--------|
| `openai_compatible` | `POST {base_url}/chat/completions`, `GET {base_url}/models` | OpenAI, DeepSeek, OpenRouter, Groq, Ollama (`http://localhost:11434/v1`), vLLM, LM Studio (`http://localhost:1234/v1`), one-api/new-api gateways, any OpenAI-shaped API |
| `gemini`            | Google `generateContent` REST v1beta             | Google AI Studio / Gemini API |

Everything else in the workbench — Stylometry, Fast-DetectGPT, Binoculars,
LM-perplexity, HF classifiers — runs **local models via Hugging Face**, no
key involved.

## Configuring a provider

Easiest: the **Providers** tab in the web UI — pick a template, paste the
key, `POST /api/keys/{id}/test` runs a cheap auth check before you save
anything.

Programmatically, `PUT /api/keys` with:

```json
{
  "id": "openai:work",
  "kind": "openai_compatible",
  "base_url": "https://api.openai.com/v1",
  "default_model": "gpt-4o-mini",
  "api_key": "sk-…",
  "enabled": true,
  "note": "work account"
}
```

* `id` — anything unique. The part before `:` links it to a template for
  defaults (`openai:work` inherits OpenAI's base URL, model hint and env var).
* `api_key` may be **empty** — see env fallback below. On update (same `id`),
  an empty key means *"keep the existing one"*; the frontend relies on this
  and never echoes keys back.
* `GET /api/keys` always returns a **masked** key (`sk-1…9f2a`) plus
  `has_key`/`key_mask` — never the key itself.

## Key resolution order

At call time each provider resolves its key like this:

1. explicit `api_key` stored in `data/providers.json`;
2. else the template's `env_key` environment variable.

So both of these work and can even coexist:

```bash
export DEEPSEEK_API_KEY=sk-xxxx           # no UI step needed at all
python -m aitextjury
```

| Template     | `env_key`          | `base_url`                                          | model hint |
|--------------|--------------------|-----------------------------------------------------|------------|
| `openai`     | `OPENAI_API_KEY`   | `https://api.openai.com/v1`                        | `gpt-4o-mini` |
| `gemini`     | `GEMINI_API_KEY`   | `https://generativelanguage.googleapis.com/v1beta` | `gemini-2.0-flash` |
| `deepseek`   | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1`                      | `deepseek-chat` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1`                  | `openrouter/auto` |
| `groq`       | `GROQ_API_KEY`     | `https://api.groq.com/openai/v1`                  | `llama-3.3-70b-versatile` |
| `ollama`     | *(none)*           | `http://localhost:11434/v1`                       | `qwen2.5:7b-instruct` |
| `custom`     | *(none)*           | `http://localhost:8000/v1` — change freely        | `your-model` |

In Docker, environment fallback is the cleanest approach — pass keys via
compose:

```yaml
environment:
  AITEXTJURY_HOME: /data
  OPENAI_API_KEY: ${OPENAI_API_KEY:-}
```

### Fully local, zero keys

With Ollama or any local OpenAI-compatible server, the whole workbench —
including the judge — runs offline:

```json
{
  "id": "ollama",
  "kind": "openai_compatible",
  "base_url": "http://localhost:11434/v1",
  "default_model": "qwen2.5:14b-instruct",
  "api_key": ""
}
```

(Docker users: `http://host.docker.internal:11434/v1` for an Ollama running on
the host.)

## What the key is used for

Only one built-in consumer today: the **LLM-Judge detector** (`llm_judge`).
It sends the text under analysis plus a fixed, auditable system prompt
(`apps/api/aitextjury/detectors/llm_judge.py`) and requests strict JSON:
`{"verdict": "ai"|"human"|"mixed", "confidence": 0-100, "reason": "…",
"flagged": ["…"]}`. It uses `temperature=0`, `json_mode` when the provider
supports it, and honors `timeout_llm` (default 120 s) from settings.

If no usable provider is configured, the Judge reports
`status: "unavailable"` with a hint — it never blocks the rest of the
analysis run.

## Privacy & storage model

* `data/providers.json` is gitignored; keys never appear in analysis
  results, history entries, caches, or the calibration corpus.
* Keys are sent to exactly one place: `Authorization: Bearer …` to the
  configured `base_url` (Gemini uses the key as a query parameter — that's
  Google's own REST API shape; use HTTPS.)
* The Judge prompt contains the analyzed text — that is unavoidable — so
  route sensitive text to a local provider (Ollama/vLLM) instead.
* `DELETE /api/keys/{pid}` removes a provider and its key from disk
  immediately.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `401 unauthorized` in the provider test | key wrong, or env var not visible to the server process — restart after `export` |
| Judge `unavailable: no provider` | provider disabled, or `enabled: false`; check `GET /api/keys` |
| Ollama connection refused | server not running (`ollama serve`), or wrong port/base_url; container → `host.docker.internal` |
| Judge JSON parse fallback | some providers ignore `json_mode`; the Judge has a brace-balancing parser, but strongly non-JSON replies will surface as `error` with the raw tail kept in evidence |
| Slow judge → timeout | raise `settings.detect.timeout_llm`, or use a faster model |

See [ARCHITECTURE.md](ARCHITECTURE.md) for how provider availability flows
into `AnalysisContext`, and [ROADMAP.md](ROADMAP.md) for where BYOK is headed
(per-detector model pinning, caching of judge verdicts).

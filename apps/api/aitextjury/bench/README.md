# Built-in calibration sets

| file         | what                                                    | license |
|--------------|---------------------------------------------------------|---------|
| `demo.jsonl` | 24 hand-written "AI-style" / "human-style" snippets — a smoke test, not a benchmark (neither side is real) | MIT (this repo) |
| `hc3.jsonl`  | 200 real samples: 100 English + 100 Chinese, half human answers, half ChatGPT answers to the *same* questions | per sample, see below |

## hc3.jsonl

A balanced subset of the **Human ChatGPT Comparison Corpus (HC3)**:

> Guo, Biyang, et al. *How Close is ChatGPT to Human Experts? Comparison
> Corpus, Evaluation, and Detection.* arXiv:2301.07597 (2023).
> <https://huggingface.co/datasets/Hello-SimpleAI/HC3>,
> <https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese>

Only HC3 splits whose upstream source has a clear, redistributable license
were used; each sample carries its `license` field:

| lang | HC3 split    | upstream source            | license      | pairs |
|------|--------------|----------------------------|--------------|-------|
| en   | `wiki_csai`  | Wikipedia                  | CC-BY-SA-4.0 | 50 |
| zh   | `finance`    | FinanceZhidao              | CC-BY-4.0    | 17 |
| zh   | `open_qa`    | WebTextQA & BaikeQA        | MIT          | 17 |
| zh   | `psychology` | Baidu AI Studio dataset    | CC0-1.0      | 16 |

Excluded on purpose: `reddit_eli5`, `open_qa` (en), `medicine`, `finance`
(en), `baike`, `law`, `nlpcc_dbqa` (unknown, missing or non-commercial
licenses).

Changes from the original: answers were whitespace-normalized; one human
and one ChatGPT answer were taken per question (length 250–2500 chars for
English, 120–2000 for Chinese), sampled with a fixed seed. As an adaptation
of CC-BY-SA material, this file is distributed under **CC-BY-SA-4.0**
(<https://creativecommons.org/licenses/by-sa/4.0/>), separately from the
MIT-licensed code.

Caveats: the AI side is ChatGPT (GPT-3.5, early 2023) — newer models write
differently; the English side is all Wikipedia-style CS/AI topics. Treat
the resulting AUCs as a sanity check and calibrate on your own domain.

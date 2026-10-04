"""Shared plumbing for local language-model detectors.

Design:
* optional heavy imports — availability() reports a friendly hint when
  `torch`/`transformers` are missing instead of crashing the API;
* a process-wide `ModelHub` so multiple detectors share model weights
  (gpt2 is used by several detectors — load once, run many);
* offset-aware chunking: texts longer than the model context are split at
  sentence boundaries with overlap; per-token stats are aggregated back to
  sentences for the heatmap.

Everything here runs on CPU by default; CUDA is used automatically when
available (set detector setting `device: cpu` to force CPU).
"""
from __future__ import annotations

import bisect
import math
import threading
from dataclasses import dataclass, field

_HAS_ML = None
_ML_ERR = ""


def ml_status() -> tuple[bool, str]:
    global _HAS_ML, _ML_ERR
    if _HAS_ML is None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            _HAS_ML, _ML_ERR = True, ""
        except Exception as e:  # pragma: no cover
            _HAS_ML, _ML_ERR = False, str(e)
    return _HAS_ML, _ML_ERR


def ml_requirement_hint() -> str:
    return ('rerun setup with -Ml: scripts\\setup.ps1 -Ml (Windows) / '
            './scripts/setup.sh --with-ml (Linux/macOS) - or: pip install '
            '"aitextjury[ml]"')


class ModelHub:
    """Thread-safe lazy cache of HF models/tokenizers."""

    def __init__(self):
        self._models: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}
        self._lock = threading.Lock()

    def tokenizer(self, model_id: str):
        import transformers
        with self._lock:
            if model_id not in self._tokenizers:
                self._tokenizers[model_id] = transformers.AutoTokenizer.from_pretrained(
                    model_id)
            return self._tokenizers[model_id]

    def causal(self, model_id: str, device: str | None = None):
        import torch
        import transformers
        with self._lock:
            if model_id not in self._models:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_id)
                model.eval()
                device = device or resolve_device()
                model.to(device)
                self._models[model_id] = model
            return self._models[model_id]

    def device_of(self, model) -> str:
        import torch
        return next(model.parameters()).device.type


HUB = ModelHub()


def resolve_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------- chunking --

@dataclass
class TokenizedText:
    """Whole-text tokenization with char offsets (byte-accurate for CJK too)."""
    input_ids: list[int]
    offsets: list[tuple[int, int]]   # char start/end per token
    sent_start_chars: list[int]      # sorted sentence start offsets
    sentences_by_index: dict[int, str]  # ordinal -> sentence segment id
    max_ctx: int
    model_id: str


@dataclass
class Chunk:
    text_span: tuple[int, int]                  # char coverage (informational)
    token_ids: list[int] = field(default_factory=list)
    token_pos: list[int] = field(default_factory=list)  # global token indices

    def __len__(self) -> int:
        return len(self.token_ids)


def tokenize_for_heatmap(model_id: str, text: str, sentences, max_ctx: int) -> TokenizedText:
    """Tokenize once and prepare sentence attribution data."""
    tok = HUB.tokenizer(model_id)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
              truncation=False)
    ids = list(enc["input_ids"])
    offsets = [(o[0], o[1]) for o in enc["offset_mapping"]]
    # keep only non-empty token spans (skip zero-width artifacts)
    keep = [i for i, (s, e) in enumerate(offsets) if e > s]
    ids = [ids[i] for i in keep]
    offsets = [offsets[i] for i in keep]

    sent_start_chars = [s.start for s in sentences]
    sentence_ids = [s.id for s in sentences]
    # map each token to the sentence that contains its start char
    return TokenizedText(
        input_ids=ids, offsets=offsets,
        sent_start_chars=sent_start_chars,
        sentences_by_index={i: sid for i, sid in enumerate(sentence_ids)},
        max_ctx=max_ctx, model_id=model_id,
    )


def token_sentence(tt: TokenizedText, token_idx: int) -> int:
    """Sentence ordinal for a token (by char start), or -1 if before first."""
    import bisect as _b
    char = tt.offsets[token_idx][0]
    return _b.bisect_right(tt.sent_start_chars, char) - 1


def build_chunks(tt: TokenizedText, overlap: int = 128,
                 chunk_tokens: int | None = None) -> list[Chunk]:
    """Split tokens into model-sized chunks aligned at sentence boundaries.

    The next chunk starts up to `overlap` tokens before the previous chunk's
    end (sentence-aligned when possible) so boundary sentences get fully
    contextualised scores. Tokens covered twice contribute twice to the
    later sentence-average — cheap and robust.
    """
    n = len(tt.input_ids)
    limit = chunk_tokens or tt.max_ctx
    if limit < 32:
        raise ValueError("chunk size too small")
    if n == 0:
        return []
    chunks: list[Chunk] = []
    start = 0
    guard = 0
    while start < n and guard < 10_000:
        guard += 1
        end = min(n, start + limit)
        if end < n:
            # prefer cutting at a sentence boundary inside the last 30%
            window_lo = start + int((end - start) * 0.7)
            cut = end
            for i in range(end, max(window_lo, start + 1), -1):
                if token_sentence(tt, i) >= 0 and \
                        token_sentence(tt, i) != token_sentence(tt, i - 1):
                    cut = i
                    break
            end = cut
        chunks.append(Chunk(
            text_span=(tt.offsets[start][0], tt.offsets[end - 1][1]),
            token_ids=tt.input_ids[start:end],
            token_pos=list(range(start, end)),
        ))
        if end >= n:
            break
        nxt = max(start + 1, end - overlap)
        for i in range(nxt, max(start + 1, nxt - overlap), -1):
            if token_sentence(tt, i) >= 0 and \
                    token_sentence(tt, i) != token_sentence(tt, i - 1):
                nxt = i
                break
        start = nxt
    return chunks


# ------------------------------------------------------------- per-token ops
#
# Memory note: log_softmax over a chunk is O(T·V). With ~512-token chunks and
# gpt2's 50k vocab that is ~100 MB fp32 per model per chunk on CPU — the
# default `chunk_tokens` of 512 keeps pair-model detectors under ~300 MB.


def forward_stats(model, token_ids: list[int]):
    """One teacher-forced pass -> (logp rows, target log-prob, entropy).

    logp[i] is the row predicting token_ids[i+1] given token_ids[:i+1]:
    alignment is shift-by-one (row i predicts position i+1).
    """
    import torch
    dev = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        logits = model(ids).logits[0]          # (T, V)
        logp = torch.log_softmax(logits, dim=-1)
    # rows 0..T-2 predict tokens 1..T-1
    logp_rows = logp[:-1, :]
    tgt = torch.tensor(token_ids[1:], dtype=torch.long, device=dev)
    target_logp = logp_rows.gather(1, tgt.view(-1, 1)).view(-1)      # (T-1,)
    probs = logp_rows.exp()
    entropy = -(probs * logp_rows).sum(dim=-1)                       # (T-1,)
    return logp_rows, target_logp, entropy


def cross_ent_terms(logp_detector, logp_reference):
    """H(R_t, M_t) = sum_w R_t(w) * (−log M_t(w)) per position."""
    import torch
    with torch.no_grad():
        cross = (logp_reference.exp() * (-logp_detector)).sum(dim=-1)
    return cross


def aggregate_to_sentences(tt: TokenizedText,
                           token_values: list[float | None]):
    """Average per-token values onto sentences; returns {segment_id: (mean, n)}.

    `token_values` is indexed by *global token index* across the whole text;
    chunk overlap therefore folds naturally into the average.
    """
    sums: dict[int, list[float]] = {}
    for i, v in enumerate(token_values):
        if v is None:
            continue
        s_ord = token_sentence(tt, i)
        if s_ord < 0:
            continue
        sums.setdefault(s_ord, []).append(v)
    out: dict[str, tuple[float, int]] = {}
    for s_ord, vals in sums.items():
        sid = tt.sentences_by_index.get(s_ord)
        if sid is None:
            continue
        out[sid] = (sum(vals) / len(vals), len(vals))
    return out


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

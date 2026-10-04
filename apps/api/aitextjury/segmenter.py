"""Paragraph & sentence segmentation with absolute character offsets.

Offsets are the backbone of the heatmap: every detector reports scores keyed
by segment ids, and the UI re-uses the same offsets to paint the original
text. Boundary rules handle both Latin and CJK punctuation.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .utils import is_cjk_char

# Split paragraphs on one or more blank lines, tolerating spaces/tabs.
_PARA_SPLIT = re.compile(r"\n[ \t]*\n+")

# Sentence enders optionally followed by closing quotes/brackets and CJK
# closer marks. Latin abbreviations are over-split; we re-merge tiny gaps.
#
# Semantics: the whitespace after a Latin ender belongs to the NEXT sentence
# (offsets remain contiguous with the paragraph text). CJK enders (。！？；…)
# never need a following space.
_SENT_SPLIT = re.compile(
    r"""
    (?<=[.!?])                      # ASCII enders …
    (?=\s|["'”’」』）)\]])            # … then a space or closing mark
    |(?<=[…。！？；])                  # CJK enders split unconditionally
    |(?<=\n)                         # hard lines split inside paragraphs
    """,
    re.VERBOSE,
)

_MIN_SENT_CHARS = 14  # Latin; scaled down for CJK
_MIN_SENT_CJK = 4     # CJK sentences are naturally compact ("不是吗？")


@dataclass
class Segment:
    id: str
    kind: str      # "paragraph" | "sentence"
    text: str
    start: int
    end: int
    parent_id: str | None = None
    index: int = 0  # order within its kind

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "parent_id": self.parent_id,
        }


@dataclass
class Segmentation:
    paragraphs: list[Segment]
    sentences: list[Segment]

    def all(self) -> list[Segment]:
        return self.paragraphs + self.sentences


def _short_uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def segment_text(text: str) -> Segmentation:
    paragraphs: list[Segment] = []

    # custom split preserving offsets
    bounds: list[tuple[int, int]] = []
    last = 0
    for m in _PARA_SPLIT.finditer(text):
        bounds.append((last, m.start()))
        last = m.end()
    bounds.append((last, len(text)))

    para_idx = 0
    for s, e in bounds:
        raw = text[s:e]
        if not raw.strip():
            continue
        para = Segment(
            id=_short_uid("p"), kind="paragraph",
            text=raw, start=s, end=e, index=para_idx,
        )
        paragraphs.append(para)
        para_idx += 1

    if not paragraphs:  # no blank lines -> whole text is one paragraph
        para = Segment(id=_short_uid("p"), kind="paragraph",
                       text=text, start=0, end=len(text), index=0)
        paragraphs.append(para)

    sentences: list[Segment] = []
    for para in paragraphs:
        pieces = _split_sentences(para.text, para.start)
        for i, (s, e) in enumerate(pieces):
            raw = text[s:e]
            sentences.append(Segment(
                id=_short_uid("s"), kind="sentence", text=raw,
                start=s, end=e, parent_id=para.id, index=len(sentences),
            ))

    return Segmentation(paragraphs=paragraphs, sentences=sentences)


def _split_sentences(par: str, offset: int) -> list[tuple[int, int]]:
    """:returns absolute (start, end) char ranges of sentences in the paragraph."""
    out: list[tuple[int, int]] = []
    last = 0
    for m in _SENT_SPLIT.finditer(par):
        cut = m.end() if m.end() > m.start() else m.start()
        cand = (last, cut)
        out.append(cand)
        last = cut
    out.append((last, len(par)))

    # post-process: merge fragments that are too small into the previous one
    merged: list[list[int]] = []
    for s, e in out:
        frag = par[s:e]
        if not frag.strip() and not merged:
            continue
        min_len = _MIN_SENT_CJK if _has_cjk(frag) else _MIN_SENT_CHARS
        if merged and (len(frag.strip()) < min_len):
            merged[-1][1] = e
        elif not frag.strip() and merged:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    # trailing whitespace belongs to no sentence
    result = [(s + offset, e + offset) for s, e in merged if par[s:e].strip()]
    return result


def _has_cjk(s: str) -> bool:
    return any(is_cjk_char(c) for c in s)

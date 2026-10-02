"""Text utilities: language heuristics, tokenization helpers, common word lists.

Bilingual (Latin + CJK) from day one — the workbench should not silently
assume English inputs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CJK_RANGES = (
    "\u3000-\u303F"   # CJK punctuation
    "\u3040-\u30FF"   # hiragana+katakana
    "\u4E00-\u9FFF"   # CJK unified
    "\u3400-\u4DBF"   # ext A
    "\uF900-\uFAFF"
)

_WORD_RE = re.compile(r"[A-Za-z']+")
_CJK_WORD_RE = re.compile(rf"[{CJK_RANGES}]")
_LANGUAGE_WEIGHT_CJK = re.compile(rf"[{CJK_RANGES}]")

SENTENCE_BOUNDARY = "。！？!?…;.；。".replace("。", "。")  # keep readable; see regex below

EN_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into
    is it its just me more most my no nor not now of off on once only or other our out
    over own same she should so some such than that the their them then there these they
    this those through to too under until up very was we were what when where which while
    who whom why will with you your would could shall may might must""".split()
)

# Phrases/words strongly associated with LLM register in English prose.
# None of these *prove* anything — they are stylometric features, and a human
# copywriter can use them too. They feed the stylometry detector + evidence UI.
AI_TELL_PATTERNS_EN: list[tuple[str, str]] = [
    (r"\bdelve[sd]?\b", "delve"),
    (r"\btapestry\b", "tapestry"),
    (r"\bit('?s| is) important to note\b", "it is important to note"),
    (r"\bit('s| is) worth (noting|mentioning)\b", "it's worth noting"),
    (r"\bin today'?s fast-paced (world|landscape|digital)\b", "today's fast-paced world"),
    (r"\bwe (must|should) (also )?(consider|recognize|acknowledge)\b", "we must consider"),
    (r"\b(?<!\w)(furthermore|moreover|additionally)\b", "boilerplate connective (furthermore/moreover/additionally)"),
    (r"\bin conclusion\b", "in conclusion"),
    (r"\baltogether,?\b", "altogether,"),
    (r"\bfirstly\b|\bsecondly\b|\bthirdly\b", "firstly/secondly/thirdly"),
    (r"\bembark(ed)? on\b", "embark on"),
    (r"\bunlock(ing|ed)? (the )?(full )?potential\b", "unlock the potential"),
    (r"\bharness(ing)? (the )?power\b", "harness the power"),
    (r"\bleverag(e|ing|ed)\b", "leverage"),
    (r"\bseamless(ly)?\b", "seamlessly"),
    (r"\brobust\b", "robust"),
    (r"\bpivotal\b", "pivotal"),
    (r"\btestament to\b", "testament to"),
    (r"\bin the realm of\b", "in the realm of"),
    (r"\bnavigat(e|ing|ion) the (complex|ever|landscape)\b", "navigate the landscape"),
    (r"\bunderscore[sd]?\b", "underscore"),
    (r"\bfoster(s|ing|ed)?\b", "foster"),
    (r"\bcutting-edge\b", "cutting-edge"),
    (r"\bgame-?chang(er|ing)\b", "game-changer"),
    (r"\bnot only\b(?=.{0,60}\bbut also\b)", "not only… but also"),
]

AI_TELL_PATTERNS_ZH: list[tuple[str, str]] = [
    (r"在当今(这个)?(快节奏的)?(时代|世界|数字|社会)", "在当今…时代"),
    (r"随着.{0,12}的(不断)?发展", "随着…的发展"),
    (r"(总而言之|综上(所述|来看)|总的来说)", "综上所述"),
    (r"值得注意的是", "值得注意的是"),
    (r"让我们(一同|一起|共同)", "让我们一同"),
    (r"扮演(着)?(重要|关键)?角色", "扮演重要角色"),
    (r"发挥(着)?(重要|关键)(的)?作用", "发挥重要作用"),
    (r"不可或缺(的一部分)?", "不可或缺"),
    (r"旨在", "旨在"),
    (r"致力于", "致力于"),
    (r"(赋能|助力)", "赋能/助力"),
    (r"探讨(一下)?", "探讨"),
    (r"首先.{0,40}其次.{0,40}(最后|再次)", "首先/其次/最后式结构"),
    (r"在这个(充满|快速变化)", "在这个充满…"),
    (r"重要的(是|在于)", "重要的是"),
]


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    hits = len(_LANGUAGE_WEIGHT_CJK.findall(text))
    return hits / max(1, len(text))


def detect_language(text: str) -> tuple[str, float]:
    """Return ('en'|'zh'|'mixed'|'unknown'), cjk_ratio."""
    r = cjk_ratio(text)
    if r == 0.0:
        return "en", 0.0
    if r > 0.15:
        return "zh", r
    return "mixed", r


def word_tokens(text: str, lang: str = "en") -> list[str]:
    if lang in ("zh", "mixed"):
        base = _WORD_RE.findall(text)
        cjk = _CJK_WORD_RE.findall(text)
        # Chinese has no whitespace morphology; approximate a "word" with
        # bigrams of characters (common lightweight approach).
        cjk_grams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        return base + cjk_grams
    return _WORD_RE.findall(text)


HL = re.compile(rf"[{CJK_RANGES}]")
def is_cjk_char(ch: str) -> bool:
    return bool(ch) and bool(HL.match(ch))


@dataclass
class BasicStats:
    chars: int
    words: int


def basic_stats(text: str, lang: str) -> BasicStats:
    return BasicStats(chars=len(text), words=len(word_tokens(text, lang)))

"""Stylometry detector — linguistic fingerprint, zero dependencies.

No model downloads: this detector *always* runs, even before optional ML
packages are installed. It computes a bundle of well-known statistical
features that separate casual / human writing from fluent LLM register:

* burstiness (sentence-length coefficient of variation) — humans are spiky
* adjacent-sentence uniformity — machines keep steady rhythm
* repetition profile (hapax ratio, repeated-bigram rate) — humans repeat
* connective boilerplate rate (furthermore / moreover / 综上所述…)
* LLM-register "tell" phrase hits (delve, tapestry, 值得注意的是…)
* punctuation diversity, exclamations, paragraph-length uniformity

Each feature produces a *pull* in [-1, +1] (positive pulls toward "AI-like"),
weighted and combined into a raw score. This is stylometric evidence — a
careful human writer can be smooth; a machine can be told to write raggedly.
The UI shows per-feature evidence so the user judges the evidence, not a
black-box number.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..schemas import Availability, EvidenceItem
from ..utils import (AI_TELL_PATTERNS_EN, AI_TELL_PATTERNS_ZH,
                     EN_STOPWORDS, detect_language, is_cjk_char,
                     word_tokens)
from .base import AnalysisContext, BaseDetector, RawOutcome, RawSegment

CONNECTIVES_EN = re.compile(
    r"\b(furthermore|moreover|additionally|in addition|consequently|"
    r"nevertheless|nonetheless|therefore|thus|overall|firstly|secondly|"
    r"thirdly|finally|in summary|to conclude)\b", re.I)
CONNECTIVES_ZH = re.compile(
    r"(此外|另外|因此|然而|其次|首先|最后|综上所述|总之|总而言之|由此可见|"
    r"总的来说|与此同时|值得注意的是)")


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class StylometryDetector(BaseDetector):
    id = "stylometry"
    name = "Stylometry"
    family = "stylometry"
    description = (
        "Statistical linguistic fingerprint: burstiness, repetition profile, "
        "connective boilerplate, LLM-register tell phrases, punctuation "
        "habits. Zero dependencies, always available.")
    link = "https://github.com/YiCQi/AITextJury/blob/main/docs/DETECTOR_API.md"
    requires: list[str] = []
    default_enabled = True
    heavy = False
    DEFAULT_BANDS = (0.50, 0.20)

    def raw_direction(self) -> str:
        return "higher_is_ai"

    def availability(self, ctx: AnalysisContext | None = None) -> Availability:
        return Availability(ok=True)

    # ------------------------------------------------------------- analyze --

    async def analyze(self, ctx: AnalysisContext) -> RawOutcome:
        text = ctx.text
        lang, cjk = detect_language(text)
        sentences = ctx.segmentation.sentences
        paragraphs = ctx.segmentation.paragraphs

        sent_units = [self._unit_len(s.text) for s in sentences]
        sent_units = [u for u in sent_units if u > 0] or [1]
        tokens = word_tokens(text, lang)
        signals: dict[str, float] = {}
        pulls: list[tuple[str, float, float, str]] = []  # (key, pull, weight, note)

        # 1. burstiness: CV of sentence length (spiky -> human)
        mean_len = sum(sent_units) / len(sent_units)
        var = sum((x - mean_len) ** 2 for x in sent_units) / max(1, len(sent_units) - 1)
        cv = math.sqrt(var) / max(mean_len, 1e-6)
        signals["burstiness_cv"] = cv
        pulls.append(("burstiness", clamp((0.40 - cv) / 0.22),
                      1.3, f"sentence-length CV={cv:.2f} (machine prose tends <0.3, human prose >0.45)"))

        # 2. adjacent uniformity: mean |Δ| between successive sentences,
        #    normalized by mean length
        if len(sent_units) >= 3:
            deltas = [abs(sent_units[i + 1] - sent_units[i])
                      for i in range(len(sent_units) - 1)]
            adj = (sum(deltas) / len(deltas)) / max(mean_len, 1e-6)
            signals["adjacent_len_var"] = adj
            pulls.append(("adjacent_uniformity", clamp((0.35 - adj) / 0.25),
                          0.9, f"successive-sentence length jumps={adj:.2f} of mean — machines keep steady rhythm"))

        # 3. repetition profile: LLMs paraphrase-diverse-ify; humans re-use words
        total_tokens = max(1, len(tokens))
        counts = Counter(t.lower() for t in tokens if len(t) > 1)
        hapax = sum(1 for _, c in counts.items() if c == 1) / max(1, len(counts))
        signals["hapax_ratio"] = hapax
        # AI prose: many once-used "sophisticated" words -> high hapax
        pulls.append(("word_repetition", clamp((hapax - 0.62) / 0.20),
                      1.0, f"hapax ratio={hapax:.2f} (high = low re-use, common in LLM register)"))

        # 4. connective boilerplate
        conn_hits = len(CONNECTIVES_EN.findall(text)) + len(CONNECTIVES_ZH.findall(text))
        per_1000 = conn_hits / total_tokens * 1000
        signals["connectives_per_1000"] = per_1000
        pulls.append(("connective_boilerplate", clamp((per_1000 - 12.0) / 9.0),
                      1.0, f"boilerplate connectives={per_1000:.1f}/1k tokens"))

        # 5. LLM tell phrases
        tell_hits, tell_names = self._tell_hits(text, lang)
        per_para = tell_hits / max(1, len(paragraphs))
        signals["tell_phrase_hits"] = tell_hits
        pulls.append(("tell_phrases", clamp((tell_hits - 0.5) / 2.0),
                      1.6, f"{tell_hits} LLM-register phrase hits"
                      + (f" ({', '.join(tell_names[:6])})" if tell_names else "")))

        # 6. paragraph length uniformity
        if len(paragraphs) >= 3:
            plens = [self._unit_len(p.text) for p in paragraphs]
            pmean = sum(plens) / len(plens)
            pvar = sum((x - pmean) ** 2 for x in plens) / max(1, len(plens) - 1)
            pcv = math.sqrt(pvar) / max(pmean, 1e-6)
            signals["para_len_cv"] = pcv
            pulls.append(("para_uniformity", clamp((0.45 - pcv) / 0.35),
                          0.8, f"paragraph-length CV={pcv:.2f} — templates produce even blocks"))

        # 7. punctuation diversity & exclamations
        puncts = ".,;:!?—–()\"\u201c\u201d\u2018\u2019''…、。，！？；："
        distinct_punct = len({c for c in text if c in puncts})
        signals["punct_diversity"] = distinct_punct
        pull_punct = clamp((9 - distinct_punct) / 5.0) * 0.4
        exclam = len(re.findall(r"[!?！？]", text)) / max(1, len(sentences))
        signals["exclam_per_sentence"] = exclam
        pull_excl = clamp((0.04 - exclam) / 0.06) * 0.3
        pulls.append(("punctuation_profile", clamp(pull_punct + pull_excl),
                      0.7, f"punctuation diversity={distinct_punct}, exclam/sentence={exclam:.2f}"))

        # 8. stopword rate (casual human writing leans on function words)
        en_tokens = [t.lower() for t in tokens if t.isalpha()]
        stop_rate = sum(1 for t in en_tokens if t in EN_STOPWORDS) / max(1, len(en_tokens))
        signals["stopword_rate"] = stop_rate
        if len(en_tokens) >= 40 and cjk < 0.1:
            pulls.append(("function_words", clamp((0.44 - stop_rate) / 0.10),
                          0.6, f"function-word rate={stop_rate:.2f} — casual human text runs high; dense LLM prose runs low"))

        # ------------------------------------------------------ aggregate --
        wsum = sum(w for _, _, w, _ in pulls)
        combined = sum(p * w for _, p, w, _ in pulls) / max(wsum, 1e-6)
        raw = 0.5 + 0.5 * clamp(combined, -1, 1)

        evidence = self._evidence(pulls)
        # per-sentence scores --------------------------------------------
        med = sorted(sent_units)[len(sent_units) // 2] if sent_units else 0
        seg_scores: list[RawSegment] = []
        for s in sentences:
            u = self._unit_len(s.text)
            dev = abs(u - med) / max(med, 1e-6)
            uniform_pull = clamp((0.45 - dev) / 0.45) if med else 0.0
            hits = self._tell_hits(s.text, lang)[0]
            tell_pull = clamp(hits / 2.5)
            neighbors = self._neighbor_mean(s, sentences, ctx)
            if neighbors:
                ndev = abs(u - neighbors) / max(neighbors, 1e-6)
                uniform_pull = clamp(0.7 * clamp((0.5 - ndev) / 0.5) + 0.3 * uniform_pull)
            value = clamp(0.65 * uniform_pull + 0.35 * tell_pull, -1, 1)
            seg_scores.append(RawSegment(
                segment_id=s.id, value=value,
                extras={"uniformity": round(uniform_pull, 3),
                        "tell_hits": hits}))

        return RawOutcome(
            raw_score=raw,
            raw_direction="higher_is_ai",
            signals=signals,
            segment_scores=seg_scores,
            evidence=evidence,
            model="stylometric-v1",
            reference_note=("Stylometric fingerprint; see docs/DETECTOR_API.md "
                           "for the full feature list."))

    # ------------------------------------------------------------- helpers --

    def _unit_len(self, text: str) -> int:
        """Approximate length unit: words for Latin, characters for CJK-heavy text."""
        if sum(1 for c in text if is_cjk_char(c)) > len(text) // 4:
            return len([c for c in text if not c.isspace()])
        return len(text.split())

    def _tell_hits(self, text: str, lang: str) -> tuple[int, list[str]]:
        hits = 0
        names: list[str] = []
        patterns = AI_TELL_PATTERNS_EN + (AI_TELL_PATTERNS_ZH if lang in ("zh", "mixed") else [])
        for pattern, label in patterns:
            n = len(re.findall(pattern, text, re.IGNORECASE if pattern.isascii() else 0))
            if n:
                hits += n
                names.append(label)
        return hits, names

    def _neighbor_mean(self, sent, sentences, ctx) -> float:
        idx = next((i for i, s in enumerate(sentences) if s.id == sent.id), None)
        if idx is None:
            return 0.0
        lens = [self._unit_len(sentences[i].text)
                for i in (idx - 1, idx + 1) if 0 <= i < len(sentences)]
        return sum(lens) / len(lens) if lens else 0.0

    def _evidence(self, pulls) -> list[EvidenceItem]:
        out: list[EvidenceItem] = []
        for key, pull, weight, note in sorted(pulls, key=lambda t: -abs(t[1]) * t[2]):
            strength = abs(pull) * weight
            if strength < 0.35 and pull > -0.45:
                continue
            if pull > 0:
                sev = "high" if pull > 0.6 and weight >= 1.2 else "warn"
                out.append(EvidenceItem(title=f"{key}: leans AI", detail=note, severity=sev))
            else:
                out.append(EvidenceItem(title=f"{key}: leans human",
                                        detail=note, severity="info"))
            if len(out) >= 6:
                break
        return out

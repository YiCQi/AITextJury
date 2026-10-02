"""Pydantic schemas — the wire contract shared by the API, store and the frontend.

Score semantics (critical for comparability):

* Every normalized `score` is a *P(AI)-style* probability in [0, 1]
  ("higher = more likely AI"), regardless of the detector's raw direction.
* `raw_score` keeps the detector's native statistic (perplexity, curvature,
  cross-agreement, judge probability...) for transparency.
* `verdict` derives from the detector's calibrated threshold; near-threshold
  results deliberately land on "uncertain" — an honest workbench never turns
  a coin-flip into a confident answer.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    likely_ai = "likely_ai"
    likely_human = "likely_human"
    uncertain = "uncertain"


class DetectorFamily(str, Enum):
    stylometry = "stylometry"          # statistical / linguistic fingerprint
    local_lm = "local_lm"              # locally-run language models
    classifier = "classifier"          # fine-tuned sequence classifiers
    byok_llm = "byok_llm"              # remote LLM, user-provided key
    plugin = "plugin"                  # third-party plugin
    meta = "meta"                      # combines other detectors


class Availability(BaseModel):
    ok: bool = True
    reason: str = ""      # human-readable why-not if ok == False
    hints: list[str] = Field(default_factory=list)
    uncalibrated: bool = False  # available but no calibration fit yet


class EvidenceItem(BaseModel):
    title: str
    detail: str
    severity: str = "info"   # info | warn | high


class SegmentScore(BaseModel):
    id: str
    kind: str                # paragraph | sentence
    text: str
    start: int
    end: int
    parent_id: Optional[str] = None
    score: Optional[float] = None       # normalized [0,1], None = untouched
    raw_score: Optional[float] = None    # detector-native value for this span
    extras: dict[str, Any] = Field(default_factory=dict)


class CalibrationInfo(BaseModel):
    status: str = "uncalibrated"   # calibrated | uncalibrated | failed
    dataset: str = ""
    n_samples: int = 0
    auc: Optional[float] = None
    ece: Optional[float] = None
    brier: Optional[float] = None
    accuracy: Optional[float] = None
    threshold: float = 0.5
    fitted_at: Optional[str] = None


class DetectorResult(BaseModel):
    detector_id: str
    name: str
    family: str
    link: Optional[str] = None
    reference_note: Optional[str] = None
    score: Optional[float] = None       # normalized P(AI)-style probability
    raw_score: Optional[float] = None
    raw_direction: str = "higher_is_ai"  # or "lower_is_ai"
    verdict: Optional[Verdict] = None
    confidence: Optional[float] = None
    threshold: float = 0.5
    signals: dict[str, float] = Field(default_factory=dict)
    segment_scores: list[SegmentScore] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    calibration: CalibrationInfo = Field(default_factory=CalibrationInfo)
    runtime_ms: int = 0
    model: Optional[str] = None
    error: Optional[str] = None


class ConsensusEntry(BaseModel):
    detector_id: str
    score: float
    verdict: Verdict
    weight: float
    calibrated: bool


class ConsensusResult(BaseModel):
    score: float
    verdict: Verdict
    agreement: float                       # [0,1] — pairwise detector agreement
    contributors: list[ConsensusEntry] = Field(default_factory=list)
    segment_scores: list[SegmentScore] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TextStats(BaseModel):
    chars: int
    words: int
    sentences: int
    paragraphs: int
    language: str = "unknown"   # en | zh | mixed | unknown
    cjk_ratio: float = 0.0


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=60000)
    detector_ids: Optional[list[str]] = None   # None = all available defaults
    force_refresh: bool = False


class AnalyzeReport(BaseModel):
    id: str
    created_at: str
    title: str
    stats: TextStats
    text: str
    paragraphs: list[SegmentScore] = Field(default_factory=list)  # segmentation
    sentences: list[SegmentScore] = Field(default_factory=list)   # segmentation
    results: list[DetectorResult]
    consensus: ConsensusResult
    options: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0


class HistoryEntry(BaseModel):
    id: str
    created_at: str
    title: str
    stats: TextStats
    consensus_score: float
    consensus_verdict: Verdict
    detectors_used: list[str]


# ------------------------------------------------------------- providers ----

class ProviderConfig(BaseModel):
    id: str
    kind: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""            # stored locally, never echoed back in full
    default_model: str = ""
    enabled: bool = True
    note: str = ""


class PublicProvider(BaseModel):
    id: str
    kind: str
    base_url: str
    default_model: str
    enabled: bool
    note: str
    key_mask: str            # e.g. "sk-…4f2a" or "(env)" or "—"
    has_key: bool


class ProviderTestResult(BaseModel):
    id: str
    ok: bool
    detail: str = ""
    latency_ms: int = 0

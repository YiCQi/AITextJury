// Types mirroring apps/api/zeroaibench/schemas.py — keep in sync.

export type Verdict = "likely_ai" | "likely_human" | "uncertain";

export type DetectorFamily =
  | "stylometry" | "local_lm" | "classifier" | "byok_llm" | "plugin" | "meta";

export interface Availability {
  ok: boolean;
  reason: string;
  hints: string[];
}

export interface DetectorInfo {
  id: string;
  name: string;
  family: DetectorFamily;
  description: string;
  link: string | null;
  requires: string[];
  default_enabled: boolean;
  heavy: boolean;
  available: boolean;
  reason: string;
  hints: string[];
  uncalibrated: boolean;
  bands_help: string;
}

export interface EvidenceItem {
  title: string;
  detail: string;
  severity: "info" | "warn" | "high" | string;
}

export interface SegmentScore {
  id: string;
  kind: string;
  text: string;
  start: number;
  end: number;
  parent_id: string | null;
  score: number | null;
  raw_score: number | null;
  extras: Record<string, unknown>;
}

export interface CalibrationInfo {
  status: "calibrated" | "uncalibrated" | "failed" | string;
  dataset: string;
  n_samples: number;
  auc: number | null;
  ece: number | null;
  brier: number | null;
  accuracy: number | null;
  threshold: number;
  fitted_at: string | null;
}

export interface DetectorResult {
  detector_id: string;
  name: string;
  family: DetectorFamily;
  link: string | null;
  reference_note: string | null;
  score: number | null;
  raw_score: number | null;
  raw_direction: string;
  verdict: Verdict | null;
  confidence: number | null;
  threshold: number;
  signals: Record<string, number>;
  segment_scores: SegmentScore[];
  evidence: EvidenceItem[];
  calibration: CalibrationInfo;
  runtime_ms: number;
  model: string | null;
  error: string | null;
}

export interface ConsensusEntry {
  detector_id: string;
  score: number;
  verdict: Verdict;
  weight: number;
  calibrated: boolean;
}

export interface ConsensusResult {
  score: number;
  verdict: Verdict;
  agreement: number;
  contributors: ConsensusEntry[];
  segment_scores: SegmentScore[];
  notes: string[];
}

export interface TextStats {
  chars: number;
  words: number;
  sentences: number;
  paragraphs: number;
  language: string;
  cjk_ratio: number;
}

export interface AnalyzeReport {
  id: string;
  created_at: string;
  title: string;
  stats: TextStats;
  text: string;
  /** segmentation of the analyzed text (for heatmap offsets) */
  paragraphs: SegmentScore[];
  sentences: SegmentScore[];
  results: DetectorResult[];
  consensus: ConsensusResult;
  options: Record<string, unknown>;
  duration_ms: number;
}

export interface HistoryEntry {
  id: string;
  created_at: string;
  title: string;
  stats: TextStats;
  consensus_score: number;
  consensus_verdict: Verdict;
  detectors_used: string[];
}

export interface PublicProvider {
  id: string;
  kind: string;
  base_url: string;
  default_model: string;
  enabled: boolean;
  note: string;
  key_mask: string;
  has_key: boolean;
  env_key?: string;
}

export interface ProviderTemplate {
  kind: string;
  base_url: string;
  env_key: string;
  default_model: string;
}

export interface CalibrationFitPublic {
  status: string;
  dataset?: string;
  n?: number;
  auc: number | null;
  ece: number | null;
  brier: number | null;
  accuracy?: number | null;
  threshold?: number;
  when?: string;
  reason?: string;
  ok?: boolean;
  errors?: string[];
}

export interface Settings {
  detect: {
    timeout_local: number;
    timeout_llm: number;
    max_text_chars: number;
    max_history: number;
    cache_enabled: boolean;
    [k: string]: unknown;
  };
  detectors: Record<string, Record<string, unknown>>;
  [k: string]: unknown;
}

export interface BenchDataset {
  name: string;
  n: number;
  labels: { ai: number; human: number };
  where: string;
}

import type { Verdict } from "./types";

/** Color helpers shared by gauge + heatmap. */
export function scoreColor(score: number | null | undefined): string {
  if (score == null) return "transparent";
  // 0 -> human blue, .5 -> neutral, 1 -> AI hot
  if (score <= 0.35) return "#4fb3ff";
  if (score < 0.45) return "#6ba4b8";
  if (score < 0.55) return "#7d8698";
  if (score < 0.65) return "#b8845a";
  return "#f0644c";
}

export function verdictClass(v: Verdict | null | undefined): string {
  switch (v) {
    case "likely_ai": return "ai";
    case "likely_human": return "human";
    default: return "uncertain";
  }
}

export const VERDICT_LABEL: Record<string, string> = {
  likely_ai: "likely AI",
  likely_human: "likely human",
  uncertain: "uncertain",
};

export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function familyLabel(f: string): string {
  switch (f) {
    case "stylometry": return "stylometry";
    case "local_lm": return "local LM";
    case "classifier": return "classifier";
    case "byok_llm": return "BYOK LLM";
    case "plugin": return "plugin";
    case "meta": return "meta";
    default: return f;
  }
}

export function timeAgo(iso: string): string {
  const t = new Date(iso).getTime();
  const s = Math.max(1, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400 * 2) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

import type { Verdict } from "./types";

/**
 * Color helpers shared by gauge + heatmap — theme-aware. Dark keeps the
 * original luminous palette (colors sit on tinted dark backgrounds);
 * light darkens the same hues for contrast on white. setDarkUI() is
 * called by App on mount and on every theme toggle.
 */
const SCALE_DARK = ["#4fb3ff", "#6ba4b8", "#7d8698", "#b8845a", "#f0644c"];
const SCALE_LIGHT = ["#1683cf", "#54798f", "#5c6c7e", "#a0620b", "#d4402b"];

let darkUI = true;
export const setDarkUI = (d: boolean) => { darkUI = d; };
export const isDarkUI = () => darkUI;

// 0 -> human blue, .5 -> neutral, 1 -> AI hot
export function scoreColor(score: number | null | undefined): string {
  const s = darkUI ? SCALE_DARK : SCALE_LIGHT;
  if (score == null) return "transparent";
  if (score <= 0.35) return s[0];
  if (score < 0.45) return s[1];
  if (score < 0.55) return s[2];
  if (score < 0.65) return s[3];
  return s[4];
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

// ---------------------------------------------------------- text helpers --

const CJK_RE = /[㐀-鿿豈-﫿]/g;

/** Rough length in a language-aware way: CJK counts characters, the rest words. */
export function textSize(text: string): { cjk: number; words: number; label: string } {
  const cjk = (text.match(CJK_RE) || []).length;
  const words = text.replace(CJK_RE, " ").trim().split(/\s+/).filter(Boolean).length;
  const label = cjk > words ? `${cjk.toLocaleString()} 字` : `${words.toLocaleString()} words`;
  return { cjk, words, label };
}

/** True when the text is too short for any detector to say much. */
export function isTooShort(text: string): boolean {
  const { cjk, words } = textSize(text);
  return cjk + words < 60 && !(cjk >= 100);
}

// ------------------------------------------------------ detector helpers --

/** One plain-language line per built-in detector (full method text stays in "more"). */
export const DETECTOR_PLAIN: Record<string, string> = {
  llm_judge: "Asks your own AI model (via API key) to judge the writing and point at suspicious paragraphs.",
  binoculars: "Two small language models compare notes: machine text looks unusually predictable to both. Strong on English.",
  fast_detect_gpt: "Checks whether the text sits at a “too likely” peak for a language model. English only.",
  lm_perplexity: "How predictable each sentence is to a small language model — machine text tends to be smoother. English only.",
  hf_classifier: "A trained AI-text classifier you pick from Hugging Face.",
  stylometry: "Writing-style statistics: sentence rhythm, word reuse, stock phrases. Instant; English and Chinese.",
  length_rhythm: "Whether paragraph lengths are suspiciously even. Instant; a weak single signal.",
};

export function speedLabel(d: { heavy: boolean; family: string }): string {
  if (d.family === "byok_llm") return "uses your API";
  return d.heavy ? "local model · slower" : "instant";
}

export const VERDICT_HEADLINE: Record<string, string> = {
  likely_ai: "Likely AI-generated",
  likely_human: "Likely human-written",
  uncertain: "Inconclusive",
};

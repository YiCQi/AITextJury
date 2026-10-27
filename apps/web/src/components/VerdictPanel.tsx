import { useState } from "react";
import type { AnalyzeReport } from "../types";
import type { SampleText } from "../api";
import { Gauge } from "./Gauge";
import { textSize, VERDICT_HEADLINE, VERDICT_LABEL, verdictClass } from "../util";

/**
 * The top of the results: one plain sentence first, the evidence trail
 * (who voted what, with which weight) one click away.
 */
export function VerdictPanel({ report, sample }:
{
  report: AnalyzeReport;
  /** set when the text came from "try a real example" — reveals the truth */
  sample: SampleText | null;
}) {
  const [copied, setCopied] = useState(false);
  const c = report.consensus;
  const names = Object.fromEntries(report.results.map((r) => [r.detector_id, r.name]));
  const agreeing = c.contributors.filter((e) => e.verdict === c.verdict).length;
  const abstained = report.results.filter((r) => r.error?.startsWith("abstained: ")).length;
  const failed = report.results.filter((r) => r.error && !r.error.startsWith("abstained: ")).length;

  const subline = c.contributors.length === 0
    ? "No detector produced a score."
    : c.verdict === "uncertain"
      ? `The detectors don't agree clearly (score ${c.score.toFixed(2)}).`
      : `${agreeing} of ${c.contributors.length} detector${c.contributors.length > 1 ? "s" : ""} point this way · score ${c.score.toFixed(2)}`;

  const copySummary = async () => {
    const lines = [
      `AITextJury: ${VERDICT_HEADLINE[c.verdict]} (score ${c.score.toFixed(2)}, agreement ${(c.agreement * 100).toFixed(0)}%)`,
      ...report.results.map((r) => r.score != null
        ? `- ${r.name}: ${r.score.toFixed(2)} (${VERDICT_LABEL[r.verdict ?? "uncertain"]})`
        : `- ${r.name}: ${r.error}`),
      ...c.notes.map((n) => `note: ${n}`),
    ];
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked: the JSON download still works */ }
  };

  const download = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `aitextjury-${report.id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className={`panel verdict ${verdictClass(c.verdict)}`}>
      <div className="verdict-top">
        <Gauge score={c.score} verdict={c.verdict} threshold={0.5} size={1.05} />
        <div style={{ flex: 1, minWidth: 220 }}>
          <div className="verdict-headline">{VERDICT_HEADLINE[c.verdict]}</div>
          <div className="verdict-sub">{subline}</div>
          <div className="verdict-meta">
            {report.stats.language === "zh" ? "Chinese" : report.stats.language === "en" ? "English" : report.stats.language}
            {" · "}{textSize(report.text).label}
            {abstained > 0 && <> · {abstained} sat out (language)</>}
            {failed > 0 && <> · <span style={{ color: "var(--warn)" }}>{failed} failed</span></>}
            {" · "}{report.duration_ms < 1000
              ? `${report.duration_ms} ms`
              : `${(report.duration_ms / 1000).toFixed(1)} s`}
          </div>
          {sample && (
            <div className={`truth ${sample.label === 1 ? "ai" : "human"}`}>
              Ground truth: <b>{sample.label === 1 ? "AI-generated (ChatGPT)" : "human-written"}</b>
              {" "}— {sample.source.replace(/^hc3\//, "HC3 ")}
            </div>
          )}
        </div>
        <div className="verdict-actions">
          <button className="ghost small" onClick={copySummary}>{copied ? "copied ✓" : "Copy summary"}</button>
          <button className="ghost small" onClick={download}>Download JSON</button>
        </div>
      </div>
      <p className="hint" style={{ margin: "10px 0 0" }}>
        Evidence, not proof: edited or paraphrased AI text and very formal human
        writing both fool detectors. Read the highlights below before deciding.
      </p>

      {c.notes.length > 0 && (
        <ul className="notes">
          {c.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}

      {c.contributors.length > 0 && (
        <details className="more" style={{ marginTop: 8 }}>
          <summary>how this score was computed</summary>
          <table className="flat" style={{ marginTop: 6 }}>
            <thead>
              <tr><th>detector</th><th>score</th><th>vote</th><th>weight</th><th>calibration</th></tr>
            </thead>
            <tbody>
              {c.contributors.map((e) => (
                <tr key={e.detector_id}>
                  <td>{names[e.detector_id] ?? e.detector_id}</td>
                  <td className="num">{e.score.toFixed(2)}</td>
                  <td><span className={`chip ${verdictClass(e.verdict)}`}>{VERDICT_LABEL[e.verdict]}</span></td>
                  <td className="num">{e.weight.toFixed(2)}</td>
                  <td>{e.calibrated ? "calibrated" : "default bands"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            Weighted mean of the scores. Weight grows with a detector's measured
            accuracy (AUC) on the calibration set; uncalibrated detectors count 0.5.
          </p>
        </details>
      )}
    </div>
  );
}

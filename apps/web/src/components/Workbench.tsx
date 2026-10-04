import { useEffect, useState } from "react";
import type { AnalyzeReport, DetectorInfo } from "../types";
import { analyze, fetchDetectors } from "../api";
import { Heatmap } from "./Heatmap";
import { DetectorCard } from "./DetectorCard";
import { Gauge } from "./Gauge";
import { fmtPct } from "../util";

/** The main Workbench tab: input → detector panel → run → evidence. */
export function Workbench() {
  const [text, setText] = useState("");
  const [detectors, setDetectors] = useState<DetectorInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<AnalyzeReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const { detectors } = await fetchDetectors();
        setDetectors(detectors);
        setSelected(new Set(detectors.filter(d => d.default_enabled).map(d => d.id)));
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id); else next.add(id);
    setSelected(next);
  };

  const run = async () => {
    if (!text.trim() || selected.size === 0) return;
    setRunning(true);
    setError(null);
    try {
      const ids = [...selected];
      const rep = await analyze(text, ids);
      setReport(rep);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  const nw = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.25fr 2fr", gap: 18, alignItems: "start" }}>
      {/* ------------------------------------------------- left: input column */}
      <div>
        <div className="panel">
          <h3>Text</h3>
          <p className="hint">
            Paste an article, essay, review… anything. Everything runs on your
            machine (except BYOK LLM-Judge calls you enable).
          </p>
          <textarea
            className="main"
            value={text}
            placeholder={"Paste text here…\n\nParagraphs get split on blank lines; sentences on punctuation (Chinese 。！？ supported)."}
            onChange={(e) => setText(e.target.value)}
          />
          <div style={{ display: "flex", gap: 14, marginTop: 10,
                       color: "var(--text-faint)", fontSize: 12, fontFamily: "var(--mono)" }}>
            <span>{text.length} chars</span>
            <span>{nw} words</span>
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 12, alignItems: "center" }}>
            <button className="primary" disabled={running || !text.trim() || selected.size === 0}
              onClick={run}>
              {running ? (<><span className="spin" />analyzing…</>) : "Analyze"}
            </button>
            {report && (
              <span style={{ color: "var(--text-faint)", fontSize: 12 }}>
                last run {report.duration_ms} ms
              </span>
            )}
          </div>
          {error && (
            <div style={{ color: "var(--warn)", marginTop: 10, fontSize: 12.5 }}>
              {error}
            </div>
          )}
        </div>

        <DetectorPanel detectors={detectors} selected={selected}
          onToggle={toggle} />
      </div>

      {/* --------------------------------------------------- right: results */}
      <div>
        {!report && (
          <div className="panel" style={{ textAlign: "center", color: "var(--text-faint)" }}>
            <h3>Results</h3>
            <p style={{ minHeight: 120 }}>
              No analysis yet. AITextJury will show each detector's score
              <i> with its evidence</i> — heatmaps, surprisal, calibration
              status — not a single opaque percentage.
            </p>
          </div>
        )}
        {report && (
          <>
            <div className="panel">
              <h3>Consensus</h3>
              <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
                <Gauge score={report.consensus.score}
                  verdict={report.consensus.verdict}
                  threshold={0.5}
                  size={1.18} />
                <div style={{ flex: 1, minWidth: 260 }}>
                  <div className="kv">
                    <span className="k">agreement</span>
                    <span className="v">{fmtPct(report.consensus.agreement)} across detectors</span>
                  </div>
                  <div className="kv">
                    <span className="k">detectors</span>
                    <span className="v">
                      {report.consensus.contributors.length} contributed ·{" "}
                      {report.results.filter((r) => r.error).length} failed
                    </span>
                  </div>
                  {report.consensus.contributors.map((c) => (
                    <div className="kv" key={c.detector_id}>
                      <span className="k">{c.detector_id}</span>
                      <span className="v">
                        {c.score.toFixed(2)} · w={c.weight.toFixed(2)} ·{" "}
                        <span style={{ color: c.calibrated ? "var(--accent)" : "var(--warn)" }}>
                          {c.calibrated ? "calibrated" : "uncalibrated"}
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              {report.consensus.notes.length > 0 && (
                <ul className="notes" style={{ marginTop: 10 }}>
                  {report.consensus.notes.map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              )}
            </div>

            <Heatmap report={report} />

            <div className="panel">
              <h3>Detectors</h3>
              <div className="result-grid">
                {report.results.map((r) => (
                  <DetectorCard key={r.detector_id} result={r} />
                ))}
              </div>
            </div>

            <div className="panel">
              <h3>Run metadata</h3>
              <button className="ghost" onClick={() => setShowRaw(!showRaw)}>
                {showRaw ? "hide" : "show"} raw JSON
              </button>
              {showRaw && (
                <pre style={{
                  maxHeight: 360, overflow: "auto", fontSize: 11,
                  background: "var(--bg-soft)", padding: 12, borderRadius: 8,
                }}>
                  {JSON.stringify(report, null, 2)}
                </pre>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function DetectorPanel({ detectors, selected, onToggle }:
{
  detectors: DetectorInfo[];
  selected: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="panel">
      <h3>Detector panel</h3>
      <p className="hint">
        Combine independent methods. Unavailable ones explain why — and how to
        enable them.
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {detectors.map((d) => {
          const on = selected.has(d.id);
          return (
            <label key={d.id} style={{
              display: "flex", gap: 10, alignItems: "flex-start",
              padding: "9px 10px", borderRadius: 8,
              border: `1px solid ${on ? "var(--accent-dim)" : "var(--line-soft)"}`,
              background: on ? "rgba(33,201,147,0.045)" : "transparent",
              cursor: d.available ? "pointer" : "not-allowed",
              opacity: d.available ? 1 : 0.55,
            }}>
              <input type="checkbox" checked={on} disabled={!d.available}
                onChange={() => onToggle(d.id)}
                style={{ marginTop: 3 }} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <b>{d.name}</b>
                  <span className={`chip ${d.uncalibrated ? "warn" : "accent"}`}>
                    {d.uncalibrated ? "uncalibrated" : "calibrated"}
                  </span>
                  {d.heavy && <span className="chip">heavy</span>}
                  {!d.available && <span className="chip">unavailable</span>}
                </div>
                <div style={{ color: "var(--text-dim)", fontSize: 12.5, marginTop: 3 }}>
                  {d.available ? d.description : d.reason}
                </div>
                {!d.available && d.hints.length > 0 && (
                  <div style={{ color: "var(--text-faint)", fontSize: 11.5, marginTop: 3 }}>
                    {d.hints.join(" · ")}
                  </div>
                )}
                {d.available && d.uncalibrated && (
                  <div style={{ color: "var(--text-faint)", fontSize: 11.5, marginTop: 3 }}
                       title={d.bands_help}>
                    runs on default bands — calibrate it on a labeled set
                  </div>
                )}
              </div>
            </label>
          );
        })}
      </div>
    </div>
  );
}

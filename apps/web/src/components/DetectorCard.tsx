import { useState } from "react";
import type { DetectorResult } from "../types";
import { familyLabel, fmtPct } from "../util";
import { Gauge } from "./Gauge";

/** One detector's verdict card: gauge, signals, evidence list, provenance. */
export function DetectorCard({ result }:{ result: DetectorResult }) {
  const [open, setOpen] = useState(false);
  const calib = result.calibration;
  const hasSignals = Object.keys(result.signals).length > 0;

  if (result.error) {
    return (
      <div className="det-card">
        <div className="head">
          <span className="nm">{result.name}</span>
          <span className="chip">error</span>
        </div>
        <div style={{ color: "var(--warn)", fontSize: 12.5 }}>
          {result.error}
        </div>
      </div>
    );
  }

  return (
    <div className="det-card">
      <div className="head">
        <span className="nm">{result.name}</span>
        <span className="chip">{familyLabel(result.family)}</span>
        <span className="model">{result.model ?? ""}</span>
      </div>

      <div className="det-mid">
        <Gauge score={result.score} verdict={result.verdict}
          confidence={result.confidence} threshold={result.threshold} />
        <div className="numbers" style={{ flex: 1 }}>
          <div>
            raw score <b style={{ color: "var(--text)" }}>
              {result.raw_score?.toFixed(3) ?? "—"}
            </b>{" "}
            <span style={{ color: "var(--text-faint)" }}>
              ({result.raw_direction === "lower_is_ai" ? "lower = AI" : "higher = AI"})
            </span>
          </div>
          <div>
            threshold <b style={{ color: "var(--text)" }}>
              {result.threshold.toFixed(2)}
            </b>
            {" · "}
            <span className={
              calib.status === "calibrated" ? "chip accent" : "chip warn"}>
              {calib.status === "calibrated"
                ? `calibrated (n=${calib.n_samples}, ${
                    calib.auc != null ? `AUC ${calib.auc.toFixed(2)}` : "—"})`
                : "uncalibrated — default bands"}
            </span>
          </div>
          <div>{result.runtime_ms} ms · {result.segment_scores.length} segments scored</div>
          {result.link && (
            <div>
              <a href={result.link} target="_blank" rel="noreferrer">method ↗</a>
            </div>
          )}
        </div>
      </div>

      {hasSignals && (
        <details open={!result.evidence.length}>
          <summary style={{ cursor: "pointer", color: "var(--text-faint)", fontSize: 12.5 }}>
            signals
          </summary>
          <div className="kv-grid" style={{
            display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
            gap: 2, fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--text-dim)",
          }}>
            {Object.entries(result.signals).map(([k, v]) => (
              <div key={k} title={k}>
                {k} = {Number.isFinite(v) && Math.abs(v) > 0 && Math.abs(v) < 1e6
                  ? v.toFixed(3) : String(v)}
              </div>
            ))}
          </div>
        </details>
      )}

      {result.evidence.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {result.evidence.slice(0, open ? 99 : 3).map((e, i) => (
            <div key={i} className={`ev-item sev-${e.severity}`}>
              <div className="t">{e.title}</div>
              <div className="d">{e.detail}</div>
            </div>
          ))}
          {result.evidence.length > 3 && (
            <button className="ghost" style={{ alignSelf: "flex-start", padding: "4px 10px" }}
              onClick={() => setOpen(!open)}>
              {open ? "less" : `+${result.evidence.length - 3} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

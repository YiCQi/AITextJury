import { useEffect, useState } from "react";
import { deleteHistoryReport, fetchHistory, fetchHistoryReport } from "../api";
import type { AnalyzeReport, HistoryEntry } from "../types";
import { Heatmap } from "./Heatmap";
import { DetectorCard } from "./DetectorCard";
import { fmtPct, timeAgo, VERDICT_LABEL, verdictClass } from "../util";

export function HistoryView({ onOpen }:
{
  /** notify the parent so it can, e.g., re-run on the Workbench */
  onOpen?: (report: AnalyzeReport) => void;
}) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [report, setReport] = useState<AnalyzeReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const { entries } = await fetchHistory();
      setEntries(entries);
    } catch (e) {
      setErr(String(e));
    }
  };

  useEffect(() => { void refresh(); }, []);

  const open = async (id: string) => {
    try {
      const rep = await fetchHistoryReport(id);
      setReport(rep);
    } catch (e) {
      setErr(String(e));
    }
  };

  return (
    <div>
      <div className="panel">
        <h3>Run history</h3>
        <p className="hint">
          Runs are stored locally under <code>data/history/</code> — nothing is
          uploaded anywhere. Full reports include every detector's evidence,
          so a run stays auditable forever.
        </p>
        {err && <div style={{ color: "var(--warn)", fontSize: 12.5 }}>{err}</div>}
        <table className="flat">
          <thead>
            <tr>
              <th>when</th><th>title</th><th>consensus</th>
              <th>detectors</th><th>lang</th><th></th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.id}>
                <td className="num" style={{ whiteSpace: "nowrap" }}>
                  {timeAgo(e.created_at)}
                </td>
                <td style={{ maxWidth: 380, overflow: "hidden",
                             textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  <a href="#" onClick={(ev) => { ev.preventDefault(); void open(e.id); }}
                     style={{ color: "var(--text)" }}>
                    {e.title || e.id}
                  </a>
                </td>
                <td>
                  <span className={`chip ${verdictClass(e.consensus_verdict)}`}>
                    {e.consensus_score.toFixed(2)} · {VERDICT_LABEL[e.consensus_verdict]}
                  </span>
                </td>
                <td className="num" style={{ fontSize: 12 }}>
                  {e.detectors_used.join(", ")}
                </td>
                <td className="num">{e.stats?.language}</td>
                <td>
                  <button className="ghost" style={{ padding: "2px 9px", fontSize: 12 }}
                    onClick={async () => {
                      await deleteHistoryReport(e.id);
                      setReport(null);
                      void refresh();
                    }}>del</button>
                </td>
              </tr>
            ))}
            {entries.length === 0 && (
              <tr><td colSpan={6} style={{ color: "var(--text-faint)" }}>
                no runs yet — analyze something on the Workbench tab
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      {report && (
        <>
          <div className="panel">
            <h3>Report {report.id} · {report.created_at}</h3>
            {onOpen && (
              <button className="ghost" onClick={() => onOpen(report)}>
                → open in workbench context
              </button>
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
        </>
      )}
    </div>
  );
}

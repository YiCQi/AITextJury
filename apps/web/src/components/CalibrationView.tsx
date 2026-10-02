import { useEffect, useState } from "react";
import { fetchBench, fetchCalibration, runCalibration } from "../api";
import type { BenchDataset, CalibrationFitPublic } from "../types";
import { fmtPct } from "../util";

/**
 * Calibration tab — the honesty engine.
 * Shows, per detector: fitted status, dataset, AUC, accuracy, ECE, Brier,
 * threshold — and lets the user re-fit on any labeled corpus (built-in demo
 * set or their own JSONL dropped into data/bench/).
 */
export function CalibrationView() {
  const [fits, setFits] = useState<Record<string, CalibrationFitPublic>>({});
  const [datasets, setDatasets] = useState<BenchDataset[]>([]);
  const [dataset, setDataset] = useState("demo");
  const [busy, setBusy] = useState(false);
  const [lastRun, setLastRun] = useState<Record<string, CalibrationFitPublic> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const [{ calibration }, { datasets }] =
        await Promise.all([fetchCalibration(), fetchBench()]);
      setFits(calibration);
      setDatasets(datasets);
      if (datasets.length && !datasets.some((d) => d.name === dataset)) {
        setDataset(datasets[0].name);
      }
    } catch (e) {
      setErr(String(e));
    }
  };

  useEffect(() => { void refresh(); }, []);

  const doRun = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await runCalibration(null, dataset);
      setLastRun(r.results);
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="panel">
        <h3>Calibration</h3>
        <p className="hint">
          Raw detector statistics mean little until they are mapped to
          probabilities via a labeled corpus. Here you can inspect the
          <b> honesty metrics</b> — AUC, accuracy, ECE, Brier — and refit any
          detector on a labeled set. Uncalibrated detectors keep running on
          documented default bands and are always labeled as such.
        </p>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} — {d.labels.ai} AI / {d.labels.human} human
              </option>
            ))}
          </select>
          <button className="primary" disabled={busy} onClick={doRun}>
            {busy ? (<><span className="spin" />calibrating…</>)
                  : "Recalibrate all detectors"}
          </button>
        </div>
        <p className="hint" style={{ marginTop: 8 }}>
          He note: local-LM detectors will download small models (~500 MB) on
          first use and take a few minutes on CPU. The built-in demo set is
          <i> a demo, not a benchmark</i> — for real-world use, drop your own
          labeled <code>data/bench/*.jsonl</code> and fit on that.
        </p>
        {err && <div style={{ color: "var(--warn)" }}>{err}</div>}
        {lastRun && (
          <div style={{ marginTop: 10 }}>
            {Object.entries(lastRun).map(([id, r]) => (
              <div className="kv" key={id}>
                <span className="k">{id}</span>
                <span className="v" style={{
                  color: r.ok ? "var(--accent)" : "var(--warn)" }}>
                  {r.ok
                    ? `fitted — AUC ${r.auc?.toFixed(2) ?? "—"}, acc ${fmtPct(r.accuracy)}`
                    : `failed: ${r.reason ?? "?"}`}
                  {r.errors?.length ? ` (${r.errors.length} sample errors)` : ""}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <h3>Current fits</h3>
        <table className="flat">
          <thead>
            <tr>
              <th>detector</th><th>status</th><th>dataset</th><th>n</th>
              <th>AUC</th><th>accuracy</th><th>ECE</th><th>Brier</th>
              <th>threshold</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(fits).map(([id, f]) => (
              <tr key={id}>
                <td>{id}</td>
                <td><span className={`chip ${f.status === "calibrated" ? "accent" : "warn"}`}>
                  {f.status}
                </span></td>
                <td className="num">{f.dataset || "—"}</td>
                <td className="num">{f.n}</td>
                <td className="num">{f.auc != null ? f.auc.toFixed(3) : "—"}</td>
                <td className="num">{fmtPct(f.accuracy)}</td>
                <td className="num">{f.ece != null ? f.ece.toFixed(3) : "—"}</td>
                <td className="num">{f.brier != null ? f.brier.toFixed(3) : "—"}</td>
                <td className="num">{f.threshold?.toFixed(2) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint" style={{ marginTop: 10 }}>
          ECE = expected calibration error (bin-level |confidence − accuracy|);
          Brier = mean squared probability error. Lower is better for both.
        </p>
      </div>
    </div>
  );
}

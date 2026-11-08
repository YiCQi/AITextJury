import { useEffect, useState } from "react";
import { fetchBench, fetchCalibration, fetchSettings, runCalibration, saveSettings } from "../api";
import type { BenchDataset, CalibrationFitPublic, Settings } from "../types";
import { fmtPct } from "../util";
import { EvaluationReport } from "./EvaluationReport";

const JUDGE_SAMPLES = 40;

/**
 * Calibration tab — the honesty engine.
 * Shows, per juror: fitted status, dataset, AUC, accuracy, ECE, Brier,
 * threshold and consensus weight — lets the user re-fit on any labeled
 * corpus (built-in sets or their own JSONL in data/bench/), calibrate the
 * paid LLM jurors on a small sample, and scale any juror's vote.
 */
export function CalibrationView() {
  const [fits, setFits] = useState<Record<string, CalibrationFitPublic>>({});
  const [datasets, setDatasets] = useState<BenchDataset[]>([]);
  const [dataset, setDataset] = useState("hc3");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<Record<string, CalibrationFitPublic> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [runs, setRuns] = useState(0);     // bumps the evaluation report

  const refresh = async () => {
    try {
      const [{ calibration }, { datasets }, { settings }] =
        await Promise.all([fetchCalibration(), fetchBench(), fetchSettings()]);
      setFits(calibration);
      setDatasets(datasets);
      setSettings(settings);
      if (datasets.length && !datasets.some((d) => d.name === dataset)) {
        setDataset(datasets[0].name);
      }
    } catch (e) {
      setErr(String(e));
    }
  };

  useEffect(() => { void refresh(); }, []);

  const doRun = async (key: string, ids: string[] | null, maxSamples?: number) => {
    setBusy(key);
    setErr(null);
    try {
      const r = await runCalibration(ids, dataset, maxSamples);
      setLastRun(r.results);
      setRuns((n) => n + 1);
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  };

  const setMultiplier = async (id: string, value: number) => {
    if (!settings || Number.isNaN(value)) return;
    const current = settings.detectors[id] ?? {};
    try {
      // PUT merges per detector id, so resend that detector's other settings too
      await saveSettings({ detectors: { [id]: { ...current, weight: value } } });
      await refresh();
    } catch (e) {
      setErr(String(e));
    }
  };

  const judges = Object.entries(fits).filter(([, f]) => f.family === "byok_llm");
  const realJudges = judges.filter(([id]) => id.includes("@"));

  return (
    <div>
      <div className="panel">
        <h3>Calibration</h3>
        <p className="hint">
          Raw detector statistics mean little until they are mapped to
          probabilities via a labeled corpus. Calibrating also sets each
          juror's <b>vote weight</b>: the better it separates human from AI
          text (AUC), the more it counts in the verdict.
        </p>
        <div className="run-row" style={{ flexWrap: "wrap" }}>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}
            style={{ width: "auto", maxWidth: "100%" }}>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} — {d.labels.ai} AI / {d.labels.human} human
              </option>
            ))}
          </select>
          <button className="primary" disabled={busy !== null} onClick={() => doRun("all", null)}>
            {busy === "all" ? (<><span className="spin" />calibrating…</>)
                            : "Recalibrate free detectors"}
          </button>
        </div>
        <p className="hint" style={{ marginTop: 8 }}>
          Covers every local detector (LLM jurors cost API calls — calibrate
          them below). Local-LM detectors download small models (~500 MB) on
          first use and take a few minutes on CPU. The built-in <b>hc3</b> set
          is 200 real human vs ChatGPT answers (100 English, 100 Chinese);
          <b> demo</b> is a tiny hand-written smoke test. Your own domain
          matters most — drop labeled <code>data/bench/*.jsonl</code> and fit on that.
        </p>
        {err && <div className="error-line">{err}</div>}
        {lastRun && (
          <div style={{ marginTop: 10 }}>
            {Object.entries(lastRun).map(([id, r]) => (
              <div className="kv" key={id}>
                <span className="k">{fits[id]?.name ?? id}</span>
                <span className="v" style={{
                  color: r.ok ? "var(--accent)" : "var(--warn)" }}>
                  {r.ok
                    ? `fitted on ${r.n} — AUC ${r.auc?.toFixed(2) ?? "—"}, acc ${fmtPct(r.accuracy)}`
                    : `failed: ${r.reason ?? "?"}`}
                  {r.errors?.length ? ` (${r.errors.length} sample errors)` : ""}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <h3>LLM jurors</h3>
        {realJudges.length === 0 ? (
          <p className="hint">
            No API configured yet — add one or more on the Providers tab. Each
            provider becomes its own juror.
          </p>
        ) : (
          <>
            <p className="hint">
              Each provider judges independently. Uncalibrated, an LLM juror
              votes with weight 1.0; calibrating on {JUDGE_SAMPLES} balanced
              samples (≈{JUDGE_SAMPLES} API calls each, English + Chinese,
              human + AI) replaces that with its measured accuracy.
            </p>
            <div className="det-list">
              {realJudges.map(([id, f]) => (
                <div key={id} className="det-row" style={{ cursor: "default", alignItems: "center" }}>
                  <div className="det-row-body">
                    <b>{f.name ?? id}</b>{" "}
                    <span className={`chip ${f.status === "calibrated" ? "accent" : "warn"}`}>
                      {f.status === "calibrated"
                        ? `AUC ${f.auc?.toFixed(2) ?? "—"} on ${f.n}`
                        : "uncalibrated"}
                    </span>
                    <span className="hint"> · weight {f.weight?.toFixed(2) ?? "—"}</span>
                  </div>
                  <button className="ghost small" disabled={busy !== null}
                    onClick={() => doRun(id, [id], JUDGE_SAMPLES)}>
                    {busy === id ? (<><span className="spin" />calibrating…</>) : `Calibrate (${JUDGE_SAMPLES} calls)`}
                  </button>
                </div>
              ))}
            </div>
            {realJudges.length > 1 && (
              <button className="ghost small" style={{ marginTop: 10 }} disabled={busy !== null}
                onClick={() => doRun("judges", ["llm_judge"], JUDGE_SAMPLES)}>
                {busy === "judges" ? (<><span className="spin" />calibrating…</>)
                  : `Calibrate all ${realJudges.length} LLM jurors (${realJudges.length * JUDGE_SAMPLES} calls)`}
              </button>
            )}
          </>
        )}
      </div>

      <div className="panel">
        <h3>Jurors and weights</h3>
        <table className="flat">
          <thead>
            <tr>
              <th>juror</th><th>status</th><th>dataset</th><th>n</th>
              <th>AUC</th><th>accuracy</th><th>ECE</th>
              <th title="vote weight in the consensus (after your multiplier)">weight</th>
              <th title="your multiplier: 0 mutes a juror, 2 doubles it">× multiplier</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(fits).map(([id, f]) => (
              <tr key={id}>
                <td>{f.name ?? id}</td>
                <td><span className={`chip ${f.status === "calibrated" ? "accent" : "warn"}`}>
                  {f.status}
                </span></td>
                <td className="num">{f.dataset || "—"}</td>
                <td className="num">{f.n ?? "—"}</td>
                <td className="num">{f.auc != null ? f.auc.toFixed(3) : "—"}</td>
                <td className="num">{fmtPct(f.accuracy)}</td>
                <td className="num">{f.ece != null ? f.ece.toFixed(3) : "—"}</td>
                <td className="num"><b>{f.weight?.toFixed(2) ?? "—"}</b></td>
                <td>
                  <input type="number" min={0} max={5} step={0.25}
                    key={`${id}-${f.multiplier ?? 1}`}
                    defaultValue={Number(f.multiplier ?? 1)}
                    style={{ width: 72 }}
                    onBlur={(e) => {
                      const v = Number(e.target.value);
                      if (v !== Number(f.multiplier ?? 1)) void setMultiplier(id, v);
                    }}
                    onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint" style={{ marginTop: 10 }}>
          Weight = measured skill (2·AUC−1, mapped to 0.25–2.0 and pulled
          toward the juror's default when the fit used few samples) × your
          multiplier. Defaults while uncalibrated: 0.5, LLM jurors 1.0. A
          multiplier set on <code>llm_judge</code> in settings applies to every
          LLM juror. ECE = expected calibration error; lower is better. Once
          per-sample scores are stored, the verdict itself uses the
          statistical fusion instead (see the report below).
        </p>
      </div>

      <EvaluationReport dataset={dataset} version={runs} />
    </div>
  );
}

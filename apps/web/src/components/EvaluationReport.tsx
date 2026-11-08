import { useEffect, useState } from "react";
import { fetchEvaluation } from "../api";
import type { EvaluationReport as Report } from "../types";

const LANG: Record<string, string> = { all: "all", en: "English", zh: "Chinese" };
const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);

/**
 * How well each juror — and the fused jury — actually separates human from
 * AI text on a labeled set: AUROC with a 95% bootstrap interval and the
 * detection rate at 1% / 5% false positives, overall and per language.
 * Computed from the per-sample scores the last calibration runs stored.
 */
export function EvaluationReport({ dataset, version }: { dataset: string; version: number }) {
  const [report, setReport] = useState<Report | null>(null);
  const [lang, setLang] = useState("all");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setErr(null);
    fetchEvaluation(dataset).then(setReport).catch((e) => setErr(String(e)));
  }, [dataset, version]);

  const rows = (report?.rows ?? []).filter((r) => r.language === lang)
    .sort((a, b) => Number(b.detector_id === "fusion") - Number(a.detector_id === "fusion")
      || b.auroc - a.auroc);
  const langs = ["all", ...(report?.languages ?? [])];

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Evaluation report — {dataset}</h3>
        <div className="segmented">
          {langs.map((l) => (
            <button key={l} className={lang === l ? "active" : ""} onClick={() => setLang(l)}>
              {LANG[l] ?? l}
            </button>
          ))}
        </div>
      </div>
      {err && <div className="error-line">{err}</div>}
      {report && report.rows.length === 0 ? (
        <p className="hint">
          No per-sample scores for <b>{dataset}</b> yet — run a calibration above
          and the report fills in.
        </p>
      ) : (
        <table className="flat">
          <thead>
            <tr>
              <th>detector</th><th>n</th><th>AUROC [95% CI]</th>
              <th title="share of AI texts caught when at most 1% of human texts may be flagged">TPR @ 1% FPR</th>
              <th title="share of AI texts caught when at most 5% of human texts may be flagged">TPR @ 5% FPR</th>
              <th>ECE</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.detector_id + r.language}
                className={`eval-row${r.detector_id === "fusion" ? " fusion" : ""}`}>
                <td title={r.members?.join(", ")}>{r.name}</td>
                <td className="num">{r.n}</td>
                <td className="num">
                  {r.auroc.toFixed(3)}{" "}
                  {r.auroc_ci && (
                    <span className="eval-ci">[{r.auroc_ci[0].toFixed(2)}–{r.auroc_ci[1].toFixed(2)}]</span>
                  )}
                </td>
                <td className="num" title={r.fpr_resolution && r.fpr_resolution > 0.01
                  ? `only ${r.n_human} human texts: the finest FPR step is ${pct(r.fpr_resolution)}` : undefined}>
                  {pct(r.tpr_at_1fpr)}{r.fpr_resolution && r.fpr_resolution > 0.01 ? "*" : ""}
                </td>
                <td className="num">{pct(r.tpr_at_5fpr)}</td>
                <td className="num">{r.ece != null ? r.ece.toFixed(3) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="hint" style={{ marginTop: 10 }}>
        AUROC: chance that a random AI text scores above a random human text
        (0.5 = coin flip); the bracket is a 95% bootstrap interval — wide
        brackets mean the bench is too small to tell. TPR @ x% FPR: share of AI
        texts caught when at most x% of human texts may be wrongly flagged, the
        number that matters when false accusations are costly. * = fewer than
        100 human texts, so "1%" can't be resolved exactly. The jury row is
        5-fold cross-validated (never scored on its own training rows). Detector
        rows rank their raw scores directly, so no fit flatters them; ECE uses
        the in-sample fit.
      </p>
    </div>
  );
}

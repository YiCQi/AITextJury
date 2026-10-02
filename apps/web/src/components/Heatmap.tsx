import { useMemo, useState } from "react";
import type { AnalyzeReport, DetectorResult, SegmentScore } from "../types";
import { scoreColor } from "../util";

/**
 * Sentence-level heatmap over the original text.
 *
 * Layers: a detector (sentence scores from one result) or the consensus
 * (cross-detector mean). Colors encode normalized score; hover shows every
 * detector's value for that sentence — the "why" behind the paint.
 */
export function Heatmap({ report }:{ report: AnalyzeReport }) {
  const [layer, setLayer] = useState<string>("consensus");
  const [hover, setHover] = useState<{ x: number; y: number; segId: string } | null>(null);

  const layers = useMemo(() => {
    const out: { id: string; name: string; scores: Map<string, number> }[] = [];
    if (report.consensus.segment_scores.length) {
      out.push({
        id: "consensus", name: "Consensus (cross-detector mean)",
        scores: new Map(
          report.consensus.segment_scores
            .filter((s) => s.score != null)
            .map((s) => [s.id, s.score as number])),
      });
    }
    for (const r of report.results) {
      if (r.segment_scores.some((s) => s.score != null)) {
        out.push({
          id: r.detector_id, name: r.name,
          scores: new Map(
            r.segment_scores
              .filter((s) => s.score != null)
              .map((s) => [s.id, s.score as number])),
        });
      }
    }
    return out;
  }, [report]);

  const active = layers.find((l) => l.id === layer) ?? layers[0];

  // per-segment values for the tooltip: all detectors + consensus
  const allBySeg = useMemo(() => {
    const m = new Map<string, { name: string; score: number; extras?: Record<string, unknown> }[]>();
    const push = (id: string, name: string, score: number | null, extras?: Record<string, unknown>) => {
      if (score == null) return;
      if (!m.has(id)) m.set(id, []);
      m.get(id)!.push({ name, score, extras });
    };
    for (const r of report.results) {
      for (const s of r.segment_scores) push(s.id, r.name, s.score, s.extras);
    }
    for (const s of report.consensus.segment_scores) push(s.id, "consensus", s.score);
    return m;
  }, [report]);

  if (!active) {
    return (
      <div className="panel">
        <h3>Heatmap</h3>
        <p className="hint">
          No detector produced sentence-level scores yet. LLM-based detectors
          and stylometry both supply them.
        </p>
      </div>
    );
  }

  const byPara = new Map<string, SegmentScore[]>();
  for (const seg of report.sentences) {
    const key = seg.parent_id ?? "_";
    if (!byPara.has(key)) byPara.set(key, []);
    byPara.get(key)!.push(seg);
  }

  return (
    <div className="panel">
      <h3>Heatmap — why each sentence looks AI (or not)</h3>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <div className="heatmap-legend" style={{ margin: 0 }}>
          <span>human</span>
          <div className="bar" />
          <span>AI-like</span>
        </div>
        <select
          value={active.id}
          onChange={(e) => setLayer(e.target.value)}
          style={{ marginLeft: "auto" }}
        >
          {layers.map((l) => (
            <option key={l.id} value={l.id}>{l.name}</option>
          ))}
        </select>
      </div>
      <p className="hint">
        Hover a sentence for per-detector values · colored = that layer's normalized P(AI)-style score
      </p>

      <div className="heatmap">
        {report.paragraphs.map((p) => (
          <span className="para" key={p.id}>
            {(byPara.get(p.id) ?? []).map((s) => {
              const v = active.scores.get(s.id);
              const bg = v != null
                ? hexAlpha(scoreColor(v), 0.16 + Math.abs(v - 0.5) * 0.5)
                : "transparent";
              return (
                <span
                  key={s.id}
                  className="seg"
                  style={{ background: bg, color: v != null ? scoreColor(v) : undefined }}
                  onMouseEnter={(e) =>
                    setHover({ x: e.clientX + 14, y: e.clientY + 14, segId: s.id })}
                  onMouseLeave={() => setHover(null)}
                >
                  {s.text}
                </span>
              );
            })}
          </span>
        ))}
      </div>

      {hover && (
        <div className="seg-tip" style={{ left: hover.x, top: hover.y }}>
          <div style={{ color: "#8b9bb4", marginBottom: 6, fontSize: 11 }}>
            segment scores · {hoverSegText(report, hover.segId)}
          </div>
          {(allBySeg.get(hover.segId) ?? []).map((row, i) => (
            <div key={i} style={{ display: "flex", gap: 10 }}>
              <span style={{ color: scoreColor(row.score), minWidth: 54 }}>
                {row.score.toFixed(2)}
              </span>
              <span>{row.name}</span>
            </div>
          ))}
          {tooltipExtras(report, hover.segId)}
        </div>
      )}
    </div>
  );
}

function hoverSegText(report: AnalyzeReport, segId: string): string {
  const seg = report.sentences.find((s) => s.id === segId);
  if (!seg) return "";
  const t = seg.text.trim().replace(/\s+/g, " ");
  return t.length > 90 ? t.slice(0, 90) + "…" : t;
}

function tooltipExtras(report: AnalyzeReport, segId: string) {
  const reasons: string[] = [];
  for (const r of report.results as DetectorResult[]) {
    for (const s of r.segment_scores) {
      if (s.id === segId && s.extras) {
        const reason = s.extras["reason"];
        if (typeof reason === "string" && reason) reasons.push(`${r.name}: ${reason}`);
      }
    }
  }
  if (!reasons.length) return null;
  return (
    <div style={{ marginTop: 6, borderTop: "1px solid #1e2839", paddingTop: 6 }}>
      {reasons.slice(0, 3).map((r, i) => (
        <div key={i} style={{ color: "#f0a35c" }}>“{r}”</div>
      ))}
    </div>
  );
}

/** #rrggbb -> rgba with alpha */
function hexAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${Math.max(0.06, Math.min(alpha, 0.85))})`;
}

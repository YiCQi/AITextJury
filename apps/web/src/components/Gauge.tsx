import { scoreColor, VERDICT_LABEL, verdictClass } from "../util";
import type { Verdict } from "../types";

/**
 * Half-donut gauge. Big number = normalized P(AI)-style score;
 * sub-caption = verdict + confidence. Needle-free by design: the arc fill
 * position already encodes the score; a needle would imply fake precision.
 */
export function Gauge({ score, verdict, confidence, threshold = 0.5, size = 1 }:
{
  score: number | null;
  verdict: Verdict | null;
  confidence?: number | null;
  threshold?: number;
  size?: number;
}) {
  const W = 132, H = 78, R = 56, CX = W / 2, CY = 74;
  const start = Math.PI * 1.0;         // left
  const sweep = Math.PI;               // 180° …
  const pt = (ang: number, r: number) =>
    `${CX + Math.cos(ang) * r},${CY + Math.sin(ang) * r}`;
  const ang = start + sweep * (score ?? 0);

  const arcPath = (a0: number, a1: number, r: number) => {
    const large = 0;
    const sweepFlag = 1;
    return `M ${pt(a0, r)} A ${r} ${r} 0 ${large} ${sweepFlag} ${pt(a1, r)}`;
  };

  const track = start + sweep;
  const color = scoreColor(score);
  const label = score == null ? "—" : score.toFixed(2);

  return (
    <div className="gauge" style={{ transform: `scale(${size})`, transformOrigin: "top left" }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        <path d={arcPath(start, track, R)}
          stroke="#1c2636" strokeWidth="9" fill="none" strokeLinecap="round" />
        {score != null && (
          <>
            <path d={arcPath(start, ang, R)}
              stroke={color} strokeWidth="9" fill="none" strokeLinecap="round" />
            {/* calibration threshold tick */}
            <line
              x1={pt(start + sweep * threshold, R - 7).split(",")[0]}
              y1={pt(start + sweep * threshold, R - 7).split(",")[1]}
              x2={pt(start + sweep * threshold, R + 7).split(",")[0]}
              y2={pt(start + sweep * threshold, R + 7).split(",")[1]}
              stroke="#5c6a80" strokeWidth="1.5" strokeDasharray="2 2" />
          </>
        )}
      </svg>
      <div className="label" style={{ color }}>
        {label}
      </div>
      <div className="sub">
        <span className={`chip ${verdictClass(verdict)}`}>
          {VERDICT_LABEL[String(verdict)] ?? "—"}
        </span>
        {" "}
        {confidence != null && `conf ${(confidence * 100).toFixed(0)}%`}
      </div>
    </div>
  );
}

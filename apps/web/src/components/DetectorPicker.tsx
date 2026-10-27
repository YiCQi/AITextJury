import type { DetectorInfo } from "../types";
import { DETECTOR_PLAIN, speedLabel } from "../util";

export type Preset = "fast" | "standard" | "everything";

/** Which detectors a preset ticks (only ever available ones). */
export function presetIds(detectors: DetectorInfo[], preset: Preset): string[] {
  const usable = detectors.filter((d) => d.available);
  switch (preset) {
    case "fast":
      return usable.filter((d) => !d.heavy && d.family !== "byok_llm").map((d) => d.id);
    case "standard":
      return usable.filter((d) => !d.heavy).map((d) => d.id);
    case "everything":
      return usable.map((d) => d.id);
  }
}

const PRESETS: [Preset, string, string][] = [
  ["fast", "Fast", "instant local checks only"],
  ["standard", "Standard", "instant checks + your LLM Judge"],
  ["everything", "Everything", "adds local language models (slower; English)"],
];

/**
 * The detector picker: presets for the common cases, one plain-language
 * line per detector, unavailable ones folded away with their fix.
 */
export function DetectorPicker({ detectors, selected, onChange, onNavigate }:
{
  detectors: DetectorInfo[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
  onNavigate: (tab: "providers" | "calibration") => void;
}) {
  const available = detectors.filter((d) => d.available);
  const unavailable = detectors.filter((d) => !d.available);
  const activePreset = PRESETS.find(([p]) => {
    const ids = presetIds(detectors, p);
    return ids.length === selected.size && ids.every((id) => selected.has(id));
  })?.[0];
  const uncalibrated = available.filter((d) => selected.has(d.id) && d.uncalibrated);
  const judge = detectors.find((d) => d.id === "llm_judge");

  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id); else next.add(id);
    onChange(next);
  };

  return (
    <div className="panel">
      <h3>Detectors</h3>
      <div className="segmented" role="group" aria-label="detector presets">
        {PRESETS.map(([p, label, help]) => (
          <button key={p} className={activePreset === p ? "active" : ""} title={help}
            onClick={() => onChange(new Set(presetIds(detectors, p)))}>
            {label}
          </button>
        ))}
      </div>
      <p className="hint" style={{ margin: "6px 0 10px" }}>
        {activePreset
          ? PRESETS.find(([p]) => p === activePreset)![2]
          : "custom selection"}
      </p>

      {judge && !judge.available && (
        <div className="callout">
          <span>
            <b>LLM Judge</b> is the strongest juror — it needs an API key
            (or a local Ollama).
          </span>
          <button className="ghost small" onClick={() => onNavigate("providers")}>
            Add a provider →
          </button>
        </div>
      )}

      <div className="det-list">
        {available.map((d) => {
          const on = selected.has(d.id);
          return (
            <label key={d.id} className={`det-row${on ? " on" : ""}`}>
              <input type="checkbox" checked={on} onChange={() => toggle(d.id)} />
              <div className="det-row-body">
                <div className="det-row-head">
                  <b>{d.name}</b>
                  {d.recommended && <span className="chip recommended">★ recommended</span>}
                  <span className="chip">{speedLabel(d)}</span>
                  {!d.uncalibrated && <span className="chip accent">calibrated</span>}
                </div>
                <div className="det-row-plain">{DETECTOR_PLAIN[d.id] ?? d.description}</div>
                {DETECTOR_PLAIN[d.id] && (
                  <details className="more" onClick={(e) => e.stopPropagation()}>
                    <summary>how it works</summary>
                    <div>{d.description}</div>
                    {d.link && <a href={d.link} target="_blank" rel="noreferrer">method ↗</a>}
                  </details>
                )}
              </div>
            </label>
          );
        })}
      </div>

      {uncalibrated.length > 0 && (
        <div className="callout warn">
          <span>
            {uncalibrated.length === 1 ? "1 selected detector is" : `${uncalibrated.length} selected detectors are`}{" "}
            not calibrated yet, so their scores use rough defaults.
          </span>
          <button className="ghost small" onClick={() => onNavigate("calibration")}>
            Calibrate →
          </button>
        </div>
      )}

      {unavailable.length > 0 && (
        <details className="unavailable">
          <summary>Not available on this machine ({unavailable.length})</summary>
          {unavailable.map((d) => (
            <div key={d.id} className="unavail-row">
              <b>{d.name}</b> — {d.reason}
              {d.hints.length > 0 && <div className="hint">{d.hints.join(" · ")}</div>}
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import type { AnalyzeReport, DetectorInfo } from "../types";
import { analyze, fetchDetectors, fetchSample, type SampleText } from "../api";
import { Heatmap } from "./Heatmap";
import { DetectorCard } from "./DetectorCard";
import { DetectorPicker, presetIds } from "./DetectorPicker";
import { VerdictPanel } from "./VerdictPanel";
import { isTooShort, textSize } from "../util";

const MAX_CHARS = 60000;
const DRAFT_KEY = "aitj.draft";
const PICK_KEY = "aitj.detectors";

// per-viewer conveniences only — storage may be unavailable (private mode)
function load(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function save(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* ignore */ }
}

const EXAMPLES: [string, string, 0 | 1][] = [
  ["Human · EN", "en", 0], ["AI · EN", "en", 1],
  ["人类 · 中文", "zh", 0], ["AI · 中文", "zh", 1],
];

/** The main Workbench tab: input → detector panel → run → evidence. */
export function Workbench({ active, onNavigate }:
{
  /** the tab is visible — refresh detector availability on re-entry */
  active: boolean;
  onNavigate: (tab: "providers" | "calibration") => void;
}) {
  const [text, setText] = useState(() => load(DRAFT_KEY) ?? "");
  const [detectors, setDetectors] = useState<DetectorInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<AnalyzeReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sample, setSample] = useState<SampleText | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!active) return;
    (async () => {
      try {
        const { detectors } = await fetchDetectors();
        setDetectors(detectors);
        setSelected((prev) => {
          const ok = new Set(detectors.filter((d) => d.available).map((d) => d.id));
          const stored = prev.size ? [...prev] : JSON.parse(load(PICK_KEY) ?? "null");
          const kept = Array.isArray(stored) ? stored.filter((id: string) => ok.has(id)) : [];
          return new Set(kept.length ? kept : presetIds(detectors, "standard"));
        });
      } catch (e) {
        setError(String(e));
      }
    })();
  }, [active]);

  useEffect(() => { save(DRAFT_KEY, text); }, [text]);

  const pick = (next: Set<string>) => {
    setSelected(next);
    save(PICK_KEY, JSON.stringify([...next]));
  };

  const edit = (value: string) => {
    setText(value);
    setSample(null);   // edited text is no longer the labeled example
  };

  const run = async () => {
    if (!text.trim() || selected.size === 0 || running) return;
    setRunning(true);
    setError(null);
    try {
      setReport(await analyze(text, [...selected]));
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  const tryExample = async (language: string, label: 0 | 1) => {
    setError(null);
    try {
      const s = await fetchSample(language, label);
      setText(s.text);
      setSample(s);
      setReport(null);
    } catch (e) {
      setError(String(e));
    }
  };

  const openFile = async (file: File | undefined) => {
    if (!file) return;
    if (file.size > 2_000_000) {
      setError(`${file.name} is too large — paste the part you want checked`);
      return;
    }
    edit(await file.text());
    setError(null);
  };

  const size = textSize(text);
  const tooLong = text.length > MAX_CHARS;
  const short = text.trim().length > 0 && isTooShort(text);
  const heavyCount = detectors.filter((d) => d.heavy && selected.has(d.id)).length;

  return (
    <div className="workbench">
      {/* ------------------------------------------------- left: input column */}
      <div className="stack">
        <div className="panel">
          <div className="panel-head">
            <h3>Text</h3>
            <div className="toolbar">
              <button className="link" onClick={() => fileInput.current?.click()}>open file</button>
              {text && <button className="link" onClick={() => { edit(""); setReport(null); }}>clear</button>}
            </div>
            <input ref={fileInput} type="file" hidden accept=".txt,.md,.markdown,.csv,.html,text/*"
              onChange={(e) => { void openFile(e.target.files?.[0]); e.target.value = ""; }} />
          </div>
          <textarea
            className={`main${dragging ? " dragging" : ""}`}
            value={text}
            placeholder={"Paste text here, or drop a .txt / .md file.\n\nAt least a paragraph or two works best — very short texts give unreliable scores."}
            onChange={(e) => edit(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); void run(); }
            }}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              void openFile(e.dataTransfer.files?.[0]);
            }}
          />
          <div className="input-meta">
            <span>{text.length.toLocaleString()} chars · {size.label}</span>
            {sample && <span className="chip accent">real example — true label shown after analysis</span>}
            {short && <span className="chip warn">short text — scores will be shaky</span>}
            {tooLong && <span className="chip warn">over {MAX_CHARS.toLocaleString()} chars</span>}
          </div>
          {!text.trim() && (
            <div className="examples">
              <span className="hint">No text handy? Try a real example:</span>
              {EXAMPLES.map(([label, lang, y]) => (
                <button key={label} className="ghost small" onClick={() => void tryExample(lang, y)}>
                  {label}
                </button>
              ))}
            </div>
          )}
          <div className="run-row">
            <button className="primary" disabled={running || !text.trim() || tooLong || selected.size === 0}
              onClick={run}>
              {running ? (<><span className="spin" />analyzing…</>) : "Analyze"}
            </button>
            <span className="hint">
              {selected.size === 0 ? "pick at least one detector" : <><kbd>Ctrl</kbd>+<kbd>Enter</kbd></>}
            </span>
          </div>
          {error && <div className="error-line">{error}</div>}
        </div>

        <DetectorPicker detectors={detectors} selected={selected}
          onChange={pick} onNavigate={onNavigate} />
      </div>

      {/* --------------------------------------------------- right: results */}
      <div className="stack">
        {running && (
          <div className="panel running">
            <span className="spin" />
            <div>
              <b>Running {selected.size} detector{selected.size > 1 ? "s" : ""}…</b>
              {heavyCount > 0 && (
                <div className="hint">
                  Local language models run on your CPU; the first run also
                  downloads their weights, which can take a few minutes.
                </div>
              )}
            </div>
          </div>
        )}
        {!report && !running && (
          <div className="panel empty-state">
            <h3>How it works</h3>
            <ol>
              <li><b>Paste text</b> on the left — or try a real example.</li>
              <li><b>Pick detectors.</b> <i>Standard</i> is a good start; add an LLM Judge under Providers for the strongest single opinion.</li>
              <li><b>Read the evidence</b>: an overall verdict, which sentences look machine-written, and why each detector thinks so.</li>
            </ol>
            <p className="hint">
              Everything runs on this machine, except calls to the LLM provider
              you configure. No detector is proof — treat results as evidence.
            </p>
          </div>
        )}
        {report && (
          <>
            <VerdictPanel report={report} sample={sample} />
            <Heatmap report={report} />
            <div className="panel">
              <h3>Detectors</h3>
              <div className="result-grid">
                {report.results.map((r) => (
                  <DetectorCard key={r.detector_id} result={r} />
                ))}
              </div>
            </div>
            <details className="panel raw">
              <summary>raw report JSON</summary>
              <pre>{JSON.stringify(report, null, 2)}</pre>
            </details>
          </>
        )}
      </div>
    </div>
  );
}

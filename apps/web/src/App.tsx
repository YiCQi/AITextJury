import { useState } from "react";
import { Workbench } from "./components/Workbench";
import { HistoryView } from "./components/HistoryView";
import { CalibrationView } from "./components/CalibrationView";
import { ProvidersView } from "./components/ProvidersView";
import { DocsView } from "./components/DocsView";

type Tab = "workbench" | "history" | "calibration" | "providers" | "methods";

export default function App() {
  const [tab, setTab] = useState<Tab>("workbench");

  return (
    <div className="app">
      <header className="topbar">
        <div className="logo">
          <span className="name">Zero<em>AI</em>Bench</span>
          <span className="tag">
            open workbench for AI text detection · evidence, not verdicts ·
            BYOK
          </span>
        </div>
        <nav className="tabs">
          {([["workbench", "Workbench"], ["history", "History"],
            ["calibration", "Calibration"], ["providers", "Providers"],
            ["methods", "Methods"]] as [Tab, string][]).map(([id, label]) => (
            <button key={id} className={tab === id ? "active" : ""}
              onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </nav>
      </header>

      <div className="masthead">
        <b>multi-detector consensus</b> · sentence heatmaps · perplexity &
        curvature signals · calibration (AUC/ECE) · plugin API —{" "}
        <span>no account, no telemetry, your keys stay on this machine</span>
      </div>

      <main className="page">
        {tab === "workbench" && <Workbench />}
        {tab === "history" && <HistoryView />}
        {tab === "calibration" && <CalibrationView />}
        {tab === "providers" && <ProvidersView />}
        {tab === "methods" && <DocsView />}
      </main>

      <footer className="pagefoot">
        ZeroAIBench v0.1.0 · MIT license · detectors are evidence engines, not
        arbiters — adversarial or edited text can defeat any known method.
      </footer>
    </div>
  );
}

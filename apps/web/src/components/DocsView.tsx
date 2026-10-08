/**
 * Methods & philosophy tab — a snapshot of docs/ inside the workbench itself,
 * because a detector you can't inspect is a detector you can't trust.
 */
export function DocsView() {
  return (
    <div>
      <div className="panel">
        <h3>Why a workbench, not a single “AIometer”</h3>
        <p>
          Single-number AI detectors fail because they hide their evidence.
          AITextJury's core object is the <b>Detector API</b>: every method —
          stylometry, local-LM surprisal, Fast-DetectGPT-style curvature,
          Binoculars-style cross-model agreement, HF classifiers, your own
          LLM-as-judge, community plugins — returns the same shape:
        </p>
        <pre style={{ background: "var(--bg-soft)", padding: 14, borderRadius: 8, fontSize: 12.5 }}>{`{
  score:        0.62,          // normalized P(AI)-style probability
  raw_score:    14.2,          // the detector's native statistic
  raw_direction:"lower_is_ai", // how to read the raw value
  verdict:      "uncertain",   // threshold-based, near-threshold ⇒ uncertain
  confidence:   0.58,
  threshold:    0.71,          // from calibration or documented bands
  signals:      { ... },       // e.g. perplexity, burstiness, crossH
  segment_scores:[ ... ],      // per-sentence values → heatmap
  evidence:     [ ... ],       // human-readable reasons
  calibration:  { status, auc, ece, brier, n, ... }
}`}</pre>
        <p style={{ marginTop: 10 }}>
          Consensus is a meta-detector, not a majority vote you must obey:
          weighted by calibration quality, with disagreement surfaced as a
          first-class signal.
        </p>
      </div>

      <div className="panel">
        <h3>The detectors</h3>
        <p className="hint">
          Listed strongest-first by general judgment strength — the same order
          as the Detector panel.
        </p>
        <table className="flat">
          <thead><tr><th>method</th><th>type</th><th>core idea</th></tr></thead>
          <tbody>
            <tr><td><b>LLM Judge</b> <span className="chip recommended">★ recommended</span></td><td>BYOK</td>
              <td>your OpenAI/Gemini/DeepSeek/OpenRouter/Ollama/… model judges
                  with a strict-JSON protocol; flags paragraphs and explains.
                  The strongest single juror: full-context reasoning plus
                  quoted evidence.</td></tr>
            <tr><td><b>Fast-DetectGPT</b></td><td>local LM</td>
              <td>conditional probability curvature: machine text defends its
                  probability peak against token perturbation
                  (<a href="https://arxiv.org/abs/2310.05130" target="_blank" rel="noreferrer">Bao et al., ICLR'24 ↗</a>).
                  The strongest zero-shot local method.
                  This is a documented variant — see its in-card note.</td></tr>
            <tr><td><b>Binoculars</b></td><td>local LM pair</td>
              <td>cross-model agreement between a performer and an observer
                  (<a href="https://arxiv.org/abs/2401.12070" target="_blank" rel="noreferrer">Hans et al. 2024 ↗</a>).
                  Strong, but its threshold is domain-sensitive:
                  faithful-in-spirit closed form; original thresholds don't
                  transfer — calibrate locally.</td></tr>
            <tr><td><b>HF Classifier</b></td><td>BYOM</td>
              <td>any HuggingFace text-classification model (HC3-style
                  detectors). Strong in its training domain, transfers poorly
                  across domains — measure it.</td></tr>
            <tr><td><b>LLM Perplexity</b></td><td>local LM</td>
              <td>mean per-token surprisal under a small causal LM
                  (default gpt2; point it at a multilingual model for CJK).
                  A soft signal — best as corroboration.</td></tr>
            <tr><td><b>Stylometry</b></td><td>local stats</td>
              <td>burstiness, repetition profile, connective boilerplate,
                  LLM-register tells (EN+ZH). Zero dependency — the weakest
                  evidence alone, but instant and useful for bulk screening.</td></tr>
            <tr><td><b>Plugins</b></td><td>community</td>
              <td>any Python file with <code>register(registry)</code> — see
                  the bundled <code>length_rhythm</code> example
                  (30 lines) and <code>docs/DETECTOR_API.md</code>.</td></tr>
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h3>Calibration & honesty</h3>
        <p>
          Every score carries a <b>calibration status</b>. Out of the box,
          detectors use documented default bands and are flagged
          <span className="chip warn">uncalibrated</span>. Drop a labeled
          corpus (<code>data/bench/*.jsonl</code>: one <code>{`{"text":…, "label":0|1}`}</code>
          per line) and hit <b>Recalibrate</b>: the backend fits a logistic
          map, picks a max-accuracy threshold and computes AUC / ECE / Brier.
          Consensus weights detectors by their measured accuracy.
        </p>
      </div>

      <div className="panel">
        <h3>What this is NOT</h3>
        <ul style={{ color: "var(--text-dim)", lineHeight: 1.8, margin: 0 }}>
          <li><b>Not a plagiarism checker</b> — it doesn't find sources.</li>
          <li><b>Not a judgment machine</b> — scores are evidence for a human
              decision. Adversarially rewritten text fools most known
              detectors; honest tools say so.</li>
          <li><b>Not a scientific benchmark</b> — the bundled demo set is a
              demo. Calibrate on data from <i>your</i> domain
              (essays? code comments? marketing copy?).</li>
          <li><b>Not a stalker</b> — history, keys and fits stay in local
              files; BYOK calls go only to providers you configure.</li>
        </ul>
      </div>
    </div>
  );
}

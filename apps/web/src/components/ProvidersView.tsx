import { useEffect, useState } from "react";
import { deleteProvider, fetchKeys, saveProvider, testProvider } from "../api";
import type { PublicProvider, ProviderTemplate } from "../types";

type Draft = {
  kind: string; base_url: string; api_key: string;
  default_model: string; note: string; template: string;
};

const BLANK: Draft = {
  kind: "openai_compatible", base_url: "",
  api_key: "", default_model: "", note: "", template: "",
};

/**
 * BYOK manager. Keys are stored ONLY in data/providers.json on this machine
 * and used solely toward the endpoint you configured. The UI never displays
 * a full key after saving.
 */
export function ProvidersView() {
  const [providers, setProviders] = useState<PublicProvider[]>([]);
  const [templates, setTemplates] = useState<Record<string, ProviderTemplate>>({});
  const [storage, setStorage] = useState("");
  const [draft, setDraft] = useState<Draft>(BLANK);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tests, setTests] = useState<Record<string, { ok: boolean; detail: string; latency_ms: number }>>({});
  const [feedback, setFeedback] = useState<string | null>(null);

  const refresh = async () => {
    const data = await fetchKeys();
    setProviders(data.providers);
    setTemplates(data.templates);
    setStorage(data.storage);
  };

  useEffect(() => { void refresh(); }, []);

  const pickTemplate = (name: string) => {
    const t = templates[name];
    if (!t) return;
    setDraft({
      ...draft,
      template: name,
      kind: t.kind,
      base_url: t.base_url,
      default_model: t.default_model,
    });
  };

  const save = async () => {
    setBusy(true);
    setFeedback(null);
    try {
      // only send api_key if non-empty — empty means “keep existing”.
      // no id = new entry; the server generates one from the preset.
      const r = await saveProvider({
        id: editingId ?? undefined,
        kind: draft.kind, base_url: draft.base_url,
        api_key: draft.api_key || undefined,
        default_model: draft.default_model, note: draft.note,
        template: editingId ? undefined : (draft.template || undefined),
      });
      setDraft(BLANK);
      setEditingId(null);
      await refresh();
      const label = r.id ?? editingId ?? "provider";
      setFeedback(`saved “${label}” — its key now lives (only) in ${storage}`);
    } catch (e) {
      setFeedback(String(e));
    } finally {
      setBusy(false);
    }
  };

  const startEdit = (p: PublicProvider) => {
    setEditingId(p.id);
    setDraft({
      kind: p.kind, base_url: p.base_url, api_key: "",
      default_model: p.default_model, note: p.note, template: "",
    });
    setFeedback(`editing “${p.id}” — leave the key empty to keep it`);
  };

  const doTest = async (id: string) => {
    setTests({ ...tests, [id]: { ok: false, detail: "…", latency_ms: 0 } });
    try {
      const r = await testProvider(id);
      setTests({ ...tests, [id]: r });
    } catch (e) {
      setTests({ ...tests, [id]: { ok: false, detail: String(e), latency_ms: 0 } });
    }
  };

  return (
    <div>
      <div className="panel">
        <h3>Bring Your Own Key</h3>
        <p className="hint">
          Plug in OpenAI, Gemini, DeepSeek, OpenRouter, Groq, a local Ollama
          (no key needed) or any OpenAI-compatible endpoint (vLLM, LM Studio…).
          Keys are stored in <code>{storage || "data/providers.json"}</code> on
          this machine, sent <b>only toward the provider you configure</b>, and
          never rendered back in full.
        </p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
          {Object.keys(templates).map((t) => (
            <button key={t} className="ghost" onClick={() => pickTemplate(t)}>
              + {t}
            </button>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))", gap: 10 }}>
          <select value={draft.kind}
            onChange={(e) => setDraft({ ...draft, kind: e.target.value })}>
            <option value="openai_compatible">openai-compatible</option>
            <option value="gemini">gemini</option>
          </select>
          <input placeholder="base URL (from template)" value={draft.base_url}
            onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} />
          <input placeholder="model (e.g. gpt-4o-mini / qwen2.5:7b)"
            value={draft.default_model}
            onChange={(e) => setDraft({ ...draft, default_model: e.target.value })} />
          <input placeholder="API key (empty = keep existing / env)"
            value={draft.api_key} type="password"
            onKeyDown={(e) => { if (e.key === "Enter" && draft.base_url.trim()) void save(); }}
            onChange={(e) => setDraft({ ...draft, api_key: e.target.value })} />
          <input placeholder="note — your label for it (optional)" value={draft.note}
            onChange={(e) => setDraft({ ...draft, note: e.target.value })} />
        </div>
        <div style={{ marginTop: 10 }}>
          <button className="primary" disabled={busy || !draft.base_url.trim()}
            onClick={() => void save()}>
            {busy ? "saving…" : editingId ? `save “${editingId}”` : "add provider"}
          </button>
          {editingId && (
            <button className="ghost" style={{ marginLeft: 8, padding: "2px 9px", fontSize: 12 }}
              onClick={() => { setEditingId(null); setDraft(BLANK); setFeedback(null); }}>
              cancel (new entry)
            </button>
          )}
          {feedback && (
            <span style={{ marginLeft: 12, fontSize: 12.5, color: "var(--warn)" }}>
              {feedback}
            </span>
          )}
        </div>
        <p className="hint" style={{ marginTop: 8 }}>
          The provider id is generated for you from the preset you picked
          (<code>deepseek</code>, then <code>deepseek:2</code>…) — the note
          is the label you see. Env fallback: leaving the key empty makes the
          provider read its template env var (
          <code>OPENAI_API_KEY</code>, <code>GEMINI_API_KEY</code>,
          <code>DEEPSEEK_API_KEY</code>…).
        </p>
      </div>

      <div className="panel">
        <h3>Configured providers</h3>
        <table className="flat">
          <thead>
            <tr><th>provider</th><th>kind</th><th>base URL</th><th>model</th>
                <th>key</th><th>connection</th><th></th></tr>
          </thead>
          <tbody>
            {providers.map((p) => {
              const t = tests[p.id];
              return (
                <tr key={p.id}>
                  <td><b>{p.note || p.id}</b>{p.note ? <span style={{ color: "var(--text-faint)" }}> · {p.id}</span> : null}</td>
                  <td className="num">{p.kind}</td>
                  <td className="num" style={{ fontSize: 12 }}>{p.base_url}</td>
                  <td className="num" style={{ fontSize: 12 }}>{p.default_model}</td>
                  <td>
                    <span className={`chip ${p.has_key ? "accent" : ""}`}>
                      {p.has_key ? p.key_mask : "no key"}
                    </span>
                    {p.env_key && <span className="chip">env: {p.env_key}</span>}
                  </td>
                  <td className="num" style={{ fontSize: 12 }}>
                    {t ? (t.ok
                      ? <span style={{ color: "var(--accent)" }}>ok ({t.latency_ms} ms)</span>
                      : <span style={{ color: "var(--warn)" }}>{t.detail}</span>)
                      : "—"}
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button className="ghost" style={{ padding: "2px 9px", fontSize: 12, marginRight: 6 }}
                      onClick={() => void doTest(p.id)}>test</button>
                    <button className="ghost" style={{ padding: "2px 9px", fontSize: 12, marginRight: 6 }}
                      onClick={() => startEdit(p)}>edit</button>
                    <button className="ghost" style={{ padding: "2px 9px", fontSize: 12 }}
                      onClick={async () => {
                        await deleteProvider(p.id);
                        void refresh();
                      }}>del</button>
                  </td>
                </tr>
              );
            })}
            {providers.length === 0 && (
              <tr><td colSpan={7} style={{ color: "var(--text-faint)" }}>
                none yet — pick a template above. Ollama at
                <code> http://localhost:11434/v1</code> works with no key.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

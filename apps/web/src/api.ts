import type {
  AnalyzeReport, BenchDataset, CalibrationFitPublic, DetectorInfo,
  HistoryEntry, PublicProvider, ProviderTemplate, Settings,
} from "./types";

const BASE = "/api";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      detail = (await res.json())["detail"] ?? detail;
    } catch { /* ignore */ }
    throw new Error(`GET ${path}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function sendJSON<T>(method: string, path: string,
                          body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      detail = (await res.json())["detail"] ?? detail;
    } catch { /* ignore */ }
    throw new Error(`${method} ${path}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------- read APIs

export const fetchHealth = () => getJSON<{ status: string; version: string }>(
  "/health");

export const fetchDetectors = () => getJSON<{ detectors: DetectorInfo[] }>(
  "/detectors");

export const fetchHistory = () => getJSON<{ entries: HistoryEntry[] }>(
  "/history");

export const fetchHistoryReport = (id: string) => getJSON<AnalyzeReport>(
  `/history/${id}`);

export const deleteHistoryReport = (id: string) =>
  sendJSON<{ deleted: string }>("DELETE", `/history/${id}`);

export const fetchKeys = () => getJSON<{
  providers: PublicProvider[]; templates: Record<string, ProviderTemplate>;
  storage: string;
}>("/keys");

export const fetchSettings = () => getJSON<{
  settings: Settings; storage: string;
}>("/settings");

export const fetchCalibration = () => getJSON<{
  calibration: Record<string, CalibrationFitPublic>;
}>("/calibration");

export const fetchBench = () => getJSON<{ datasets: BenchDataset[] }>("/bench");

// ------------------------------------------------------------- write APIs

export const analyze = (text: string, detector_ids: string[] | null,
                        force_refresh = false) =>
  sendJSON<AnalyzeReport>("POST", "/analyze",
    { text, detector_ids, force_refresh });

export const saveProvider = (p: {
  id: string; kind: string; base_url: string; api_key?: string;
  default_model: string; enabled?: boolean; note?: string;
}) => sendJSON<{ providers: PublicProvider[] }>("PUT", "/keys", p);

export const deleteProvider = (id: string) =>
  sendJSON<{ providers: PublicProvider[] }>("DELETE", `/keys/${id}`);

export const testProvider = (id: string) =>
  sendJSON<{ id: string; ok: boolean; detail: string; latency_ms: number }>(
    "POST", `/keys/${id}/test`);

export const saveSettings = (settings: Partial<Settings>) =>
  sendJSON<{ settings: Settings }>("PUT", "/settings", settings);

export const runCalibration = (detector_ids: string[] | null, dataset: string) =>
  sendJSON<{
    dataset: string; n_samples: number;
    labels: { ai: number; human: number };
    results: Record<string, CalibrationFitPublic & { ok: boolean }>;
  }>("POST", "/calibration/run", { detector_ids, dataset });

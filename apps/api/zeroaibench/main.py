"""FastAPI application — the workbench's local server.

Design: an explicit WORLD singleton bundle (settings / registry / providers /
calibration / engine / history), so state is inspectable and the app is easy
to reason about in tests.

Endpoints (all JSON, all prefixed /api):

  GET  /health                     — liveness
  GET  /meta                       — version, dirs, plugin load errors, ML status
  GET  /detectors                  — registry + live availability
  POST /analyze                    — run the workbench on a text
  GET  /history                    — recent runs
  GET  /history/{id}               — full saved report
  DEL  /history/{id}
  GET  /keys                       — BYOK provider configs (masked)
  PUT  /keys                       — upsert a provider (id, kind, base_url...)
  DEL  /keys/{id}
  POST /keys/{id}/test             — connectivity/auth check
  GET  /keys/{id}/models           — list models the provider exposes
  GET  /settings                   — global + per-detector settings
  PUT  /settings                   — update settings (persisted locally)
  GET  /calibration                — stored calibration fits
  POST /calibration/run            — recalibrate detectors on a labeled set
  GET  /bench                      — labeled corpora (demo + user JSONL)
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import __version__, config as cfg
from .calibration import (CalibrationStore, all_corpora, calibrate_detector,
                          fit_to_public)
from .detectors import DetectorRegistry, build_default_registry
from .detectors.lm_common import ml_status
from .engine import Engine
from .history import HistoryStore
from .providers.manager import ProviderManager
from .schemas import (AnalyzeRequest, AnalyzeReport, Availability,
                      CalibrationInfo, HistoryEntry, ProviderConfig,
                      ProviderTestResult, PublicProvider, Verdict)


# ------------------------------------------------------------------- state --

def load_settings() -> dict:
    path = cfg.data_dir() / "settings.json"
    settings = json.loads(json.dumps(cfg.DEFAULT_SETTINGS))  # deep copy
    try:
        if path.exists():
            user = json.loads(path.read_text(encoding="utf-8"))
            for section in ("detect", "detectors"):
                if isinstance(user.get(section), dict):
                    settings[section].update(user[section])
    except Exception:
        pass
    return settings


def save_settings(settings: dict) -> dict:
    path = cfg.data_dir() / "settings.json"
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return settings


class CalibrationRunRequest(BaseModel):
    detector_ids: list[str] | None = None
    dataset: str = "demo"
    max_chars: int = 4000


@dataclass
class World:
    settings: dict = field(default_factory=load_settings)
    registry: DetectorRegistry = field(default_factory=build_default_registry)
    providers: ProviderManager = field(default_factory=ProviderManager)
    calibration: CalibrationStore = field(default_factory=CalibrationStore)
    history: HistoryStore = field(default_factory=lambda: HistoryStore(
        max_entries=load_settings()["detect"]["max_history"]))
    engine: Engine | None = None

    def rebuild_engine(self) -> None:
        self.engine = Engine(
            registry=self.registry,
            providers=self.providers,
            calibration=self.calibration,
            settings=self.settings,
            history_store=self.history)


def build_app(world: World | None = None) -> FastAPI:
    world = world or World()
    world.rebuild_engine()

    app = FastAPI(title=cfg.APP_NAME, version=__version__,
                  description=__doc__ if isinstance(__doc__, str) else None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # local-first workbench; typical dev = localhost
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.world = world

    # ---------------------------------------------------------------- health

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__,
                "app": cfg.APP_NAME}

    @app.get("/api/meta")
    def meta():
        w: World = app.state.world
        ml_ok, ml_err = ml_status()
        return {
            "version": __version__,
            "data_dir": str(cfg.data_dir()),
            "plugin_errors": w.registry.plugin_errors,
            "ml_available": ml_ok,
            "ml_note": "" if ml_ok else ml_err,
        }

    # ------------------------------------------------------------ detectors

    @app.get("/api/detectors")
    def detectors():
        w: World = app.state.world
        return {"detectors": w.engine.available_detectors()}

    @app.post("/api/analyze")
    async def analyze(req: AnalyzeRequest) -> AnalyzeReport:
        w: World = app.state.world
        try:
            report = await w.engine.analyze(
                req.text, req.detector_ids, req.force_refresh)
            # persist full report
            try:
                w.history.save_report(report)
            except Exception:
                pass
            return report
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"{type(e).__name__}: {e}")

    # -------------------------------------------------------------- history

    @app.get("/api/history")
    def history_list():
        w: World = app.state.world
        return {"entries": [e.model_dump() for e in w.history.list()]}

    @app.get("/api/history/{rid}")
    def history_get(rid: str):
        w: World = app.state.world
        report = w.history.get(rid)
        if report is None:
            raise HTTPException(404, "run not found")
        return report

    @app.delete("/api/history/{rid}")
    def history_delete(rid: str):
        w: World = app.state.world
        if not w.history.delete(rid):
            raise HTTPException(404, "run not found")
        return {"deleted": rid}

    # ------------------------------------------------------------- BYOK keys

    @app.get("/api/keys")
    def keys_list():
        w: World = app.state.world
        return {
            "providers": w.providers.public_view(),
            "templates": w.providers.templates,
            "storage": str(w.providers.path),
        }

    @app.put("/api/keys")
    def keys_upsert(body: ProviderConfig):
        w: World = app.state.world
        try:
            data = body.model_dump()
            data = {k: v for k, v in data.items() if k not in ("key_mask", "has_key")}
            w.providers.upsert(data)
        except ValueError as e:
            raise HTTPException(422, detail=str(e))
        except Exception as e:
            raise HTTPException(500, detail=f"{type(e).__name__}: {e}")
        return {"providers": w.providers.public_view()}

    @app.delete("/api/keys/{pid}")
    def keys_delete(pid: str):
        w: World = app.state.world
        w.providers.remove(pid)
        return {"providers": w.providers.public_view()}

    @app.post("/api/keys/{pid}/test", response_model=ProviderTestResult)
    async def keys_test(pid: str):
        w: World = app.state.world
        provider = w.providers.build(pid)
        if provider is None:
            raise HTTPException(404, "provider not configured")
        t0 = time.monotonic()
        ok, detail = await provider.health()
        return ProviderTestResult(
            id=pid, ok=ok, detail=detail,
            latency_ms=int((time.monotonic() - t0) * 1000))

    @app.get("/api/keys/{pid}/models")
    async def keys_models(pid: str):
        w: World = app.state.world
        provider = w.providers.build(pid)
        if provider is None:
            raise HTTPException(404, "provider not configured")
        return {"models": await provider.list_models()}

    # ------------------------------------------------------------- settings

    @app.get("/api/settings")
    def settings_get():
        w: World = app.state.world
        return {"settings": w.settings,
                "storage": str(cfg.data_dir() / "settings.json")}

    @app.put("/api/settings")
    async def settings_put(body: dict):
        w: World = app.state.world
        try:
            for section in ("detect", "detectors"):
                incoming = body.get(section, {})
                if isinstance(incoming, dict):
                    w.settings.setdefault(section, {}).update(incoming)
            w.settings["detect"]["max_history"] = int(
                w.settings["detect"].get("max_history", 100))
            save_settings(w.settings)
        except Exception as e:
            raise HTTPException(422, detail=f"invalid settings: {e}")
        w.history.max_entries = w.settings["detect"]["max_history"]
        w.rebuild_engine()
        return {"settings": w.settings}

    # ----------------------------------------------------------- calibration

    @app.get("/api/calibration")
    def calibration_summary():
        w: World = app.state.world
        out = {}
        for det in w.registry.ordered():
            fit = w.calibration.get(det.id)
            out[det.id] = fit_to_public(fit) if fit else {
                "status": "uncalibrated", "threshold": 0.5}
        return {"calibration": out}

    @app.post("/api/calibration/run")
    async def calibration_run(req: CalibrationRunRequest):
        w: World = app.state.world
        corpora = all_corpora()
        corpus = corpora.get(req.dataset)
        if corpus is None:
            raise HTTPException(404, f"dataset '{req.dataset}' not found "
                                     f"(available: {list(corpora.keys())})")
        det_ids = [d for d in w.registry.ids()
                   if d in (req.detector_ids or w.registry.ids())]
        results: dict[str, dict] = {}
        for det_id in det_ids:
            det = w.registry.get(det_id)
            avail = det.availability(None)
            if not avail.ok:
                results[det_id] = {"ok": False,
                                   "reason": f"unavailable: {avail.reason}"}
                continue
            try:
                fit = await calibrate_detector(
                    det, corpus, max_chars=req.max_chars)
            except Exception as e:
                fit = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
            if fit.get("ok") and fit.get("fit"):
                w.calibration.put(det_id, fit["fit"])
                pub = fit_to_public(fit["fit"])
                pub["ok"] = True
                pub["errors"] = fit.get("errors", [])
                results[det_id] = pub
            else:
                results[det_id] = {"ok": False,
                                   "reason": fit.get("reason", ""),
                                   "errors": fit.get("errors", [])}
        return {"dataset": corpus.name, "n_samples": corpus.n,
                "labels": corpus.label_counts(), "results": results}

    # ----------------------------------------------------------------- bench

    @app.get("/api/bench")
    def bench():
        corpora = all_corpora()
        out = []
        for name, c in corpora.items():
            counts = c.label_counts()
            out.append({"name": name, "n": c.n, "labels": counts,
                        "where": _corpus_where(name)})
        return {"datasets": out}

    def _corpus_where(name: str) -> str:
        if name == "demo":
            return str(Path(__file__).parent / "bench" / "demo.jsonl")
        return str(cfg.data_dir() / "bench" / f"{name}.jsonl")

    # ------------------------------------------------- static UI (docker image)
    # If a built frontend (apps/web/dist) was copied into zeroaibench/static
    # at image build time, serve it as a fallback for non-API GETs. The Vite
    # dev server (npm run dev) never hits this path during normal development.
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(static_dir), html=True),
                  name="static")

    return app


app = build_app()


def run(host: str = "127.0.0.1", port: int = cfg.API_PORT) -> None:  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")

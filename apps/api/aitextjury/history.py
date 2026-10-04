"""Run history — a plain JSON store, local-first and easy to export.

Reports live in data/history/<id>.json; manifest.json keeps a lightweight
index. The count is capped (settings.detect.max_history) and old entries are
pruned oldest-first. No database, no telemetry — your runs stay your own.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from . import config as cfg
from .schemas import AnalyzeReport, HistoryEntry, TextStats, Verdict


class HistoryStore:
    def __init__(self, root: Path | None = None, max_entries: int = 100):
        self.root = root or cfg.subdir("history")
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------ internals

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _load_manifest(self) -> list[dict]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return data.get("entries", [])
        except Exception:
            return []

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps({"entries": self._manifest}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def _path(self, rid: str) -> Path:
        return self.root / f"{rid}.json"

    # ------------------------------------------------------------------ api

    def save(self, entry: HistoryEntry, report: AnalyzeReport | None = None,
             max_entries: int | None = None) -> None:
        with self.lock:
            cap = max_entries or self.max_entries
            rec = entry.model_dump()
            self._manifest = [e for e in self._manifest if e["id"] != rec["id"]]
            self._manifest.append(rec)
            self._manifest.sort(key=lambda e: e["created_at"], reverse=True)
            self._save_manifest()
            if report is not None:
                self._path(rec["id"]).write_text(
                    report.model_dump_json(indent=2), encoding="utf-8")
            # prune
            while len(self._manifest) > cap:
                old = self._manifest.pop()
                try:
                    self._path(old["id"]).unlink(missing_ok=True)
                except Exception:
                    pass
            # fix ordering by recency (newest first)
            self._manifest.sort(key=lambda e: e["created_at"], reverse=True)

    def save_report(self, report: AnalyzeReport) -> None:
        entry = HistoryEntry(
            id=report.id,
            created_at=report.created_at,
            title=report.title,
            stats=TextStats(
                chars=report.stats.chars, words=report.stats.words,
                sentences=report.stats.sentences,
                paragraphs=report.stats.paragraphs,
                language=report.stats.language,
                cjk_ratio=report.stats.cjk_ratio),
            consensus_score=report.consensus.score,
            consensus_verdict=report.consensus.verdict,
            detectors_used=[r.detector_id for r in report.results])
        self.save(entry, report)

    def list(self) -> list[HistoryEntry]:
        return [HistoryEntry(**{
            **e,
            "consensus_verdict": e.get("consensus_verdict", "uncertain"),
        }) for e in self._manifest]

    def get(self, rid: str) -> AnalyzeReport | None:
        p = self._path(rid)
        if not p.exists():
            return None
        try:
            return AnalyzeReport.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete(self, rid: str) -> bool:
        with self.lock:
            before = len(self._manifest)
            self._manifest = [e for e in self._manifest if e["id"] != rid]
            if len(self._manifest) == before:
                return False
            self._save_manifest()
            self._path(rid).unlink(missing_ok=True)
            return True

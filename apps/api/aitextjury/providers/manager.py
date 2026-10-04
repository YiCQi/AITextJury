"""Provider manager: BYOK configuration storage + resolution.

`data/providers.json` holds the user's provider configs. Keys are:
  * saved locally only (this file is gitignored and never included in any
    HTTP payload except as auth toward the configured provider itself),
  * resolved at call time: explicit key > template env var,
  * echoed back only as masked strings via the API.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .. import config as cfg
from . import build_provider, BaseProvider


def mask_key(key: str) -> str:
    if not key:
        return "—"
    if len(key) <= 8:
        return key[:2] + "…"
    return f"{key[:4]}…{key[-4:]}"


class ProviderManager:
    def __init__(self, path: Path | None = None,
                 templates: dict | None = None):
        self.path = path or cfg.data_dir() / "providers.json"
        self.lock = threading.Lock()
        self.templates = templates or cfg.DEFAULT_SETTINGS["providers"]["templates"]
        self._providers: list[dict] | None = None
        self._load()

    # ------------------------------------------------------------- storage --

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._providers = raw.get("providers", [])
            except Exception:
                self._providers = []
        else:
            self._providers = []

    def _save(self) -> None:
        with self.lock:
            self.path.write_text(
                json.dumps({"providers": self._providers},
                           ensure_ascii=False, indent=2),
                encoding="utf-8")

    # ------------------------------------------------------------- access --

    def all_configs(self) -> list[dict]:
        assert self._providers is not None
        return list(self._providers)

    def upsert(self, cfg_dict: dict) -> dict:
        assert self._providers is not None
        pid = cfg_dict.get("id", "").strip()
        if not pid:
            raise ValueError("provider id required")
        cfg_dict["id"] = pid
        existing = [p for p in self._providers if p["id"] == pid]
        for cleaned in ("api_key", "base_url", "note"):
            cfg_dict.setdefault(cleaned, "")
        if existing:
            # empty api_key means "keep existing" (frontend never resends keys)
            if not (cfg_dict.get("api_key") or "").strip():
                cfg_dict["api_key"] = existing[0].get("api_key", "")
            self._providers = [p for p in self._providers if p["id"] != pid] + [cfg_dict]
        else:
            template = self.templates.get(pid.split(":")[0], {})
            if not cfg_dict.get("base_url") and template.get("base_url"):
                cfg_dict["base_url"] = template["base_url"]
            if not cfg_dict.get("default_model") and template.get("default_model"):
                cfg_dict["default_model"] = template["default_model"]
            if not cfg_dict.get("env_key") and template.get("env_key"):
                cfg_dict["env_key"] = template["env_key"]
            self._providers.append(cfg_dict)
        self._save()
        return cfg_dict

    def remove(self, pid: str) -> None:
        assert self._providers is not None
        self._providers = [p for p in self._providers if p["id"] != pid]
        self._save()

    def get(self, pid: str) -> dict | None:
        for p in self.all_configs():
            if p.get("id") == pid:
                return p
        return None

    def enabled(self) -> list[dict]:
        return [p for p in self.all_configs() if p.get("enabled", True)]

    def has_usable(self) -> bool:
        return bool(self.enabled())

    # -------------------------------------------------------------- build --

    def resolve_env_fallback(self, pconfig: dict) -> dict:
        """Attach env_key hint from templates so keys can be optional."""
        base = pconfig.get("id", "").split(":")[0]
        tmpl = self.templates.get(base, {})
        merged = dict(pconfig)
        merged.setdefault("env_key", tmpl.get("env_key", ""))
        merged.setdefault("base_url", tmpl.get("base_url", ""))
        if not merged.get("default_model"):
            merged["default_model"] = tmpl.get("default_model", "")
        return merged

    def build(self, pid: str) -> BaseProvider | None:
        raw = self.get(pid)
        if not raw:
            return None
        return build_provider(self.resolve_env_fallback(raw))

    def build_all_enabled(self) -> list[BaseProvider]:
        providers = []
        for raw in self.enabled():
            try:
                providers.append(build_provider(self.resolve_env_fallback(raw)))
            except Exception:
                continue
        return providers

    # ------------------------------------------------------------- public --

    def public_view(self) -> list[dict]:
        out = []
        for raw in self.all_configs():
            raw = self.resolve_env_fallback(raw)
            key = (raw.get("api_key") or "").strip()
            env_alt = ""
            if not key and raw.get("env_key"):
                key = os.environ.get(raw["env_key"], "").strip()
                env_alt = raw["env_key"] if key else ""
            out.append({
                "id": raw.get("id"),
                "kind": raw.get("kind", "openai_compatible"),
                "base_url": raw.get("base_url", ""),
                "default_model": raw.get("default_model", ""),
                "enabled": bool(raw.get("enabled", True)),
                "note": raw.get("note", ""),
                "key_mask": mask_key(key) if key else ("—"),
                "has_key": bool(key),
                **({"env_key": env_alt} if env_alt else {}),
            })
        return out

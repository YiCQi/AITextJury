"""Runtime configuration & paths.

Everything user-generated (history, API keys, cache, calibration fits,
plugin overrides) lives under a data directory whose location is:

    1. $AITEXTJURY_HOME (explicit override)
    2. <repo>/data            (default in a dev checkout)

BYOK keys never leave this machine except as auth headers toward the exact
provider endpoint the user configured.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "AITextJury"
VERSION = "0.1.0"
API_PORT = int(os.environ.get("AITEXTJURY_PORT", "8000"))


def repo_root() -> Path:
    """Path to the git repo root (works in installed-package layouts too)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "apps" / "api").is_dir() or (parent / ".git").exists():
            return parent
    return here.parents[2]


def data_dir() -> Path:
    override = os.environ.get("AITEXTJURY_HOME")
    base = Path(override) if override else repo_root() / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def subdir(name: str) -> Path:
    p = data_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- defaults --

DEFAULT_SETTINGS: dict = {
    "providers": {
        # kind: openai_compatible | gemini
        # Keys may be left empty -> the provider adapter falls back to the
        # corresponding environment variable (see providers/manager.py).
        "templates": {
            "openai": {
                "kind": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "env_key": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
            },
            "deepseek": {
                "kind": "openai_compatible",
                "base_url": "https://api.deepseek.com/v1",
                "env_key": "DEEPSEEK_API_KEY",
                "default_model": "deepseek-chat",
            },
            "openrouter": {
                "kind": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "env_key": "OPENROUTER_API_KEY",
                "default_model": "openrouter/auto",
            },
            "groq": {
                "kind": "openai_compatible",
                "base_url": "https://api.groq.com/openai/v1",
                "env_key": "GROQ_API_KEY",
                "default_model": "llama-3.3-70b-versatile",
            },
            "ollama": {
                "kind": "openai_compatible",
                "base_url": "http://localhost:11434/v1",
                "env_key": "",
                "default_model": "qwen2.5:7b-instruct",
            },
            "gemini": {
                "kind": "gemini",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "env_key": "GEMINI_API_KEY",
                "default_model": "gemini-2.0-flash",
            },
            "custom": {
                "kind": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "env_key": "",
                "default_model": "your-model",
            },
        }
    },
    "detect": {
        # per-call timeouts (seconds)
        "timeout_local": 90,
        "timeout_llm": 120,
        "max_text_chars": 60000,
        "max_history": 100,
        "cache_enabled": True,
    },
    "detectors": {
        # user overrides merged over each detector's own defaults, e.g.:
        # "binoculars": {"observer_model": "gpt2-medium", "performer_model": "gpt2"},
        # "lm_perplexity": {"model": "gpt2"},
        # "hf_classifier": {"model": "MyOrg/my-ai-detector"},
    },
}


def is_windows() -> bool:
    return sys.platform.startswith("win")

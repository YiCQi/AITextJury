"""Detector registry — every detector, built-in or plugin, is born equal here.

The registry instantiates the core detectors and scans the user's plugin dirs
for `register(registry)` hooks (see docs/DETECTOR_API.md). Loading failures
never break the workbench: a broken plugin simply reports unavailable.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

from .. import config as cfg
from .base import AnalysisContext, BaseDetector, DetectorError, RawOutcome, \
    RawSegment
from .binoculars import BinocularsDetector
from .fast_detect_gpt import FastDetectGPTDetector
from .hf_classifier import HFClassifierDetector
from .lm_perplexity import LMPerplexityDetector
from .llm_judge import LLMJudgeDetector
from .stylometry import StylometryDetector

CORE_DETECTORS: list[type[BaseDetector]] = [
    StylometryDetector,
    LMPerplexityDetector,
    FastDetectGPTDetector,
    BinocularsDetector,
    HFClassifierDetector,
    LLMJudgeDetector,
]


class DetectorRegistry:
    def __init__(self):
        self.detectors: dict[str, BaseDetector] = {}
        self.plugin_errors: list[dict] = []

    # ------------------------------------------------------------- core ---

    def register(self, detector: BaseDetector, replace: bool = True) -> None:
        if detector.id in self.detectors and not replace:
            raise ValueError(f"duplicate detector id: {detector.id}")
        self.detectors[detector.id] = detector

    def get(self, detector_id: str) -> BaseDetector | None:
        return self.detectors.get(detector_id)

    def require(self, detector_id: str) -> BaseDetector:
        d = self.detectors.get(detector_id)
        if d is None:
            raise DetectorError(f"unknown detector: {detector_id}")
        return d

    def ids(self) -> list[str]:
        return list(self.detectors.keys())

    def ordered(self) -> list[BaseDetector]:
        return [self.detectors[k] for k in self.ids()]

    # --------------------------------------------------------- loading ---

    def load_core(self) -> None:
        for cls in CORE_DETECTORS:
            try:
                self.register(cls())
            except Exception as e:  # pragma: no cover
                self.plugin_errors.append(
                    {"id": cls.id, "error": f"{type(e).__name__}: {e}"})

    def load_plugins(self, extra_dirs: list[Path] | None = None) -> None:
        """Load `plugins/*.py` exposing `register(registry)`."""
        dirs: list[Path] = []
        bundled = cfg.repo_root() / "plugins"
        if bundled.is_dir():
            dirs.append(bundled)
        user = cfg.data_dir() / "plugins"
        if user.is_dir():
            dirs.append(user)
        for d in extra_dirs or []:
            p = Path(d)
            if p.is_dir():
                dirs.append(p)
        seen: set[Path] = set()
        for d in dirs:
            for py in sorted(d.glob("*.py")):
                if py.name.startswith("_") or py.resolve() in seen:
                    continue
                seen.add(py.resolve())
                name = f"aitextjury_plugin_{py.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(name, py)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[name] = mod
                    spec.loader.exec_module(mod)
                    register = getattr(mod, "register", None)
                    if callable(register):
                        before = set(self.detectors.keys())
                        register(self)
                        new = [k for k in self.detectors if k not in before]
                        for nid in new:
                            self.detectors[nid].family = \
                                self.detectors[nid].family or "plugin"
                except Exception as e:
                    self.plugin_errors.append(
                        {"id": py.name, "error":
                         f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"})

    def load_all(self) -> None:
        self.load_core()
        self.load_plugins()


def build_default_registry() -> DetectorRegistry:
    r = DetectorRegistry()
    r.load_core()
    r.load_plugins()
    return r


__all__ = [
    "AnalysisContext", "BaseDetector", "DetectorError", "RawOutcome",
    "RawSegment", "DetectorRegistry", "build_default_registry",
    "StylometryDetector", "LMPerplexityDetector", "FastDetectGPTDetector",
    "BinocularsDetector", "HFClassifierDetector", "LLMJudgeDetector",
]

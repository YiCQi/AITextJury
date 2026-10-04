"""AITextJury — an open workbench for AI-generated text detection.

Core philosophy: *evidence, not verdicts*. Instead of one opaque "AI rate",
AITextJury runs multiple independent detectors through a unified Detector API
and surfaces the underlying evidence — per-segment heatmaps, perplexity,
cross-model agreement, stylometric fingerprints, calibration quality — so a
human can ask "why does this text look AI-generated?" and get an answer.

All heavy local-ML detectors are optional at runtime: the workbench is fully
functional with zero ML dependencies (stylometry + calibration + BYOK LLM
Judge) and unlocks the rest when `pip install aitextjury[ml]` is installed.
"""

__version__ = "0.1.0"

"""AITextJury CLI.

  python -m aitextjury.cli analyze report.txt -d stylometry,lm_perplexity
  python -m aitextjury.cli calibrate -d stylometry --dataset hc3
  python -m aitextjury.cli detectors

The CLI runs the same engine as the web UI — same detectors, same
calibration store, same cache — so results are interchangeable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from . import __version__ as VERSION
from .calibration import CalibrationStore, all_corpora, calibrate_detector, \
    fit_to_public
from .detectors.lm_common import ml_status
from .main import JUDGE_CALIB_SAMPLES, World
from .providers.manager import ProviderManager


def _world() -> World:
    w = World()
    w.rebuild_engine()
    return w


def cmd_detectors(_args) -> int:
    w = _world()
    for info in w.engine.available_detectors():
        status = "ready" if info["available"] else f"unavailable ({info['reason']})"
        print(f"[{info['id']:>22}] {info['name']:<36} {status}")
    if w.registry.plugin_errors:
        print("\nplugin errors:", file=sys.stderr)
        for e in w.registry.plugin_errors:
            print(f"  {e['id']}: {e['error']}", file=sys.stderr)
    return 0


def cmd_analyze(args) -> int:
    w = _world()
    text = open(args.file, encoding="utf-8", errors="replace").read()
    det_ids = [s.strip() for s in args.detectors.split(",")] if args.detectors else None
    report = asyncio.run(w.engine.analyze(text, det_ids,
                                          force_refresh=args.force))
    payload = json.loads(report.model_dump_json(indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"report written to {args.out}")
    print(f"\n— AITextJury report {report.id} —")
    for r in report.results:
        if r.error:
            print(f"  {r.detector_id:>22}  ERROR: {r.error}")
        else:
            print(f"  {r.detector_id:>22}  "
                  f"score={r.score:.3f}  verdict={r.verdict.value}  "
                  f"({r.runtime_ms} ms)")
    c = report.consensus
    print(f"  {'consensus':>22}  score={c.score:.3f}  "
          f"verdict={c.verdict.value}  agreement={c.agreement:.2f}")
    for note in c.notes:
        print(f"     · {note}")
    return 0


def cmd_calibrate(args) -> int:
    w = _world()
    corpora = all_corpora()
    corpus = corpora.get(args.dataset)
    if corpus is None:
        print(f"dataset '{args.dataset}' not found. available: "
              f"{list(corpora.keys())}", file=sys.stderr)
        return 2
    try:
        # LLM jurors cost API calls: only when named (-d llm_judge = all of them)
        dets = (w.engine.resolve([s.strip() for s in args.detectors.split(",")])
                if args.detectors
                else [d for d in w.engine.jurors() if d.family != "byok_llm"])
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    for det in dets:
        det_id = det.id
        paid = det.family == "byok_llm"
        avail = det.availability(w.providers if paid else None)
        if not avail.ok:
            print(f"  {det_id:>22}  skip ({avail.reason})")
            continue
        n = args.max_samples or (JUDGE_CALIB_SAMPLES if paid else 0)
        print(f"  {det_id:>22}  calibrating on {corpus.name} "
              f"(n={min(n, corpus.n) if n else corpus.n}) …", end="", flush=True)
        fit = asyncio.run(calibrate_detector(
            det, corpus, max_samples=n or None, concurrency=4 if paid else 1,
            providers=w.providers,
            det_settings={det_id: w.engine.detector_eff(det_id)}))
        if fit.get("ok"):
            w.calibration.put(det_id, fit["fit"])
            m = fit["fit"]
            print(f"  auc={m['auc']:.3f}  acc={m['acc']:.3f}  "
                  f"ece={m['ece']:.3f}  thr={m['threshold']:.2f}")
        else:
            print(f"  failed ({fit.get('reason')})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aitextjury",
        description="AITextJury — open workbench for AI text detection. "
                    "Evidence, not verdicts.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_det = sub.add_parser("detectors", help="list detectors & availability")
    p_det.set_defaults(func=cmd_detectors)

    p_an = sub.add_parser("analyze", help="analyze a text file")
    p_an.add_argument("file")
    p_an.add_argument("-d", "--detectors", default="",
                      help="comma-separated detector ids (default: defaults)")
    p_an.add_argument("-o", "--out", default="", help="write JSON report here")
    p_an.add_argument("--force", action="store_true",
                      help="bypass the result cache")
    p_an.set_defaults(func=cmd_analyze)

    p_cal = sub.add_parser("calibrate", help="fit calibration on a labeled set")
    p_cal.add_argument("-d", "--detectors", default="",
                       help="comma-separated detector ids (default: all)")
    p_cal.add_argument("--dataset", default="hc3")
    p_cal.add_argument("--max-samples", type=int, default=0,
                       help="stratified subset size (LLM jurors default to "
                            f"{JUDGE_CALIB_SAMPLES})")
    p_cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
run_evaluation.py
=================
Programmatic evaluation of the ShopCo support agent with DeepEval's evaluate().

For every test case from agent_eval.test_dataset, this script:
  1. builds the metrics declared in the case's `target_metrics`
     (thresholds come from agent_eval.metrics),
  2. calls deepeval.evaluate() for that case only,
  3. prints each metric's score, threshold, verdict and the judge's reason,
  4. saves a JSON report in reports/.

Examples
--------
    python scripts/run_evaluation.py                     # static dataset, 1 run
    python scripts/run_evaluation.py --mode live         # evaluate the live SUT
    python scripts/run_evaluation.py --runs 2            # double run + variability
    python scripts/run_evaluation.py --cases TC-01,TC-02 --judge-model gpt-4o

Exit codes
----------
    0  every metric verdict matched its expected verdict (including TC-02's FAIL)
    1  at least one unexpected result or evaluation error
    2  configuration error (e.g. missing OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List, Optional

# Fallback for running without `pip install -e .`: put src/ on the path.
try:
    import agent_eval  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase

try:  # DeepEval >= 3.x exposes the config objects here
    from deepeval.evaluate import AsyncConfig, DisplayConfig, ErrorConfig
except ImportError:  # pragma: no cover - layout differs in some versions
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

from agent_eval.metrics import build_metrics_for, get_judge_model, get_spec, metric_display_name
from agent_eval.reporting import (
    CaseResult, MetricResult, RunResult, compute_variability, print_case_result,
    print_run_header, print_run_summary, print_variability_report, save_run_report,
    save_variability_report,
)
from agent_eval.test_dataset import get_all_test_cases, get_metadata
from support_agent.config import PROJECT_ROOT, Settings
from support_agent.prompts import PROMPT_VERSION

logger = logging.getLogger("run_evaluation")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the ShopCo support agent with DeepEval.")
    p.add_argument("--mode", choices=["static", "live"],
                   default=os.getenv("EVAL_DATASET_MODE", "static"),
                   help="static = recorded outputs, live = call the SUT (default: env or static)")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of repeated runs (>=2 enables variability analysis)")
    p.add_argument("--cases", type=str, default="",
                   help="Comma-separated case IDs to run, e.g. TC-01,TC-04 (default: all)")
    p.add_argument("--judge-model", type=str, default=None,
                   help="Override EVAL_JUDGE_MODEL for the LLM-as-a-judge metrics")
    p.add_argument("--reports-dir", type=str, default=os.getenv("EVAL_REPORTS_DIR", "reports"))
    p.add_argument("--no-save", action="store_true", help="Do not write JSON reports")
    p.add_argument("--sequential", action="store_true",
                   help="Disable async metric execution (slower; easier to debug)")
    p.add_argument("--deepeval-verbose", action="store_true",
                   help="Also print DeepEval's intermediate verbose logs")
    args = p.parse_args(argv)
    if args.runs < 1:
        p.error("--runs must be >= 1")
    return args


# =============================================================================
# Helpers
# =============================================================================

def _deepeval_version() -> str:
    try:
        return version("deepeval")
    except PackageNotFoundError:
        return "unknown"


def build_run_config(args: argparse.Namespace, judge_model: str, kb_version: str) -> dict:
    """Everything needed to reproduce or compare a run later."""
    s = Settings.from_env()
    return {
        "dataset_mode": args.mode,
        "judge_model": judge_model,
        "sut_backend": s.backend if args.mode == "live" else "n/a (static)",
        "sut_model": s.model if args.mode == "live" else "n/a (static)",
        "sut_temperature": s.temperature if args.mode == "live" else "n/a (static)",
        "prompt_version": PROMPT_VERSION,
        "kb_version": kb_version,
        "deepeval_version": _deepeval_version(),
        "async": not args.sequential,
    }


def load_cases(mode: str, selected_ids: List[str]):
    """Build test cases. In live mode the SUT is called again on every run."""
    cases = get_all_test_cases(mode=mode)
    if selected_ids:
        cases = [c for c in cases if get_metadata(c)["id"] in selected_ids]
        missing = set(selected_ids) - {get_metadata(c)["id"] for c in cases}
        if missing:
            raise KeyError(f"Unknown case id(s): {sorted(missing)}")
    return cases


def _describe_case(test_case) -> dict:
    """Extract printable input/output fields for both test-case types."""
    if isinstance(test_case, ConversationalTestCase):
        assistant = [t.content for t in test_case.turns if t.role == "assistant"]
        return {"test_type": "conversational", "input": test_case.scenario or "",
                "actual_output": assistant[-1] if assistant else "",
                "num_turns": len(test_case.turns)}
    return {"test_type": "single_turn", "input": test_case.input,
            "actual_output": test_case.actual_output or "", "num_turns": 1}


# =============================================================================
# Core: evaluate one test case with its routed metrics
# =============================================================================

def evaluate_case(test_case, judge_model: str, async_mode: bool, verbose: bool) -> CaseResult:
    meta = get_metadata(test_case)
    expected_pass = meta.get("expected_verdict", "pass") == "pass"

    case = CaseResult(
        case_id=meta["id"],
        name=getattr(test_case, "name", None) or meta["id"],
        category=meta.get("category", ""),
        expected_verdict=meta.get("expected_verdict", "pass"),
        output_source=meta.get("output_source", "static"),
        **_describe_case(test_case),
    )

    # 1) Route fresh metric instances to this case.
    routed = build_metrics_for(test_case, model=judge_model, async_mode=async_mode)
    name_to_key = {metric_display_name(m): key for key, m in routed}

    # 2) Run DeepEval. Rendering is ours, so DeepEval's own printing is disabled.
    started = time.perf_counter()
    try:
        result = evaluate(
            test_cases=[test_case],
            metrics=[m for _, m in routed],
            async_config=AsyncConfig(run_async=async_mode),
            display_config=DisplayConfig(print_results=False, show_indicator=True,
                                         verbose_mode=verbose),
            error_config=ErrorConfig(ignore_errors=True),  # record errors, keep going
        )
    except Exception as exc:  # e.g. invalid API key, network failure
        logger.exception("evaluate() failed for %s", case.case_id)
        case.error = f"{type(exc).__name__}: {exc}"
        case.duration_s = time.perf_counter() - started
        for key, _ in routed:
            spec = get_spec(key)
            case.metrics.append(MetricResult(
                key=key, name=spec.display_name, threshold=spec.threshold,
                lower_is_better=spec.lower_is_better, expected_pass=expected_pass,
                error=case.error))
        return case
    case.duration_s = time.perf_counter() - started

    # 3) Convert DeepEval MetricData into our result model.
    test_results = getattr(result, "test_results", None) or []
    metrics_data = (test_results[0].metrics_data or []) if test_results else []
    seen = set()
    for md in metrics_data:
        key = name_to_key.get(md.name, md.name)
        seen.add(key)
        spec = get_spec(key) if key in name_to_key.values() else None
        case.metrics.append(MetricResult(
            key=key,
            name=spec.display_name if spec else md.name,
            threshold=md.threshold,
            lower_is_better=spec.lower_is_better if spec else False,
            expected_pass=expected_pass,
            score=md.score,
            success=md.success,
            reason=md.reason,
            evaluation_model=getattr(md, "evaluation_model", None),
            cost=getattr(md, "evaluation_cost", None),
            error=getattr(md, "error", None),
        ))

    # 4) Metrics that produced no data are reported as errors, not hidden.
    for key, _ in routed:
        if key not in seen:
            spec = get_spec(key)
            case.metrics.append(MetricResult(
                key=key, name=spec.display_name, threshold=spec.threshold,
                lower_is_better=spec.lower_is_better, expected_pass=expected_pass,
                error="No metric data returned by DeepEval."))
    return case


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args(argv)

    # The judge always needs an API key, even in static mode.
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
              file=sys.stderr)
        return 2

    judge_model = get_judge_model(args.judge_model)
    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    runs: List[RunResult] = []

    for run_index in range(1, args.runs + 1):
        try:
            cases = load_cases(args.mode, selected)
        except Exception as exc:
            print(f"ERROR while building the dataset: {exc}", file=sys.stderr)
            return 2

        kb_version = get_metadata(cases[0]).get("kb_version", "unknown") if cases else "unknown"
        run = RunResult(
            run_id=batch_id,
            run_index=run_index,
            started_at=datetime.now(timezone.utc).isoformat(),
            config=build_run_config(args, judge_model, kb_version),
        )
        print_run_header(run, args.runs)

        for test_case in cases:
            case_result = evaluate_case(test_case, judge_model,
                                        async_mode=not args.sequential,
                                        verbose=args.deepeval_verbose)
            run.cases.append(case_result)
            print_case_result(case_result)

        print_run_summary(run)
        if not args.no_save:
            print(f" Report saved: {save_run_report(run, reports_dir)}")
        runs.append(run)

    if len(runs) > 1:
        rows = compute_variability(runs)
        print_variability_report(rows, len(runs))
        if not args.no_save:
            print(f" Variability report saved: {save_variability_report(rows, runs, reports_dir)}")

    return 0 if all(r.all_as_expected for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
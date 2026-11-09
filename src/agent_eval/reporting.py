"""
agent_eval.reporting
====================
Result models, console rendering, JSON persistence and multi-run variability
analysis.

Console output includes, for every metric: score, threshold and direction,
PASS/FAIL, expected-vs-actual verdict, and the judge's textual reason.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LINE = "=" * 88
SUBLINE = "-" * 88
WRAP_WIDTH = 84
UNSTABLE_RANGE = 0.10   # score spread across runs that we flag as unstable


# =============================================================================
# 1. Terminal colours (disabled for NO_COLOR or non-TTY output)
# =============================================================================

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str: return _c(t, "32")
def red(t: str) -> str: return _c(t, "31")
def yellow(t: str) -> str: return _c(t, "33")
def bold(t: str) -> str: return _c(t, "1")
def dim(t: str) -> str: return _c(t, "2")


# =============================================================================
# 2. Result models
# =============================================================================

@dataclass
class MetricResult:
    key: str
    name: str
    threshold: float
    lower_is_better: bool
    expected_pass: bool
    score: Optional[float] = None
    success: Optional[bool] = None
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    cost: Optional[float] = None
    error: Optional[str] = None

    @property
    def as_expected(self) -> bool:
        """True when the metric verdict matches the test case's expected verdict."""
        return self.error is None and self.success is not None and self.success == self.expected_pass

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "as_expected": self.as_expected}


@dataclass
class CaseResult:
    case_id: str
    name: str
    category: str
    test_type: str              # "single_turn" | "conversational"
    expected_verdict: str       # "pass" | "fail"
    output_source: str          # "static" | "live"
    input: str
    actual_output: str
    num_turns: int = 1
    metrics: List[MetricResult] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def as_expected(self) -> bool:
        return self.error is None and bool(self.metrics) and all(m.as_expected for m in self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = [m.to_dict() for m in self.metrics]
        data["as_expected"] = self.as_expected
        return data


@dataclass
class RunResult:
    run_id: str
    run_index: int
    started_at: str
    config: Dict[str, Any]
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def all_as_expected(self) -> bool:
        return bool(self.cases) and all(c.as_expected for c in self.cases)

    @property
    def total_cost(self) -> float:
        return sum(m.cost or 0.0 for c in self.cases for m in c.metrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_index": self.run_index,
            "started_at": self.started_at,
            "config": self.config,
            "summary": {
                "cases": len(self.cases),
                "cases_as_expected": sum(c.as_expected for c in self.cases),
                "all_as_expected": self.all_as_expected,
                "total_cost_usd": round(self.total_cost, 6),
            },
            "cases": [c.to_dict() for c in self.cases],
        }


# =============================================================================
# 3. Console rendering
# =============================================================================

def _fmt_score(score: Optional[float]) -> str:
    return "  n/a" if score is None else f"{score:.3f}"


def _shorten(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _wrap(text: str, indent: str) -> str:
    return textwrap.fill(
        " ".join((text or "").split()) or "(no reason returned)",
        width=WRAP_WIDTH, initial_indent=indent, subsequent_indent=indent,
    )


def print_run_header(run: RunResult, total_runs: int) -> None:
    cfg = run.config
    print("\n" + LINE)
    print(bold(f" DeepEval run {run.run_index}/{total_runs}  |  run_id={run.run_id}"))
    print(LINE)
    print(f" Dataset mode : {cfg.get('dataset_mode')}")
    print(f" Judge model  : {cfg.get('judge_model')}")
    print(f" SUT          : backend={cfg.get('sut_backend')} model={cfg.get('sut_model')} "
          f"temperature={cfg.get('sut_temperature')}")
    print(f" Versions     : prompt={cfg.get('prompt_version')} kb={cfg.get('kb_version')} "
          f"deepeval={cfg.get('deepeval_version')}")
    print(LINE)


def print_case_result(case: CaseResult) -> None:
    verdict = green("AS EXPECTED") if case.as_expected else red("UNEXPECTED")
    print("\n" + SUBLINE)
    print(bold(f" {case.name}"))
    print(f" type={case.test_type} | category={case.category} | expected={case.expected_verdict} "
          f"| source={case.output_source} | {case.duration_s:.1f}s | {verdict}")
    print(SUBLINE)

    if case.test_type == "conversational":
        print(f" Scenario     : {_shorten(case.input, 200)}")
        print(f" Dialogue     : {case.num_turns} turns")
        print(f" Last reply   : {_shorten(case.actual_output, 200)}")
    else:
        print(f" Input        : {_shorten(case.input, 200)}")
        print(f" Actual output: {_shorten(case.actual_output, 300)}")

    if case.error:
        print(red(f" ERROR: {case.error}"))

    for m in case.metrics:
        direction = "<=" if m.lower_is_better else ">="
        status = green("PASS") if m.success else red("FAIL")
        if m.error:
            status = yellow("ERROR")
        expectation = green("as expected") if m.as_expected else red("UNEXPECTED")
        print(f"\n   * {bold(m.name):<28} score={_fmt_score(m.score)}  "
              f"(pass if {direction} {m.threshold})  ->  {status}  [{expectation}]")
        if m.error:
            print(red(_wrap(f"Error: {m.error}", "       ")))
        else:
            print(dim("     Reason:"))
            print(_wrap(m.reason or "", "       "))


def print_run_summary(run: RunResult) -> None:
    print("\n" + LINE)
    print(bold(f" SUMMARY - run {run.run_index}"))
    print(LINE)
    print(f" {'Case':<8}{'Metric':<28}{'Score':>8}{'Threshold':>12}{'Verdict':>10}{'Expected?':>13}")
    print(" " + "-" * 79)
    for case in run.cases:
        for m in case.metrics:
            cond = f"{'<=' if m.lower_is_better else '>='} {m.threshold}"
            verdict = "ERROR" if m.error else ("PASS" if m.success else "FAIL")
            expected = "yes" if m.as_expected else "NO"
            line = (f" {case.case_id:<8}{m.name:<28}{_fmt_score(m.score):>8}"
                    f"{cond:>12}{verdict:>10}{expected:>13}")
            print(line if m.as_expected else red(line))
    print(" " + "-" * 79)
    ok = sum(c.as_expected for c in run.cases)
    overall = green("ALL AS EXPECTED") if run.all_as_expected else red("UNEXPECTED RESULTS")
    print(f" Cases as expected: {ok}/{len(run.cases)}  |  "
          f"judge cost: ${run.total_cost:.4f}  |  {overall}")
    print(LINE)


# =============================================================================
# 4. Multi-run variability analysis (stochasticity)
# =============================================================================

def compute_variability(runs: List[RunResult]) -> List[Dict[str, Any]]:
    """
    For each (case, metric) pair, compare scores across runs.
    A pair is 'unstable' if its verdict flips or its score range exceeds UNSTABLE_RANGE.
    """
    buckets: Dict[tuple, Dict[str, Any]] = {}
    for run in runs:
        for case in run.cases:
            for m in case.metrics:
                b = buckets.setdefault((case.case_id, m.key), {
                    "case_id": case.case_id, "metric": m.name,
                    "threshold": m.threshold, "scores": [], "verdicts": [],
                    "valid_runs": 0,
                })
                if (
                    case.error is None and m.error is None
                    and m.score is not None and m.success is not None
                ):
                    b["scores"].append(round(m.score, 4))
                    b["verdicts"].append(m.success)
                    b["valid_runs"] += 1
                else:
                    b["verdicts"].append(None)

    rows: List[Dict[str, Any]] = []
    for b in buckets.values():
        scores = b["scores"]
        spread = (max(scores) - min(scores)) if scores else None
        flipped = len({v for v in b["verdicts"] if v is not None}) > 1
        rows.append({
            **b,
            "mean": round(statistics.mean(scores), 4) if scores else None,
            "stdev": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "range": round(spread, 4) if spread is not None else None,
            "verdict_flip": flipped,
            "unstable": flipped or (spread is not None and spread > UNSTABLE_RANGE),
            "complete": b["valid_runs"] == len(runs),
        })
    return rows


def print_variability_report(rows: List[Dict[str, Any]], num_runs: int) -> None:
    print("\n" + LINE)
    print(bold(f" STOCHASTIC VARIABILITY ACROSS {num_runs} RUNS"))
    print(LINE)
    print(f" {'Case':<8}{'Metric':<28}{'Scores':<22}{'Mean':>7}{'Range':>8}{'Flip':>6}  Status")
    print(" " + "-" * 86)
    for r in rows:
        scores = ", ".join(f"{s:.2f}" for s in r["scores"]) or "n/a"
        rng = "n/a" if r["range"] is None else f"{r['range']:.2f}"
        mean = "n/a" if r["mean"] is None else f"{r['mean']:.2f}"
        if not r["complete"]:
            status = yellow("INCOMPLETE")
        else:
            status = red("UNSTABLE") if r["unstable"] else green("stable")
        print(f" {r['case_id']:<8}{r['metric']:<28}{scores:<22}{mean:>7}{rng:>8}"
              f"{('yes' if r['verdict_flip'] else 'no'):>6}  {status}")
    unstable = sum(r["unstable"] and r["complete"] for r in rows)
    incomplete = sum(not r["complete"] for r in rows)
    print(" " + "-" * 86)
    print(f" Unstable metric/case pairs: {unstable}/{len(rows)} "
          f"(flip in verdict or score range > {UNSTABLE_RANGE})")
    print(f" Incomplete metric/case pairs: {incomplete}/{len(rows)}")
    print(LINE)


# =============================================================================
# 5. JSON persistence (reports/ folder)
# =============================================================================

def _write_json(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_run_report(run: RunResult, reports_dir: Path) -> Path:
    mode = run.config.get("dataset_mode", "unknown")
    return _write_json(run.to_dict(),
                       reports_dir / f"eval_{run.run_id}_{mode}_run{run.run_index}.json")


def save_variability_report(rows: List[Dict[str, Any]], runs: List[RunResult],
                            reports_dir: Path) -> Path:
    data = {
        "run_ids": [r.run_id for r in runs],
        "num_runs": len(runs),
        "config": runs[0].config if runs else {},
        "unstable_threshold": UNSTABLE_RANGE,
        "rows": rows,
    }
    return _write_json(data, reports_dir / f"variability_{runs[0].run_id}_{len(runs)}runs.json")
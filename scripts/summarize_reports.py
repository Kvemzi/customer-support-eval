#!/usr/bin/env python
"""
summarize_reports.py
====================
Print Markdown tables from the newest reports in reports/.

    python scripts/summarize_reports.py                 # newest batch
    python scripts/summarize_reports.py --reports-dir reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _newest(files: List[Path]) -> List[Path]:
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def per_run_table(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        f"**Run {report['run_index']}** "
        f"(mode: `{cfg.get('dataset_mode')}`, judge: `{cfg.get('judge_model')}`, "
        f"judge cost: ${report['summary']['total_cost_usd']})",
        "",
        "| Case | Metric | Score | Pass condition | Verdict | As expected |",
        "|---|---|---:|---|---|---|",
    ]
    for case in report["cases"]:
        for m in case["metrics"]:
            op = "<=" if m["lower_is_better"] else ">="
            score = "n/a" if m["score"] is None else f"{m['score']:.2f}"
            verdict = "ERROR" if m["error"] else ("PASS" if m["success"] else "FAIL")
            lines.append(f"| {case['case_id']} | {m['name']} | {score} | "
                         f"{op} {m['threshold']} | {verdict} | "
                         f"{'yes' if m['as_expected'] else '**NO**'} |")
    return "\n".join(lines)


def variability_table(data: Dict[str, Any]) -> str:
    lines = [
        f"**Variability across {data['num_runs']} runs** "
        f"(unstable = verdict flip or score range > {data['unstable_threshold']})",
        "",
        "| Case | Metric | Scores per run | Mean | Std dev | Range | Verdict flip | Status |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in data["rows"]:
        scores = ", ".join(f"{s:.2f}" for s in r["scores"]) or "n/a"
        mean = "n/a" if r["mean"] is None else f"{r['mean']:.2f}"
        rng = "n/a" if r["range"] is None else f"{r['range']:.2f}"
        lines.append(f"| {r['case_id']} | {r['metric']} | {scores} | {mean} | "
                     f"{r['stdev']:.2f} | {rng} | {'yes' if r['verdict_flip'] else 'no'} | "
                     f"{'UNSTABLE' if r['unstable'] else 'stable'} |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="reports")
    args = p.parse_args()

    root = Path(args.reports_dir)
    variability = _newest(list(root.glob("variability_*.json")))
    if variability:
        data = _load(variability[0])
        batch = data["run_ids"][0]
        print(f"<!-- batch: {batch} -->\n")
        for run_file in sorted(root.glob(f"eval_{batch}_*_run*.json")):
            print(per_run_table(_load(run_file)) + "\n")
        print(variability_table(data))
        return 0

    runs = _newest(list(root.glob("eval_*_run*.json")))
    if not runs:
        print(f"No reports found in {root}/. Run scripts/run_evaluation.py first.",
              file=sys.stderr)
        return 1
    print(per_run_table(_load(runs[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
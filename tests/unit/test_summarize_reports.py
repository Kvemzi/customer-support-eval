"""Tests for rendering archived evaluation reports."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from summarize_reports import main, per_run_table, variability_table


def test_archived_metric_direction_uses_current_registry():
    report = {
        "run_index": 1,
        "config": {"dataset_mode": "static", "judge_model": "gpt-4o-mini"},
        "summary": {"total_cost_usd": 0.001},
        "cases": [
            {
                "case_id": "TC-09",
                "metrics": [
                    {
                        "key": "hallucination",
                        "name": "Hallucination",
                        "score": 1.0,
                        "threshold": 0.5,
                        "lower_is_better": True,
                        "success": True,
                        "as_expected": True,
                        "error": None,
                    }
                ],
            }
        ],
    }

    assert "| TC-09 | Hallucination | 1.00 | >= 0.5 | PASS | yes |" in per_run_table(report)


def test_incomplete_variability_is_not_reported_as_stable():
    report = {
        "num_runs": 2, "unstable_threshold": 0.1,
        "rows": [{
            "case_id": "TC-01", "metric": "Hallucination", "scores": [],
            "mean": None, "stdev": 0.0, "range": None, "verdict_flip": False,
            "unstable": False, "complete": False,
        }],
    }
    assert "INCOMPLETE" in variability_table(report)


def test_newer_single_run_takes_precedence_over_old_variability(tmp_path, monkeypatch, capsys):
    report = {
        "run_id": "new", "run_index": 1,
        "config": {"dataset_mode": "static", "judge_model": "new-judge"},
        "summary": {"total_cost_usd": 0}, "cases": [],
    }
    variability = {
        "run_ids": ["old"], "num_runs": 2, "unstable_threshold": 0.1, "rows": [],
    }
    old = tmp_path / "variability_old_2runs.json"
    new = tmp_path / "eval_new_static_run1.json"
    old.write_text(json.dumps(variability), encoding="utf-8")
    new.write_text(json.dumps(report), encoding="utf-8")
    os.utime(old, (10, 10))
    os.utime(new, (20, 20))
    monkeypatch.setattr(sys, "argv", ["summarize_reports.py", "--reports-dir", str(tmp_path)])
    assert main() == 0
    output = capsys.readouterr().out
    assert "new-judge" in output
    assert "batch: old" not in output


def test_latest_batch_includes_matching_variability_even_if_run_written_later(
    tmp_path, monkeypatch, capsys
):
    report = {
        "run_id": "new", "run_index": 1,
        "config": {"dataset_mode": "static", "judge_model": "judge"},
        "summary": {"total_cost_usd": 0}, "cases": [],
    }
    variability = {
        "run_ids": ["new"], "num_runs": 2, "unstable_threshold": 0.1, "rows": [],
    }
    summary = tmp_path / "variability_new_2runs.json"
    run = tmp_path / "eval_new_static_run1.json"
    summary.write_text(json.dumps(variability), encoding="utf-8")
    run.write_text(json.dumps(report), encoding="utf-8")
    os.utime(summary, (10, 10))
    os.utime(run, (20, 20))
    monkeypatch.setattr(sys, "argv", ["summarize_reports.py", "--reports-dir", str(tmp_path)])
    assert main() == 0
    assert "batch: new" in capsys.readouterr().out

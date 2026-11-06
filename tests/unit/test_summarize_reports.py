"""Tests for rendering archived evaluation reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from summarize_reports import per_run_table


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

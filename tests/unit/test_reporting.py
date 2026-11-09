from agent_eval.reporting import CaseResult, MetricResult, RunResult, compute_variability


def make_run(index, score=None, success=None, error=None, include_metric=True):
    metric = MetricResult(
        key="hallucination", name="Hallucination", threshold=0.5,
        lower_is_better=False, expected_pass=True, score=score,
        success=success, error=error,
    )
    case = CaseResult(
        case_id="TC-01", name="case", category="test", test_type="single_turn",
        expected_verdict="pass", output_source="static", input="", actual_output="",
        metrics=[metric] if include_metric else [],
    )
    return RunResult(run_id="batch", run_index=index, started_at="", config={}, cases=[case])


def test_failed_measurements_are_incomplete_not_stable(capsys):
    from agent_eval.reporting import print_variability_report

    runs = [make_run(1, error="judge error"), make_run(2, error="judge error")]
    rows = compute_variability(runs)
    assert rows[0]["complete"] is False
    assert rows[0]["valid_runs"] == 0
    print_variability_report(rows, 2)
    assert "INCOMPLETE" in capsys.readouterr().out


def test_missing_metric_is_incomplete():
    rows = compute_variability([
        make_run(1, score=0.8, success=True), make_run(2, include_metric=False)
    ])
    assert rows[0]["complete"] is False
    assert rows[0]["valid_runs"] == 1


def test_complete_runs_report_verdict_flips():
    rows = compute_variability([
        make_run(1, score=0.8, success=True), make_run(2, score=0.4, success=False)
    ])
    assert rows[0]["complete"] is True
    assert rows[0]["verdict_flip"] is True
    assert rows[0]["unstable"] is True

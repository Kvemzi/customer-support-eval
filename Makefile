.PHONY: install chat test-unit eval test-llm test-llm-twice lint

install:            ## Install dependencies + project in editable mode
	pip install -r requirements.txt

chat:               ## Start the interactive support agent
	python -m support_agent.cli

test-unit:          ## Fast, deterministic tests (no LLM calls, no cost)
	pytest tests/unit -q

eval:               ## Static dataset, single run (isolates judge behaviour)
	python scripts/run_evaluation.py

eval-live:          ## Evaluate the live SUT
	python scripts/run_evaluation.py --mode live

eval-twice:         ## Double run + stochastic variability report
	python scripts/run_evaluation.py --mode live --runs 2

test-llm:           ## LLM-as-a-judge test suite via DeepEval CLI (Prompt 4)
	deepeval test run tests/evals/test_suite.py

test-llm-twice:     ## Two consecutive runs for the stochastic variability analysis
	deepeval test run tests/evals/test_suite.py && deepeval test run tests/evals/test_suite.py

lint:
	ruff check src tests scripts

summary:            ## Markdown tables from the latest JSON reports (for the README)
	python scripts/summarize_reports.py

test-fast:          ## Everything that does NOT call an LLM
	pytest -m "not llm" -q
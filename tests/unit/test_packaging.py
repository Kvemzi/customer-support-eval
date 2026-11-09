import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_source_tree_loads_policy_without_installed_policy_package(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys, importlib.resources; sys.path.insert(0, sys.argv[1]); "
        "from unittest.mock import patch; "
        "p = patch('importlib.resources.files', side_effect=ModuleNotFoundError); "
        "p.start(); "
        "from support_agent.config import DEFAULT_KB_PATH; "
        "assert DEFAULT_KB_PATH.is_file()"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(root / "src")], cwd=tmp_path,
        check=True, capture_output=True, text=True,
    )


def test_wheel_contains_and_loads_default_policy(tmp_path):
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(root / name, source / name)
    for name in ("src", "data"):
        shutil.copytree(
            root / name, source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
            "--no-index", "--wheel-dir", str(wheels), str(source),
        ],
        check=True, capture_output=True, text=True,
    )
    wheel = next(wheels.glob("*.whl"))
    target = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
         "--target", str(target), str(wheel)],
        check=True, capture_output=True, text=True,
    )
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from support_agent.config import DEFAULT_KB_PATH; "
        "from support_agent.knowledge_base import KnowledgeBase; "
        "assert DEFAULT_KB_PATH.is_relative_to(Path(sys.argv[1])); "
        "assert '30 days' in KnowledgeBase.from_json().get('returns')"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(target)], cwd=tmp_path,
        check=True, capture_output=True, text=True,
    )

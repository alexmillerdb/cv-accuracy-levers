import os
from pathlib import Path
import subprocess
import sys


def test_tiny_sample_script_runs_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "scripts/run_tiny_sample.py"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "tiny_sample_metrics" in result.stdout
    assert "recall_defective=" in result.stdout


def test_tiny_sample_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/run_tiny_sample.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_tiny_sample_script_main_exec_does_not_raise_system_exit_zero():
    script_path = Path("scripts/run_tiny_sample.py")
    namespace = {"__name__": "__main__"}
    original_argv = sys.argv

    try:
        sys.argv = [str(script_path)]
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
    finally:
        sys.argv = original_argv

    assert namespace["exit_code"] == 0

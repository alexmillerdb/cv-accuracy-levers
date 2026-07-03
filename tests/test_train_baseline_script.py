import os
from pathlib import Path
import subprocess
import sys


def test_train_baseline_script_runs_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline.py",
            "--sample-mode",
            "true",
            "--runtime",
            "local_cpu",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "baseline_metrics" in result.stdout
    assert "recall_defective=" in result.stdout
    assert "auc_pr=" in result.stdout
    assert "baseline_params" in result.stdout
    assert "cv_runtime=local_cpu" in result.stdout


def test_train_baseline_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/train_baseline.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_train_baseline_script_main_exec_does_not_raise_system_exit_zero():
    script_path = Path("scripts/train_baseline.py")
    namespace = {"__name__": "__main__"}
    original_argv = sys.argv

    try:
        sys.argv = [str(script_path), "--sample-mode", "true"]
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
    finally:
        sys.argv = original_argv

    assert namespace["exit_code"] == 0

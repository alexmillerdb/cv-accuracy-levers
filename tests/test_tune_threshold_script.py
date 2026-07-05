import os
from pathlib import Path
import subprocess
import sys


def test_tune_threshold_script_runs_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tune_threshold.py",
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

    assert "threshold_tuning_metrics" in result.stdout
    assert "tuned.recall_defective=" in result.stdout
    assert "fixed_0_5.recall_defective=" in result.stdout
    assert "delta.recall_defective=" in result.stdout
    assert "threshold_tuning_params" in result.stdout
    assert "cv_runtime=local_cpu" in result.stdout
    assert "lever_name=threshold_tuning" in result.stdout


def test_tune_threshold_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/tune_threshold.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_tune_threshold_script_main_exec_does_not_raise_system_exit_zero():
    script_path = Path("scripts/tune_threshold.py")
    namespace = {"__name__": "__main__"}
    original_argv = sys.argv

    try:
        sys.argv = [str(script_path), "--sample-mode", "true"]
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
    finally:
        sys.argv = original_argv

    assert namespace["exit_code"] == 0

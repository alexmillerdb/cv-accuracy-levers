import os
from pathlib import Path
import subprocess
import sys


def test_review_false_negatives_script_runs_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/review_false_negatives.py",
            "--sample-mode",
            "true",
            "--runtime",
            "local_cpu",
            "--review-threshold",
            "0.95",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "false_negative_review_metrics" in result.stdout
    assert "review.false_negatives=" in result.stdout
    assert "review.total_false_negatives=1.0" in result.stdout
    assert "false_negative_review_params" in result.stdout
    assert "cv_runtime=local_cpu" in result.stdout
    assert "lever_name=false_negative_review" in result.stdout
    assert "false_negative_review_rows" in result.stdout
    assert "synthetic/group_09/view_00.jpg" in result.stdout


def test_review_false_negatives_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/review_false_negatives.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_review_false_negatives_script_main_exec_does_not_raise_system_exit_zero():
    script_path = Path("scripts/review_false_negatives.py")
    namespace = {"__name__": "__main__"}
    original_argv = sys.argv

    try:
        sys.argv = [str(script_path), "--sample-mode", "true"]
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
    finally:
        sys.argv = original_argv

    assert namespace["exit_code"] == 0

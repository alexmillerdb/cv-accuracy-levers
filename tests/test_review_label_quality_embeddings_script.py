import os
from pathlib import Path
import subprocess
import sys


def test_review_label_quality_embeddings_script_runs_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/review_label_quality_embeddings.py",
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

    assert "label_quality_metrics" in result.stdout
    assert "review.reviewed_issue_count=" in result.stdout
    assert "label_quality_params" in result.stdout
    assert "cv_runtime=local_cpu" in result.stdout
    assert "lever_name=label_quality_embeddings" in result.stdout
    assert "embedding_source=sample_baseline_synthetic_features_v1" in result.stdout
    assert "label_quality_review_rows" in result.stdout


def test_review_label_quality_embeddings_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/review_label_quality_embeddings.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_review_label_quality_embeddings_script_main_exec_does_not_raise_system_exit_zero():
    script_path = Path("scripts/review_label_quality_embeddings.py")
    namespace = {"__name__": "__main__"}
    original_argv = sys.argv

    try:
        sys.argv = [str(script_path), "--sample-mode", "true"]
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
    finally:
        sys.argv = original_argv

    assert namespace["exit_code"] == 0


def test_review_label_quality_embeddings_synthetic_issue_produces_review_rows():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/review_label_quality_embeddings.py",
            "--sample-mode",
            "true",
            "--runtime",
            "local_cpu",
            "--inject-synthetic-label-issue",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "review.reviewed_issue_count=0.0" not in result.stdout
    assert "issue_row=" in result.stdout
    assert "synthetic/injected_label_issue/view_00.jpg" in result.stdout
    assert "synthetic_issue:true" in result.stdout

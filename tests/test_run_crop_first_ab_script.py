import os
from pathlib import Path
import subprocess
import sys

from levers.config import ProjectConfig
from levers.crop_first import run_sample_crop_first_ab


def test_run_crop_first_ab_script_runs_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_crop_first_ab.py",
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

    assert "crop_first_ab_metrics" in result.stdout
    assert "baseline.recall_defective=" in result.stdout
    assert "crop_first.recall_defective=" in result.stdout
    assert "delta.recall_defective=" in result.stdout
    assert "crop_first_ab_params" in result.stdout
    assert "cv_runtime=local_cpu" in result.stdout
    assert "lever_name=crop_first_ab" in result.stdout
    assert "crop_feature_source=sample_crop_region_emphasis_features_v1" in result.stdout
    assert "crop_first_ab_leaderboard" in result.stdout


def test_run_crop_first_ab_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/run_crop_first_ab.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_run_crop_first_ab_script_main_exec_does_not_raise_system_exit_zero():
    script_path = Path("scripts/run_crop_first_ab.py")
    namespace = {"__name__": "__main__"}
    original_argv = sys.argv

    try:
        sys.argv = [str(script_path), "--sample-mode", "true"]
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
    finally:
        sys.argv = original_argv

    assert namespace["exit_code"] == 0


def test_run_crop_first_ab_mlflow_payload_shape_contains_runtime_context():
    script_path = Path("scripts/run_crop_first_ab.py")
    namespace = {"__name__": "payload_shape_test"}
    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    config = ProjectConfig.from_env()
    result = run_sample_crop_first_ab()
    metrics = namespace["build_metric_payload"](result)
    params = namespace["build_runtime_params"](
        config=config,
        result=result,
        runtime="databricks_serverless_cpu",
    )
    leaderboard = result.leaderboard_row_payload()

    assert params["runtime"] == "databricks_serverless_cpu"
    assert params["cv_runtime"] == "databricks_serverless_cpu"
    assert params["sample_mode"] == "true"
    assert params["catalog"] == config.catalog
    assert params["schema"] == config.schema
    assert params["volume"] == config.volume
    assert params["lever_name"] == "crop_first_ab"
    assert "baseline_selected_threshold" in params
    assert "crop_first_selected_threshold" in params
    assert params["leaderboard_artifact"] == "leaderboard_row.json"
    assert "baseline.recall_defective" in metrics
    assert "crop_first.recall_defective" in metrics
    assert "delta.false_negatives" in metrics
    assert leaderboard["lever_name"] == "crop_first_ab"
    assert leaderboard["baseline_score_source"] == "baseline_whole_image_sample"
    assert leaderboard["crop_first_score_source"] == "crop_first_sample"

"""Run a sample-mode crop-first A/B comparison."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import sys
from typing import Sequence


def _script_path() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve()
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Cannot resolve script path")
    return Path(frame.f_code.co_filename).resolve()


REPO_ROOT = _script_path().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from levers.config import ProjectConfig
from levers.crop_first import CropFirstABResult, run_sample_crop_first_ab
from levers.mlflow_utils import flatten_metrics


def _bool_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ProjectConfig.from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-mode", type=_bool_text, default=os.getenv("SAMPLE_MODE", "true"))
    parser.add_argument("--runtime", default=os.getenv("CV_RUNTIME", "local_cpu"))
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--feature-seed", type=int, default=17)
    parser.add_argument("--crop-emphasis", type=float, default=1.35)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="crop_first_ab_sample")
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args(argv)


def build_metric_payload(result: CropFirstABResult) -> dict[str, float]:
    return flatten_metrics(result.metric_payload())


def build_runtime_params(
    *,
    config: ProjectConfig,
    result: CropFirstABResult,
    runtime: str,
) -> dict[str, str | int | float]:
    params = result.param_payload()
    params.update(
        {
            "runtime": runtime,
            "cv_runtime": runtime,
            "catalog": config.catalog,
            "schema": config.schema,
            "volume": config.volume,
            "volume_subpath": config.volume_subpath,
            "baseline_selected_threshold": (
                result.baseline.selected_point.threshold
            ),
            "crop_first_selected_threshold": (
                result.crop_first.selected_point.threshold
            ),
            "leaderboard_artifact": "leaderboard_row.json",
        }
    )
    return params


def _set_mlflow_experiment(
    *,
    config: ProjectConfig,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
) -> None:
    import mlflow

    resolved_tracking_uri = tracking_uri or config.mlflow_tracking_uri
    if resolved_tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_registry_uri(config.mlflow_registry_uri)

    if resolved_tracking_uri.startswith("file:") and not experiment_id:
        mlflow.set_experiment(experiment_name or "cv-accuracy-levers-crop-first")
        return

    resolved_experiment_id = experiment_id or config.mlflow_experiment_id
    resolved_experiment_name = experiment_name or config.mlflow_experiment_name
    if resolved_experiment_id:
        mlflow.set_experiment(experiment_id=resolved_experiment_id)
    elif resolved_experiment_name:
        mlflow.set_experiment(resolved_experiment_name)
    else:
        config.require_mlflow_experiment()


def maybe_log_mlflow(
    *,
    args: argparse.Namespace,
    config: ProjectConfig,
    result: CropFirstABResult,
    metrics: dict[str, float],
    params: dict[str, str | int | float],
) -> None:
    if not args.log_mlflow:
        return

    import mlflow

    _set_mlflow_experiment(
        config=config,
        tracking_uri=args.tracking_uri,
        experiment_id=args.experiment_id,
        experiment_name=args.experiment_name,
    )
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            result.comparison_rows_payload(),
            "crop_first_comparison_rows.json",
        )
        mlflow.log_dict(
            result.review_rows_payload(),
            "crop_first_review_rows.json",
        )
        mlflow.log_dict(result.summary_payload(), "crop_first_summary.json")
        mlflow.log_dict(result.validation_sweeps_payload(), "validation_sweeps.json")
        mlflow.log_dict(result.leaderboard_row_payload(), "leaderboard_row.json")
        mlflow.log_dict(result.predictions_payload(), "predictions.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProjectConfig.from_env()
    result = run_sample_crop_first_ab(
        sample_mode=args.sample_mode,
        split_seed=args.split_seed,
        feature_seed=args.feature_seed,
        min_recall=args.min_recall,
        crop_emphasis=args.crop_emphasis,
    )
    metrics = build_metric_payload(result)
    params = build_runtime_params(
        config=config,
        result=result,
        runtime=args.runtime,
    )
    maybe_log_mlflow(
        args=args,
        config=config,
        result=result,
        metrics=metrics,
        params=params,
    )

    print("crop_first_ab_metrics")
    for key, value in metrics.items():
        print(f"{key}={value}")
    print("crop_first_ab_params")
    for key, value in params.items():
        print(f"{key}={value}")
    print("crop_first_ab_leaderboard")
    for key, value in result.leaderboard_row_payload().items():
        print(f"{key}={value}")
    print("crop_first_ab_review_rows")
    if not result.changed_rows:
        print("review_rows=0")
    for row in result.changed_rows:
        print(
            "review_row="
            f"rank:{row.rank},"
            f"change_type:{row.change_type},"
            f"image_path:{row.image_path},"
            f"group_id:{row.group_id},"
            f"label:{row.label},"
            f"baseline_score:{row.baseline_score},"
            f"crop_first_score:{row.crop_first_score},"
            f"recall_impact:{row.recall_impact}"
        )
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

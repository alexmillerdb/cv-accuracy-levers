"""Create a false-negative review grid from sample baseline predictions."""

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
from levers.error_review import (
    FalseNegativeReviewResult,
    run_sample_false_negative_review,
)
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
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
    parser.add_argument("--review-threshold", type=float, default=None)
    parser.add_argument("--review-split", default="test")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--borderline-margin", type=float, default=0.05)
    parser.add_argument("--high-confidence-margin", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--feature-seed", type=int, default=17)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="false_negative_review_sample")
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args(argv)


def build_metric_payload(result: FalseNegativeReviewResult) -> dict[str, float]:
    return flatten_metrics(result.metric_payload(), prefix="review")


def build_runtime_params(
    *,
    config: ProjectConfig,
    result: FalseNegativeReviewResult,
    runtime: str,
    sample_mode: bool,
    min_recall: float,
    baseline_threshold: float,
    split_seed: int,
    feature_seed: int,
    top_k: int,
) -> dict[str, str | int | float]:
    return {
        "runtime": runtime,
        "cv_runtime": runtime,
        "sample_mode": str(sample_mode).lower(),
        "catalog": config.catalog,
        "schema": config.schema,
        "volume": config.volume,
        "volume_subpath": config.volume_subpath,
        "lever_name": "false_negative_review",
        "sample_size": result.sample_size,
        "group_count": result.group_count,
        "train_size": result.train_size,
        "val_size": result.val_size,
        "test_size": result.test_size,
        "min_recall": min_recall,
        "baseline_threshold": baseline_threshold,
        "selected_threshold": result.selected_threshold,
        "review_threshold": result.review_threshold,
        "review_split": result.review_split,
        "review_top_k": top_k,
        "split_seed": split_seed,
        "feature_seed": feature_seed,
        "threshold_source": "validation_or_review_override",
        "score_source": "baseline_whole_image_sample",
        "model_family": "centroid_sample_baseline",
    }


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
        mlflow.set_experiment(experiment_name or "cv-accuracy-levers-error-review")
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
    result: FalseNegativeReviewResult,
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
        mlflow.log_dict(result.review_rows_payload(), "false_negative_review_rows.json")
        mlflow.log_dict(result.summary_payload(), "false_negative_review_summary.json")
        mlflow.log_dict(result.leaderboard_row_payload(), "leaderboard_row.json")
        mlflow.log_dict(
            [prediction.__dict__ for prediction in result.predictions],
            "predictions.json",
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProjectConfig.from_env()
    result = run_sample_false_negative_review(
        sample_mode=args.sample_mode,
        split_seed=args.split_seed,
        feature_seed=args.feature_seed,
        min_recall=args.min_recall,
        baseline_threshold=args.baseline_threshold,
        review_threshold=args.review_threshold,
        review_split=args.review_split,
        top_k=args.top_k,
        borderline_margin=args.borderline_margin,
        high_confidence_margin=args.high_confidence_margin,
    )
    metrics = build_metric_payload(result)
    params = build_runtime_params(
        config=config,
        result=result,
        runtime=args.runtime,
        sample_mode=args.sample_mode,
        min_recall=args.min_recall,
        baseline_threshold=args.baseline_threshold,
        split_seed=args.split_seed,
        feature_seed=args.feature_seed,
        top_k=args.top_k,
    )
    maybe_log_mlflow(
        args=args,
        config=config,
        result=result,
        metrics=metrics,
        params=params,
    )

    print("false_negative_review_metrics")
    for key, value in metrics.items():
        print(f"{key}={value}")
    print("false_negative_review_params")
    for key, value in params.items():
        print(f"{key}={value}")
    print("false_negative_review_rows")
    if not result.review_rows:
        print("review_rows=0")
    for row in result.review_rows:
        print(
            "review_row="
            f"rank:{row.rank},"
            f"image_path:{row.image_path},"
            f"group_id:{row.group_id},"
            f"score:{row.score:.6f},"
            f"threshold:{row.threshold:.6f},"
            f"margin:{row.margin:.6f},"
            f"bucket:{row.review_bucket}"
        )
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

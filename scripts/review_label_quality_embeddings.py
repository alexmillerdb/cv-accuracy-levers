"""Create label-quality embedding review artifacts from sample features."""

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
from levers.label_quality import (
    LabelQualityResult,
    run_sample_label_quality_embeddings,
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
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--feature-seed", type=int, default=17)
    parser.add_argument("--label-conflict-similarity-threshold", type=float, default=0.98)
    parser.add_argument("--cross-split-similarity-threshold", type=float, default=0.999)
    parser.add_argument("--neighbor-count", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--inject-synthetic-label-issue", action="store_true")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="label_quality_embeddings_sample")
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args(argv)


def build_metric_payload(result: LabelQualityResult) -> dict[str, float]:
    return flatten_metrics(result.metric_payload(), prefix="review")


def build_runtime_params(
    *,
    config: ProjectConfig,
    result: LabelQualityResult,
    runtime: str,
    sample_mode: bool,
    min_recall: float,
    split_seed: int,
    feature_seed: int,
) -> dict[str, str | int | float]:
    return {
        "runtime": runtime,
        "cv_runtime": runtime,
        "sample_mode": str(sample_mode).lower(),
        "catalog": config.catalog,
        "schema": config.schema,
        "volume": config.volume,
        "volume_subpath": config.volume_subpath,
        "lever_name": "label_quality_embeddings",
        "sample_size": result.sample_size,
        "group_count": result.group_count,
        "train_size": result.train_size,
        "val_size": result.val_size,
        "test_size": result.test_size,
        "min_recall": min_recall,
        "selected_threshold": result.selected_threshold,
        "split_seed": split_seed,
        "feature_seed": feature_seed,
        "embedding_source": result.embedding_source,
        "label_conflict_similarity_threshold": (
            result.label_conflict_similarity_threshold
        ),
        "cross_split_similarity_threshold": result.cross_split_similarity_threshold,
        "neighbor_count": result.neighbor_count,
        "review_top_k": -1 if result.top_k is None else result.top_k,
        "inject_synthetic_label_issue": str(result.synthetic_issue_injected).lower(),
        "threshold_source": "validation",
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
        mlflow.set_experiment(experiment_name or "cv-accuracy-levers-label-quality")
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
    result: LabelQualityResult,
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
        mlflow.log_dict(result.review_rows_payload(), "label_quality_review_rows.json")
        mlflow.log_dict(result.summary_payload(), "label_quality_summary.json")
        mlflow.log_dict(result.neighbor_rows_payload(), "embedding_neighbors.json")
        mlflow.log_dict(result.leaderboard_row_payload(), "leaderboard_row.json")
        mlflow.log_dict(result.predictions_payload(), "predictions.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProjectConfig.from_env()
    result = run_sample_label_quality_embeddings(
        sample_mode=args.sample_mode,
        split_seed=args.split_seed,
        feature_seed=args.feature_seed,
        min_recall=args.min_recall,
        label_conflict_similarity_threshold=args.label_conflict_similarity_threshold,
        cross_split_similarity_threshold=args.cross_split_similarity_threshold,
        neighbor_count=args.neighbor_count,
        top_k=args.top_k,
        inject_synthetic_label_issue=args.inject_synthetic_label_issue,
    )
    metrics = build_metric_payload(result)
    params = build_runtime_params(
        config=config,
        result=result,
        runtime=args.runtime,
        sample_mode=args.sample_mode,
        min_recall=args.min_recall,
        split_seed=args.split_seed,
        feature_seed=args.feature_seed,
    )
    maybe_log_mlflow(
        args=args,
        config=config,
        result=result,
        metrics=metrics,
        params=params,
    )

    print("label_quality_metrics")
    for key, value in metrics.items():
        print(f"{key}={value}")
    print("label_quality_params")
    for key, value in params.items():
        print(f"{key}={value}")
    print("label_quality_review_rows")
    if not result.issue_rows:
        print("review_rows=0")
    for row in result.issue_rows:
        print(
            "issue_row="
            f"rank:{row.rank},"
            f"type:{row.issue_type},"
            f"image_path:{row.image_path},"
            f"label:{row.label},"
            f"predicted_label:{row.predicted_label},"
            f"neighbor_image_path:{row.neighbor_image_path},"
            f"similarity:{row.similarity},"
            f"synthetic_issue:{str(row.synthetic_issue).lower()}"
        )
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

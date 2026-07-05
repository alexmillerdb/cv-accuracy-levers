"""Tune a recall-first threshold over sample baseline predictions."""

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
from levers.mlflow_utils import flatten_metrics
from levers.threshold_tuning import ThresholdTuningResult, run_sample_threshold_tuning


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
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--feature-seed", type=int, default=17)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="threshold_tuning_sample")
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args(argv)


def build_metric_payload(result: ThresholdTuningResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(flatten_metrics(result.tuned_metric_payload(), prefix="tuned"))
    metrics.update(flatten_metrics(result.validation_metric_payload(), prefix="val"))
    metrics.update(flatten_metrics(result.fixed_metric_payload(), prefix="fixed_0_5"))
    metrics.update(flatten_metrics(result.delta_metric_payload(), prefix="delta"))
    metrics["selected_threshold"] = float(result.selected_point.threshold)
    return metrics


def build_runtime_params(
    *,
    config: ProjectConfig,
    result: ThresholdTuningResult,
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
        "lever_name": "threshold_tuning",
        "sample_size": result.sample_size,
        "group_count": result.group_count,
        "train_size": result.train_size,
        "val_size": result.val_size,
        "test_size": result.test_size,
        "min_recall": min_recall,
        "baseline_threshold": result.baseline_threshold,
        "split_seed": split_seed,
        "feature_seed": feature_seed,
        "threshold_source": "validation",
        "threshold_count": len(result.validation_sweep),
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
    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_registry_uri(config.mlflow_registry_uri)

    if resolved_tracking_uri.startswith("file:") and not experiment_id:
        mlflow.set_experiment(experiment_name or "cv-accuracy-levers-threshold")
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
    result: ThresholdTuningResult,
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
        mlflow.log_dict(result.validation_sweep_payload(), "threshold_sweep.json")
        mlflow.log_dict(
            {
                "selected_threshold": result.selected_point.threshold,
                "baseline_threshold": result.baseline_threshold,
                "tuned_test_metrics": result.tuned_metric_payload(),
                "fixed_0_5_test_metrics": result.fixed_metric_payload(),
                "delta_metrics": result.delta_metric_payload(),
            },
            "threshold_tuning_summary.json",
        )
        mlflow.log_dict(
            [prediction.__dict__ for prediction in result.predictions],
            "predictions.json",
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProjectConfig.from_env()
    result = run_sample_threshold_tuning(
        sample_mode=args.sample_mode,
        split_seed=args.split_seed,
        feature_seed=args.feature_seed,
        min_recall=args.min_recall,
        baseline_threshold=args.baseline_threshold,
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

    print("threshold_tuning_metrics")
    for key, value in metrics.items():
        print(f"{key}={value}")
    print("threshold_tuning_params")
    for key, value in params.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

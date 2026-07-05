"""Train a manifest-backed image baseline for the AIR CLI GPU path."""

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
from levers.gpu_baseline import (
    DEFAULT_BACKBONE,
    GpuBaselineRunResult,
    GpuTrainingConfig,
    artifact_payloads,
    run_gpu_baseline,
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
    parser.add_argument(
        "--manifest-path",
        default=os.getenv("CV_DATA_MANIFEST"),
        help="Path to a CSV, JSON, or JSONL public image manifest.",
    )
    parser.add_argument("--data-dir", default=os.getenv("CV_DATA_DIR"))
    parser.add_argument(
        "--sample-mode",
        type=_bool_text,
        default=os.getenv("SAMPLE_MODE", "true"),
    )
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--runtime", default=os.getenv("CV_RUNTIME", "local_cpu"))
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backbone",
        choices=("tiny_cnn", "resnet18"),
        default=DEFAULT_BACKBONE,
    )
    parser.add_argument("--allow-full-data", action="store_true")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="gpu_baseline_real_image_sample")
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args(argv)


def build_metric_payload(result: GpuBaselineRunResult) -> dict[str, float]:
    metrics = flatten_metrics(result.metric_payload())
    metrics.update(flatten_metrics(result.validation_metric_payload(), prefix="val"))
    return metrics


def build_runtime_params(
    *,
    config: ProjectConfig,
    result: GpuBaselineRunResult,
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
            "data_source": config.data_source,
            "data_manifest_configured": str(bool(config.data_manifest)).lower(),
            "data_dir_configured": str(bool(config.data_dir)).lower(),
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
        mlflow.set_experiment(experiment_name or "cv-accuracy-levers-gpu-baseline")
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
    metrics: dict[str, float],
    params: dict[str, str | int | float],
    result: GpuBaselineRunResult,
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
        for artifact_path, payload in artifact_payloads(result).items():
            mlflow.log_dict(payload, artifact_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.manifest_path:
        raise ValueError("Set CV_DATA_MANIFEST or pass --manifest-path.")

    config = ProjectConfig.from_env()
    result = run_gpu_baseline(
        GpuTrainingConfig(
            manifest_path=args.manifest_path,
            data_dir=args.data_dir,
            sample_mode=args.sample_mode,
            sample_size=args.sample_size,
            split_seed=args.split_seed,
            min_recall=args.min_recall,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            device=args.device,
            backbone=args.backbone,
            seed=args.seed,
            allow_full_data=args.allow_full_data,
        )
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
        metrics=metrics,
        params=params,
        result=result,
    )

    print("gpu_baseline_metrics")
    for key, value in metrics.items():
        print(f"{key}={value}")
    print("gpu_baseline_params")
    for key, value in params.items():
        print(f"{key}={value}")
    print("gpu_baseline_leaderboard")
    for key, value in result.leaderboard_row_payload().items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

"""Tiny CPU sample for local and Databricks smoke verification."""

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
from levers.data import DatasetRecord, make_grouped_split, validate_group_splits
from levers.eval import binary_classification_metrics
from levers.mlflow_utils import flatten_metrics
from levers.thresholds import choose_threshold_for_recall, threshold_sweep


def build_records() -> list[DatasetRecord]:
    return [
        DatasetRecord("fixture/group_a_0.jpg", 1, "group_a"),
        DatasetRecord("fixture/group_a_1.jpg", 1, "group_a"),
        DatasetRecord("fixture/group_b_0.jpg", 0, "group_b"),
        DatasetRecord("fixture/group_b_1.jpg", 0, "group_b"),
        DatasetRecord("fixture/group_c_0.jpg", 1, "group_c"),
        DatasetRecord("fixture/group_d_0.jpg", 0, "group_d"),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ProjectConfig.from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default=os.environ.get("CV_RUNTIME", "local_cpu"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="cv-accuracy-levers-tiny-sample")
    parser.add_argument("--experiment-id", default=os.environ.get("MLFLOW_EXPERIMENT_ID"))
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args(argv)


def maybe_log_mlflow(
    *,
    args: argparse.Namespace,
    metrics: dict[str, float],
    params: dict[str, str | int | float],
) -> None:
    if not args.log_mlflow:
        return

    import mlflow

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", "databricks-uc"))
    if args.experiment_id:
        mlflow.set_experiment(experiment_id=args.experiment_id)
    else:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="tiny_sample_smoke"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProjectConfig.from_env()
    split_records = make_grouped_split(build_records(), seed=11)
    validate_group_splits(split_records)

    # Deterministic stand-in scores keep this smoke test model-free and CPU-only.
    y_true = [record.label for record in split_records]
    y_score = [0.92, 0.72, 0.35, 0.18, 0.62, 0.41]
    points = threshold_sweep(y_true, y_score, thresholds=[0.3, 0.5, 0.7])
    chosen = choose_threshold_for_recall(points, min_recall=2 / 3)
    metrics = binary_classification_metrics(
        y_true,
        y_score,
        threshold=chosen.threshold,
    )

    payload = flatten_metrics(metrics.as_dict())
    params = {
        "runtime": args.runtime,
        "cv_runtime": args.runtime,
        "sample_mode": str(config.sample_mode).lower(),
        "catalog": config.catalog,
        "schema": config.schema,
        "volume": config.volume,
        "volume_subpath": config.volume_subpath,
        "lever_name": "tiny_sample_smoke",
        "sample_size": len(split_records),
        "group_count": len({record.group_id for record in split_records}),
    }
    maybe_log_mlflow(args=args, metrics=payload, params=params)

    print("tiny_sample_metrics")
    for key, value in payload.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

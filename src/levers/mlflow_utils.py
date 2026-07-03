"""Small helpers that keep MLflow payloads stable across notebooks/scripts."""

from __future__ import annotations

from collections.abc import Mapping

from levers.config import ProjectConfig


def flatten_metrics(
    metrics: Mapping[str, int | float | None],
    *,
    prefix: str | None = None,
) -> dict[str, float]:
    """Return MLflow-safe numeric metrics, dropping missing values."""

    result: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        metric_key = f"{prefix}.{key}" if prefix else key
        result[metric_key] = float(value)
    return result


def configure_mlflow_from_env() -> ProjectConfig:
    """Load .env/environment settings and apply MLflow configuration."""

    config = ProjectConfig.from_env()
    config.apply_mlflow()
    return config

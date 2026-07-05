"""Project configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os


def load_project_env(path: str | None = None) -> None:
    """Load a local .env file when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=path)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _required_env(name: str, default: str | None = None) -> str:
    value = _optional_env(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"Missing required environment variable: {name}")


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime-neutral configuration shared by scripts and notebooks."""

    catalog: str
    schema: str
    volume: str
    volume_subpath: str
    mlflow_tracking_uri: str
    mlflow_registry_uri: str
    mlflow_experiment_id: str | None
    mlflow_experiment_name: str | None
    databricks_config_profile: str | None
    databricks_serverless_compute_id: str | None
    runtime: str
    sample_mode: bool
    data_source: str
    data_manifest: str | None
    data_dir: str | None
    data_uc_table: str

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "ProjectConfig":
        if load_dotenv_file:
            load_project_env()

        return cls(
            catalog=_required_env("CV_CATALOG", "serverless_stable_yau46e_catalog"),
            schema=_required_env("CV_SCHEMA", "cv_accuracy_levers"),
            volume=_required_env("CV_VOLUME", "cv_accuracy_levers"),
            volume_subpath=_required_env("CV_VOLUME_SUBPATH", "artifacts"),
            mlflow_tracking_uri=_required_env("MLFLOW_TRACKING_URI", "databricks"),
            mlflow_registry_uri=_required_env("MLFLOW_REGISTRY_URI", "databricks-uc"),
            mlflow_experiment_id=_optional_env("MLFLOW_EXPERIMENT_ID"),
            mlflow_experiment_name=_optional_env("MLFLOW_EXPERIMENT_NAME"),
            databricks_config_profile=_optional_env("DATABRICKS_CONFIG_PROFILE"),
            databricks_serverless_compute_id=_optional_env(
                "DATABRICKS_SERVERLESS_COMPUTE_ID"
            ),
            runtime=_required_env("CV_RUNTIME", "local_cpu"),
            sample_mode=_required_env("SAMPLE_MODE", "true").lower()
            in {"1", "true", "yes", "y"},
            data_source=_required_env("CV_DATA_SOURCE", "manifest"),
            data_manifest=_optional_env("CV_DATA_MANIFEST"),
            data_dir=_optional_env("CV_DATA_DIR"),
            data_uc_table=_required_env("CV_UC_TABLE", "image_manifest"),
        )

    def uc_table(self, table_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{table_name}"

    def volume_uri(self, *parts: str) -> str:
        normalized = [self.volume_subpath, *parts]
        tail = "/".join(part.strip("/") for part in normalized if part and part.strip("/"))
        base = f"dbfs:/Volumes/{self.catalog}/{self.schema}/{self.volume}"
        return f"{base}/{tail}" if tail else base

    def volume_path(self, *parts: str) -> str:
        normalized = [self.volume_subpath, *parts]
        tail = "/".join(part.strip("/") for part in normalized if part and part.strip("/"))
        base = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"
        return f"{base}/{tail}" if tail else base

    def require_mlflow_experiment(self) -> None:
        if not self.mlflow_experiment_id and not self.mlflow_experiment_name:
            raise ValueError(
                "Set MLFLOW_EXPERIMENT_ID or MLFLOW_EXPERIMENT_NAME before "
                "logging to Databricks MLflow."
            )

    def apply_mlflow(self) -> None:
        """Apply MLflow tracking and experiment settings."""

        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_registry_uri(self.mlflow_registry_uri)

        if self.mlflow_experiment_id:
            mlflow.set_experiment(experiment_id=self.mlflow_experiment_id)
        elif self.mlflow_experiment_name:
            mlflow.set_experiment(self.mlflow_experiment_name)
        else:
            raise ValueError(
                "Set MLFLOW_EXPERIMENT_ID or MLFLOW_EXPERIMENT_NAME before "
                "logging to MLflow."
            )

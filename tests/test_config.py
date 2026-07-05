from levers.config import ProjectConfig


def test_project_config_defaults(monkeypatch):
    for key in (
        "CV_CATALOG",
        "CV_SCHEMA",
        "CV_VOLUME",
        "CV_VOLUME_SUBPATH",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI",
        "MLFLOW_EXPERIMENT_ID",
        "MLFLOW_EXPERIMENT_NAME",
        "DATABRICKS_CONFIG_PROFILE",
        "DATABRICKS_SERVERLESS_COMPUTE_ID",
        "CV_RUNTIME",
        "SAMPLE_MODE",
        "CV_DATA_SOURCE",
        "CV_DATA_MANIFEST",
        "CV_DATA_DIR",
        "CV_UC_TABLE",
    ):
        monkeypatch.delenv(key, raising=False)

    config = ProjectConfig.from_env(load_dotenv_file=False)

    assert config.catalog == "serverless_stable_yau46e_catalog"
    assert config.schema == "cv_accuracy_levers"
    assert config.volume == "cv_accuracy_levers"
    assert config.mlflow_tracking_uri == "databricks"
    assert config.mlflow_registry_uri == "databricks-uc"
    assert config.runtime == "local_cpu"
    assert config.sample_mode is True
    assert config.data_source == "manifest"
    assert config.data_manifest is None
    assert config.data_dir is None
    assert config.data_uc_table == "image_manifest"


def test_project_config_builds_uc_names(monkeypatch):
    monkeypatch.setenv("CV_CATALOG", "dev")
    monkeypatch.setenv("CV_SCHEMA", "vision")
    monkeypatch.setenv("CV_VOLUME", "images")
    monkeypatch.setenv("CV_VOLUME_SUBPATH", "cv-demo")
    monkeypatch.setenv("SAMPLE_MODE", "false")
    monkeypatch.setenv("CV_DATA_SOURCE", "manifest")
    monkeypatch.setenv("CV_DATA_MANIFEST", "data/input.jsonl")
    monkeypatch.setenv("CV_DATA_DIR", "data/images")
    monkeypatch.setenv("CV_UC_TABLE", "metadata")

    config = ProjectConfig.from_env(load_dotenv_file=False)

    assert config.uc_table("metadata") == "dev.vision.metadata"
    assert config.volume_uri("raw", "sample") == (
        "dbfs:/Volumes/dev/vision/images/cv-demo/raw/sample"
    )
    assert config.volume_path("raw", "sample") == (
        "/Volumes/dev/vision/images/cv-demo/raw/sample"
    )
    assert config.sample_mode is False
    assert config.data_source == "manifest"
    assert config.data_manifest == "data/input.jsonl"
    assert config.data_dir == "data/images"
    assert config.data_uc_table == "metadata"


def test_project_config_requires_mlflow_experiment(monkeypatch):
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    config = ProjectConfig.from_env(load_dotenv_file=False)

    try:
        config.require_mlflow_experiment()
    except ValueError as error:
        assert "MLFLOW_EXPERIMENT_ID" in str(error)
    else:
        raise AssertionError("expected missing MLflow experiment to fail")

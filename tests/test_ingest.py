import json

import pytest

from levers.ingest import (
    copy_images_to_uc_volume,
    ensure_uc_objects,
    load_manifest_rows,
    normalize_manifest_row,
    normalize_manifest_rows,
    persist_ingest_to_uc,
    validate_source_license,
    write_manifest_jsonl,
)
from levers.config import ProjectConfig


class _FakeWriter:
    def __init__(self):
        self.calls = []

    def mode(self, value):
        self.calls.append(("mode", value))
        return self

    def format(self, value):
        self.calls.append(("format", value))
        return self

    def saveAsTable(self, value):
        self.calls.append(("saveAsTable", value))
        return None

    def json(self, value):
        self.calls.append(("json", value))
        return None


class _FakeDataFrame:
    def __init__(self, rows):
        self.rows = rows
        self.write = _FakeWriter()
        self.coalesce_calls = []

    def coalesce(self, partitions):
        self.coalesce_calls.append(partitions)
        return self


class _FakeSpark:
    def __init__(self):
        self.sql_calls = []
        self.dataframes = []

    def sql(self, statement):
        self.sql_calls.append(statement)

    def createDataFrame(self, rows):
        dataframe = _FakeDataFrame(rows)
        self.dataframes.append(dataframe)
        return dataframe


class _FakeFilesClient:
    def __init__(self):
        self.created_directories = []
        self.uploads = []

    def create_directory(self, directory_path):
        self.created_directories.append(directory_path)

    def upload(self, file_path, contents, *, overwrite=None):
        self.uploads.append(
            {
                "file_path": file_path,
                "contents": contents.read(),
                "overwrite": overwrite,
            }
        )


def _config() -> ProjectConfig:
    return ProjectConfig(
        catalog="demo_catalog",
        schema="demo_schema",
        volume="demo_volume",
        volume_subpath="cv",
        mlflow_tracking_uri="databricks",
        mlflow_registry_uri="databricks-uc",
        mlflow_experiment_id=None,
        mlflow_experiment_name=None,
        databricks_config_profile=None,
        databricks_serverless_compute_id="auto",
        runtime="local_cpu",
        sample_mode=True,
        data_source="manifest",
        data_manifest=None,
        data_dir=None,
        data_uc_table="image_manifest",
    )


def _touch_images(tmp_path, names):
    for name in names:
        image_path = tmp_path / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"sample")


def _rows():
    return [
        {
            "image_path": "good/group_a_0.jpg",
            "label": "normal",
            "group_id": "group_a",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
        {
            "image_path": "good/group_a_1.jpg",
            "label": "good",
            "group_id": "group_a",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
        {
            "image_path": "defect/group_b_0.jpg",
            "label": "crack",
            "group_id": "group_b",
            "source": "unit-test",
            "source_license": "Apache-2.0",
            "defect_types": ["crack"],
        },
        {
            "image_path": "defect/group_c_0.jpg",
            "label": "defect",
            "group_id": "group_c",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
        {
            "image_path": "good/group_d_0.jpg",
            "label": "normal",
            "group_id": "group_d",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
        {
            "image_path": "defect/group_e_0.jpg",
            "label": "defective",
            "group_id": "group_e",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
    ]


def test_normalize_manifest_row_maps_labels_and_metadata(tmp_path):
    _touch_images(tmp_path, ["defect/group_b_0.jpg"])
    row = _rows()[2]

    normalized = normalize_manifest_row(row, data_dir=tmp_path)

    assert normalized.record.label == 1
    assert normalized.record.group_id == "group_b"
    assert normalized.record.defect_types == ("crack",)
    assert normalized.source == "unit-test"
    assert normalized.source_license == "apache-2.0"
    assert str(tmp_path) in normalized.record.image_path


def test_non_commercial_license_is_rejected():
    with pytest.raises(ValueError, match="non-commercial"):
        validate_source_license("CC-BY-NC-SA-4.0")


def test_unapproved_license_is_rejected():
    with pytest.raises(ValueError, match="allowed license"):
        validate_source_license("custom-research-only")


def test_missing_license_is_rejected():
    with pytest.raises(ValueError, match="source_license is required"):
        validate_source_license(None)


def test_missing_image_path_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="image_path does not exist"):
        normalize_manifest_row(_rows()[0], data_dir=tmp_path)


def test_missing_group_fails(tmp_path):
    _touch_images(tmp_path, ["good/group_a_0.jpg"])
    row = dict(_rows()[0])
    row.pop("group_id")

    with pytest.raises(ValueError, match="group_id is required"):
        normalize_manifest_row(row, data_dir=tmp_path)


def test_normalize_manifest_rows_caps_sample_and_assigns_grouped_splits(tmp_path):
    rows = _rows()
    _touch_images(tmp_path, [row["image_path"] for row in rows])

    result = normalize_manifest_rows(
        rows,
        data_dir=tmp_path,
        source="manifest",
        sample_mode=True,
        sample_size=4,
        split_seed=7,
    )

    assert result.record_count == 4
    assert result.group_count == 3
    assert result.positive_count == 2
    assert result.negative_count == 2
    assert set(result.split_counts()) == {"train", "val", "test"}

    split_by_group = {}
    for row in result.records:
        existing = split_by_group.setdefault(row.record.group_id, row.record.split)
        assert existing == row.record.split


def test_load_manifest_rows_jsonl_and_write_output(tmp_path):
    rows = _rows()[:2]
    _touch_images(tmp_path, [row["image_path"] for row in rows])
    manifest_path = tmp_path / "input.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    loaded = load_manifest_rows(manifest_path)
    result = normalize_manifest_rows(loaded, data_dir=tmp_path, sample_mode=False)
    output = write_manifest_jsonl(result, tmp_path / "normalized" / "manifest.jsonl")

    assert output.output_path is not None
    written_rows = [
        json.loads(line)
        for line in output.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written_rows[0]["source_license"] == "apache-2.0"
    assert written_rows[0]["split"] in {"train", "val", "test"}


def test_copy_images_to_uc_volume_copies_and_rewrites_paths(tmp_path):
    rows = _rows()[:2]
    _touch_images(tmp_path, [row["image_path"] for row in rows])
    result = normalize_manifest_rows(rows, data_dir=tmp_path, sample_mode=False)
    image_dir = tmp_path / "volume" / "images"

    copied = copy_images_to_uc_volume(
        result,
        config=_config(),
        image_dir=image_dir,
    )

    assert copied.uc_image_dir == str(image_dir)
    assert copied.records[0].record.image_path.startswith(str(image_dir))
    assert (image_dir / copied.records[0].record.split / "group_a" / "group_a_0.jpg").exists()


def test_copy_images_to_uc_volume_can_upload_with_sdk_client(tmp_path):
    rows = _rows()[:2]
    _touch_images(tmp_path, [row["image_path"] for row in rows])
    result = normalize_manifest_rows(rows, data_dir=tmp_path, sample_mode=False)
    files_client = _FakeFilesClient()
    image_dir = "/Volumes/demo_catalog/demo_schema/demo_volume/cv/images"

    copied = copy_images_to_uc_volume(
        result,
        config=_config(),
        image_dir=image_dir,
        upload_mode="sdk",
        files_client=files_client,
    )

    assert copied.uc_image_dir == image_dir
    assert copied.records[0].record.image_path.startswith(image_dir)
    assert files_client.created_directories
    assert files_client.uploads[0]["file_path"].startswith(image_dir)
    assert files_client.uploads[0]["contents"] == b"sample"
    assert files_client.uploads[0]["overwrite"] is True


def test_ensure_uc_objects_uses_configured_catalog_schema_volume():
    spark = _FakeSpark()

    ensure_uc_objects(config=_config(), spark=spark)

    assert spark.sql_calls == [
        "CREATE CATALOG IF NOT EXISTS `demo_catalog`",
        "CREATE SCHEMA IF NOT EXISTS `demo_catalog`.`demo_schema`",
        "CREATE VOLUME IF NOT EXISTS `demo_catalog`.`demo_schema`.`demo_volume`",
    ]


def test_persist_ingest_to_uc_writes_delta_table_and_volume_manifest(tmp_path):
    rows = _rows()[:2]
    _touch_images(tmp_path, [row["image_path"] for row in rows])
    result = normalize_manifest_rows(rows, data_dir=tmp_path, sample_mode=False)
    spark = _FakeSpark()

    persisted = persist_ingest_to_uc(
        result,
        config=_config(),
        table_name="image_manifest",
        mode="overwrite",
        create_uc_objects=True,
        spark=spark,
    )

    assert persisted.uc_table_name == "demo_catalog.demo_schema.image_manifest"
    assert persisted.uc_manifest_uri == (
        "dbfs:/Volumes/demo_catalog/demo_schema/demo_volume/cv/ingest/image_manifest"
    )
    assert spark.sql_calls == [
        "CREATE CATALOG IF NOT EXISTS `demo_catalog`",
        "CREATE SCHEMA IF NOT EXISTS `demo_catalog`.`demo_schema`",
        "CREATE VOLUME IF NOT EXISTS `demo_catalog`.`demo_schema`.`demo_volume`",
    ]
    dataframe = spark.dataframes[0]
    assert dataframe.rows[0]["source_license"] == "apache-2.0"
    assert ("saveAsTable", "`demo_catalog`.`demo_schema`.`image_manifest`") in (
        dataframe.write.calls
    )
    assert (
        "json",
        "dbfs:/Volumes/demo_catalog/demo_schema/demo_volume/cv/ingest/image_manifest",
    ) in dataframe.write.calls
    assert dataframe.coalesce_calls == [1]

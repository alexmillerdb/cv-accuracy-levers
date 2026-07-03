"""Public dataset ingest helpers for manifest-backed image records."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

from levers.config import ProjectConfig
from levers.data import DatasetRecord, make_grouped_split, validate_group_splits


PERMISSIVE_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "mit",
    "odc-by-1.0",
    "public-domain",
}
NON_COMMERCIAL_MARKERS = ("non-commercial", "noncommercial", "-nc", "cc-by-nc")
DEFAULT_POSITIVE_LABELS = (
    "1",
    "anomaly",
    "bad",
    "crack",
    "defect",
    "defective",
    "positive",
)
DEFAULT_NEGATIVE_LABELS = ("0", "false", "good", "negative", "normal", "ok")


@dataclass(frozen=True)
class IngestedRecord:
    """Normalized image record with public-source metadata."""

    record: DatasetRecord
    source: str
    source_license: str
    original_label: str

    def as_manifest_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "image_path": self.record.image_path,
            "label": self.record.label,
            "group_id": self.record.group_id,
            "split": self.record.split,
            "source": self.source,
            "source_license": self.source_license,
            "original_label": self.original_label,
        }
        if self.record.defect_types:
            row["defect_types"] = list(self.record.defect_types)
        if self.record.bbox is not None:
            row["bbox"] = list(self.record.bbox)
        if self.record.comment_category is not None:
            row["comment_category"] = self.record.comment_category
        return row


@dataclass(frozen=True)
class IngestResult:
    """Output from a manifest ingest run."""

    records: tuple[IngestedRecord, ...]
    source: str
    sample_mode: bool
    output_path: Path | None = None
    uc_table_name: str | None = None
    uc_manifest_uri: str | None = None
    uc_image_dir: str | None = None

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def group_count(self) -> int:
        return len({row.record.group_id for row in self.records})

    @property
    def positive_count(self) -> int:
        return sum(1 for row in self.records if row.record.label == 1)

    @property
    def negative_count(self) -> int:
        return sum(1 for row in self.records if row.record.label == 0)

    def split_counts(self) -> dict[str, int]:
        counts = {"train": 0, "val": 0, "test": 0}
        for row in self.records:
            if row.record.split is not None:
                counts[row.record.split] = counts.get(row.record.split, 0) + 1
        return counts

    def summary(self) -> dict[str, object]:
        return {
            "source": self.source,
            "sample_mode": self.sample_mode,
            "record_count": self.record_count,
            "group_count": self.group_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "split_counts": self.split_counts(),
            "output_path": str(self.output_path) if self.output_path else None,
            "uc_table_name": self.uc_table_name,
            "uc_manifest_uri": self.uc_manifest_uri,
            "uc_image_dir": self.uc_image_dir,
        }


def _replace_result_records(
    result: IngestResult,
    records: Sequence[IngestedRecord],
    *,
    output_path: Path | None | object = ...,
    uc_table_name: str | None | object = ...,
    uc_manifest_uri: str | None | object = ...,
    uc_image_dir: str | None | object = ...,
) -> IngestResult:
    return IngestResult(
        records=tuple(records),
        source=result.source,
        sample_mode=result.sample_mode,
        output_path=(
            result.output_path
            if output_path is ...
            else output_path  # type: ignore[arg-type]
        ),
        uc_table_name=(
            result.uc_table_name
            if uc_table_name is ...
            else uc_table_name  # type: ignore[arg-type]
        ),
        uc_manifest_uri=(
            result.uc_manifest_uri
            if uc_manifest_uri is ...
            else uc_manifest_uri  # type: ignore[arg-type]
        ),
        uc_image_dir=(
            result.uc_image_dir
            if uc_image_dir is ...
            else uc_image_dir  # type: ignore[arg-type]
        ),
    )


def _normalize_license(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


def validate_source_license(
    license_value: object,
    *,
    allowed_licenses: Sequence[str] = tuple(sorted(PERMISSIVE_LICENSES)),
) -> str:
    """Return a normalized license string or raise for public-safety issues."""

    if license_value is None or str(license_value).strip() == "":
        raise ValueError("source_license is required")
    normalized = _normalize_license(license_value)
    if any(marker in normalized for marker in NON_COMMERCIAL_MARKERS):
        raise ValueError(f"non-commercial datasets are not allowed by default: {normalized}")
    allowed = {_normalize_license(value) for value in allowed_licenses}
    if normalized not in allowed:
        raise ValueError(
            f"source_license={normalized!r} is not in the allowed license list: "
            f"{', '.join(sorted(allowed))}"
        )
    return normalized


def normalize_binary_label(
    value: object,
    *,
    positive_labels: Sequence[str] = DEFAULT_POSITIVE_LABELS,
    negative_labels: Sequence[str] = DEFAULT_NEGATIVE_LABELS,
) -> int:
    """Map a source label into the demo's binary defective/non-defective target."""

    normalized = str(value).strip().lower()
    positives = {label.strip().lower() for label in positive_labels}
    negatives = {label.strip().lower() for label in negative_labels}
    if normalized in positives:
        return 1
    if normalized in negatives:
        return 0
    raise ValueError(
        f"label {value!r} is not mapped; add it to positive_labels or negative_labels"
    )


def load_manifest_rows(path: str | Path) -> list[dict[str, object]]:
    """Load manifest rows from CSV, JSON list, or JSONL."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows: list[dict[str, object]] = []
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"manifest line {line_number} is not an object")
                rows.append(row)
        return rows
    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must contain a list of row objects")
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError("JSON manifest rows must be objects")
        return list(payload)
    raise ValueError("manifest must be a .csv, .json, or .jsonl file")


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} is required")
    return str(value).strip()


def _row_label(row: Mapping[str, object]) -> object:
    for key in ("label", "class_name", "category"):
        if row.get(key) is not None and str(row[key]).strip() != "":
            return row[key]
    raise ValueError("label, class_name, or category is required")


def _resolve_image_path(image_path: str, data_dir: str | Path | None) -> Path:
    path = Path(image_path)
    if not path.is_absolute() and data_dir is not None:
        return Path(data_dir) / path
    return path


def normalize_manifest_row(
    row: Mapping[str, object],
    *,
    data_dir: str | Path | None = None,
    default_source: str = "manifest",
    positive_labels: Sequence[str] = DEFAULT_POSITIVE_LABELS,
    negative_labels: Sequence[str] = DEFAULT_NEGATIVE_LABELS,
    validate_images: bool = True,
) -> IngestedRecord:
    """Normalize one source manifest row into an ingest record."""

    image_path = _required_text(row, "image_path")
    group_id = _required_text(row, "group_id")
    original_label = str(_row_label(row)).strip()
    label = normalize_binary_label(
        original_label,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
    )
    source = str(row.get("source") or default_source).strip()
    source_license = validate_source_license(row.get("source_license") or row.get("license"))

    resolved_image_path = _resolve_image_path(image_path, data_dir)
    if validate_images and not resolved_image_path.exists():
        raise FileNotFoundError(f"image_path does not exist: {resolved_image_path}")

    record = DatasetRecord.from_mapping(
        {
            "image_path": str(resolved_image_path),
            "label": label,
            "group_id": group_id,
            "split": row.get("split"),
            "defect_types": row.get("defect_types", ()),
            "bbox": row.get("bbox"),
            "comment_category": row.get("comment_category"),
        }
    )
    return IngestedRecord(
        record=record,
        source=source,
        source_license=source_license,
        original_label=original_label,
    )


def normalize_manifest_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    data_dir: str | Path | None = None,
    source: str = "manifest",
    positive_labels: Sequence[str] = DEFAULT_POSITIVE_LABELS,
    negative_labels: Sequence[str] = DEFAULT_NEGATIVE_LABELS,
    sample_mode: bool = True,
    sample_size: int = 100,
    split_seed: int = 42,
    validate_images: bool = True,
) -> IngestResult:
    """Normalize rows, assign missing splits, and validate grouped splits."""

    normalized = [
        normalize_manifest_row(
            row,
            data_dir=data_dir,
            default_source=source,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            validate_images=validate_images,
        )
        for row in rows
    ]
    if not normalized:
        raise ValueError("manifest contains no records")
    if sample_mode:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        normalized = normalized[:sample_size]

    records = [row.record for row in normalized]
    if any(record.split is None for record in records):
        split_records = make_grouped_split(records, seed=split_seed)
    else:
        split_records = records
    validate_group_splits(split_records)

    by_identity = {
        (row.record.image_path, row.record.group_id): row
        for row in normalized
    }
    split_normalized = tuple(
        IngestedRecord(
            record=record,
            source=by_identity[(record.image_path, record.group_id)].source,
            source_license=by_identity[
                (record.image_path, record.group_id)
            ].source_license,
            original_label=by_identity[
                (record.image_path, record.group_id)
            ].original_label,
        )
        for record in split_records
    )
    return IngestResult(
        records=split_normalized,
        source=source,
        sample_mode=sample_mode,
    )


def write_manifest_jsonl(result: IngestResult, output_path: str | Path) -> IngestResult:
    """Write normalized manifest rows as JSONL and return an updated result."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in result.records:
            handle.write(json.dumps(row.as_manifest_row(), sort_keys=True))
            handle.write("\n")
    return IngestResult(
        records=result.records,
        source=result.source,
        sample_mode=result.sample_mode,
        output_path=path,
        uc_table_name=result.uc_table_name,
        uc_manifest_uri=result.uc_manifest_uri,
        uc_image_dir=result.uc_image_dir,
    )


def _quote_uc_identifier(*parts: str) -> str:
    return ".".join(f"`{part.replace('`', '``')}`" for part in parts)


def _qualified_table_name(config: ProjectConfig, table_name: str) -> str:
    parts = [part for part in table_name.split(".") if part]
    if len(parts) == 1:
        return config.uc_table(parts[0])
    if len(parts) == 3:
        return ".".join(parts)
    raise ValueError("table_name must be a plain table name or catalog.schema.table")


def _quoted_table_name(config: ProjectConfig, table_name: str) -> str:
    return _quote_uc_identifier(*_qualified_table_name(config, table_name).split("."))


def _get_spark_session():
    try:
        from databricks.connect import DatabricksSession

        builder = DatabricksSession.builder
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        if profile:
            builder = builder.profile(profile)
        cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")
        if cluster_id:
            builder = builder.clusterId(cluster_id)
        elif os.getenv("DATABRICKS_SERVERLESS_COMPUTE_ID") == "auto":
            builder = builder.serverless(True)
        return builder.getOrCreate()
    except ImportError:
        pass

    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise RuntimeError(
            "Writing to Unity Catalog requires PySpark or databricks-connect. "
            "Install the databricks optional dependencies or run this in "
            "Databricks."
        ) from error
    return SparkSession.builder.getOrCreate()


def ensure_uc_objects(
    *,
    config: ProjectConfig,
    spark=None,
) -> None:
    """Create configured UC catalog/schema/volume when caller has privileges."""

    resolved_spark = spark or _get_spark_session()
    resolved_spark.sql(f"CREATE CATALOG IF NOT EXISTS {_quote_uc_identifier(config.catalog)}")
    resolved_spark.sql(
        "CREATE SCHEMA IF NOT EXISTS "
        f"{_quote_uc_identifier(config.catalog, config.schema)}"
    )
    resolved_spark.sql(
        "CREATE VOLUME IF NOT EXISTS "
        f"{_quote_uc_identifier(config.catalog, config.schema, config.volume)}"
    )


def copy_images_to_uc_volume(
    result: IngestResult,
    *,
    config: ProjectConfig,
    image_dir: str | Path | None = None,
    overwrite: bool = True,
    upload_mode: str = "local",
    databricks_profile: str | None = None,
    files_client=None,
) -> IngestResult:
    """Copy referenced image files into the configured UC volume path."""

    if upload_mode not in {"local", "sdk"}:
        raise ValueError("upload_mode must be local or sdk")
    if upload_mode == "sdk" and files_client is None:
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as error:
            raise RuntimeError(
                "SDK image upload requires the databricks-sdk package."
            ) from error
        files_client = WorkspaceClient(
            profile=databricks_profile or config.databricks_config_profile
        ).files

    target_dir = Path(image_dir or config.volume_path("images"))
    copied_rows: list[IngestedRecord] = []
    for row in result.records:
        source_path = Path(row.record.image_path)
        if not source_path.exists():
            raise FileNotFoundError(f"image_path does not exist: {source_path}")
        relative_parts = [
            row.record.split or "unsplit",
            row.record.group_id,
            source_path.name,
        ]
        destination = target_dir.joinpath(*relative_parts)
        if upload_mode == "sdk":
            files_client.create_directory(str(destination.parent))
            with source_path.open("rb") as handle:
                files_client.upload(
                    str(destination),
                    handle,
                    overwrite=overwrite,
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                raise FileExistsError(f"image already exists: {destination}")
            shutil.copy2(source_path, destination)
        copied_rows.append(
            IngestedRecord(
                record=DatasetRecord.from_mapping(
                    {
                        **row.record.__dict__,
                        "image_path": str(destination),
                    }
                ),
                source=row.source,
                source_license=row.source_license,
                original_label=row.original_label,
            )
        )
    return _replace_result_records(
        result,
        copied_rows,
        uc_image_dir=str(target_dir),
    )


def persist_ingest_to_uc(
    result: IngestResult,
    *,
    config: ProjectConfig,
    table_name: str = "image_manifest",
    manifest_uri: str | None = None,
    mode: str = "overwrite",
    create_uc_objects: bool = False,
    spark=None,
) -> IngestResult:
    """Persist normalized manifest rows to a UC Delta table and volume path."""

    if mode not in {"append", "error", "errorifexists", "ignore", "overwrite"}:
        raise ValueError("mode must be append, error, errorifexists, ignore, or overwrite")
    resolved_spark = spark or _get_spark_session()
    if create_uc_objects:
        ensure_uc_objects(config=config, spark=resolved_spark)

    rows = [row.as_manifest_row() for row in result.records]
    dataframe = resolved_spark.createDataFrame(rows)
    qualified_table = _qualified_table_name(config, table_name)
    dataframe.write.mode(mode).format("delta").saveAsTable(
        _quoted_table_name(config, table_name)
    )

    resolved_manifest_uri = manifest_uri or config.volume_uri(
        "ingest",
        qualified_table.split(".")[-1],
    )
    dataframe.coalesce(1).write.mode(mode).json(resolved_manifest_uri)
    return _replace_result_records(
        result,
        result.records,
        uc_table_name=qualified_table,
        uc_manifest_uri=resolved_manifest_uri,
    )

"""Prepare a public dataset manifest for training and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from levers.ingest import (
    DEFAULT_NEGATIVE_LABELS,
    DEFAULT_POSITIVE_LABELS,
    IngestResult,
    copy_images_to_uc_volume,
    ensure_uc_objects,
    load_manifest_rows,
    normalize_manifest_rows,
    persist_ingest_to_uc,
    write_manifest_jsonl,
)


def _bool_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _label_list(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return tuple(default)
    return tuple(label.strip() for label in value.split(",") if label.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config = ProjectConfig.from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=config.data_source, choices=("manifest", "rf100vl"))
    parser.add_argument("--manifest-path", default=config.data_manifest)
    parser.add_argument("--data-dir", default=config.data_dir)
    parser.add_argument(
        "--output-path",
        default="artifacts/ingest/normalized_manifest.jsonl",
    )
    parser.add_argument("--sample-mode", type=_bool_text, default=config.sample_mode)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--runtime", default=config.runtime)
    parser.add_argument("--catalog", default=config.catalog)
    parser.add_argument("--schema", default=config.schema)
    parser.add_argument("--volume", default=config.volume)
    parser.add_argument("--volume-subpath", default=config.volume_subpath)
    parser.add_argument(
        "--positive-labels",
        default=None,
        help="Comma-separated source labels to map to defective.",
    )
    parser.add_argument(
        "--negative-labels",
        default=None,
        help="Comma-separated source labels to map to non-defective.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Normalize metadata without checking local image file existence.",
    )
    parser.add_argument(
        "--copy-images-to-uc-volume",
        action="store_true",
        help="Copy referenced images into the configured UC volume path.",
    )
    parser.add_argument(
        "--uc-image-dir",
        default=None,
        help="Override the target image directory. Defaults to CV_VOLUME/images.",
    )
    parser.add_argument(
        "--uc-image-upload-mode",
        default="local",
        choices=("local", "sdk"),
        help="Use local /Volumes copy inside Databricks or SDK upload from an IDE.",
    )
    parser.add_argument(
        "--databricks-profile",
        default=config.databricks_config_profile,
        help="Databricks profile for SDK image upload.",
    )
    parser.add_argument(
        "--write-uc",
        action="store_true",
        help="Write normalized manifest rows to a UC Delta table and volume path.",
    )
    parser.add_argument("--uc-table", default=config.data_uc_table)
    parser.add_argument("--uc-manifest-uri", default=None)
    parser.add_argument(
        "--uc-write-mode",
        default="overwrite",
        choices=("append", "error", "errorifexists", "ignore", "overwrite"),
    )
    parser.add_argument(
        "--create-uc-objects",
        action="store_true",
        help="Create the configured catalog, schema, and volume if permitted.",
    )
    return parser.parse_args(argv)


def _rf100vl_not_wired() -> None:
    if not os.getenv("ROBOFLOW_API_KEY"):
        raise ValueError(
            "ROBOFLOW_API_KEY is required for --source rf100vl. For offline "
            "or CI runs, export a permissively licensed dataset manifest and "
            "use --source manifest."
        )
    raise NotImplementedError(
        "Direct RF100-VL download is intentionally not wired in this public "
        "repo yet. Export a Roboflow Universe manifest locally, then run "
        "--source manifest with --manifest-path and --data-dir."
    )


def run_prepare_dataset(args: argparse.Namespace) -> IngestResult:
    config = replace(
        ProjectConfig.from_env(),
        catalog=args.catalog,
        schema=args.schema,
        volume=args.volume,
        volume_subpath=args.volume_subpath,
    )
    if args.source == "rf100vl":
        _rf100vl_not_wired()

    if not args.manifest_path:
        raise ValueError("Set --manifest-path or CV_DATA_MANIFEST for --source manifest")

    rows = load_manifest_rows(args.manifest_path)
    result = normalize_manifest_rows(
        rows,
        data_dir=args.data_dir,
        source=args.source,
        positive_labels=_label_list(args.positive_labels, DEFAULT_POSITIVE_LABELS),
        negative_labels=_label_list(args.negative_labels, DEFAULT_NEGATIVE_LABELS),
        sample_mode=args.sample_mode,
        sample_size=args.sample_size,
        split_seed=args.split_seed,
        validate_images=not args.allow_missing_images,
    )
    if args.copy_images_to_uc_volume and args.create_uc_objects:
        ensure_uc_objects(config=config)
    if args.copy_images_to_uc_volume:
        result = copy_images_to_uc_volume(
            result,
            config=config,
            image_dir=args.uc_image_dir,
            upload_mode=args.uc_image_upload_mode,
            databricks_profile=args.databricks_profile,
        )
    result = write_manifest_jsonl(result, args.output_path)
    if args.write_uc:
        result = persist_ingest_to_uc(
            result,
            config=config,
            table_name=args.uc_table,
            manifest_uri=args.uc_manifest_uri,
            mode=args.uc_write_mode,
            create_uc_objects=args.create_uc_objects,
        )
    return result


def _print_summary(args: argparse.Namespace, result: IngestResult) -> None:
    split_counts = result.split_counts()
    print("dataset_ingest_summary")
    print(f"source={result.source}")
    print(f"runtime={args.runtime}")
    print(f"sample_mode={str(result.sample_mode).lower()}")
    print(f"record_count={result.record_count}")
    print(f"group_count={result.group_count}")
    print(f"positive_count={result.positive_count}")
    print(f"negative_count={result.negative_count}")
    print(f"split_train={split_counts.get('train', 0)}")
    print(f"split_val={split_counts.get('val', 0)}")
    print(f"split_test={split_counts.get('test', 0)}")
    print(f"output_path={result.output_path}")
    if result.uc_image_dir:
        print(f"uc_image_dir={result.uc_image_dir}")
    if result.uc_table_name:
        print(f"uc_table_name={result.uc_table_name}")
    if result.uc_manifest_uri:
        print(f"uc_manifest_uri={result.uc_manifest_uri}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_prepare_dataset(args)
    _print_summary(args, result)
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

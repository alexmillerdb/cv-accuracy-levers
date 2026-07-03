import json
import os
from pathlib import Path
import subprocess
import sys


def _write_manifest(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "images"
    rows = [
        {
            "image_path": "good/group_a_0.jpg",
            "label": "normal",
            "group_id": "group_a",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
        {
            "image_path": "defect/group_b_0.jpg",
            "label": "defect",
            "group_id": "group_b",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
        {
            "image_path": "good/group_c_0.jpg",
            "label": "good",
            "group_id": "group_c",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
        {
            "image_path": "defect/group_d_0.jpg",
            "label": "crack",
            "group_id": "group_d",
            "source": "unit-test",
            "source_license": "Apache-2.0",
        },
    ]
    for row in rows:
        image_path = data_dir / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"sample")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return manifest_path, data_dir


def test_prepare_dataset_script_runs_without_pythonpath(tmp_path):
    manifest_path, data_dir = _write_manifest(tmp_path)
    output_path = tmp_path / "out" / "normalized.jsonl"
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_dataset.py",
            "--source",
            "manifest",
            "--manifest-path",
            str(manifest_path),
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--sample-mode",
            "true",
            "--sample-size",
            "3",
            "--runtime",
            "local_cpu",
            "--catalog",
            "demo_catalog",
            "--schema",
            "demo_schema",
            "--volume",
            "demo_volume",
            "--volume-subpath",
            "cv",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "dataset_ingest_summary" in result.stdout
    assert "record_count=3" in result.stdout
    assert "runtime=local_cpu" in result.stdout
    assert output_path.exists()


def test_prepare_dataset_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/prepare_dataset.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_prepare_dataset_script_main_exec_does_not_raise_system_exit_zero(tmp_path):
    manifest_path, data_dir = _write_manifest(tmp_path)
    output_path = tmp_path / "normalized.jsonl"
    script_path = Path("scripts/prepare_dataset.py")
    namespace = {"__name__": "__main__"}
    original_argv = sys.argv

    try:
        sys.argv = [
            str(script_path),
            "--manifest-path",
            str(manifest_path),
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--sample-mode",
            "true",
        ]
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
    finally:
        sys.argv = original_argv

    assert namespace["exit_code"] == 0
    assert output_path.exists()

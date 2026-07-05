import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from levers.eval import binary_classification_metrics
from levers.gpu_baseline import (
    GpuBaselinePrediction,
    GpuBaselineRunResult,
    GpuTrainingConfig,
    artifact_payloads,
    load_gpu_manifest_records,
    require_sample_or_explicit_full_data,
)
from levers.thresholds import ThresholdPoint, threshold_sweep


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=color).save(path)


def _manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "image_path": "normal/group_a.jpg",
            "label": "normal",
            "group_id": "group_a",
            "source": "unit-test",
            "source_license": "Apache-2.0",
            "split": "train",
        },
        {
            "image_path": "defect/group_b.jpg",
            "label": "defect",
            "group_id": "group_b",
            "source": "unit-test",
            "source_license": "Apache-2.0",
            "split": "train",
        },
        {
            "image_path": "defect/group_c.jpg",
            "label": "defect",
            "group_id": "group_c",
            "source": "unit-test",
            "source_license": "Apache-2.0",
            "split": "val",
        },
        {
            "image_path": "normal/group_d.jpg",
            "label": "normal",
            "group_id": "group_d",
            "source": "unit-test",
            "source_license": "Apache-2.0",
            "split": "val",
        },
        {
            "image_path": "normal/group_e.jpg",
            "label": "normal",
            "group_id": "group_e",
            "source": "unit-test",
            "source_license": "Apache-2.0",
            "split": "test",
        },
        {
            "image_path": "defect/group_f.jpg",
            "label": "defect",
            "group_id": "group_f",
            "source": "unit-test",
            "source_license": "Apache-2.0",
            "split": "test",
        },
    ]


def _write_manifest_fixture(tmp_path: Path) -> Path:
    colors = {
        "normal/group_a.jpg": (24, 48, 80),
        "defect/group_b.jpg": (220, 40, 40),
        "defect/group_c.jpg": (210, 55, 55),
        "normal/group_d.jpg": (30, 50, 85),
        "normal/group_e.jpg": (35, 60, 90),
        "defect/group_f.jpg": (225, 35, 35),
    }
    for image_path, color in colors.items():
        _write_image(tmp_path / image_path, color)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in _manifest_rows()),
        encoding="utf-8",
    )
    return manifest_path


def test_load_gpu_manifest_records_preserves_existing_grouped_splits(tmp_path):
    manifest_path = _write_manifest_fixture(tmp_path)

    rows = load_gpu_manifest_records(
        manifest_path=manifest_path,
        data_dir=tmp_path,
        sample_mode=True,
        sample_size=6,
    )

    assert len(rows) == 6
    split_by_group = {
        row.record.group_id: row.record.split
        for row in rows
    }
    assert split_by_group["group_a"] == "train"
    assert split_by_group["group_c"] == "val"
    assert split_by_group["group_f"] == "test"


def test_load_gpu_manifest_records_rejects_split_leakage(tmp_path):
    manifest_path = _write_manifest_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["group_id"] = "group_a"
    rows[1]["split"] = "test"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="appears in both"):
        load_gpu_manifest_records(
            manifest_path=manifest_path,
            data_dir=tmp_path,
            sample_mode=True,
            sample_size=6,
        )


def test_full_data_gpu_training_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="Full-data GPU training is disabled"):
        require_sample_or_explicit_full_data(
            sample_mode=False,
            allow_full_data=False,
        )


def test_gpu_baseline_artifact_payloads_are_stable():
    points = tuple(threshold_sweep([0, 1], [0.2, 0.8], thresholds=[0.5]))
    metrics = binary_classification_metrics([0, 1], [0.2, 0.8], threshold=0.5)
    result = GpuBaselineRunResult(
        threshold_point=ThresholdPoint(threshold=0.5, metrics=metrics),
        threshold_sweep=points,
        metrics=metrics,
        predictions=(
            GpuBaselinePrediction(
                image_path="/Volumes/demo/test/good.jpg",
                group_id="good",
                split="test",
                label=0,
                score=0.2,
                predicted_label=0,
            ),
            GpuBaselinePrediction(
                image_path="/Volumes/demo/test/bad.jpg",
                group_id="bad",
                split="test",
                label=1,
                score=0.8,
                predicted_label=1,
            ),
        ),
        train_losses=(0.8,),
        sample_size=6,
        group_count=6,
        train_size=2,
        val_size=2,
        test_size=2,
        min_recall=0.75,
        image_size=24,
        batch_size=2,
        epochs=1,
        learning_rate=0.001,
        requested_device="cpu",
        resolved_device="cpu",
        backbone="tiny_cnn",
        sample_mode=True,
        split_seed=42,
        seed=17,
        manifest_path="/tmp/manifest.jsonl",
        data_dir="/tmp/images",
        cuda_available=False,
    )

    payloads = artifact_payloads(result)

    assert set(payloads) == {
        "leaderboard_row.json",
        "predictions.json",
        "threshold_sweep.json",
        "training_summary.json",
    }
    assert result.leaderboard_row_payload()["comparison_status"] == (
        "needs_same_split_baseline_comparison"
    )
    assert result.training_summary_payload()["backbone"] == "tiny_cnn"


def test_air_gpu_baseline_yaml_parses():
    yaml = pytest.importorskip("yaml")

    payload = yaml.safe_load(Path("air/gpu_baseline_sample.yaml").read_text())

    assert payload["compute"]["accelerator_type"] == "GPU_1xA10"
    assert payload["compute"]["num_accelerators"] == 1
    assert payload["code_source"]["type"] == "snapshot"
    assert "scripts/train_gpu_baseline.py" in payload["command"]


def test_train_gpu_baseline_script_imports_when_file_global_is_missing():
    script_path = Path("scripts/train_gpu_baseline.py")
    namespace = {"__name__": "databricks_exec_simulation"}

    exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)

    assert "main" in namespace


def test_gpu_training_config_defaults_to_sample_mode():
    config = GpuTrainingConfig(manifest_path="/tmp/manifest.jsonl")

    assert config.sample_mode is True
    assert config.allow_full_data is False
    assert config.backbone == "tiny_cnn"


def test_train_gpu_baseline_script_local_cpu_torch_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    manifest_path = _write_manifest_fixture(tmp_path)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_gpu_baseline.py",
            "--manifest-path",
            str(manifest_path),
            "--data-dir",
            str(tmp_path),
            "--sample-mode",
            "true",
            "--sample-size",
            "6",
            "--runtime",
            "local_cpu",
            "--device",
            "cpu",
            "--image-size",
            "24",
            "--batch-size",
            "2",
            "--epochs",
            "1",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "gpu_baseline_metrics" in result.stdout
    assert "recall_defective=" in result.stdout
    assert "gpu_baseline_params" in result.stdout
    assert "cv_runtime=local_cpu" in result.stdout

"""Manifest-backed Torch image baseline for the Phase 6 GPU path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from levers.baseline import DEFAULT_THRESHOLDS
from levers.data import DatasetRecord, validate_group_splits
from levers.eval import BinaryMetrics, binary_classification_metrics
from levers.ingest import IngestedRecord, load_manifest_rows, normalize_manifest_rows
from levers.thresholds import ThresholdPoint, choose_threshold_for_recall, threshold_sweep


LEVER_NAME = "gpu_baseline_real_image"
MODEL_FAMILY = "torch_image_classifier"
DEFAULT_BACKBONE = "tiny_cnn"


@dataclass(frozen=True)
class GpuTrainingConfig:
    """Configuration for a small manifest-backed image training run."""

    manifest_path: str
    data_dir: str | None = None
    sample_mode: bool = True
    sample_size: int = 32
    split_seed: int = 42
    min_recall: float = 0.75
    image_size: int = 160
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 0.001
    num_workers: int = 0
    device: str = "auto"
    backbone: str = DEFAULT_BACKBONE
    seed: int = 17
    allow_full_data: bool = False
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class GpuBaselinePrediction:
    """One prediction row from the image baseline."""

    image_path: str
    group_id: str
    split: str
    label: int
    score: float
    predicted_label: int

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "image_path": self.image_path,
            "group_id": self.group_id,
            "split": self.split,
            "label": self.label,
            "score": self.score,
            "predicted_label": self.predicted_label,
        }


@dataclass(frozen=True)
class GpuBaselineRunResult:
    """Metrics, predictions, and artifact payloads from a GPU baseline run."""

    threshold_point: ThresholdPoint
    threshold_sweep: tuple[ThresholdPoint, ...]
    metrics: BinaryMetrics
    predictions: tuple[GpuBaselinePrediction, ...]
    train_losses: tuple[float, ...]
    sample_size: int
    group_count: int
    train_size: int
    val_size: int
    test_size: int
    min_recall: float
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    requested_device: str
    resolved_device: str
    backbone: str
    sample_mode: bool
    split_seed: int
    seed: int
    manifest_path: str
    data_dir: str | None
    cuda_available: bool
    cuda_device_name: str | None = None

    def metric_payload(self) -> dict[str, float | int | None]:
        return self.metrics.as_dict()

    def validation_metric_payload(self) -> dict[str, float | int | None]:
        return self.threshold_point.metrics.as_dict()

    def threshold_sweep_payload(self) -> list[dict[str, float | int | None]]:
        return [point.as_dict() for point in self.threshold_sweep]

    def predictions_payload(self) -> list[dict[str, str | int | float]]:
        return [prediction.as_dict() for prediction in self.predictions]

    def training_summary_payload(self) -> dict[str, object]:
        return {
            "lever_name": LEVER_NAME,
            "model_family": MODEL_FAMILY,
            "backbone": self.backbone,
            "sample_mode": self.sample_mode,
            "sample_size": self.sample_size,
            "group_count": self.group_count,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "min_recall": self.min_recall,
            "selected_threshold": self.threshold_point.threshold,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "train_losses": list(self.train_losses),
            "manifest_path": self.manifest_path,
            "data_dir": self.data_dir,
        }

    def leaderboard_row_payload(self) -> dict[str, str | int | float | None]:
        return {
            "lever_name": LEVER_NAME,
            "lever_type": "gpu_baseline",
            "model_family": MODEL_FAMILY,
            "backbone": self.backbone,
            "threshold": self.threshold_point.threshold,
            "recall_defective": self.metrics.recall_defective,
            "precision_defective": self.metrics.precision_defective,
            "f1_defective": self.metrics.f1_defective,
            "false_negatives": self.metrics.false_negatives,
            "auc_pr": self.metrics.auc_pr,
            "auc_roc": self.metrics.auc_roc,
            "sample_size": self.sample_size,
            "image_size": self.image_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "resolved_device": self.resolved_device,
            "comparison_status": "needs_same_split_baseline_comparison",
        }

    def param_payload(self) -> dict[str, str | int | float]:
        payload: dict[str, str | int | float] = {
            "lever_name": LEVER_NAME,
            "model_family": MODEL_FAMILY,
            "backbone": self.backbone,
            "sample_mode": str(self.sample_mode).lower(),
            "sample_size": self.sample_size,
            "group_count": self.group_count,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "min_recall": self.min_recall,
            "threshold_source": "validation",
            "selected_threshold": self.threshold_point.threshold,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "cuda_available": str(self.cuda_available).lower(),
            "split_seed": self.split_seed,
            "seed": self.seed,
        }
        if self.cuda_device_name:
            payload["cuda_device_name"] = self.cuda_device_name
        return payload


def require_sample_or_explicit_full_data(
    *,
    sample_mode: bool,
    allow_full_data: bool,
) -> None:
    """Keep Phase 6 from accidentally training a full dataset by default."""

    if not sample_mode and not allow_full_data:
        raise ValueError(
            "Full-data GPU training is disabled by default. Re-run with "
            "sample_mode=True for the first gate, or pass --allow-full-data "
            "after reviewing the public dataset, cost, and runtime target."
        )


def load_gpu_manifest_records(
    *,
    manifest_path: str | Path,
    data_dir: str | Path | None = None,
    sample_mode: bool = True,
    sample_size: int = 32,
    split_seed: int = 42,
) -> tuple[IngestedRecord, ...]:
    """Load a public manifest and return grouped train/val/test records."""

    rows = load_manifest_rows(manifest_path)
    result = normalize_manifest_rows(
        rows,
        data_dir=data_dir,
        source="manifest",
        sample_mode=sample_mode,
        sample_size=sample_size,
        split_seed=split_seed,
        validate_images=True,
    )
    validate_group_splits([row.record for row in result.records])
    return result.records


def _split_records(records: Sequence[DatasetRecord], split: str) -> list[int]:
    return [index for index, record in enumerate(records) if record.split == split]


def _validate_training_frame(records: Sequence[DatasetRecord]) -> None:
    if not records:
        raise ValueError("at least one image record is required")

    validate_group_splits(records)
    for split in ("train", "val", "test"):
        split_labels = [record.label for record in records if record.split == split]
        if not split_labels:
            raise ValueError(f"manifest must include at least one {split} record")
        if split in {"train", "test"} and len(set(split_labels)) < 2:
            raise ValueError(f"{split} split must contain both classes")
        if split == "val" and 1 not in split_labels:
            raise ValueError("val split must contain at least one defective record")


def _require_torch_stack():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from torchvision.models import resnet18
    except ImportError as error:
        raise RuntimeError(
            "GPU baseline training requires torch, torchvision, and Pillow. "
            "Install the gpu extra locally or run through the AIR YAML."
        ) from error
    return torch, nn, DataLoader, Dataset, transforms, resnet18


def _resolve_device(torch_module, requested_device: str):
    if requested_device == "auto":
        return torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
    return torch_module.device(requested_device)


def _build_model(*, nn_module, resnet18_factory, backbone: str):
    if backbone == "tiny_cnn":
        return nn_module.Sequential(
            nn_module.Conv2d(3, 8, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.MaxPool2d(2),
            nn_module.Conv2d(8, 16, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.AdaptiveAvgPool2d((1, 1)),
            nn_module.Flatten(),
            nn_module.Linear(16, 2),
        )
    if backbone == "resnet18":
        model = resnet18_factory(weights=None)
        model.fc = nn_module.Linear(model.fc.in_features, 2)
        return model
    raise ValueError("backbone must be tiny_cnn or resnet18")


def _make_dataset_class(torch_module, dataset_base, transforms_module, image_size: int):
    from PIL import Image

    transform = transforms_module.Compose(
        [
            transforms_module.Resize((image_size, image_size)),
            transforms_module.ToTensor(),
        ]
    )

    class ManifestImageDataset(dataset_base):
        def __init__(self, rows: Sequence[IngestedRecord]):
            self.rows = tuple(rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(row.record.image_path).convert("RGB")
            return transform(image), torch_module.tensor(row.record.label).long()

    return ManifestImageDataset


def _train_one_epoch(*, model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("training loader produced no examples")
    return total_loss / total_examples


def _predict_scores(*, torch_module, model, loader, device) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch_module.no_grad():
        for images, _labels in loader:
            logits = model(images.to(device))
            probabilities = torch_module.softmax(logits, dim=1)[:, 1]
            scores.extend(float(value) for value in probabilities.detach().cpu())
    return scores


def run_gpu_baseline(config: GpuTrainingConfig) -> GpuBaselineRunResult:
    """Train a small Torch image classifier over a manifest-backed sample."""

    require_sample_or_explicit_full_data(
        sample_mode=config.sample_mode,
        allow_full_data=config.allow_full_data,
    )
    if config.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")

    torch_module, nn_module, data_loader, dataset_base, transforms_module, resnet18_factory = (
        _require_torch_stack()
    )
    torch_module.manual_seed(config.seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(config.seed)

    rows = load_gpu_manifest_records(
        manifest_path=config.manifest_path,
        data_dir=config.data_dir,
        sample_mode=config.sample_mode,
        sample_size=config.sample_size,
        split_seed=config.split_seed,
    )
    records = [row.record for row in rows]
    _validate_training_frame(records)

    train_indices = _split_records(records, "train")
    val_indices = _split_records(records, "val")
    test_indices = _split_records(records, "test")
    row_by_split = {
        "train": [rows[index] for index in train_indices],
        "val": [rows[index] for index in val_indices],
        "test": [rows[index] for index in test_indices],
    }

    dataset_class = _make_dataset_class(
        torch_module,
        dataset_base,
        transforms_module,
        config.image_size,
    )
    generator = torch_module.Generator()
    generator.manual_seed(config.seed)
    train_loader = data_loader(
        dataset_class(row_by_split["train"]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = data_loader(
        dataset_class(row_by_split["val"]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = data_loader(
        dataset_class(row_by_split["test"]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = _resolve_device(torch_module, config.device)
    model = _build_model(
        nn_module=nn_module,
        resnet18_factory=resnet18_factory,
        backbone=config.backbone,
    ).to(device)
    criterion = nn_module.CrossEntropyLoss()
    optimizer = torch_module.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_losses = tuple(
        _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        for _ in range(config.epochs)
    )

    val_scores = _predict_scores(
        torch_module=torch_module,
        model=model,
        loader=val_loader,
        device=device,
    )
    val_labels = [row.record.label for row in row_by_split["val"]]
    points = tuple(
        threshold_sweep(
            val_labels,
            val_scores,
            thresholds=config.thresholds,
        )
    )
    selected = choose_threshold_for_recall(points, min_recall=config.min_recall)

    test_scores = _predict_scores(
        torch_module=torch_module,
        model=model,
        loader=test_loader,
        device=device,
    )
    test_labels = [row.record.label for row in row_by_split["test"]]
    metrics = binary_classification_metrics(
        test_labels,
        test_scores,
        threshold=selected.threshold,
    )
    predictions = tuple(
        GpuBaselinePrediction(
            image_path=row.record.image_path,
            group_id=row.record.group_id,
            split=str(row.record.split),
            label=row.record.label,
            score=float(score),
            predicted_label=int(float(score) >= selected.threshold),
        )
        for row, score in zip(row_by_split["test"], test_scores)
    )

    cuda_device_name = None
    if torch_module.cuda.is_available() and device.type == "cuda":
        cuda_device_name = torch_module.cuda.get_device_name(device)

    return GpuBaselineRunResult(
        threshold_point=selected,
        threshold_sweep=points,
        metrics=metrics,
        predictions=predictions,
        train_losses=train_losses,
        sample_size=len(rows),
        group_count=len({row.record.group_id for row in rows}),
        train_size=len(train_indices),
        val_size=len(val_indices),
        test_size=len(test_indices),
        min_recall=config.min_recall,
        image_size=config.image_size,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        requested_device=config.device,
        resolved_device=str(device),
        backbone=config.backbone,
        sample_mode=config.sample_mode,
        split_seed=config.split_seed,
        seed=config.seed,
        manifest_path=str(config.manifest_path),
        data_dir=str(config.data_dir) if config.data_dir is not None else None,
        cuda_available=bool(torch_module.cuda.is_available()),
        cuda_device_name=cuda_device_name,
    )


def artifact_payloads(result: GpuBaselineRunResult) -> dict[str, Any]:
    """Return MLflow artifact payloads with stable filenames."""

    return {
        "predictions.json": result.predictions_payload(),
        "threshold_sweep.json": result.threshold_sweep_payload(),
        "training_summary.json": result.training_summary_payload(),
        "leaderboard_row.json": result.leaderboard_row_payload(),
    }

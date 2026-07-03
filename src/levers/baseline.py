"""Sample-mode baseline training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from levers.data import DatasetRecord, make_grouped_split, validate_group_splits
from levers.eval import BinaryMetrics, binary_classification_metrics
from levers.thresholds import ThresholdPoint, choose_threshold_for_recall, threshold_sweep


DEFAULT_THRESHOLDS = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))


@dataclass(frozen=True)
class CentroidBaselineModel:
    """Tiny trainable stand-in for a whole-image classifier."""

    negative_centroid: tuple[float, ...]
    positive_centroid: tuple[float, ...]
    scale: float

    def score_samples(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Return defective-class scores in [0, 1]."""

        feature_array = _as_2d_features(features)
        negative = np.asarray(self.negative_centroid, dtype=float)
        positive = np.asarray(self.positive_centroid, dtype=float)
        negative_distance = np.linalg.norm(feature_array - negative, axis=1)
        positive_distance = np.linalg.norm(feature_array - positive, axis=1)
        margin = negative_distance - positive_distance
        return 1.0 / (1.0 + np.exp(-margin * self.scale))


@dataclass(frozen=True)
class BaselinePrediction:
    image_path: str
    group_id: str
    split: str
    label: int
    score: float


@dataclass(frozen=True)
class BaselineRunResult:
    model: CentroidBaselineModel
    threshold_point: ThresholdPoint
    metrics: BinaryMetrics
    predictions: tuple[BaselinePrediction, ...]
    sample_size: int
    group_count: int
    train_size: int
    val_size: int
    test_size: int

    def metric_payload(self) -> dict[str, float | int | None]:
        return self.metrics.as_dict()

    def validation_metric_payload(self) -> dict[str, float | int | None]:
        return self.threshold_point.metrics.as_dict()


def _as_2d_features(features: Sequence[Sequence[float]]) -> np.ndarray:
    feature_array = np.asarray(features, dtype=float)
    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array-like value")
    if feature_array.shape[0] == 0:
        raise ValueError("at least one feature row is required")
    return feature_array


def _labels_array(labels: Sequence[int]) -> np.ndarray:
    label_array = np.asarray(labels, dtype=int)
    if label_array.ndim != 1:
        raise ValueError("labels must be a 1D array-like value")
    if label_array.shape[0] == 0:
        raise ValueError("at least one label is required")
    return label_array


def _split_indices(records: Sequence[DatasetRecord], split: str) -> list[int]:
    return [index for index, record in enumerate(records) if record.split == split]


def build_sample_baseline_dataset(
    *,
    group_count: int = 12,
    views_per_group: int = 2,
    seed: int = 17,
) -> tuple[list[DatasetRecord], np.ndarray]:
    """Build public-safe sample records and synthetic whole-image features."""

    if group_count < 6:
        raise ValueError("sample baseline needs at least six groups")
    if views_per_group < 1:
        raise ValueError("views_per_group must be positive")

    rng = np.random.default_rng(seed)
    records: list[DatasetRecord] = []
    features: list[np.ndarray] = []

    for group_index in range(group_count):
        label = group_index % 2
        group_id = f"group_{group_index:02d}"
        base_signal = np.array(
            [
                0.20 + 0.58 * label,
                0.72 - 0.44 * label,
                0.28 + 0.34 * label,
                0.52 + 0.12 * ((group_index % 3) - 1),
            ],
            dtype=float,
        )
        for view_index in range(views_per_group):
            image_path = f"synthetic/{group_id}/view_{view_index:02d}.jpg"
            records.append(
                DatasetRecord(
                    image_path=image_path,
                    label=label,
                    group_id=group_id,
                    defect_types=("synthetic_defect",) if label else (),
                )
            )
            view_noise = rng.normal(loc=0.0, scale=0.035, size=base_signal.shape)
            features.append(base_signal + view_noise + (view_index * 0.015))

    return records, np.vstack(features)


def train_centroid_baseline(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    positive_label: int = 1,
) -> CentroidBaselineModel:
    """Train a simple centroid scorer for the sample whole-image baseline."""

    feature_array = _as_2d_features(features)
    label_array = _labels_array(labels)
    if feature_array.shape[0] != label_array.shape[0]:
        raise ValueError("features and labels must have the same row count")

    positive_mask = label_array == positive_label
    negative_mask = ~positive_mask
    if not positive_mask.any() or not negative_mask.any():
        raise ValueError("baseline training requires both positive and negative labels")

    positive_centroid = feature_array[positive_mask].mean(axis=0)
    negative_centroid = feature_array[negative_mask].mean(axis=0)
    centroid_distance = float(np.linalg.norm(positive_centroid - negative_centroid))
    if centroid_distance == 0.0:
        raise ValueError("positive and negative centroids are identical")

    return CentroidBaselineModel(
        negative_centroid=tuple(float(value) for value in negative_centroid),
        positive_centroid=tuple(float(value) for value in positive_centroid),
        scale=4.0 / centroid_distance,
    )


def run_sample_baseline(
    *,
    sample_mode: bool = True,
    split_seed: int = 1,
    feature_seed: int = 17,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> BaselineRunResult:
    """Train on sample records, choose a recall-first threshold, and test."""

    if not sample_mode:
        raise NotImplementedError(
            "Full-dataset baseline is deferred. Use sample_mode=True until "
            "open dataset ingest is implemented."
        )

    records, features = build_sample_baseline_dataset(seed=feature_seed)
    split_records = make_grouped_split(records, seed=split_seed)
    validate_group_splits(split_records)

    labels = np.asarray([record.label for record in split_records], dtype=int)
    train_indices = _split_indices(split_records, "train")
    val_indices = _split_indices(split_records, "val")
    test_indices = _split_indices(split_records, "test")
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("sample baseline requires train, val, and test splits")

    model = train_centroid_baseline(features[train_indices], labels[train_indices])
    scores = model.score_samples(features)
    val_points = threshold_sweep(
        labels[val_indices],
        scores[val_indices],
        thresholds=thresholds,
    )
    threshold_point = choose_threshold_for_recall(val_points, min_recall=min_recall)
    metrics = binary_classification_metrics(
        labels[test_indices],
        scores[test_indices],
        threshold=threshold_point.threshold,
    )
    predictions = tuple(
        BaselinePrediction(
            image_path=record.image_path,
            group_id=record.group_id,
            split=str(record.split),
            label=record.label,
            score=float(scores[index]),
        )
        for index, record in enumerate(split_records)
    )

    return BaselineRunResult(
        model=model,
        threshold_point=threshold_point,
        metrics=metrics,
        predictions=predictions,
        sample_size=len(split_records),
        group_count=len({record.group_id for record in split_records}),
        train_size=len(train_indices),
        val_size=len(val_indices),
        test_size=len(test_indices),
    )

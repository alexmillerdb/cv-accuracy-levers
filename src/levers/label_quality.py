"""Label-quality embedding review helpers for sample-mode analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from levers.baseline import (
    DEFAULT_THRESHOLDS,
    BaselinePrediction,
    build_sample_baseline_dataset,
)
from levers.threshold_tuning import run_sample_threshold_tuning


EMBEDDING_SOURCE = "sample_baseline_synthetic_features_v1"


@dataclass(frozen=True)
class EmbeddingRow:
    """One sample embedding with baseline score context."""

    image_path: str
    group_id: str
    split: str
    label: int
    score: float
    predicted_label: int
    embedding: tuple[float, ...]
    synthetic_issue: bool = False

    def prediction_dict(self) -> dict[str, str | int | float | bool]:
        return {
            "image_path": self.image_path,
            "group_id": self.group_id,
            "split": self.split,
            "label": self.label,
            "score": self.score,
            "predicted_label": self.predicted_label,
            "synthetic_issue": self.synthetic_issue,
        }

    def embedding_dict(self) -> dict[str, str | int | float | bool | list[float]]:
        payload = self.prediction_dict()
        payload["embedding"] = list(self.embedding)
        return payload


@dataclass(frozen=True)
class EmbeddingNeighborRow:
    """One nearest-neighbor relationship used by label-quality review."""

    image_path: str
    group_id: str
    split: str
    label: int
    neighbor_rank: int
    neighbor_image_path: str
    neighbor_group_id: str
    neighbor_split: str
    neighbor_label: int
    similarity: float
    same_label: bool
    same_group: bool
    synthetic_issue_pair: bool

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return {
            "image_path": self.image_path,
            "group_id": self.group_id,
            "split": self.split,
            "label": self.label,
            "neighbor_rank": self.neighbor_rank,
            "neighbor_image_path": self.neighbor_image_path,
            "neighbor_group_id": self.neighbor_group_id,
            "neighbor_split": self.neighbor_split,
            "neighbor_label": self.neighbor_label,
            "similarity": self.similarity,
            "same_label": self.same_label,
            "same_group": self.same_group,
            "synthetic_issue_pair": self.synthetic_issue_pair,
        }


@dataclass(frozen=True)
class LabelQualityIssueRow:
    """One ranked candidate for human label-quality review."""

    rank: int
    issue_type: str
    image_path: str
    group_id: str
    split: str
    label: int
    score: float
    predicted_label: int
    selected_threshold: float
    severity: float
    reason: str
    neighbor_image_path: str | None = None
    neighbor_group_id: str | None = None
    neighbor_split: str | None = None
    neighbor_label: int | None = None
    neighbor_score: float | None = None
    neighbor_predicted_label: int | None = None
    similarity: float | None = None
    synthetic_issue: bool = False

    def as_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "rank": self.rank,
            "issue_type": self.issue_type,
            "image_path": self.image_path,
            "group_id": self.group_id,
            "split": self.split,
            "label": self.label,
            "score": self.score,
            "predicted_label": self.predicted_label,
            "selected_threshold": self.selected_threshold,
            "severity": self.severity,
            "reason": self.reason,
            "neighbor_image_path": self.neighbor_image_path,
            "neighbor_group_id": self.neighbor_group_id,
            "neighbor_split": self.neighbor_split,
            "neighbor_label": self.neighbor_label,
            "neighbor_score": self.neighbor_score,
            "neighbor_predicted_label": self.neighbor_predicted_label,
            "similarity": self.similarity,
            "synthetic_issue": self.synthetic_issue,
        }


@dataclass(frozen=True)
class LabelQualityResult:
    """Label-quality review candidates plus embedding-neighbor context."""

    issue_rows: tuple[LabelQualityIssueRow, ...]
    embedding_rows: tuple[EmbeddingRow, ...]
    neighbor_rows: tuple[EmbeddingNeighborRow, ...]
    selected_threshold: float
    label_conflict_similarity_threshold: float
    cross_split_similarity_threshold: float
    neighbor_count: int
    top_k: int | None
    embedding_source: str
    synthetic_issue_injected: bool
    total_label_conflict_candidates: int
    total_cross_split_near_duplicate_candidates: int
    total_model_label_mismatches: int
    sample_size: int
    group_count: int
    train_size: int
    val_size: int
    test_size: int

    def review_rows_payload(self) -> list[dict[str, str | int | float | bool | None]]:
        return [row.as_dict() for row in self.issue_rows]

    def neighbor_rows_payload(self) -> list[dict[str, str | int | float | bool]]:
        return [row.as_dict() for row in self.neighbor_rows]

    def predictions_payload(self) -> list[dict[str, str | int | float | bool]]:
        return [row.prediction_dict() for row in self.embedding_rows]

    def embedding_rows_payload(
        self,
    ) -> list[dict[str, str | int | float | bool | list[float]]]:
        return [row.embedding_dict() for row in self.embedding_rows]

    def issue_type_counts_payload(self) -> dict[str, int]:
        return {
            "suspected_label_conflict": self.total_label_conflict_candidates,
            "cross_split_near_duplicate": (
                self.total_cross_split_near_duplicate_candidates
            ),
            "model_label_mismatch": self.total_model_label_mismatches,
        }

    def metric_payload(self) -> dict[str, float | int]:
        max_similarity = max(
            (row.similarity for row in self.issue_rows if row.similarity is not None),
            default=0.0,
        )
        return {
            "selected_threshold": self.selected_threshold,
            "reviewed_issue_count": len(self.issue_rows),
            "total_label_conflict_candidates": self.total_label_conflict_candidates,
            "total_cross_split_near_duplicate_candidates": (
                self.total_cross_split_near_duplicate_candidates
            ),
            "total_model_label_mismatches": self.total_model_label_mismatches,
            "max_issue_similarity": max_similarity,
        }

    def summary_payload(self) -> dict[str, str | int | float | bool | dict[str, int]]:
        return {
            "lever_name": "label_quality_embeddings",
            "lever_type": "analysis_artifact",
            "embedding_source": self.embedding_source,
            "selected_threshold": self.selected_threshold,
            "label_conflict_similarity_threshold": (
                self.label_conflict_similarity_threshold
            ),
            "cross_split_similarity_threshold": self.cross_split_similarity_threshold,
            "neighbor_count": self.neighbor_count,
            "top_k": -1 if self.top_k is None else self.top_k,
            "synthetic_issue_injected": self.synthetic_issue_injected,
            "reviewed_issue_count": len(self.issue_rows),
            "issue_type_counts": self.issue_type_counts_payload(),
            "sample_size": self.sample_size,
            "group_count": self.group_count,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
        }

    def leaderboard_row_payload(self) -> dict[str, str | int | float | bool]:
        return {
            "lever_name": "label_quality_embeddings",
            "lever_type": "analysis_artifact",
            "embedding_source": self.embedding_source,
            "selected_threshold": self.selected_threshold,
            "reviewed_issue_count": len(self.issue_rows),
            "label_conflict_candidates": self.total_label_conflict_candidates,
            "cross_split_near_duplicate_candidates": (
                self.total_cross_split_near_duplicate_candidates
            ),
            "model_label_mismatches": self.total_model_label_mismatches,
            "synthetic_issue_injected": self.synthetic_issue_injected,
            "recommended_next_action": "inspect_label_quality_review_rows",
        }


def _validate_issue_settings(
    *,
    label_conflict_similarity_threshold: float,
    cross_split_similarity_threshold: float,
    neighbor_count: int,
    top_k: int | None,
) -> None:
    for name, value in {
        "label_conflict_similarity_threshold": label_conflict_similarity_threshold,
        "cross_split_similarity_threshold": cross_split_similarity_threshold,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding rows must have the same dimensionality")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding rows must have non-zero norm")
    return float(
        sum(left_value * right_value for left_value, right_value in zip(left, right))
        / (left_norm * right_norm)
    )


def _prediction_by_path(
    predictions: Iterable[BaselinePrediction],
) -> dict[str, BaselinePrediction]:
    prediction_by_path: dict[str, BaselinePrediction] = {}
    for prediction in predictions:
        prediction_by_path[prediction.image_path] = prediction
    return prediction_by_path


def _embedding_rows_from_predictions(
    predictions: Iterable[BaselinePrediction],
    features: np.ndarray,
    *,
    selected_threshold: float,
    positive_label: int = 1,
) -> tuple[EmbeddingRow, ...]:
    prediction_rows = tuple(predictions)
    if features.shape[0] != len(prediction_rows):
        raise ValueError("features and predictions must have the same row count")

    rows: list[EmbeddingRow] = []
    for prediction, feature in zip(prediction_rows, features):
        predicted_label = positive_label if prediction.score >= selected_threshold else 0
        rows.append(
            EmbeddingRow(
                image_path=prediction.image_path,
                group_id=prediction.group_id,
                split=prediction.split,
                label=prediction.label,
                score=float(prediction.score),
                predicted_label=predicted_label,
                embedding=tuple(float(value) for value in feature),
            )
        )
    return tuple(rows)


def _inject_synthetic_label_issue(
    rows: Sequence[EmbeddingRow],
    *,
    selected_threshold: float,
    positive_label: int = 1,
) -> tuple[EmbeddingRow, ...]:
    if not rows:
        raise ValueError("synthetic label issue injection requires embedding rows")

    source = next((row for row in rows if row.split != "test"), rows[0])
    injected_label = 0 if source.label == positive_label else positive_label
    injected_score = min(source.score, selected_threshold - 0.10)
    injected_predicted_label = (
        positive_label if injected_score >= selected_threshold else 0
    )
    injected = EmbeddingRow(
        image_path="synthetic/injected_label_issue/view_00.jpg",
        group_id="synthetic_injected_label_issue",
        split="test",
        label=injected_label,
        score=float(injected_score),
        predicted_label=injected_predicted_label,
        embedding=source.embedding,
        synthetic_issue=True,
    )
    return (*rows, injected)


def build_sample_embedding_rows(
    *,
    sample_mode: bool = True,
    split_seed: int = 1,
    feature_seed: int = 17,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    inject_synthetic_label_issue: bool = False,
) -> tuple[EmbeddingRow, ...]:
    """Build deterministic sample embeddings from the baseline sample path."""

    if not sample_mode:
        raise NotImplementedError(
            "Full-dataset label-quality embeddings are deferred. Use "
            "sample_mode=True for the CPU smoke path."
        )

    threshold_result = run_sample_threshold_tuning(
        sample_mode=sample_mode,
        split_seed=split_seed,
        feature_seed=feature_seed,
        min_recall=min_recall,
        thresholds=thresholds,
    )
    records, features = build_sample_baseline_dataset(seed=feature_seed)
    prediction_by_path = _prediction_by_path(threshold_result.predictions)
    ordered_predictions = tuple(
        prediction_by_path[record.image_path] for record in records
    )
    rows = _embedding_rows_from_predictions(
        ordered_predictions,
        features,
        selected_threshold=threshold_result.selected_point.threshold,
    )
    if inject_synthetic_label_issue:
        rows = _inject_synthetic_label_issue(
            rows,
            selected_threshold=threshold_result.selected_point.threshold,
        )
    return rows


def _build_neighbor_rows(
    rows: Sequence[EmbeddingRow],
    *,
    neighbor_count: int,
) -> tuple[EmbeddingNeighborRow, ...]:
    neighbor_rows: list[EmbeddingNeighborRow] = []
    for row in rows:
        ranked_neighbors = sorted(
            (
                (_cosine_similarity(row.embedding, candidate.embedding), candidate)
                for candidate in rows
                if candidate.image_path != row.image_path
            ),
            key=lambda item: (-item[0], item[1].image_path),
        )[:neighbor_count]
        for rank, (similarity, neighbor) in enumerate(ranked_neighbors, start=1):
            neighbor_rows.append(
                EmbeddingNeighborRow(
                    image_path=row.image_path,
                    group_id=row.group_id,
                    split=row.split,
                    label=row.label,
                    neighbor_rank=rank,
                    neighbor_image_path=neighbor.image_path,
                    neighbor_group_id=neighbor.group_id,
                    neighbor_split=neighbor.split,
                    neighbor_label=neighbor.label,
                    similarity=similarity,
                    same_label=row.label == neighbor.label,
                    same_group=row.group_id == neighbor.group_id,
                    synthetic_issue_pair=(
                        row.synthetic_issue or neighbor.synthetic_issue
                    ),
                )
            )
    return tuple(neighbor_rows)


def _pair_key(row: EmbeddingNeighborRow) -> tuple[str, str]:
    return tuple(sorted([row.image_path, row.neighbor_image_path]))


def _issue_from_neighbor(
    *,
    issue_type: str,
    row: EmbeddingRow,
    neighbor: EmbeddingRow,
    selected_threshold: float,
    similarity: float,
    reason: str,
) -> LabelQualityIssueRow:
    return LabelQualityIssueRow(
        rank=0,
        issue_type=issue_type,
        image_path=row.image_path,
        group_id=row.group_id,
        split=row.split,
        label=row.label,
        score=row.score,
        predicted_label=row.predicted_label,
        selected_threshold=selected_threshold,
        severity=similarity,
        reason=reason,
        neighbor_image_path=neighbor.image_path,
        neighbor_group_id=neighbor.group_id,
        neighbor_split=neighbor.split,
        neighbor_label=neighbor.label,
        neighbor_score=neighbor.score,
        neighbor_predicted_label=neighbor.predicted_label,
        similarity=similarity,
        synthetic_issue=row.synthetic_issue or neighbor.synthetic_issue,
    )


def _rank_issue_rows(
    issue_rows: Sequence[LabelQualityIssueRow],
) -> tuple[LabelQualityIssueRow, ...]:
    ranked = sorted(
        issue_rows,
        key=lambda row: (
            -row.severity,
            row.issue_type,
            row.image_path,
            row.neighbor_image_path or "",
        ),
    )
    return tuple(
        LabelQualityIssueRow(
            rank=index,
            issue_type=row.issue_type,
            image_path=row.image_path,
            group_id=row.group_id,
            split=row.split,
            label=row.label,
            score=row.score,
            predicted_label=row.predicted_label,
            selected_threshold=row.selected_threshold,
            severity=row.severity,
            reason=row.reason,
            neighbor_image_path=row.neighbor_image_path,
            neighbor_group_id=row.neighbor_group_id,
            neighbor_split=row.neighbor_split,
            neighbor_label=row.neighbor_label,
            neighbor_score=row.neighbor_score,
            neighbor_predicted_label=row.neighbor_predicted_label,
            similarity=row.similarity,
            synthetic_issue=row.synthetic_issue,
        )
        for index, row in enumerate(ranked, start=1)
    )


def find_label_quality_issues(
    embedding_rows: Iterable[EmbeddingRow],
    *,
    selected_threshold: float,
    label_conflict_similarity_threshold: float = 0.98,
    cross_split_similarity_threshold: float = 0.999,
    neighbor_count: int = 5,
    top_k: int | None = 20,
) -> LabelQualityResult:
    """Rank suspected label conflicts, leakage candidates, and model mismatches."""

    _validate_issue_settings(
        label_conflict_similarity_threshold=label_conflict_similarity_threshold,
        cross_split_similarity_threshold=cross_split_similarity_threshold,
        neighbor_count=neighbor_count,
        top_k=top_k,
    )
    rows = tuple(embedding_rows)
    if not rows:
        raise ValueError("label-quality review requires at least one embedding row")

    neighbor_rows = _build_neighbor_rows(rows, neighbor_count=neighbor_count)
    rows_by_path = {row.image_path: row for row in rows}
    pair_neighbors: dict[tuple[str, str], EmbeddingNeighborRow] = {}
    for neighbor_row in neighbor_rows:
        key = _pair_key(neighbor_row)
        existing = pair_neighbors.get(key)
        if existing is None or neighbor_row.similarity > existing.similarity:
            pair_neighbors[key] = neighbor_row

    label_conflicts: list[LabelQualityIssueRow] = []
    cross_split_candidates: list[LabelQualityIssueRow] = []
    for neighbor_row in pair_neighbors.values():
        row = rows_by_path[neighbor_row.image_path]
        neighbor = rows_by_path[neighbor_row.neighbor_image_path]
        if (
            row.label != neighbor.label
            and neighbor_row.similarity >= label_conflict_similarity_threshold
        ):
            label_conflicts.append(
                _issue_from_neighbor(
                    issue_type="suspected_label_conflict",
                    row=row,
                    neighbor=neighbor,
                    selected_threshold=selected_threshold,
                    similarity=neighbor_row.similarity,
                    reason="nearest_neighbor_has_different_label",
                )
            )
        if (
            row.split != neighbor.split
            and neighbor_row.similarity >= cross_split_similarity_threshold
        ):
            cross_split_candidates.append(
                _issue_from_neighbor(
                    issue_type="cross_split_near_duplicate",
                    row=row,
                    neighbor=neighbor,
                    selected_threshold=selected_threshold,
                    similarity=neighbor_row.similarity,
                    reason="nearest_neighbor_crosses_train_val_test_boundary",
                )
            )

    model_mismatches = [
        LabelQualityIssueRow(
            rank=0,
            issue_type="model_label_mismatch",
            image_path=row.image_path,
            group_id=row.group_id,
            split=row.split,
            label=row.label,
            score=row.score,
            predicted_label=row.predicted_label,
            selected_threshold=selected_threshold,
            severity=abs(row.score - selected_threshold),
            reason="baseline_prediction_disagrees_with_label_at_selected_threshold",
            synthetic_issue=row.synthetic_issue,
        )
        for row in rows
        if row.predicted_label != row.label
    ]

    all_issues = _rank_issue_rows(
        [*label_conflicts, *cross_split_candidates, *model_mismatches]
    )
    issue_rows = all_issues if top_k is None else all_issues[:top_k]

    return LabelQualityResult(
        issue_rows=tuple(issue_rows),
        embedding_rows=rows,
        neighbor_rows=neighbor_rows,
        selected_threshold=selected_threshold,
        label_conflict_similarity_threshold=label_conflict_similarity_threshold,
        cross_split_similarity_threshold=cross_split_similarity_threshold,
        neighbor_count=neighbor_count,
        top_k=top_k,
        embedding_source=EMBEDDING_SOURCE,
        synthetic_issue_injected=any(row.synthetic_issue for row in rows),
        total_label_conflict_candidates=len(label_conflicts),
        total_cross_split_near_duplicate_candidates=len(cross_split_candidates),
        total_model_label_mismatches=len(model_mismatches),
        sample_size=len(rows),
        group_count=len({row.group_id for row in rows}),
        train_size=len([row for row in rows if row.split == "train"]),
        val_size=len([row for row in rows if row.split == "val"]),
        test_size=len([row for row in rows if row.split == "test"]),
    )


def run_sample_label_quality_embeddings(
    *,
    sample_mode: bool = True,
    split_seed: int = 1,
    feature_seed: int = 17,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    label_conflict_similarity_threshold: float = 0.98,
    cross_split_similarity_threshold: float = 0.999,
    neighbor_count: int = 5,
    top_k: int | None = 20,
    inject_synthetic_label_issue: bool = False,
) -> LabelQualityResult:
    """Run sample baseline embeddings and return label-quality review candidates."""

    threshold_result = run_sample_threshold_tuning(
        sample_mode=sample_mode,
        split_seed=split_seed,
        feature_seed=feature_seed,
        min_recall=min_recall,
        thresholds=thresholds,
    )
    rows = build_sample_embedding_rows(
        sample_mode=sample_mode,
        split_seed=split_seed,
        feature_seed=feature_seed,
        min_recall=min_recall,
        thresholds=thresholds,
        inject_synthetic_label_issue=inject_synthetic_label_issue,
    )
    return find_label_quality_issues(
        rows,
        selected_threshold=threshold_result.selected_point.threshold,
        label_conflict_similarity_threshold=label_conflict_similarity_threshold,
        cross_split_similarity_threshold=cross_split_similarity_threshold,
        neighbor_count=neighbor_count,
        top_k=top_k,
    )

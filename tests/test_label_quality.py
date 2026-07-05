import pytest

from levers.label_quality import (
    EMBEDDING_SOURCE,
    EmbeddingRow,
    find_label_quality_issues,
    run_sample_label_quality_embeddings,
)


def _row(
    name: str,
    *,
    split: str,
    label: int,
    score: float,
    embedding: tuple[float, ...],
    predicted_label: int | None = None,
) -> EmbeddingRow:
    selected_predicted_label = int(score >= 0.5) if predicted_label is None else predicted_label
    return EmbeddingRow(
        image_path=f"synthetic/{split}/{name}.jpg",
        group_id=name,
        split=split,
        label=label,
        score=score,
        predicted_label=selected_predicted_label,
        embedding=embedding,
    )


def test_sample_label_quality_output_is_deterministic():
    first = run_sample_label_quality_embeddings(split_seed=1, feature_seed=17)
    second = run_sample_label_quality_embeddings(split_seed=1, feature_seed=17)

    assert first.summary_payload() == second.summary_payload()
    assert first.review_rows_payload() == second.review_rows_payload()
    assert first.neighbor_rows_payload() == second.neighbor_rows_payload()
    assert first.sample_size == 24
    assert first.train_size == 16
    assert first.val_size == 4
    assert first.test_size == 4
    assert first.embedding_source == EMBEDDING_SOURCE
    assert first.selected_threshold == pytest.approx(0.9)


def test_label_conflict_candidates_rank_by_nearest_neighbor_similarity():
    result = find_label_quality_issues(
        [
            _row("anchor", split="train", label=0, score=0.10, embedding=(1.0, 0.0)),
            _row("near_conflict", split="train", label=1, score=0.90, embedding=(0.999, 0.001)),
            _row("far_conflict", split="train", label=1, score=0.90, embedding=(0.970, 0.030)),
        ],
        selected_threshold=0.5,
        label_conflict_similarity_threshold=0.95,
        cross_split_similarity_threshold=1.0,
        neighbor_count=2,
        top_k=None,
    )

    conflicts = [
        row
        for row in result.issue_rows
        if row.issue_type == "suspected_label_conflict"
    ]
    assert len(conflicts) == 2
    assert conflicts[0].neighbor_image_path == "synthetic/train/near_conflict.jpg"
    assert conflicts[0].similarity > conflicts[1].similarity
    assert result.total_label_conflict_candidates == 2


def test_cross_split_near_duplicate_candidates_are_detected():
    result = find_label_quality_issues(
        [
            _row("train_dup", split="train", label=1, score=0.80, embedding=(1.0, 0.0)),
            _row("test_dup", split="test", label=1, score=0.80, embedding=(0.9999, 0.0001)),
            _row("test_far", split="test", label=1, score=0.80, embedding=(0.0, 1.0)),
        ],
        selected_threshold=0.5,
        label_conflict_similarity_threshold=1.0,
        cross_split_similarity_threshold=0.999,
        neighbor_count=1,
        top_k=None,
    )

    assert result.total_cross_split_near_duplicate_candidates == 1
    candidate = next(
        row
        for row in result.issue_rows
        if row.issue_type == "cross_split_near_duplicate"
    )
    assert candidate.image_path == "synthetic/train/train_dup.jpg"
    assert candidate.neighbor_image_path == "synthetic/test/test_dup.jpg"


def test_model_label_mismatch_counting():
    result = find_label_quality_issues(
        [
            _row("false_positive", split="test", label=0, score=0.95, embedding=(1.0, 0.0)),
            _row("false_negative", split="test", label=1, score=0.05, embedding=(0.0, 1.0)),
            _row("true_positive", split="test", label=1, score=0.95, embedding=(0.0, 0.9)),
        ],
        selected_threshold=0.5,
        label_conflict_similarity_threshold=1.0,
        cross_split_similarity_threshold=1.0,
        neighbor_count=1,
        top_k=None,
    )

    mismatches = [
        row for row in result.issue_rows if row.issue_type == "model_label_mismatch"
    ]
    assert result.total_model_label_mismatches == 2
    assert len(mismatches) == 2
    assert {row.image_path for row in mismatches} == {
        "synthetic/test/false_positive.jpg",
        "synthetic/test/false_negative.jpg",
    }


def test_label_quality_rejects_invalid_thresholds_and_top_k():
    rows = [_row("one", split="test", label=1, score=0.90, embedding=(1.0, 0.0))]

    with pytest.raises(ValueError, match="label_conflict_similarity_threshold"):
        find_label_quality_issues(
            rows,
            selected_threshold=0.5,
            label_conflict_similarity_threshold=-0.1,
        )

    with pytest.raises(ValueError, match="cross_split_similarity_threshold"):
        find_label_quality_issues(
            rows,
            selected_threshold=0.5,
            cross_split_similarity_threshold=1.1,
        )

    with pytest.raises(ValueError, match="neighbor_count must be positive"):
        find_label_quality_issues(rows, selected_threshold=0.5, neighbor_count=0)

    with pytest.raises(ValueError, match="top_k must be non-negative"):
        find_label_quality_issues(rows, selected_threshold=0.5, top_k=-1)

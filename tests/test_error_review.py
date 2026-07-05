import pytest

from levers.baseline import BaselinePrediction
from levers.error_review import (
    build_false_negative_review_rows,
    review_false_negatives_from_predictions,
    run_sample_false_negative_review,
)


def _prediction(
    split: str,
    label: int,
    score: float,
    *,
    name: str,
) -> BaselinePrediction:
    return BaselinePrediction(
        image_path=f"synthetic/{split}/{name}.jpg",
        group_id=name,
        split=split,
        label=label,
        score=score,
    )


def test_false_negative_review_rows_rank_by_margin():
    rows = build_false_negative_review_rows(
        [
            _prediction("test", 1, 0.48, name="borderline"),
            _prediction("test", 1, 0.20, name="high_confidence"),
            _prediction("test", 1, 0.35, name="medium"),
            _prediction("test", 0, 0.10, name="true_negative"),
            _prediction("test", 1, 0.70, name="true_positive"),
            _prediction("val", 1, 0.10, name="wrong_split"),
        ],
        threshold=0.5,
        top_k=2,
    )

    assert [row.image_path for row in rows] == [
        "synthetic/test/high_confidence.jpg",
        "synthetic/test/medium.jpg",
    ]
    assert [row.rank for row in rows] == [1, 2]
    assert rows[0].margin == pytest.approx(0.3)
    assert rows[0].review_bucket == "high_confidence_good_prediction"
    assert rows[1].review_bucket == "medium_confidence_good_prediction"


def test_false_negative_review_result_summarizes_rows_and_metrics():
    result = review_false_negatives_from_predictions(
        [
            _prediction("train", 1, 0.80, name="train_pos"),
            _prediction("val", 1, 0.80, name="val_pos"),
            _prediction("test", 1, 0.48, name="borderline"),
            _prediction("test", 1, 0.20, name="high_confidence"),
            _prediction("test", 0, 0.10, name="true_negative"),
            _prediction("test", 1, 0.70, name="true_positive"),
        ],
        review_threshold=0.5,
        selected_threshold=0.4,
        top_k=None,
    )

    assert result.total_false_negatives == 2
    assert result.review_metrics.false_negatives == 2
    assert result.review_metrics.true_positives == 1
    assert result.metric_payload()["threshold_delta"] == pytest.approx(0.1)
    assert result.bucket_counts_payload() == {
        "high_confidence_good_prediction": 1,
        "borderline_threshold_miss": 1,
    }
    assert result.summary_payload()["lever_name"] == "false_negative_review"
    assert result.leaderboard_row_payload()["recommended_next_action"] == (
        "inspect_false_negative_review_rows"
    )


def test_sample_false_negative_review_defaults_to_selected_threshold():
    result = run_sample_false_negative_review(split_seed=1, feature_seed=17)

    assert result.sample_size == 24
    assert result.selected_threshold == pytest.approx(0.9)
    assert result.review_threshold == pytest.approx(result.selected_threshold)
    assert result.total_false_negatives == 0
    assert result.review_rows_payload() == []


def test_sample_false_negative_review_supports_threshold_override():
    result = run_sample_false_negative_review(
        split_seed=1,
        feature_seed=17,
        review_threshold=0.95,
    )

    assert result.review_threshold == pytest.approx(0.95)
    assert result.selected_threshold == pytest.approx(0.9)
    assert result.total_false_negatives == 1
    assert result.review_rows[0].image_path == "synthetic/group_09/view_00.jpg"
    assert result.review_rows[0].review_bucket == "borderline_threshold_miss"


def test_false_negative_review_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        build_false_negative_review_rows(
            [_prediction("test", 1, 0.1, name="miss")],
            threshold=0.5,
            top_k=-1,
        )

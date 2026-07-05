import pytest

from levers.baseline import BaselinePrediction, run_sample_baseline
from levers.crop_first import (
    CROP_FIRST_FEATURE_SOURCE,
    compare_crop_first_to_baseline,
    run_sample_crop_first_ab,
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


def _paired_predictions() -> tuple[list[BaselinePrediction], list[BaselinePrediction]]:
    baseline_rows = [
        _prediction("val", 1, 0.45, name="val_pos_a"),
        _prediction("val", 1, 0.44, name="val_pos_b"),
        _prediction("val", 0, 0.43, name="val_neg_a"),
        _prediction("val", 0, 0.10, name="val_neg_b"),
        _prediction("test", 1, 0.20, name="test_pos_a"),
        _prediction("test", 1, 0.18, name="test_pos_b"),
        _prediction("test", 0, 0.95, name="test_neg_a"),
        _prediction("test", 0, 0.90, name="test_neg_b"),
    ]
    crop_first_rows = [
        _prediction("val", 1, 0.60, name="val_pos_a"),
        _prediction("val", 1, 0.55, name="val_pos_b"),
        _prediction("val", 0, 0.50, name="val_neg_a"),
        _prediction("val", 0, 0.10, name="val_neg_b"),
        _prediction("test", 1, 0.70, name="test_pos_a"),
        _prediction("test", 1, 0.65, name="test_pos_b"),
        _prediction("test", 0, 0.30, name="test_neg_a"),
        _prediction("test", 0, 0.20, name="test_neg_b"),
    ]
    return baseline_rows, crop_first_rows


def test_sample_crop_first_ab_output_is_deterministic():
    first = run_sample_crop_first_ab(split_seed=1, feature_seed=17)
    second = run_sample_crop_first_ab(split_seed=1, feature_seed=17)

    assert first.metric_payload() == second.metric_payload()
    assert first.param_payload() == second.param_payload()
    assert first.comparison_rows_payload() == second.comparison_rows_payload()
    assert first.sample_size == 24
    assert first.train_size == 16
    assert first.val_size == 4
    assert first.test_size == 4
    assert first.feature_source == CROP_FIRST_FEATURE_SOURCE


def test_sample_crop_first_preserves_grouped_split_and_baseline_scores():
    result = run_sample_crop_first_ab(split_seed=1, feature_seed=17)
    baseline = run_sample_baseline(split_seed=1, feature_seed=17)

    split_by_group: dict[str, str] = {}
    for prediction in result.baseline.predictions:
        existing = split_by_group.setdefault(prediction.group_id, prediction.split)
        assert existing == prediction.split

    assert [row.image_path for row in result.baseline.predictions] == [
        row.image_path for row in result.crop_first.predictions
    ]
    assert [row.label for row in result.baseline.predictions] == [
        row.label for row in result.crop_first.predictions
    ]
    assert [row.score for row in result.baseline.predictions] == [
        row.score for row in baseline.predictions
    ]


def test_crop_first_thresholds_are_selected_from_validation_only():
    baseline_rows, crop_first_rows = _paired_predictions()
    result = compare_crop_first_to_baseline(
        baseline_rows,
        crop_first_rows,
        min_recall=1.0,
        thresholds=[0.30, 0.40, 0.44, 0.50, 0.55],
    )

    assert result.baseline.selected_point.threshold == pytest.approx(0.44)
    assert result.crop_first.selected_point.threshold == pytest.approx(0.55)
    assert result.baseline.test_metrics.recall_defective == 0.0
    assert result.crop_first.test_metrics.recall_defective == 1.0
    assert result.delta_metric_payload()["recall_defective"] == 1.0


def test_crop_first_comparison_reports_changed_test_groups():
    baseline_rows, crop_first_rows = _paired_predictions()
    result = compare_crop_first_to_baseline(
        baseline_rows,
        crop_first_rows,
        min_recall=1.0,
        thresholds=[0.30, 0.40, 0.44, 0.50, 0.55],
    )

    review_rows = result.review_rows_payload()
    assert len(review_rows) == 4
    assert {row["change_type"] for row in review_rows} == {
        "false_negative_recovered",
        "false_positive_removed",
    }
    assert result.leaderboard_row_payload()["delta_recall_defective"] == 1.0
    assert result.leaderboard_row_payload()["delta_false_negatives"] == -2
    assert result.summary_payload()["changed_prediction_count"] == 4


def test_crop_first_requires_apples_to_apples_prediction_frame():
    baseline_rows, crop_first_rows = _paired_predictions()
    crop_first_rows[0] = BaselinePrediction(
        image_path=crop_first_rows[0].image_path,
        group_id="different_group",
        split=crop_first_rows[0].split,
        label=crop_first_rows[0].label,
        score=crop_first_rows[0].score,
    )

    with pytest.raises(ValueError, match="same image, group, split, and label"):
        compare_crop_first_to_baseline(baseline_rows, crop_first_rows)


def test_full_dataset_crop_first_ab_is_deferred():
    with pytest.raises(NotImplementedError, match="Full-dataset crop-first"):
        run_sample_crop_first_ab(sample_mode=False)

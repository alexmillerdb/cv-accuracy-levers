import pytest

from levers.baseline import BaselinePrediction
from levers.threshold_tuning import (
    run_sample_threshold_tuning,
    tune_threshold_from_predictions,
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


def test_sample_threshold_tuning_is_deterministic():
    first = run_sample_threshold_tuning(split_seed=1, feature_seed=17)
    second = run_sample_threshold_tuning(split_seed=1, feature_seed=17)

    assert first.tuned_metric_payload() == second.tuned_metric_payload()
    assert first.validation_metric_payload() == second.validation_metric_payload()
    assert first.fixed_metric_payload() == second.fixed_metric_payload()
    assert first.validation_sweep_payload() == second.validation_sweep_payload()
    assert first.sample_size == 24
    assert first.train_size == 16
    assert first.val_size == 4
    assert first.test_size == 4


def test_threshold_tuning_compares_against_fixed_threshold():
    result = tune_threshold_from_predictions(
        [
            _prediction("val", 1, 0.45, name="val_pos_a"),
            _prediction("val", 1, 0.40, name="val_pos_b"),
            _prediction("val", 0, 0.42, name="val_neg_a"),
            _prediction("val", 0, 0.20, name="val_neg_b"),
            _prediction("test", 1, 0.45, name="test_pos_a"),
            _prediction("test", 1, 0.41, name="test_pos_b"),
            _prediction("test", 0, 0.44, name="test_neg_a"),
            _prediction("test", 0, 0.10, name="test_neg_b"),
        ],
        min_recall=1.0,
        thresholds=[0.3, 0.4, 0.43, 0.5],
        baseline_threshold=0.5,
    )

    assert result.selected_point.threshold == pytest.approx(0.4)
    assert result.tuned_test_metrics.recall_defective == 1.0
    assert result.fixed_test_metrics.recall_defective == 0.0
    assert result.delta_metric_payload()["recall_defective"] == 1.0
    assert result.delta_metric_payload()["precision_defective"] == pytest.approx(2 / 3)


def test_threshold_tuning_enforces_recall_floor():
    with pytest.raises(ValueError, match="no threshold satisfies min_recall"):
        tune_threshold_from_predictions(
            [
                _prediction("val", 1, 0.45, name="val_pos"),
                _prediction("val", 0, 0.20, name="val_neg"),
                _prediction("test", 1, 0.45, name="test_pos"),
                _prediction("test", 0, 0.20, name="test_neg"),
            ],
            min_recall=1.1,
            thresholds=[0.3, 0.5],
        )

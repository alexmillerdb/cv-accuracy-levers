import pytest

from levers.baseline import (
    build_sample_baseline_dataset,
    run_sample_baseline,
    train_centroid_baseline,
)
from levers.data import make_grouped_split, validate_group_splits


def test_sample_baseline_dataset_is_public_safe_and_grouped():
    records, features = build_sample_baseline_dataset()
    split_records = make_grouped_split(records, seed=1)

    validate_group_splits(split_records)
    assert features.shape == (len(records), 4)
    assert all(record.image_path.startswith("synthetic/") for record in records)
    assert {record.label for record in records} == {0, 1}


def test_run_sample_baseline_uses_validation_threshold_and_test_metrics():
    result = run_sample_baseline(split_seed=1, feature_seed=17)

    assert result.sample_size == 24
    assert result.group_count == 12
    assert result.train_size == 16
    assert result.val_size == 4
    assert result.test_size == 4
    assert result.threshold_point.metrics.recall_defective >= 0.75
    assert result.metrics.false_negatives >= 0
    assert result.metrics.auc_pr is not None
    assert result.metrics.auc_roc is not None


def test_run_sample_baseline_is_deterministic():
    first = run_sample_baseline(split_seed=1, feature_seed=17)
    second = run_sample_baseline(split_seed=1, feature_seed=17)

    assert first.metric_payload() == second.metric_payload()
    assert [row.score for row in first.predictions] == [
        row.score for row in second.predictions
    ]


def test_centroid_baseline_requires_both_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        train_centroid_baseline([[0.1], [0.2]], [1, 1])


def test_full_dataset_baseline_is_deferred():
    with pytest.raises(NotImplementedError, match="Full-dataset baseline"):
        run_sample_baseline(sample_mode=False)

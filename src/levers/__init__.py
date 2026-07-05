"""Shared helpers for the CV accuracy levers demo."""

from levers.data import DatasetRecord, make_grouped_split, validate_group_splits
from levers.eval import BinaryMetrics, binary_classification_metrics
from levers.config import ProjectConfig
from levers.crop_first import (
    CropFirstABResult,
    CropFirstArmResult,
    CropFirstReviewRow,
    build_crop_first_sample_features,
    compare_crop_first_to_baseline,
    run_sample_crop_first_ab,
)
from levers.error_review import (
    FalseNegativeReviewResult,
    FalseNegativeReviewRow,
    build_false_negative_review_rows,
    review_false_negatives_from_predictions,
    run_sample_false_negative_review,
)
from levers.ingest import (
    IngestResult,
    IngestedRecord,
    copy_images_to_uc_volume,
    normalize_manifest_rows,
    persist_ingest_to_uc,
)
from levers.label_quality import (
    EmbeddingRow,
    LabelQualityIssueRow,
    LabelQualityResult,
    build_sample_embedding_rows,
    find_label_quality_issues,
    run_sample_label_quality_embeddings,
)
from levers.thresholds import ThresholdPoint, threshold_sweep
from levers.threshold_tuning import (
    ThresholdTuningResult,
    run_sample_threshold_tuning,
    tune_threshold_from_predictions,
)

__all__ = [
    "BinaryMetrics",
    "CropFirstABResult",
    "CropFirstArmResult",
    "CropFirstReviewRow",
    "DatasetRecord",
    "EmbeddingRow",
    "FalseNegativeReviewResult",
    "FalseNegativeReviewRow",
    "IngestResult",
    "IngestedRecord",
    "LabelQualityIssueRow",
    "LabelQualityResult",
    "ProjectConfig",
    "ThresholdPoint",
    "ThresholdTuningResult",
    "binary_classification_metrics",
    "build_crop_first_sample_features",
    "build_false_negative_review_rows",
    "build_sample_embedding_rows",
    "copy_images_to_uc_volume",
    "compare_crop_first_to_baseline",
    "find_label_quality_issues",
    "make_grouped_split",
    "normalize_manifest_rows",
    "persist_ingest_to_uc",
    "review_false_negatives_from_predictions",
    "run_sample_false_negative_review",
    "run_sample_crop_first_ab",
    "run_sample_label_quality_embeddings",
    "run_sample_threshold_tuning",
    "threshold_sweep",
    "tune_threshold_from_predictions",
    "validate_group_splits",
]

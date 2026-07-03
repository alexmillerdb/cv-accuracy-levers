"""Dataset records and deterministic grouped split helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DatasetRecord:
    """Public dataset metadata used by the demo notebooks and tests."""

    image_path: str
    label: int
    group_id: str
    split: str | None = None
    defect_types: tuple[str, ...] = field(default_factory=tuple)
    bbox: tuple[float, float, float, float] | None = None
    comment_category: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "DatasetRecord":
        """Build a record from a dict-like row."""

        defect_types = row.get("defect_types", ())
        if defect_types is None:
            defect_types_tuple: tuple[str, ...] = ()
        elif isinstance(defect_types, str):
            defect_types_tuple = (defect_types,)
        else:
            defect_types_tuple = tuple(str(value) for value in defect_types)

        bbox = row.get("bbox")
        bbox_tuple = tuple(float(value) for value in bbox) if bbox is not None else None
        if bbox_tuple is not None and len(bbox_tuple) != 4:
            raise ValueError("bbox must contain exactly four numeric values")

        return cls(
            image_path=str(row["image_path"]),
            label=int(row["label"]),
            group_id=str(row["group_id"]),
            split=str(row["split"]) if row.get("split") is not None else None,
            defect_types=defect_types_tuple,
            bbox=bbox_tuple,
            comment_category=(
                str(row["comment_category"])
                if row.get("comment_category") is not None
                else None
            ),
        )


def _coerce_record(record: DatasetRecord | Mapping[str, object]) -> DatasetRecord:
    if isinstance(record, DatasetRecord):
        return record
    return DatasetRecord.from_mapping(record)


def make_grouped_split(
    records: Iterable[DatasetRecord | Mapping[str, object]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[DatasetRecord]:
    """Assign deterministic train/val/test splits without splitting groups."""

    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("at least one split ratio must be positive")

    coerced = [_coerce_record(record) for record in records]
    groups = sorted({record.group_id for record in coerced})
    rng = random.Random(seed)
    rng.shuffle(groups)

    train_cutoff = train_ratio / ratio_sum
    val_cutoff = (train_ratio + val_ratio) / ratio_sum
    split_by_group: dict[str, str] = {}
    total_groups = len(groups)

    for index, group_id in enumerate(groups):
        position = (index + 1) / total_groups if total_groups else 0
        if position <= train_cutoff:
            split = "train"
        elif position <= val_cutoff:
            split = "val"
        else:
            split = "test"
        split_by_group[group_id] = split

    return [
        replace(record, split=split_by_group[record.group_id])
        for record in coerced
    ]


def validate_group_splits(records: Sequence[DatasetRecord]) -> None:
    """Raise if the same group appears in multiple splits."""

    split_by_group: dict[str, str] = {}
    for record in records:
        if record.split is None:
            raise ValueError(f"record has no split: {record.image_path}")
        existing = split_by_group.setdefault(record.group_id, record.split)
        if existing != record.split:
            raise ValueError(
                f"group {record.group_id!r} appears in both {existing!r} "
                f"and {record.split!r}"
            )

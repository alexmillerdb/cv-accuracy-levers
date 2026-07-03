from levers.data import DatasetRecord, make_grouped_split, validate_group_splits


def test_grouped_split_keeps_groups_together():
    records = [
        DatasetRecord(image_path=f"{group}_{idx}.jpg", label=idx % 2, group_id=group)
        for group in ("a", "b", "c", "d", "e", "f")
        for idx in range(2)
    ]

    split_records = make_grouped_split(records, seed=7)

    validate_group_splits(split_records)
    assert {record.split for record in split_records} <= {"train", "val", "test"}
    assert len(split_records) == len(records)


def test_record_from_mapping_normalizes_optional_values():
    record = DatasetRecord.from_mapping(
        {
            "image_path": "x.jpg",
            "label": "1",
            "group_id": 123,
            "defect_types": ["crack", "hole"],
            "bbox": [1, 2, 3, 4],
            "comment_category": "missing_board",
        }
    )

    assert record.label == 1
    assert record.group_id == "123"
    assert record.defect_types == ("crack", "hole")
    assert record.bbox == (1.0, 2.0, 3.0, 4.0)
    assert record.comment_category == "missing_board"

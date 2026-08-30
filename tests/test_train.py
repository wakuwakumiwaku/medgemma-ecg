from __future__ import annotations

from pathlib import Path

import pytest

from medgemma_ecg.train import require_manifest_split


def test_require_manifest_split_accepts_expected_rows() -> None:
    require_manifest_split(
        [{"id": "a", "split": "train"}, {"id": "b", "split": "train"}],
        "train",
        Path("train.jsonl"),
    )


@pytest.mark.parametrize(
    ("expected_split", "actual_split"),
    [("train", "test"), ("validation", "train")],
)
def test_require_manifest_split_rejects_rows_from_another_split(
    expected_split: str, actual_split: str
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"{expected_split}\.jsonl must contain only {expected_split!r} rows; "
            rf"found {actual_split!r}"
        ),
    ):
        require_manifest_split(
            [{"id": "wrong", "split": actual_split}],
            expected_split,
            Path(f"{expected_split}.jsonl"),
        )


def test_require_manifest_split_rejects_empty_manifest() -> None:
    with pytest.raises(ValueError, match="train.jsonl is empty"):
        require_manifest_split([], "train", Path("train.jsonl"))

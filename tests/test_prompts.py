import numpy as np
import pandas as pd
import pytest

from medgemma_ecg.prepare_ptbxl import (
    normalize_lead_name,
    split_from_fold,
    statement_label_groups,
)
from medgemma_ecg.prompts import TARGET_KEYS, normalize_target, target_text


def test_recommended_ptbxl_folds() -> None:
    assert all(split_from_fold(fold) == "train" for fold in range(1, 9))
    assert split_from_fold(9) == "validation"
    assert split_from_fold(10) == "test"


@pytest.mark.parametrize("fold", [-1, 0, 11])
def test_invalid_ptbxl_folds_are_rejected(fold: int) -> None:
    with pytest.raises(ValueError, match="unexpected PTB-XL strat_fold"):
        split_from_fold(fold)


def test_wfdb_augmented_lead_names_are_normalized() -> None:
    assert normalize_lead_name("AVR") == "aVR"
    assert normalize_lead_name("avl") == "aVL"
    assert normalize_lead_name("V1") == "V1"


def test_zero_likelihood_scp_annotation_is_retained() -> None:
    statements = pd.DataFrame(
        {
            "diagnostic": [1.0, np.nan],
            "rhythm": [np.nan, 1.0],
            "form": [np.nan, np.nan],
        },
        index=["NORM", "SR"],
    )
    groups = statement_label_groups({"NORM": 100.0, "SR": 0.0}, statements)
    assert groups == {"diagnostic": ["NORM"], "rhythm": ["SR"], "form": []}


def test_target_has_stable_schema() -> None:
    target = normalize_target({"labels": ["MI", "MI"], "findings": None})
    assert tuple(target) == TARGET_KEYS
    assert target["labels"] == ["MI"]
    assert target["findings"] == []
    assert '"labels":["MI"]' in target_text(target)


def test_normalize_target_treats_string_fields_as_single_values() -> None:
    target = normalize_target({"labels": "MI", "findings": "ST elevation"})

    assert target["labels"] == ["MI"]
    assert target["findings"] == ["ST elevation"]

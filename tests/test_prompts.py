from medgemma_ecg.prepare_ptbxl import split_from_fold
from medgemma_ecg.prompts import TARGET_KEYS, normalize_target, target_text


def test_recommended_ptbxl_folds() -> None:
    assert all(split_from_fold(fold) == "train" for fold in range(1, 9))
    assert split_from_fold(9) == "validation"
    assert split_from_fold(10) == "test"


def test_target_has_stable_schema() -> None:
    target = normalize_target({"labels": ["MI", "MI"], "findings": None})
    assert tuple(target) == TARGET_KEYS
    assert target["labels"] == ["MI"]
    assert target["findings"] == []
    assert '"labels":["MI"]' in target_text(target)

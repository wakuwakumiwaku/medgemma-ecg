from medgemma_ecg.infer import extract_json


def test_extract_json_plain() -> None:
    assert extract_json('{"labels":["NORM"]}') == {"labels": ["NORM"]}


def test_extract_json_fence() -> None:
    assert extract_json('```json\n{"labels":[]}\n```') == {"labels": []}


def test_extract_json_rejects_prose() -> None:
    assert extract_json("Diagnosis: normal") is None

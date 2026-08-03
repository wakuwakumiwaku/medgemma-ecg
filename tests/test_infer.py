from medgemma_ecg.infer import extract_json


def test_extract_json_plain() -> None:
    assert extract_json('{"labels":["NORM"]}') == {"labels": ["NORM"]}


def test_extract_json_fence() -> None:
    assert extract_json('```json\n{"labels":[]}\n```') == {"labels": []}


def test_extract_json_medgemma_reasoning_envelope() -> None:
    text = (
        '<unused94>thought\nAnalyze the image first. '
        '<unused95>```json\n{"intervals":{"qrs":"normal"},"labels":["NORM"]}\n```'
    )
    assert extract_json(text) == {
        "intervals": {"qrs": "normal"},
        "labels": ["NORM"],
    }


def test_extract_json_embedded_object() -> None:
    assert extract_json('Answer: {"labels":["NORM"]} done') == {
        "labels": ["NORM"]
    }


def test_extract_json_rejects_prose() -> None:
    assert extract_json("Diagnosis: normal") is None

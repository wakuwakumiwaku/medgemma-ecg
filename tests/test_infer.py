import json

import pytest

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


@pytest.mark.parametrize("ending", ["", ",}"])
@pytest.mark.parametrize("wrapper", [
    "{}",
    "Answer: {} done",
    "```json\n{}\n```",
    "<unused94>thought\nAnalyze first.<unused95>{}",
])
def test_extract_json_rejects_nested_object_in_invalid_response(ending: str, wrapper: str) -> None:
    response = '{"intervals":{"qrs_ms":90},"labels":["NORM"]' + ending
    assert extract_json(wrapper.format(response)) is None


def test_extract_json_handles_braces_and_escaped_quotes_in_strings() -> None:
    response = {
        "impression": 'A literal {} and "quoted {text}" with a \\backslash',
        "intervals": {"qrs_ms": 90},
        "labels": ["NORM"],
    }
    text = json.dumps(response)
    assert extract_json(f"Answer: {text} done") == response
    assert extract_json(f"Answer: {text[:-1]}") is None


@pytest.mark.parametrize("quoted", ["{}", "{", 'A "quoted {brace}" and \\backslash'])
def test_extract_json_ignores_braces_in_quoted_prose(quoted: str) -> None:
    explanation = f"Explanation: {json.dumps(quoted)}."
    assert extract_json(explanation) is None
    assert extract_json(f'{explanation} Answer: {{"labels":[]}}') == {"labels": []}


def test_extract_json_skips_balanced_prose_braces() -> None:
    assert extract_json('Example: {not JSON}. Answer: {"labels":["NORM"]}') == {
        "labels": ["NORM"]
    }


def test_extract_json_prefers_largest_outer_object() -> None:
    assert extract_json('Example: {}. Answer: {"intervals":{"qrs_ms":90},"labels":[]}') == {
        "intervals": {"qrs_ms": 90},
        "labels": [],
    }

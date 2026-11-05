from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path

import pytest

from medgemma_ecg.evaluate import evaluate_rows, main
from medgemma_ecg.infer import extract_json
from medgemma_ecg.manifest import write_jsonl


def test_evaluation_import_provenance() -> None:
    source = Path(importlib.import_module("medgemma_ecg.evaluate").__file__).resolve()
    expected = os.environ.get("EXPECTED_SOURCE_ROOT")
    if expected:
        assert source.is_relative_to(Path(expected).resolve()), source


def references() -> list[dict]:
    return [
        {"id": "synthetic-positive", "patient_id": "synthetic-p1", "labels": ["SYNTHETIC"]},
        {"id": "synthetic-empty", "patient_id": "synthetic-p2", "labels": []},
    ]


def wrapped_prediction(raw: str) -> dict:
    """Synthetic fixture matching predict.main's documented output shape, not inference."""
    parsed = extract_json(raw)
    labels = parsed.get("labels", []) if parsed else []
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        labels = []
    return {"id": "synthetic-empty", "labels": sorted(set(labels)), "parsed": parsed, "raw": raw}


@pytest.mark.parametrize(
    "raw",
    [
        "Synthetic unparseable response, not a diagnosis",
        '{"intervals":{"qrs_ms":90},"labels":',
        '{"impression":"Synthetic missing label field"}',
        '{"labels":"SYNTHETIC"}',
        '{"labels":[123]}',
    ],
)
def test_predict_failure_wrappers_cannot_be_scored_as_valid_empty_sets(raw: str) -> None:
    reference_rows = references()
    predictions = [
        {"id": "synthetic-positive", "labels": ["SYNTHETIC"]},
        wrapped_prediction(raw),
    ]
    before = copy.deepcopy((reference_rows, predictions))
    assert predictions[1]["labels"] == []

    with pytest.raises(ValueError, match="prediction.*synthetic-empty"):
        evaluate_rows(reference_rows, predictions, bootstrap_samples=0)

    assert (reference_rows, predictions) == before


def test_parsed_and_exported_prediction_labels_must_agree() -> None:
    predictions = [
        {"id": "synthetic-positive", "labels": ["SYNTHETIC"]},
        {"id": "synthetic-empty", "labels": [], "parsed": {"labels": ["SYNTHETIC"]}},
    ]
    with pytest.raises(ValueError, match="prediction.*synthetic-empty.*differ"):
        evaluate_rows(references(), predictions, bootstrap_samples=0)


@pytest.mark.parametrize("wrapped", [False, True])
def test_explicit_empty_sets_and_normalized_labels_still_score(wrapped: bool) -> None:
    predictions = [
        {"id": "synthetic-positive", "labels": [" SYNTHETIC "]},
        {"id": "synthetic-empty", "labels": []},
    ]
    if wrapped:
        predictions[0]["parsed"] = {"labels": ["SYNTHETIC"]}
        predictions[1]["parsed"] = {"labels": []}
    report = evaluate_rows(references(), predictions, bootstrap_samples=0)
    assert report["exact_set_accuracy"] == 1.0
    assert report["sample_jaccard"] == 1.0
    assert report["per_label"]["SYNTHETIC"]["true_negative"] == 1


@pytest.mark.parametrize("existing", [False, True])
def test_cli_does_not_publish_metrics_for_unparsed_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture, existing: bool
) -> None:
    reference_path = tmp_path / "references.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    write_jsonl(reference_path, references())
    write_jsonl(
        prediction_path,
        [
            {"id": "synthetic-positive", "labels": ["SYNTHETIC"]},
            wrapped_prediction("Synthetic unparseable model-output fixture"),
        ],
    )
    previous = json.dumps({"previous": "synthetic report"}).encode()
    if existing:
        output.write_bytes(previous)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    with pytest.raises(SystemExit) as failure:
        main(
            [
                "--references",
                str(reference_path),
                "--predictions",
                str(prediction_path),
                "--output",
                str(output),
                "--bootstrap-samples",
                "0",
            ]
        )

    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "synthetic-empty" in captured.err
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

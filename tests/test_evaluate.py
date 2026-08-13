from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from medgemma_ecg.evaluate import _patient_bootstrap_indices, evaluate_rows, main
from medgemma_ecg.manifest import write_jsonl


def reference_rows() -> list[dict]:
    return [
        {"id": "a", "patient_id": "p1", "labels": ["A"]},
        {"id": "b", "patient_id": "p1", "labels": ["A", "B"]},
        {"id": "c", "patient_id": "p2", "labels": ["B"]},
        {"id": "d", "patient_id": "p3", "labels": []},
    ]


def prediction_rows() -> list[dict]:
    return [
        {"id": "a", "labels": ["A"]},
        {"id": "b", "labels": ["B"]},
        {"id": "c", "labels": ["A", "B"]},
        {"id": "d", "labels": []},
    ]


def test_evaluate_rows_reports_multilabel_metrics() -> None:
    report = evaluate_rows(reference_rows(), prediction_rows(), bootstrap_samples=0)

    assert report["samples"] == 4
    assert report["patients"] == 3
    assert report["labels"] == ["A", "B"]
    assert report["exact_set_accuracy"] == pytest.approx(0.5)
    assert report["hamming_loss"] == pytest.approx(0.25)
    assert report["micro"] == {
        "precision": pytest.approx(0.75),
        "recall": pytest.approx(0.75),
        "f1": pytest.approx(0.75),
    }
    assert report["per_label"]["A"]["support"] == 2
    assert report["per_label"]["A"]["specificity"] == pytest.approx(0.5)
    assert report["per_label"]["A"]["npv"] == pytest.approx(0.5)
    assert "confidence_intervals" not in report


def test_patient_bootstrap_is_reproducible() -> None:
    first = evaluate_rows(
        reference_rows(), prediction_rows(), bootstrap_samples=200, confidence_level=90, seed=17
    )
    second = evaluate_rows(
        reference_rows(), prediction_rows(), bootstrap_samples=200, confidence_level=90, seed=17
    )

    intervals = first["confidence_intervals"]
    assert intervals == second["confidence_intervals"]
    assert intervals["method"] == "patient-cluster percentile bootstrap"
    assert intervals["confidence_level"] == 90.0
    assert intervals["resamples"] == 200
    assert intervals["seed"] == 17
    exact = intervals["exact_set_accuracy"]
    assert 0 <= exact["lower"] <= exact["upper"] <= 1
    assert exact["valid_resamples"] == 200
    hamming = intervals["hamming_loss"]
    assert 0 <= hamming["lower"] <= hamming["upper"] <= 1
    assert hamming["valid_resamples"] == 200
    assert set(intervals["per_label"]) == {"A", "B"}
    assert intervals["per_label"]["A"]["recall"]["valid_resamples"] < 200


def test_patient_bootstrap_keeps_each_patients_records_together() -> None:
    indices = _patient_bootstrap_indices(["p1", "p1", "p2", "p3"], np.random.default_rng(4))
    counts = Counter(indices.tolist())

    assert counts[0] == counts[1]
    assert len(indices) >= 3


def test_perfect_predictions_keep_perfect_bootstrap_intervals() -> None:
    references = [
        {"id": "a", "patient_id": "p1", "labels": ["A"]},
        {"id": "b", "patient_id": "p2", "labels": ["B"]},
    ]
    predictions = [{"id": "a", "labels": ["A"]}, {"id": "b", "labels": ["B"]}]

    report = evaluate_rows(references, predictions, bootstrap_samples=100, seed=3)

    intervals = report["confidence_intervals"]
    for average in ("micro", "macro", "weighted"):
        assert intervals[average]["f1"]["lower"] == pytest.approx(1.0)
        assert intervals[average]["f1"]["upper"] == pytest.approx(1.0)
    for label in ("A", "B"):
        for metric in ("specificity", "npv"):
            assert intervals["per_label"][label][metric]["lower"] == pytest.approx(1.0)
            assert intervals["per_label"][label][metric]["upper"] == pytest.approx(1.0)


def test_evaluate_rows_rejects_duplicate_ids() -> None:
    references = reference_rows() + [reference_rows()[0]]

    with pytest.raises(ValueError, match="duplicate reference id"):
        evaluate_rows(references, prediction_rows(), bootstrap_samples=0)


def test_evaluate_cli_writes_bootstrap_report(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    write_jsonl(references, reference_rows())
    write_jsonl(predictions, prediction_rows())

    exit_code = main(
        [
            "--references",
            str(references),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--bootstrap-samples",
            "20",
            "--seed",
            "9",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["confidence_intervals"]["resamples"] == 20
    assert json.loads(capsys.readouterr().out) == report

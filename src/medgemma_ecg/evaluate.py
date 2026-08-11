from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from medgemma_ecg.manifest import MISSING_IDENTIFIER_VALUES, load_jsonl

AVERAGES = ("micro", "macro", "weighted")


def _labels(row: dict, source: str, sample_id: str) -> list[str]:
    values = row.get("labels", [])
    if not isinstance(values, list) or not all(
        isinstance(label, str) and label.strip() for label in values
    ):
        raise ValueError(f"{source} id {sample_id!r} has invalid labels")
    return sorted(set(values))


def _index_rows(rows: Sequence[dict], source: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        sample_id = "" if row.get("id") is None else str(row["id"]).strip()
        if sample_id.lower() in MISSING_IDENTIFIER_VALUES:
            raise ValueError(f"{source} row has an empty id")
        if sample_id in indexed:
            raise ValueError(f"duplicate {source} id {sample_id!r}")
        indexed[sample_id] = {**row, "labels": _labels(row, source, sample_id)}
    return indexed


def _score_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    *,
    zero_division: Any = 0,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        multilabel_confusion_matrix,
        precision_recall_fscore_support,
    )

    report: dict[str, Any] = {
        "exact_set_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for average in AVERAGES:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=zero_division
        )
        report[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=zero_division
    )
    confusion = multilabel_confusion_matrix(y_true, y_pred)
    per_label = {}

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float(zero_division)

    for index, label in enumerate(labels):
        true_negative, false_positive, false_negative, _ = confusion[index].ravel()
        per_label[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "specificity": ratio(true_negative, true_negative + false_positive),
            "npv": ratio(true_negative, true_negative + false_negative),
        }
    report["per_label"] = per_label
    return report


def _patient_bootstrap_indices(
    patient_ids: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        groups[patient_id].append(index)
    patients = sorted(groups)
    sampled = rng.choice(len(patients), size=len(patients), replace=True)
    return np.asarray(
        [index for patient_index in sampled for index in groups[patients[int(patient_index)]]],
        dtype=int,
    )


def _metric_paths(labels: Sequence[str]) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = [("exact_set_accuracy",)]
    paths.extend(
        (average, metric)
        for average in AVERAGES
        for metric in ("precision", "recall", "f1")
    )
    paths.extend(
        ("per_label", label, metric)
        for label in labels
        for metric in ("precision", "recall", "f1", "specificity", "npv")
    )
    return paths


def _metric_value(report: dict, path: tuple[str, ...]) -> float:
    value: Any = report
    for key in path:
        value = value[key]
    return float(value)


def _assign_interval(report: dict, path: tuple[str, ...], interval: dict[str, Any]) -> None:
    destination = report
    for key in path[:-1]:
        destination = destination.setdefault(key, {})
    destination[path[-1]] = interval


def _bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    patient_ids: Sequence[str],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    paths = _metric_paths(labels)
    samples = {path: [] for path in paths}

    for _ in range(bootstrap_samples):
        indices = _patient_bootstrap_indices(patient_ids, rng)
        replicate = _score_arrays(
            y_true[indices], y_pred[indices], labels, zero_division=np.nan
        )
        for path in paths:
            value = _metric_value(replicate, path)
            if np.isfinite(value):
                samples[path].append(value)

    tail = (100 - confidence_level) / 2
    intervals: dict[str, Any] = {
        "method": "patient-cluster percentile bootstrap",
        "confidence_level": confidence_level,
        "resamples": bootstrap_samples,
        "seed": seed,
    }
    for path, values in samples.items():
        if values:
            lower, upper = np.percentile(values, [tail, 100 - tail])
            interval = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_resamples": len(values),
            }
        else:
            interval = {"lower": None, "upper": None, "valid_resamples": 0}
        _assign_interval(
            intervals,
            path,
            interval,
        )
    return intervals


def evaluate_rows(
    reference_rows: Sequence[dict],
    prediction_rows: Sequence[dict],
    *,
    bootstrap_samples: int = 1000,
    confidence_level: float = 95.0,
    seed: int = 42,
) -> dict:
    """Score aligned multilabel predictions with patient-cluster bootstrap intervals."""

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if not 0 < confidence_level < 100:
        raise ValueError("confidence_level must be between 0 and 100")

    references = _index_rows(reference_rows, "reference")
    predictions = _index_rows(prediction_rows, "prediction")
    if not references:
        raise ValueError("references are empty")

    missing = sorted(set(references) - set(predictions))
    extra = sorted(set(predictions) - set(references))
    if missing or extra:
        raise ValueError(
            f"Prediction IDs do not match references. Missing={missing[:10]}, extra={extra[:10]}"
        )

    ids = sorted(references)
    patient_ids = []
    for sample_id in ids:
        patient_id = (
            ""
            if references[sample_id].get("patient_id") is None
            else str(references[sample_id]["patient_id"]).strip()
        )
        if patient_id.lower() in MISSING_IDENTIFIER_VALUES:
            raise ValueError(f"reference id {sample_id!r} has an empty patient_id")
        patient_ids.append(patient_id)

    y_true_sets = [references[sample_id]["labels"] for sample_id in ids]
    y_pred_sets = [predictions[sample_id]["labels"] for sample_id in ids]
    labels = sorted({label for values in y_true_sets + y_pred_sets for label in values})
    if not labels:
        raise ValueError("No labels were found in references or predictions")

    from sklearn.preprocessing import MultiLabelBinarizer

    encoder = MultiLabelBinarizer(classes=labels)
    encoder.fit([labels])
    y_true = encoder.transform(y_true_sets)
    y_pred = encoder.transform(y_pred_sets)

    report: dict[str, Any] = {
        "samples": len(ids),
        "patients": len(set(patient_ids)),
        "labels": labels,
        **_score_arrays(y_true, y_pred, labels),
    }
    if bootstrap_samples:
        report["confidence_intervals"] = _bootstrap_intervals(
            y_true,
            y_pred,
            labels,
            patient_ids,
            bootstrap_samples=bootstrap_samples,
            confidence_level=float(confidence_level),
            seed=seed,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score multilabel ECG predictions")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Patient-cluster bootstrap resamples; use 0 to disable intervals",
    )
    parser.add_argument("--confidence-level", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate_rows(
            load_jsonl(args.references),
            load_jsonl(args.predictions),
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

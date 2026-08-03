from __future__ import annotations

import argparse
import json
from pathlib import Path

from medgemma_ecg.manifest import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Score multilabel ECG predictions")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import MultiLabelBinarizer

    reference_rows = load_jsonl(args.references)
    prediction_rows = load_jsonl(args.predictions)
    references = {str(row["id"]): sorted(set(row["labels"])) for row in reference_rows}
    predictions = {str(row["id"]): sorted(set(row.get("labels", []))) for row in prediction_rows}

    missing = sorted(set(references) - set(predictions))
    extra = sorted(set(predictions) - set(references))
    if missing or extra:
        raise SystemExit(
            f"Prediction IDs do not match references. Missing={missing[:10]}, extra={extra[:10]}"
        )

    ids = sorted(references)
    y_true_sets = [references[sample_id] for sample_id in ids]
    y_pred_sets = [predictions[sample_id] for sample_id in ids]
    labels = sorted({label for values in y_true_sets + y_pred_sets for label in values})
    encoder = MultiLabelBinarizer(classes=labels)
    encoder.fit([labels])
    y_true = encoder.transform(y_true_sets)
    y_pred = encoder.transform(y_pred_sets)

    report: dict[str, object] = {
        "samples": len(ids),
        "labels": labels,
        "exact_set_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for average in ("micro", "macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        report[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    report["per_label"] = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

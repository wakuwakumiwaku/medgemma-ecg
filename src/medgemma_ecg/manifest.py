from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

ALLOWED_SPLITS = {"train", "validation", "test"}
REQUIRED_FIELDS = {"id", "patient_id", "split", "image", "labels", "target", "source"}
MISSING_IDENTIFIER_VALUES = {"", "nan", "none", "null"}


class ManifestError(ValueError):
    pass


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{line_number}: row must be a JSON object")
            row["_manifest"] = str(path)
            row["_line"] = line_number
            rows.append(row)
    return rows


def resolve_image_path(row: dict, data_root: str | Path = ".") -> Path:
    image = Path(str(row["image"]))
    return image if image.is_absolute() else Path(data_root) / image


def validate_rows(
    rows: Iterable[dict], data_root: str | Path = ".", require_images: bool = True
) -> dict:
    errors: list[str] = []
    seen_ids: set[str] = set()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)

    for index, row in enumerate(rows, start=1):
        location = f"{row.get('_manifest', '<rows>')}:{row.get('_line', index)}"
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{location}: missing fields {sorted(missing)}")
            continue

        sample_id = "" if row["id"] is None else str(row["id"]).strip()
        patient_id = (
            "" if row["patient_id"] is None else str(row["patient_id"]).strip()
        )
        split = str(row["split"])
        if sample_id.lower() in MISSING_IDENTIFIER_VALUES:
            errors.append(f"{location}: id must be non-empty")
        if patient_id.lower() in MISSING_IDENTIFIER_VALUES:
            errors.append(f"{location}: patient_id must be non-empty")
        if sample_id in seen_ids:
            errors.append(f"{location}: duplicate id {sample_id!r}")
        seen_ids.add(sample_id)

        if split not in ALLOWED_SPLITS:
            errors.append(f"{location}: invalid split {split!r}")
        else:
            split_counts[split] += 1
            patient_splits[patient_id].add(split)

        labels = row["labels"]
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            errors.append(f"{location}: labels must be a list of strings")
        else:
            for label in set(labels):
                label_counts[label] += 1

        target = row["target"]
        if not isinstance(target, dict):
            errors.append(f"{location}: target must be an object")
        elif sorted(target.get("labels", [])) != sorted(labels):
            errors.append(f"{location}: labels and target.labels differ")

        if require_images:
            image_path = resolve_image_path(row, data_root)
            if not image_path.is_file():
                errors.append(f"{location}: missing image {image_path}")

    leaked = {patient: splits for patient, splits in patient_splits.items() if len(splits) > 1}
    for patient, splits in sorted(leaked.items()):
        errors.append(f"patient {patient!r} appears in multiple splits: {sorted(splits)}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n- ... and {len(errors) - 30} more"
        raise ManifestError(
            f"Manifest validation failed with {len(errors)} error(s):\n{preview}{suffix}"
        )

    return {
        "samples": len(seen_ids),
        "patients": len(patient_splits),
        "splits": dict(sorted(split_counts.items())),
        "labels": dict(sorted(label_counts.items())),
        "patient_leakage": 0,
    }


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ECG JSONL manifests")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    for manifest in args.manifests:
        rows.extend(load_jsonl(manifest))
    report = validate_rows(rows, args.data_root, require_images=not args.skip_image_check)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

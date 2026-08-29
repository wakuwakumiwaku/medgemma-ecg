from __future__ import annotations

import json
from pathlib import Path

import pytest

from medgemma_ecg.manifest import ManifestError, validate_rows, write_jsonl


def sample(sample_id: str, patient: str, split: str, image: Path, label: str = "NORM") -> dict:
    return {
        "id": sample_id,
        "patient_id": patient,
        "split": split,
        "image": str(image),
        "labels": [label],
        "target": {"labels": [label]},
        "source": "test",
    }


def test_manifest_accepts_disjoint_patients(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"image")
    image_b.write_bytes(b"image")
    report = validate_rows(
        [sample("a", "p1", "train", image_a), sample("b", "p2", "test", image_b)]
    )
    assert report["samples"] == 2
    assert report["patient_leakage"] == 0


def test_manifest_rejects_patient_leakage(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    rows = [sample("a", "p1", "train", image), sample("b", "p1", "test", image)]
    with pytest.raises(ManifestError, match="multiple splits"):
        validate_rows(rows)


def test_missing_patient_identifier_is_rejected(tmp_path: Path) -> None:
    image = tmp_path / "ecg.png"
    image.write_bytes(b"x")
    row = sample("a", "patient-1", "train", image)
    row["patient_id"] = None
    with pytest.raises(ManifestError, match="patient_id must be non-empty"):
        validate_rows([row])


def test_manifest_rejects_label_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    row = sample("a", "p1", "train", image)
    row["target"]["labels"] = ["MI"]
    with pytest.raises(ManifestError, match="target.labels"):
        validate_rows([row])


@pytest.mark.parametrize("field", ["labels", "target.labels"])
@pytest.mark.parametrize("label", ["", " \t"])
def test_manifest_rejects_empty_labels(tmp_path: Path, field: str, label: str) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    row = sample("a", "p1", "train", image)
    destination = row if field == "labels" else row["target"]
    destination["labels"] = [label]

    with pytest.raises(
        ManifestError, match=rf"{field} must be a list of non-empty strings"
    ):
        validate_rows([row])


@pytest.mark.parametrize("field", ["labels", "target.labels"])
def test_manifest_rejects_duplicate_normalized_labels(tmp_path: Path, field: str) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    row = sample("a", "p1", "train", image)
    destination = row if field == "labels" else row["target"]
    destination["labels"] = ["NORM", " NORM "]

    with pytest.raises(
        ManifestError, match=rf"{field} contains duplicates after trimming"
    ):
        validate_rows([row])


@pytest.mark.parametrize("target_labels", [None, "MI", [1]])
def test_manifest_rejects_malformed_target_labels(
    tmp_path: Path, target_labels: object
) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    row = sample("a", "p1", "train", image)
    row["target"]["labels"] = target_labels

    with pytest.raises(ManifestError, match="target.labels must be a list of non-empty strings"):
        validate_rows([row])


def test_write_jsonl_removes_internal_metadata(tmp_path: Path) -> None:
    output = tmp_path / "manifest.jsonl"
    write_jsonl(output, [{"id": "a", "_line": 3}])
    assert json.loads(output.read_text(encoding="utf-8")) == {"id": "a"}

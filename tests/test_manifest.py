from __future__ import annotations

import io
import json
import os
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


@pytest.mark.parametrize(
    ("labels", "target_labels"),
    [
        ([" NORM "], ["NORM"]),
        (["NORM"], [" NORM "]),
        ([" NORM "], [" NORM "]),
    ],
)
def test_manifest_normalizes_validated_labels(
    tmp_path: Path, labels: list[str], target_labels: list[str]
) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    row = sample("a", "p1", "train", image)
    row["labels"] = labels
    row["target"]["labels"] = target_labels

    report = validate_rows([row])

    assert row["labels"] == ["NORM"]
    assert row["target"]["labels"] == ["NORM"]
    assert report["labels"] == {"NORM": 1}


def test_write_jsonl_removes_internal_metadata(tmp_path: Path) -> None:
    output = tmp_path / "manifest.jsonl"
    write_jsonl(output, [{"id": "a", "_line": 3}])
    assert json.loads(output.read_text(encoding="utf-8")) == {"id": "a"}


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["serialization", "iteration"])
def test_write_jsonl_preserves_output_on_row_failure(
    tmp_path: Path, existing: bool, failure: str
) -> None:
    output = tmp_path / "manifest.jsonl"
    previous = b'{"id": "previous", "labels": ["SYNTHETIC"]}\n'
    if existing:
        output.write_bytes(previous)
    before = set(tmp_path.iterdir())

    def rows():
        yield {"id": "first", "labels": ["SYNTHETIC_OTHER"]}
        if failure == "iteration":
            raise RuntimeError("synthetic row producer failed")
        yield {"id": "second", "unserializable": object()}

    expected_error = RuntimeError if failure == "iteration" else TypeError
    with pytest.raises(expected_error):
        write_jsonl(output, rows())

    assert (output.read_bytes() if output.exists() else None) == (
        previous if existing else None
    )
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("existing", [False, True])
def test_write_jsonl_preserves_output_on_stream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    output = tmp_path / "manifest.jsonl"
    previous = b'{"id": "previous"}\n'
    if existing:
        output.write_bytes(previous)
    before = set(tmp_path.iterdir())
    original_open = io.open

    class FailingWriter:
        def __init__(self, handle):
            self.handle = handle
            self.writes = 0

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, value):
            if self.writes:
                raise OSError("synthetic stream failure")
            self.writes += 1
            return self.handle.write(value)

    def failing_open(file, mode="r", *args, **kwargs):
        handle = original_open(file, mode, *args, **kwargs)
        return FailingWriter(handle) if mode == "w" else handle

    monkeypatch.setattr(io, "open", failing_open)
    with pytest.raises(OSError, match="synthetic stream failure"):
        write_jsonl(output, [{"id": "first"}, {"id": "second"}])

    assert (output.read_bytes() if output.exists() else None) == (
        previous if existing else None
    )
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("existing", [False, True])
def test_write_jsonl_preserves_output_on_replacement_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    output = tmp_path / "manifest.jsonl"
    previous = b'{"id": "previous"}\n'
    if existing:
        output.write_bytes(previous)
    before = set(tmp_path.iterdir())

    def failed_replace(*args, **kwargs):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr("os.replace", failed_replace)
    with pytest.raises(OSError, match="synthetic publication failure"):
        write_jsonl(output, [{"id": "replacement"}])

    assert (output.read_bytes() if output.exists() else None) == (
        previous if existing else None
    )
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("existing", [False, True])
def test_write_jsonl_streams_and_replaces_complete_utf8_output(
    tmp_path: Path, existing: bool
) -> None:
    output = tmp_path / "manifest.jsonl"
    if existing:
        output.write_bytes(b'{"id": "previous"}\n')
    observed = []
    values = [
        {"id": "synthetic-ä", "_line": 1, "labels": ["SYNTHETIC"]},
        {"id": "synthetic-日本語", "_manifest": "private fixture", "labels": []},
    ]

    def rows():
        for index, value in enumerate(values):
            observed.append(index)
            yield value

    write_jsonl(output, rows())

    assert observed == [0, 1]
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"id": "synthetic-ä", "labels": ["SYNTHETIC"]},
        {"id": "synthetic-日本語", "labels": []},
    ]
    assert values[0]["_line"] == 1
    assert values[1]["_manifest"] == "private fixture"
    assert set(tmp_path.iterdir()) == {output}


def test_write_jsonl_accepts_empty_iterable(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manifest.jsonl"
    write_jsonl(output, iter(()))
    assert output.read_bytes() == b""
    assert set(output.parent.iterdir()) == {output}


def test_write_jsonl_keeps_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(b'{"id": "previous"}\n')
    output = tmp_path / "linked.jsonl"
    output.symlink_to(target.name)

    write_jsonl(output, [{"id": "replacement"}])

    assert output.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"id": "replacement"}
    assert set(tmp_path.iterdir()) == {target, output}


def test_write_jsonl_keeps_existing_file_permissions(tmp_path: Path) -> None:
    output = tmp_path / "manifest.jsonl"
    output.write_bytes(b'{"id": "previous"}\n')
    output.chmod(0o640)
    before = output.stat().st_mode

    write_jsonl(output, [{"id": "replacement"}])

    assert output.stat().st_mode == before
    assert json.loads(output.read_text(encoding="utf-8")) == {"id": "replacement"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX write-permission control")
def test_write_jsonl_refuses_read_only_destination(tmp_path: Path) -> None:
    output = tmp_path / "manifest.jsonl"
    previous = b'{"id": "previous"}\n'
    output.write_bytes(previous)
    output.chmod(0o444)
    if os.access(output, os.W_OK):
        pytest.skip("The effective user can write a mode-444 file")

    with pytest.raises(PermissionError):
        write_jsonl(output, [{"id": "replacement"}])

    assert output.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {output}

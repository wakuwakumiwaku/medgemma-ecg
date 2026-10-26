from __future__ import annotations

import json
from pathlib import Path

import pytest

from medgemma_ecg.evaluate import evaluate_rows, main

REFERENCE_BYTES = (
    b'{ "id": "synthetic-a", "patient_id": "synthetic-group", "labels": ["A"] }\n\n'
    b'{"id":"synthetic-b","patient_id":"synthetic-group","labels":["B"]}\n'
)
PREDICTION_BYTES = (
    b'{"id":"synthetic-a","labels":["A"]}\n'
    b'{ "id": "synthetic-b", "labels": [] }\n'
)


def manifests(tmp_path: Path) -> tuple[Path, Path]:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    references.write_bytes(REFERENCE_BYTES)
    predictions.write_bytes(PREDICTION_BYTES)
    return references, predictions


@pytest.mark.parametrize("input_name", ["references", "predictions"])
@pytest.mark.parametrize(
    "alias",
    ["exact", "relative", "dotdot", "missing-parent-dotdot", "symlink", "input-symlink",
     "hardlink", "parent-symlink"],
)
def test_evaluate_output_refuses_input_alias_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch,
    input_name: str, alias: str,
) -> None:
    references, predictions = manifests(tmp_path)
    target = references if input_name == "references" else predictions
    output = target
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path(target.name)
    elif alias in {"dotdot", "missing-parent-dotdot"}:
        parent = tmp_path / "intermediate"
        if alias == "dotdot":
            parent.mkdir()
        output = parent / ".." / target.name
    elif alias == "input-symlink":
        output = tmp_path / "stored.jsonl"
        target.rename(output)
        target.symlink_to(output.name)
    elif alias in {"symlink", "hardlink"}:
        output = tmp_path / "metrics.json"
        if alias == "symlink":
            output.symlink_to(target.name)
        else:
            output.hardlink_to(target)
    elif alias == "parent-symlink":
        parent = tmp_path / "linked-directory"
        parent.symlink_to(tmp_path, target_is_directory=True)
        output = parent / target.name
    before = set(tmp_path.iterdir())

    try:
        status = main([
            "--references", str(references), "--predictions", str(predictions),
            "--output", str(output), "--bootstrap-samples", "0",
        ])
    except SystemExit as error:
        status = error.code

    assert references.read_bytes() == REFERENCE_BYTES, "reference manifest was overwritten"
    assert predictions.read_bytes() == PREDICTION_BYTES, "prediction manifest was overwritten"
    assert set(tmp_path.iterdir()) == before, "refusal created output parent directories"
    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--output" in captured.err
    assert f"--{input_name}" in captured.err
    assert "same file" in captured.err
    assert str(output) in captured.err
    assert str(target) in captured.err
    if alias == "symlink":
        assert output.is_symlink()
    elif alias == "input-symlink":
        assert target.is_symlink()


@pytest.mark.parametrize(
    "destination", ["new", "replace", "same-content", "shared-input", "stdout"]
)
def test_evaluate_output_keeps_safe_destinations(
    tmp_path: Path, capsys: pytest.CaptureFixture, destination: str
) -> None:
    references, predictions = manifests(tmp_path)
    prediction_path = references if destination == "shared-input" else predictions
    output = tmp_path / "reports" / "metrics.json"
    if destination in {"replace", "same-content"}:
        output.parent.mkdir()
        output.write_bytes(REFERENCE_BYTES if destination == "same-content" else b"old report\n")
    argv = [
        "--references", str(references), "--predictions", str(prediction_path),
        "--bootstrap-samples", "0",
    ]
    if destination != "stdout":
        argv.extend(["--output", str(output)])
    reference_rows = [json.loads(line) for line in REFERENCE_BYTES.splitlines() if line.strip()]
    prediction_rows = (
        reference_rows if destination == "shared-input"
        else [json.loads(line) for line in PREDICTION_BYTES.splitlines()]
    )
    expected = evaluate_rows(reference_rows, prediction_rows, bootstrap_samples=0)

    assert main(argv) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == expected
    assert references.read_bytes() == REFERENCE_BYTES
    assert predictions.read_bytes() == PREDICTION_BYTES
    if destination == "stdout":
        assert not output.parent.exists()
    else:
        assert output.read_text(encoding="utf-8") == captured.out
        assert json.loads(output.read_text(encoding="utf-8")) == expected

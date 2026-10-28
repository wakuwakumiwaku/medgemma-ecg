from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from medgemma_ecg.evaluate import main

REFERENCE_BYTES = b'{"id":"synthetic-a","patient_id":"synthetic-p","labels":["A"]}\n'
PREDICTION_BYTES = b'{"id":"synthetic-a","labels":["A"]}\n'
PREVIOUS_REPORT = b'{"previous":"synthetic experiment"}\n'


@pytest.fixture
def evaluation_args(tmp_path: Path) -> list[str]:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    references.write_bytes(REFERENCE_BYTES)
    predictions.write_bytes(PREDICTION_BYTES)
    return [
        "--references", str(references), "--predictions", str(predictions),
        "--bootstrap-samples", "0",
    ]


def report_destination(tmp_path: Path, existing: bool, symlink: bool) -> tuple[Path, Path]:
    directory = tmp_path / "reports"
    directory.mkdir()
    target = directory / "metrics.json"
    if existing:
        target.write_bytes(PREVIOUS_REPORT)
    output = directory / "linked.json" if symlink else target
    if symlink:
        output.symlink_to(target.name)
    return output, target


def assert_inputs_unchanged(tmp_path: Path) -> None:
    assert (tmp_path / "references.jsonl").read_bytes() == REFERENCE_BYTES
    assert (tmp_path / "predictions.jsonl").read_bytes() == PREDICTION_BYTES


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("symlink", [False, True])
@pytest.mark.parametrize("failure", ["write", "close"])
def test_evaluate_preserves_report_on_stream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    evaluation_args: list[str], existing: bool, symlink: bool, failure: str,
) -> None:
    output, target = report_destination(tmp_path, existing, symlink)
    before = set(output.parent.iterdir())
    original_open = io.open
    reached = []

    class FailingWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            result = self.handle.__exit__(*args)
            if failure == "close":
                reached.append("close")
                raise OSError("synthetic report close failure")
            return result

        def write(self, value):
            if failure == "write":
                # Exercise actual truncation/partial bytes, not a pre-open exception.
                self.handle.write(value[: len(value) // 2])
                self.handle.flush()
                reached.append("write")
                raise OSError("synthetic report write failure")
            return self.handle.write(value)

    def failing_open(file, mode="r", *args, **kwargs):
        handle = original_open(file, mode, *args, **kwargs)
        return FailingWriter(handle) if mode == "w" else handle

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", failing_open)
        with pytest.raises(OSError, match=f"synthetic report {failure} failure"):
            main([*evaluation_args, "--output", str(output)])

    assert reached == [failure]
    assert (target.read_bytes() if target.exists() else None) == (
        PREVIOUS_REPORT if existing else None
    ), "failed report write must preserve the previous report or leave no first output"
    assert set(output.parent.iterdir()) == before
    assert_inputs_unchanged(tmp_path)
    if symlink:
        assert output.is_symlink()

    capsys.readouterr()
    assert main([*evaluation_args, "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert target.read_text(encoding="utf-8") == captured.out
    assert json.loads(captured.out)["exact_set_accuracy"] == 1.0
    assert set(output.parent.iterdir()) == before | {target}
    assert_inputs_unchanged(tmp_path)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("symlink", [False, True])
def test_evaluate_preserves_report_on_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    evaluation_args: list[str], existing: bool, symlink: bool,
) -> None:
    output, target = report_destination(tmp_path, existing, symlink)
    before = set(output.parent.iterdir())
    reached = []

    def failed_replace(source, destination):
        staged = Path(source)
        assert staged != target
        assert staged.parent.parent == target.parent
        assert Path(destination) == target
        assert json.loads(staged.read_text(encoding="utf-8"))["samples"] == 1
        reached.append(staged.read_bytes())
        raise OSError("synthetic report publication failure")

    error = None
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failed_replace)
        try:
            main([*evaluation_args, "--output", str(output)])
        except OSError as exc:
            error = exc

    assert (target.read_bytes() if target.exists() else None) == (
        PREVIOUS_REPORT if existing else None
    ), "failed report publication must preserve the previous report or leave no first output"
    assert error is not None and "synthetic report publication failure" in str(error)
    assert reached == [capsys.readouterr().out.encode("utf-8")]
    assert set(output.parent.iterdir()) == before
    assert_inputs_unchanged(tmp_path)
    if symlink:
        assert output.is_symlink()

    assert main([*evaluation_args, "--output", str(output)]) == 0
    assert target.read_bytes() == reached[0]
    assert set(output.parent.iterdir()) == before | {target}
    assert_inputs_unchanged(tmp_path)


@pytest.mark.parametrize("symlink", [False, True])
def test_evaluate_publication_preserves_mode_and_symlink(
    tmp_path: Path, capsys: pytest.CaptureFixture, evaluation_args: list[str], symlink: bool,
) -> None:
    output, target = report_destination(tmp_path, existing=True, symlink=symlink)
    target.chmod(0o640)
    mode = target.stat().st_mode
    before = set(output.parent.iterdir())

    assert main([*evaluation_args, "--output", str(output)]) == 0

    assert target.stat().st_mode == mode
    assert target.read_text(encoding="utf-8") == capsys.readouterr().out
    assert set(output.parent.iterdir()) == before
    assert_inputs_unchanged(tmp_path)
    if symlink:
        assert output.is_symlink()


def test_evaluate_publication_replaces_only_the_report_hardlink(
    tmp_path: Path, evaluation_args: list[str],
) -> None:
    output, target = report_destination(tmp_path, existing=True, symlink=False)
    old_report = output.parent / "previous-experiment.json"
    old_report.hardlink_to(target)

    assert main([*evaluation_args, "--output", str(output)]) == 0

    assert old_report.read_bytes() == PREVIOUS_REPORT
    assert json.loads(target.read_text(encoding="utf-8"))["samples"] == 1
    assert not target.samefile(old_report)
    assert set(output.parent.iterdir()) == {target, old_report}
    assert_inputs_unchanged(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX write-permission control")
def test_evaluate_publication_refuses_read_only_report(
    tmp_path: Path, evaluation_args: list[str],
) -> None:
    output, target = report_destination(tmp_path, existing=True, symlink=False)
    target.chmod(0o444)
    if os.access(target, os.W_OK):
        pytest.skip("The effective user can write a mode-444 file")

    with pytest.raises(PermissionError):
        main([*evaluation_args, "--output", str(output)])

    assert target.read_bytes() == PREVIOUS_REPORT
    assert set(output.parent.iterdir()) == {target}
    assert_inputs_unchanged(tmp_path)

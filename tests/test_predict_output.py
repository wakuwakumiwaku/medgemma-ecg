from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

import pytest

from medgemma_ecg import predict


class ModelBoundary(RuntimeError):
    pass


def test_import_provenance() -> None:
    if root := os.environ.get("EXPECTED_SOURCE_ROOT"):
        assert Path(predict.__file__).resolve().is_relative_to(Path(root).resolve())


@pytest.fixture
def no_model_import(monkeypatch):
    original = builtins.__import__

    def stop_at_model(name, *args, **kwargs):
        if name == "torch":
            raise ModelBoundary("model loading deliberately not exercised")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stop_at_model)


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink", "missing-parent"])
def test_prediction_refuses_input_alias_before_loading_model(
    tmp_path, monkeypatch, capsys, no_model_import, alias
):
    manifest = tmp_path / "reference.jsonl"
    previous = b'{"id":"one","patient_id":"patient","labels":["A"]}\n'
    manifest.write_bytes(previous)
    output = manifest
    if alias == "symlink":
        output = tmp_path / "alias.jsonl"
        output.symlink_to(manifest.name)
    elif alias == "hardlink":
        output = tmp_path / "hardlink.jsonl"
        output.hardlink_to(manifest)
    elif alias == "missing-parent":
        output = tmp_path / "missing" / ".." / manifest.name
    monkeypatch.setattr(sys, "argv", ["predict", "--manifest", str(manifest),
                                      "--output", str(output)])
    error = None
    try:
        predict.main()
    except (SystemExit, ModelBoundary) as exc:
        error = exc
    assert isinstance(error, SystemExit), "input alias was admitted to model loading"
    assert error.code == 2
    assert "same file" in capsys.readouterr().err
    assert manifest.read_bytes() == previous
    assert not (tmp_path / "missing").exists()


def test_distinct_output_is_admitted_to_model_boundary(tmp_path, monkeypatch, no_model_import):
    manifest = tmp_path / "reference.jsonl"
    manifest.write_text('{"id":"one","labels":["A"]}\n')
    output = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(sys, "argv", ["predict", "--manifest", str(manifest),
                                      "--output", str(output)])
    with pytest.raises(ModelBoundary, match="deliberately not exercised"):
        predict.main()
    assert not output.exists()

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb
from PIL import Image

from medgemma_ecg.manifest import load_jsonl, resolve_image_path
from medgemma_ecg.prepare_ptbxl import CANONICAL_LEADS, main, render_ecg


def test_prepare_import_provenance() -> None:
    source = Path(importlib.import_module("medgemma_ecg.prepare_ptbxl").__file__).resolve()
    expected = os.environ.get("EXPECTED_SOURCE_ROOT")
    if expected:
        assert source.is_relative_to(Path(expected).resolve()), source


def write_source(root: Path, frequency: int = 1, record: str = "synthetic") -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ecg_id": 1,
                "patient_id": "synthetic-patient",
                "strat_fold": 1,
                "filename_lr": f"records100/{record}",
                "scp_codes": "{'SR': 0.0}",
                "report": "Synthetic test waveform only",
            }
        ]
    ).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame({"diagnostic": [0], "rhythm": [1], "form": [0]}, index=["SR"]).to_csv(
        root / "scp_statements.csv"
    )
    directory = root / "records100"
    directory.mkdir(exist_ok=True)
    time = np.arange(1000) / 100
    signal = np.tile(np.sin(2 * np.pi * frequency * time)[:, None], (1, 12))
    wfdb.wrsamp(
        record_name=record,
        fs=100,
        units=["mV"] * 12,
        sig_name=CANONICAL_LEADS,
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(directory),
    )


def prepare(monkeypatch: pytest.MonkeyPatch, root: Path, output: Path) -> dict:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medgemma-ecg-prepare-ptbxl",
            "--ptbxl-root",
            str(root),
            "--output-root",
            str(output),
            "--sampling-rate",
            "100",
        ],
    )
    main()
    rows = load_jsonl(output / "train.jsonl")
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("change", ["source-root", "record-path"])
def test_cache_is_bound_to_source_waveform_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    root = tmp_path / "source-a"
    output = tmp_path / "processed"
    write_source(root, 1)
    first = prepare(monkeypatch, root, output)
    first_image = resolve_image_path(first)
    first_bytes = first_image.read_bytes()

    second_root = tmp_path / "source-b" if change == "source-root" else root
    record_name = "synthetic" if change == "source-root" else "replacement"
    write_source(second_root, 3, record_name)
    record = wfdb.rdrecord(str(second_root / f"records100/{record_name}"))
    expected = tmp_path / "expected.png"
    render_ecg(record.p_signal, record.fs, record.sig_name, expected)
    reads = []
    original_read = wfdb.rdrecord

    def observed_read(*args, **kwargs):
        reads.append(args[0])
        return original_read(*args, **kwargs)

    monkeypatch.setattr(wfdb, "rdrecord", observed_read)
    second = prepare(monkeypatch, second_root, output)

    with Image.open(resolve_image_path(second)) as actual, Image.open(expected) as reference:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
    assert reads == [str(second_root / f"records100/{record_name}")]
    assert second["id"] == first["id"] == "ptbxl-00001"
    assert second["labels"] == first["labels"] == ["SR"]
    assert second["record"] == f"records100/{record_name}"
    assert resolve_image_path(second) != first_image
    assert first_image.read_bytes() == first_bytes


def test_resolved_source_alias_reuses_matching_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    write_source(root)
    output = tmp_path / "processed"
    first = prepare(monkeypatch, root, output)
    alias = tmp_path / "source-alias"
    alias.symlink_to(root, target_is_directory=True)

    def unexpected_read(*args, **kwargs):
        pytest.fail("A resolved alias must reuse the same source-specific image")

    monkeypatch.setattr(wfdb, "rdrecord", unexpected_read)
    second = prepare(monkeypatch, alias, output)
    assert second["image"] == first["image"]


def test_legacy_rate_only_cache_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    write_source(root)
    output = tmp_path / "processed"
    legacy = output / "images" / "100hz" / "train" / "ptbxl-00001.png"
    legacy.parent.mkdir(parents=True)
    previous = b"image with unknown source provenance"
    legacy.write_bytes(previous)

    row = prepare(monkeypatch, root, output)

    assert resolve_image_path(row) != legacy
    assert legacy.read_bytes() == previous
    with Image.open(resolve_image_path(row)) as image:
        image.verify()

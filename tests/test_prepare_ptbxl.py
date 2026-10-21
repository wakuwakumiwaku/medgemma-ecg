from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb
from PIL import Image

from medgemma_ecg.manifest import load_jsonl, resolve_image_path
from medgemma_ecg.prepare_ptbxl import CANONICAL_LEADS, main, render_ecg


@pytest.fixture
def ptbxl_root(tmp_path: Path) -> Path:
    """Write artificial metadata and two visibly different synthetic waveforms."""
    root = tmp_path / "synthetic-ptbxl"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "ecg_id": 1,
                "patient_id": "synthetic-patient",
                "strat_fold": 1,
                "filename_lr": "records100/synthetic",
                "filename_hr": "records500/synthetic",
                "scp_codes": "{'SR': 0.0}",
                "report": "Synthetic test waveform, not a clinical record",
            }
        ]
    ).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"diagnostic": [0], "rhythm": [1], "form": [0]}, index=["SR"]
    ).to_csv(root / "scp_statements.csv")
    for rate, frequency in [(100, 1), (500, 3)]:
        directory = root / f"records{rate}"
        directory.mkdir()
        time = np.arange(10 * rate) / rate
        signal = np.tile(np.sin(2 * np.pi * frequency * time)[:, None], (1, 12))
        wfdb.wrsamp(
            record_name="synthetic",
            fs=rate,
            units=["mV"] * 12,
            sig_name=CANONICAL_LEADS,
            p_signal=signal,
            fmt=["16"] * 12,
            write_dir=str(directory),
        )
    return root


def prepare(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    output: Path,
    rate: int,
    *extra: str,
) -> dict:
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
            str(rate),
            *extra,
        ],
    )
    main()
    rows = load_jsonl(output / "train.jsonl")
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize(("first_rate", "second_rate"), [(100, 500), (500, 100)])
def test_sampling_rate_change_uses_matching_waveform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ptbxl_root: Path,
    first_rate: int,
    second_rate: int,
) -> None:
    output = tmp_path / "processed"
    first = prepare(monkeypatch, ptbxl_root, output, first_rate)
    first_image = resolve_image_path(first)
    first_bytes = first_image.read_bytes()

    second = prepare(monkeypatch, ptbxl_root, output, second_rate)
    assert second["sampling_rate"] == second_rate
    assert second["record"] == f"records{second_rate}/synthetic"
    record = wfdb.rdrecord(str(ptbxl_root / second["record"]))
    expected_image = tmp_path / "expected.png"
    render_ecg(record.p_signal, record.fs, record.sig_name, expected_image)

    with Image.open(resolve_image_path(second)) as actual, Image.open(expected_image) as expected:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert first_image.read_bytes() == first_bytes


@pytest.mark.parametrize("rate", [100, 500])
def test_same_sampling_rate_reuses_cached_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ptbxl_root: Path, rate: int
) -> None:
    output = tmp_path / "processed"
    first = prepare(monkeypatch, ptbxl_root, output, rate)
    first_bytes = resolve_image_path(first).read_bytes()

    def unexpected_read(*args, **kwargs):
        pytest.fail("A matching cached image should not require reading the waveform")

    monkeypatch.setattr(wfdb, "rdrecord", unexpected_read)
    second = prepare(monkeypatch, ptbxl_root, output, rate)
    assert second["image"] == first["image"]
    assert resolve_image_path(second).read_bytes() == first_bytes


def test_overwrite_rerenders_matching_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ptbxl_root: Path
) -> None:
    output = tmp_path / "processed"
    first = prepare(monkeypatch, ptbxl_root, output, 100)
    image = resolve_image_path(first)
    expected_bytes = image.read_bytes()
    image.write_bytes(b"stale synthetic image")

    second = prepare(monkeypatch, ptbxl_root, output, 100, "--overwrite")
    assert second["image"] == first["image"]
    assert image.read_bytes() == expected_bytes


def test_unversioned_cache_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ptbxl_root: Path
) -> None:
    output = tmp_path / "processed"
    legacy_image = output / "images" / "train" / "ptbxl-00001.png"
    legacy_image.parent.mkdir(parents=True)
    legacy_bytes = b"synthetic legacy image with unknown sampling rate"
    legacy_image.write_bytes(legacy_bytes)

    row = prepare(monkeypatch, ptbxl_root, output, 100)

    assert resolve_image_path(row).read_bytes() != legacy_bytes
    assert legacy_image.read_bytes() == legacy_bytes
    with Image.open(resolve_image_path(row)) as image:
        image.verify()

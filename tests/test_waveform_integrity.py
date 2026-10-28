from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import wfdb
from PIL import Image

from medgemma_ecg.manifest import load_jsonl, resolve_image_path, validate_rows
from medgemma_ecg.prepare_ptbxl import CANONICAL_LEADS, main, render_ecg


def output_bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def write_synthetic_record(
    root: Path, rate: int, missing: str | None = None, *, flatline: bool = False
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    time = np.arange(10 * rate) / rate
    waveform = np.zeros(time.size) if flatline else np.sin(2 * np.pi * time)
    digital = np.tile(np.rint(1000 * waveform).astype(np.int16)[:, None], (1, 12))
    if missing == "sample":
        digital[5 * rate + 1, 1] = -32768
    elif missing == "lead":
        digital[:, 6] = -32768
    wfdb.wrsamp(
        record_name="synthetic",
        fs=rate,
        units=["mV"] * 12,
        sig_name=CANONICAL_LEADS,
        d_signal=digital,
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
        write_dir=str(root),
    )
    pd.DataFrame(
        [{
            "ecg_id": 1,
            "patient_id": "synthetic-patient",
            "strat_fold": 1,
            "filename_lr": "synthetic",
            "filename_hr": "synthetic",
            "scp_codes": "{'SR': 0.0}",
            "report": "Synthetic waveform, not a clinical record",
        }]
    ).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"diagnostic": [0], "rhythm": [1], "form": [0]}, index=["SR"]
    ).to_csv(root / "scp_statements.csv")


def prepare(
    monkeypatch: pytest.MonkeyPatch, root: Path, output: Path, rate: int, *, overwrite: bool = False
) -> None:
    argv = [
        "medgemma-ecg-prepare-ptbxl", "--ptbxl-root", str(root),
        "--output-root", str(output), "--sampling-rate", str(rate),
    ]
    if overwrite:
        argv.append("--overwrite")
    monkeypatch.setattr(sys, "argv", argv)
    main()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "negative-inf"])
@pytest.mark.parametrize("existing", [False, True])
def test_render_rejects_nonfinite_samples(tmp_path: Path, value: float, existing: bool) -> None:
    output = tmp_path / "synthetic.png"
    if existing:
        Image.new("RGB", (10, 10), "white").save(output)
    before = output_bytes(tmp_path)
    figures = set(plt.get_fignums())
    signal = np.zeros((1000, 12))
    signal[100, 0] = value

    error = None
    try:
        render_ecg(signal, 100, CANONICAL_LEADS, output)
    except ValueError as caught:
        error = caught

    assert error is not None, "Non-finite waveform samples were silently rendered"
    assert "non-finite" in str(error)
    assert output_bytes(tmp_path) == before
    assert set(plt.get_fignums()) == figures


@pytest.mark.parametrize("rate", [100, 500])
@pytest.mark.parametrize("missing", ["sample", "lead"])
@pytest.mark.parametrize("existing", [False, True])
def test_prepare_refuses_missing_wfdb_samples_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rate: int,
    missing: str,
    existing: bool,
) -> None:
    root = tmp_path / "synthetic-ptbxl"
    output = tmp_path / "processed"
    write_synthetic_record(root, rate)
    if existing:
        prepare(monkeypatch, root, output, rate)
    before = output_bytes(output)
    figures = set(plt.get_fignums())
    write_synthetic_record(root, rate, missing)
    # Prove this is WFDB's real missing-data decoding, not a mocked renderer failure.
    record = wfdb.rdrecord(str(root / "synthetic"))
    assert np.isnan(record.p_signal).sum() == (1 if missing == "sample" else 10 * rate)

    error = None
    try:
        prepare(monkeypatch, root, output, rate, overwrite=existing)
    except ValueError as caught:
        error = caught

    assert error is not None, "Missing WFDB samples were silently accepted into the dataset"
    assert "non-finite" in str(error)
    assert output_bytes(output) == before
    assert set(plt.get_fignums()) == figures
    assert not list(output.rglob(".medgemma-image-*"))
    # Repairing the synthetic source and retrying must work without manual output cleanup.
    write_synthetic_record(root, rate)
    prepare(monkeypatch, root, output, rate, overwrite=existing)
    rows = load_jsonl(output / "train.jsonl")
    assert validate_rows(rows)["samples"] == 1
    with Image.open(resolve_image_path(rows[0])) as image:
        image.verify()
    assert set(plt.get_fignums()) == figures


@pytest.mark.parametrize("rate", [100, 500])
@pytest.mark.parametrize("flatline", [False, True])
def test_prepare_valid_waveforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: int, flatline: bool
) -> None:
    root = tmp_path / "synthetic-ptbxl"
    output = tmp_path / "processed"
    write_synthetic_record(root, rate, flatline=flatline)
    record = wfdb.rdrecord(str(root / "synthetic"))
    assert np.isfinite(record.p_signal).all()
    figures = set(plt.get_fignums())

    prepare(monkeypatch, root, output, rate)

    rows = load_jsonl(output / "train.jsonl")
    assert validate_rows(rows)["samples"] == 1
    assert rows[0]["id"] == "ptbxl-00001"
    assert rows[0]["labels"] == rows[0]["target"]["labels"] == ["SR"]
    assert rows[0]["patient_id"] == "synthetic-patient"
    assert rows[0]["sampling_rate"] == rate
    assert rows[0]["record"] == "synthetic"
    assert load_jsonl(output / "validation.jsonl") == load_jsonl(output / "test.jsonl") == []
    with Image.open(resolve_image_path(rows[0])) as image:
        image.verify()
    assert set(plt.get_fignums()) == figures
    assert not list(output.rglob(".medgemma-image-*"))

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import wfdb
from matplotlib.figure import Figure
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


@pytest.fixture
def rendered_figures():
    """Keep a failing renderer test from leaking figures into other tests."""
    before = set(plt.get_fignums())
    yield before
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["save", "replace"])
def test_render_failure_does_not_publish_incomplete_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rendered_figures: set[int],
    existing: bool,
    failure: str,
) -> None:
    output = tmp_path / "synthetic.png"
    previous = b"previous synthetic cache contents"
    if existing:
        output.write_bytes(previous)
    before = set(tmp_path.iterdir())
    reached = []

    def failed_save(self, destination, **kwargs):
        Path(destination).write_bytes(b"synthetic partial PNG")
        reached.append("save")
        raise OSError("synthetic save failure")

    def failed_replace(source, destination):
        with Image.open(source) as image:
            image.verify()
        reached.append("replace")
        raise OSError("synthetic replace failure")

    if failure == "save":
        monkeypatch.setattr(Figure, "savefig", failed_save)
    else:
        monkeypatch.setattr(os, "replace", failed_replace)

    with pytest.raises(OSError, match=f"synthetic {failure} failure"):
        render_ecg(np.zeros((1000, 12)), 100, CANONICAL_LEADS, output)

    assert reached == [failure]
    assert (output.read_bytes() if output.exists() else None) == (
        previous if existing else None
    )
    assert set(tmp_path.iterdir()) == before
    assert set(plt.get_fignums()) == rendered_figures


def test_prepare_retries_render_after_partial_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ptbxl_root: Path,
    rendered_figures: set[int],
) -> None:
    output = tmp_path / "processed"
    reads = []
    original_read = wfdb.rdrecord

    def observed_read(*args, **kwargs):
        reads.append(args[0])
        return original_read(*args, **kwargs)

    def failed_save(self, destination, **kwargs):
        Path(destination).write_bytes(b"synthetic partial PNG")
        raise OSError("synthetic save failure")

    monkeypatch.setattr(wfdb, "rdrecord", observed_read)
    with monkeypatch.context() as patch:
        patch.setattr(Figure, "savefig", failed_save)
        with pytest.raises(OSError, match="synthetic save failure"):
            prepare(monkeypatch, ptbxl_root, output, 100)
    assert len(reads) == 1
    assert not list(output.glob("*.jsonl"))

    row = prepare(monkeypatch, ptbxl_root, output, 100)

    assert len(reads) == 2, "Retry must render the waveform, not reuse an incomplete PNG"
    with Image.open(resolve_image_path(row)) as image:
        image.verify()
    assert set(plt.get_fignums()) == rendered_figures
    assert not list(output.rglob(".medgemma-image-*"))


def test_failed_overwrite_preserves_cached_image_and_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ptbxl_root: Path,
    rendered_figures: set[int],
) -> None:
    output = tmp_path / "processed"
    row = prepare(monkeypatch, ptbxl_root, output, 100)
    image_path = resolve_image_path(row)
    previous_image = image_path.read_bytes()
    previous_manifests = {path: path.read_bytes() for path in output.glob("*.jsonl")}
    reached = []

    def failed_save(self, destination, **kwargs):
        Path(destination).write_bytes(b"synthetic partial replacement PNG")
        reached.append(True)
        raise OSError("synthetic overwrite failure")

    monkeypatch.setattr(Figure, "savefig", failed_save)
    with pytest.raises(OSError, match="synthetic overwrite failure"):
        prepare(monkeypatch, ptbxl_root, output, 100, "--overwrite")

    assert reached == [True]
    assert image_path.read_bytes() == previous_image
    assert {path: path.read_bytes() for path in output.glob("*.jsonl")} == previous_manifests
    assert set(plt.get_fignums()) == rendered_figures
    assert not list(output.rglob(".medgemma-image-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_successful_render_matches_direct_save_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rendered_figures: set[int],
    existing: bool,
) -> None:
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    output = output_dir / "synthetic.png"
    if existing:
        output.write_bytes(b"old synthetic contents")
    expected = tmp_path / "direct.png"
    original_save = Figure.savefig
    saves = []

    def observed_save(self, destination, **kwargs):
        saves.append(Path(destination))
        original_save(self, expected, **kwargs)
        return original_save(self, destination, **kwargs)

    monkeypatch.setattr(Figure, "savefig", observed_save)
    time = np.arange(1000) / 100
    signal = np.tile(np.sin(2 * np.pi * time)[:, None], (1, 12))
    render_ecg(signal, 100, CANONICAL_LEADS, output)

    assert len(saves) == 1
    with Image.open(output) as actual, Image.open(expected) as direct:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(direct))
    assert set(output_dir.iterdir()) == {output}
    assert set(plt.get_fignums()) == rendered_figures


def test_successful_render_preserves_symlink_and_permissions(tmp_path: Path) -> None:
    # Format follows the caller's .png name, not the resolved target's suffix.
    target = tmp_path / "target.cache"
    target.write_bytes(b"previous synthetic contents")
    target.chmod(0o640)
    previous_mode = target.stat().st_mode
    output = tmp_path / "linked.png"
    output.symlink_to(target.name)

    render_ecg(np.zeros((1000, 12)), 100, CANONICAL_LEADS, output)

    assert output.is_symlink()
    assert target.stat().st_mode == previous_mode
    with Image.open(target) as image:
        image.verify()
    assert set(tmp_path.iterdir()) == {target, output}


@pytest.mark.parametrize("name", ["synthetic", "synthetic."])
def test_render_preserves_default_format_extension(tmp_path: Path, name: str) -> None:
    output = tmp_path / name
    expected = tmp_path / "synthetic.png"

    with plt.rc_context({"savefig.format": "png"}):
        render_ecg(np.zeros((1000, 12)), 100, CANONICAL_LEADS, output)

    with Image.open(expected) as image:
        image.verify()
    assert set(tmp_path.iterdir()) == {expected}


@pytest.mark.skipif(os.name != "posix", reason="POSIX write-permission control")
def test_render_refuses_read_only_destination(
    tmp_path: Path, rendered_figures: set[int]
) -> None:
    output = tmp_path / "synthetic.png"
    previous = b"previous synthetic contents"
    output.write_bytes(previous)
    output.chmod(0o444)
    if os.access(output, os.W_OK):
        pytest.skip("The effective user can write a mode-444 file")

    with pytest.raises(PermissionError):
        render_ecg(np.zeros((1000, 12)), 100, CANONICAL_LEADS, output)

    assert output.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {output}
    assert set(plt.get_fignums()) == rendered_figures

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from medgemma_ecg.prepare_ptbxl import CANONICAL_LEADS, render_ecg


@pytest.mark.parametrize("fs", [0.0, -100.0, np.nan, np.inf, -np.inf])
def test_sampling_rate_is_validated_before_creating_a_figure(tmp_path, monkeypatch, fs) -> None:
    output = tmp_path / "existing.png"
    output.write_bytes(b"existing cache")
    created = []
    real_figure = plt.figure
    figures_before = set(plt.get_fignums())

    def observe(*args, **kwargs):
        created.append(True)
        return real_figure(*args, **kwargs)

    monkeypatch.setattr(plt, "figure", observe)
    error = None
    try:
        render_ecg(np.zeros((100, 12)), fs, CANONICAL_LEADS, output)
    except Exception as exc:
        error = exc
    finally:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)
    assert isinstance(error, ValueError), f"expected validation error, got {error!r}"
    assert "sampling rate" in str(error).lower()
    assert not created
    assert output.read_bytes() == b"existing cache"


def test_duplicate_normalized_leads_do_not_silently_replace_waveforms(tmp_path) -> None:
    leads = [*CANONICAL_LEADS, " avr "]
    with pytest.raises(ValueError, match="duplicate.*lead"):
        render_ecg(np.zeros((100, 13)), 100, leads, tmp_path / "duplicate.png")
    assert not (tmp_path / "duplicate.png").exists()


def test_empty_signal_is_not_published_as_an_ecg(tmp_path) -> None:
    with pytest.raises(ValueError, match="samples"):
        render_ecg(np.empty((0, 12)), 100, CANONICAL_LEADS, tmp_path / "empty.png")
    assert not (tmp_path / "empty.png").exists()


def test_finite_positive_sampling_rate_and_unique_leads_render(tmp_path) -> None:
    output = tmp_path / "valid.png"
    render_ecg(np.zeros((100, 12)), 100, CANONICAL_LEADS, output)
    assert output.read_bytes().startswith(b"\x89PNG")
    assert not plt.get_fignums()

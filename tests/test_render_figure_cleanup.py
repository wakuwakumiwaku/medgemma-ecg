from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from medgemma_ecg.prepare_ptbxl import CANONICAL_LEADS, render_ecg


@pytest.mark.parametrize(
    ("owner", "method"),
    [
        (Axes, "plot"), (Axes, "text"), (Axes, "grid"), (Figure, "subplots"),
        (Figure, "tight_layout"), (Figure, "savefig"),
    ],
)
def test_plotting_failures_close_only_the_owned_figure(
    tmp_path, monkeypatch, owner, method
) -> None:
    unrelated = plt.figure()
    figures_before = set(plt.get_fignums())
    output = tmp_path / "existing.png"
    output.write_bytes(b"existing image")

    def fail(*args, **kwargs):
        raise OSError("synthetic plotting failure")

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(owner, method, fail)
            with pytest.raises(OSError, match="synthetic plotting failure"):
                render_ecg(np.zeros((100, 12)), 100, CANONICAL_LEADS, output)
        figures_after_failure = set(plt.get_fignums())
        assert output.read_bytes() == b"existing image"
        assert figures_after_failure == figures_before
        render_ecg(np.zeros((100, 12)), 100, CANONICAL_LEADS, output)
        assert output.read_bytes().startswith(b"\x89PNG")
        assert set(plt.get_fignums()) == figures_before
    finally:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)
        plt.close(unrelated)

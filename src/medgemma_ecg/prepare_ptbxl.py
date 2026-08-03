from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from medgemma_ecg.manifest import write_jsonl

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DISPLAY_ORDER = [
    ("I", 0.0),
    ("aVR", 2.5),
    ("V1", 5.0),
    ("V4", 7.5),
    ("II", 0.0),
    ("aVL", 2.5),
    ("V2", 5.0),
    ("V5", 7.5),
    ("III", 0.0),
    ("aVF", 2.5),
    ("V3", 5.0),
    ("V6", 7.5),
]


def normalize_lead_name(name: str) -> str:
    normalized = str(name).strip().upper()
    return {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
    }.get(normalized, normalized)


def render_ecg(signal: np.ndarray, fs: float, lead_names: list[str], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if signal.ndim != 2 or signal.shape[1] != len(lead_names):
        raise ValueError("signal shape and lead names do not match")
    lead_index = {
        normalize_lead_name(name): index for index, name in enumerate(lead_names)
    }
    missing = set(CANONICAL_LEADS) - set(lead_index)
    if missing:
        raise ValueError(f"missing required leads: {sorted(missing)}")

    duration = min(10.0, signal.shape[0] / fs)
    figure, axes = plt.subplots(4, 1, figsize=(14, 9), dpi=120, sharex=True)
    figure.patch.set_facecolor("white")

    for row, axis in enumerate(axes[:3]):
        for column in range(4):
            lead, start = DISPLAY_ORDER[row * 4 + column]
            start_sample = int(start * fs)
            end_sample = min(start_sample + int(2.5 * fs), signal.shape[0])
            segment = signal[start_sample:end_sample, lead_index[lead]]
            time = np.arange(segment.size) / fs + start
            axis.plot(time, segment, color="#111111", linewidth=0.8)
            axis.text(start + 0.03, 1.25, lead, fontsize=9, weight="bold")
            if column:
                axis.axvline(start, color="#cc7777", linewidth=0.4)

    rhythm = signal[: min(int(10 * fs), signal.shape[0]), lead_index["II"]]
    rhythm_time = np.arange(rhythm.size) / fs
    axes[3].plot(rhythm_time, rhythm, color="#111111", linewidth=0.8)
    axes[3].text(0.03, 1.25, "II rhythm", fontsize=9, weight="bold")

    for axis in axes:
        axis.set_xlim(0, max(10.0, duration))
        axis.set_ylim(-1.8, 1.8)
        axis.set_xticks(np.arange(0, 10.01, 0.2), minor=False)
        axis.set_xticks(np.arange(0, 10.01, 0.04), minor=True)
        axis.set_yticks(np.arange(-2.0, 2.01, 0.5), minor=False)
        axis.set_yticks(np.arange(-2.0, 2.01, 0.1), minor=True)
        axis.grid(which="major", color="#e7a4a4", linewidth=0.45)
        axis.grid(which="minor", color="#f5d5d5", linewidth=0.25)
        axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle("12-lead ECG | 25 mm/s | 10 mm/mV", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.98), h_pad=0.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def split_from_fold(fold: int) -> str:
    if fold <= 8:
        return "train"
    if fold == 9:
        return "validation"
    if fold == 10:
        return "test"
    raise ValueError(f"unexpected PTB-XL strat_fold {fold}")


def statement_label_groups(scp_codes: dict, statements) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"diagnostic": [], "rhythm": [], "form": []}
    for code in scp_codes:
        if code not in statements.index:
            continue
        # A listed SCP code is an annotation even when its optional likelihood is 0.
        # PTB-XL commonly encodes labels such as sinus rhythm as {"SR": 0.0}.
        row = statements.loc[code]
        for group in groups:
            try:
                enabled = float(row.get(group, 0)) == 1.0
            except (TypeError, ValueError):
                enabled = False
            if enabled:
                groups[group].append(str(code))
    return {group: sorted(set(codes)) for group, codes in groups.items()}


def clean_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def make_target(
    labels: list[str],
    report: str | None,
    rhythm_labels: list[str],
    form_labels: list[str],
    axis: str | None,
) -> dict:
    return {
        "rhythm": ", ".join(rhythm_labels) if rhythm_labels else None,
        "rate_bpm": None,
        "axis": clean_text(axis),
        "intervals": None,
        "findings": form_labels,
        "labels": labels,
        "impression": clean_text(report),
        "uncertainty": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PTB-XL records and create manifests")
    parser.add_argument("--ptbxl-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/ptbxl"))
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    import wfdb

    database_csv = args.ptbxl_root / "ptbxl_database.csv"
    statements_csv = args.ptbxl_root / "scp_statements.csv"
    if not database_csv.is_file() or not statements_csv.is_file():
        raise SystemExit("PTB-XL metadata files were not found under --ptbxl-root")

    records = pd.read_csv(database_csv, index_col="ecg_id")
    statements = pd.read_csv(statements_csv, index_col=0)
    if args.limit is not None:
        records = records.iloc[: args.limit]

    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    filename_column = "filename_lr" if args.sampling_rate == 100 else "filename_hr"

    for number, (ecg_id, row) in enumerate(records.iterrows(), start=1):
        split = split_from_fold(int(row["strat_fold"]))
        sample_id = f"ptbxl-{int(ecg_id):05d}"
        output_image = args.output_root / "images" / split / f"{sample_id}.png"
        record_rel = str(row[filename_column])
        record_path = args.ptbxl_root / record_rel
        if args.overwrite or not output_image.is_file():
            record = wfdb.rdrecord(str(record_path))
            render_ecg(record.p_signal, float(record.fs), list(record.sig_name), output_image)

        codes = ast.literal_eval(str(row["scp_codes"]))
        label_groups = statement_label_groups(codes, statements)
        labels = sorted({code for values in label_groups.values() for code in values})
        try:
            image_value = str(output_image.relative_to(Path.cwd()))
        except ValueError:
            image_value = str(output_image.resolve())
        split_rows[split].append(
            {
                "id": sample_id,
                "patient_id": str(row["patient_id"]),
                "split": split,
                "image": image_value,
                "labels": labels,
                "target": make_target(
                    labels,
                    row.get("report"),
                    label_groups["rhythm"],
                    label_groups["form"],
                    row.get("heart_axis"),
                ),
                "source": "PTB-XL",
                "record": record_rel,
                "sampling_rate": args.sampling_rate,
            }
        )
        if number % 500 == 0:
            print(f"Prepared {number}/{len(records)} records", flush=True)

    for split, rows in split_rows.items():
        write_jsonl(args.output_root / f"{split}.jsonl", rows)
    print(json.dumps({split: len(rows) for split, rows in split_rows.items()}, indent=2))


if __name__ == "__main__":
    main()

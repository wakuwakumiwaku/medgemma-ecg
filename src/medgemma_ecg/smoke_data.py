from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from medgemma_ecg.manifest import write_jsonl


def draw_synthetic(path: Path, variant: int, seed: int) -> None:
    rng = random.Random(seed)
    width, height = 1000, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 10):
        color = (255, 230, 230) if x % 50 else (245, 180, 180)
        draw.line((x, 0, x, height), fill=color, width=1)
    for y in range(0, height, 10):
        color = (255, 230, 230) if y % 50 else (245, 180, 180)
        draw.line((0, y, width, y), fill=color, width=1)

    for lead in range(12):
        row = lead // 3
        col = lead % 3
        left = 15 + col * 330
        baseline = 75 + row * 155
        points = []
        frequency = 0.055 if variant == 0 else 0.09
        for x_local in range(300):
            phase = x_local * frequency
            y = baseline + 14 * math.sin(phase)
            if x_local % (115 if variant == 0 else 72) in range(34, 41):
                y -= 60 - 15 * abs(37 - x_local % (115 if variant == 0 else 72))
            y += rng.uniform(-1.0, 1.0)
            points.append((left + x_local, y))
        draw.line(points, fill=(25, 25, 25), width=2)
        draw.text((left, baseline - 70), f"S{lead + 1}", fill=(25, 25, 25))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def target_for(label: str) -> dict:
    return {
        "rhythm": None,
        "rate_bpm": None,
        "axis": None,
        "intervals": None,
        "findings": ["synthetic pipeline test only"],
        "labels": [label],
        "impression": "Artificial image for software testing, not a medical ECG.",
        "uncertainty": "not medically interpretable",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create non-medical pipeline smoke data")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/smoke"))
    args = parser.parse_args()

    split_sizes = {"train": 4, "validation": 2, "test": 2}
    sample_number = 0
    for split, size in split_sizes.items():
        rows = []
        for index in range(size):
            variant = index % 2
            label = f"SYNTHETIC_{variant}"
            sample_id = f"smoke-{sample_number:03d}"
            image = args.output_root / "images" / f"{sample_id}.png"
            draw_synthetic(image, variant, sample_number)
            try:
                image_value = str(image.relative_to(Path.cwd()))
            except ValueError:
                image_value = str(image.resolve())
            rows.append(
                {
                    "id": sample_id,
                    "patient_id": f"synthetic-patient-{sample_number:03d}",
                    "split": split,
                    "image": image_value,
                    "labels": [label],
                    "target": target_for(label),
                    "source": "synthetic-software-smoke-test",
                    "record": None,
                }
            )
            sample_number += 1
        write_jsonl(args.output_root / f"{split}.jsonl", rows)
    print(json.dumps({"output_root": str(args.output_root), "samples": sample_number}, indent=2))


if __name__ == "__main__":
    main()

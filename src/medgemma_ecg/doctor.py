from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect() -> dict:
    report: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "peft",
                "huggingface-hub",
                "wfdb",
            )
        },
    }
    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        }
    except ImportError:
        report["cuda"] = {"available": False, "error": "torch is not installed"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report MedGemma ECG environment readiness")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = collect()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_cuda and not report["cuda"].get("available"):
        raise SystemExit("CUDA is required but was not detected")


if __name__ == "__main__":
    main()

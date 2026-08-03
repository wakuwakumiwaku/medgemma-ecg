from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "google/medgemma-1.5-4b-it"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download gated MedGemma files locally")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, default=Path("models/medgemma-1.5-4b-it"))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    args.local_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=args.local_dir,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            "MedGemma access is gated. Sign in with `hf auth login`, open "
            "https://huggingface.co/google/medgemma-1.5-4b-it, accept the terms, "
            f"and retry. Original error: {exc}"
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"Hugging Face download failed: {exc}") from exc
    print(f"Downloaded {args.model}@{args.revision} to {downloaded}")


if __name__ == "__main__":
    main()

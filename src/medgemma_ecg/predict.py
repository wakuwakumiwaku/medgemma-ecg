from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from medgemma_ecg.infer import extract_json
from medgemma_ecg.manifest import load_jsonl, resolve_image_path, write_jsonl
from medgemma_ecg.prompts import training_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured predictions for a manifest")
    parser.add_argument("--model", default="models/medgemma-1.5-4b-it")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    model_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    processor_source = str(args.adapter) if args.adapter else args.model
    processor = AutoProcessor.from_pretrained(processor_source)

    rows = load_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    predictions = []
    for index, row in enumerate(rows, start=1):
        image_path = resolve_image_path(row, args.data_root)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        messages = training_messages()
        messages[1]["content"][0]["image"] = image
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )[0][input_length:]
        raw = processor.decode(generated, skip_special_tokens=True).strip()
        parsed = extract_json(raw)
        labels = parsed.get("labels", []) if parsed else []
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            labels = []
        predictions.append(
            {
                "id": row["id"],
                "labels": sorted(set(labels)),
                "parsed": parsed,
                "raw": raw,
            }
        )
        print(f"Predicted {index}/{len(rows)}: {row['id']}", flush=True)

    write_jsonl(args.output, predictions)
    failures = sum(prediction["parsed"] is None for prediction in predictions)
    print(json.dumps({"samples": len(predictions), "parse_failures": failures}, indent=2))


if __name__ == "__main__":
    main()

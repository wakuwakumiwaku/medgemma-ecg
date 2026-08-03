from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from medgemma_ecg.prompts import training_messages


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedGemma ECG image inference")
    parser.add_argument("--model", default="models/medgemma-1.5-4b-it")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
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

    with Image.open(args.image) as opened:
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
    text = processor.decode(generated, skip_special_tokens=True).strip()
    parsed = extract_json(text)
    print(json.dumps({"raw": text, "parsed": parsed}, indent=2, ensure_ascii=False))
    if parsed is None:
        raise SystemExit("Model response was not valid JSON")


if __name__ == "__main__":
    main()

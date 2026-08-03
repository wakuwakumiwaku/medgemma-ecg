from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from medgemma_ecg.manifest import load_jsonl, resolve_image_path, validate_rows
from medgemma_ecg.prompts import training_messages


class ECGDataset:
    def __init__(self, rows: list[dict], data_root: str | Path = "."):
        self.rows = rows
        self.data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class MultimodalCompletionCollator:
    def __init__(self, processor, data_root: str | Path, max_length: int = 1024):
        self.processor = processor
        self.data_root = Path(data_root)
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict:
        images: list[list[Image.Image]] = []
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        for example in examples:
            image_path = resolve_image_path(example, self.data_root)
            with Image.open(image_path) as image:
                images.append([image.convert("RGB")])
            full_messages = training_messages(example["target"])
            prompt_messages = training_messages()
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages, add_generation_prompt=False, tokenize=False
                ).strip()
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages, add_generation_prompt=True, tokenize=False
                ).strip()
            )

        batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, : int(prompt_length)] = -100

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        special_map = tokenizer.special_tokens_map
        for key in ("boi_token", "image_token"):
            token = special_map.get(key)
            if token:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    labels[labels == token_id] = -100
        labels[labels == 262144] = -100
        if not (labels != -100).any(dim=1).all():
            raise ValueError(
                "A sample has no assistant tokens after masking. "
                "Increase max_length or shorten targets."
            )
        batch["labels"] = labels
        return batch


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML object")
    return config


def select_rows(rows: list[dict], maximum: int | None) -> list[dict]:
    return rows if maximum is None else rows[: int(maximum)]


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune MedGemma 1.5 on ECG images")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, help="Override max_steps for a smoke run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the supplied QLoRA configuration")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The detected GPU does not support bfloat16")

    set_seed(int(config.get("seed", 42)))
    data_root = Path(config.get("data_root", "."))
    train_rows = load_jsonl(config["train_manifest"])
    validation_rows = load_jsonl(config["validation_manifest"])
    validate_rows(train_rows + validation_rows, data_root=data_root, require_images=True)
    train_rows = select_rows(train_rows, config.get("max_train_samples"))
    validation_rows = select_rows(validation_rows, config.get("max_validation_samples"))

    local_dir = Path(config.get("model_local_dir", ""))
    model_source = str(local_dir) if (local_dir / "config.json").is_file() else config["model"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=quantization,
    )
    processor = AutoProcessor.from_pretrained(model_source)
    processor.tokenizer.padding_side = "right"
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    peft_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=config.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        max_steps=int(config.get("max_steps", -1)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        logging_steps=int(config.get("logging_steps", 5)),
        eval_strategy="steps",
        eval_steps=int(config.get("eval_steps", 50)),
        save_strategy="steps",
        save_steps=int(config.get("save_steps", 50)),
        save_total_limit=2,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        warmup_steps=int(config.get("warmup_steps", 0)),
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        dataloader_num_workers=0,
        seed=int(config.get("seed", 42)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ECGDataset(train_rows, data_root),
        eval_dataset=ECGDataset(validation_rows, data_root),
        data_collator=MultimodalCompletionCollator(
            processor, data_root, int(config.get("max_length", 1024))
        ),
        processing_class=processor,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = dict(result.metrics)
    metrics["train_samples"] = len(train_rows)
    metrics["validation_samples"] = len(validation_rows)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    print(f"Saved LoRA adapter and processor to {output_dir}")


if __name__ == "__main__":
    main()

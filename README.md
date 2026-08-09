# MedGemma ECG

A reproducible research pipeline for adapting `google/medgemma-1.5-4b-it` to de-identified 12-lead ECG images with QLoRA, patient-safe dataset splits, structured outputs, and held-out evaluation.

MedGemma 1.5 is a 4B multimodal instruction-tuned model. It accepts images and text, not raw voltage arrays. This project renders waveform datasets as standardized 12-lead ECG images before training.

## Scope and safety

This repository is for research and education. It is not a medical device and must not be used as a substitute for clinician interpretation, emergency assessment, or validated diagnostic software. Fine-tuned outputs can be confidently wrong. Any intended clinical use requires representative external validation, subgroup analysis, calibration, human-factors testing, regulatory review, and prospective monitoring.

No model weights, adapters, patient data, ECG records, credentials, or generated datasets belong in Git. The included `.gitignore` excludes them.

## Hardware target

The default configuration targets:

- NVIDIA RTX 5080 with 16 GB VRAM
- WSL2/Linux
- Python 3.12
- CUDA 12.8 PyTorch wheels
- 4-bit NF4 QLoRA, batch size 1, LoRA rank 8, gradient checkpointing

Google's reference vision fine-tuning notebook specifies a 40 GB GPU. The included 16 GB profile is deliberately conservative, but it still must be verified with a one-step training smoke test after gated model access is configured. If it runs out of memory, reduce `max_length`, keep batch size at 1, and use a smaller rendered image size only after measuring the effect on ECG readability.

## 1. Install

```bash
bash scripts/setup_wsl_cuda128.sh
source .venv/bin/activate
python -m medgemma_ecg.doctor
```

On WSL, the setup script keeps the virtual environment in Linux storage at `~/.venvs/medgemma-ecg` and creates a `.venv` symlink in the repository. This avoids very slow package installation on a Windows-mounted drive. The versions verified on this hardware are pinned in `constraints-wsl-cu128.txt`.

## 2. Obtain MedGemma access

1. Sign in to Hugging Face.
2. Open https://huggingface.co/google/medgemma-1.5-4b-it
3. Accept the Health AI Developer Foundations terms.
4. Create a read token and authenticate locally:

```bash
source .venv/bin/activate
hf auth login
python -m medgemma_ecg.download_model \
  --local-dir models/medgemma-1.5-4b-it
```

The downloaded model stays under the ignored `models/` directory.

## 3. Create pipeline-only smoke data

The smoke generator creates synthetic line images and artificial labels only. It does not create medically valid ECG examples.

```bash
python -m medgemma_ecg.smoke_data --output-root data/processed/smoke
python -m medgemma_ecg.manifest \
  data/processed/smoke/train.jsonl \
  data/processed/smoke/validation.jsonl \
  data/processed/smoke/test.jsonl
```

## 4. Prepare PTB-XL

Download PTB-XL from PhysioNet and follow its license and attribution requirements. Point the converter at the directory containing `ptbxl_database.csv`, `scp_statements.csv`, and the waveform folders.

```bash
python -m medgemma_ecg.prepare_ptbxl \
  --ptbxl-root /path/to/ptb-xl/1.0.3 \
  --output-root data/processed/ptbxl \
  --sampling-rate 100

python -m medgemma_ecg.manifest \
  data/processed/ptbxl/train.jsonl \
  data/processed/ptbxl/validation.jsonl \
  data/processed/ptbxl/test.jsonl
```

The recommended PTB-XL folds are preserved: folds 1 through 8 for training, fold 9 for validation, and fold 10 for testing. The validator rejects patient leakage across splits.

## 5. Fine-tune

Edit `configs/rtx5080_16gb.yaml` if your paths differ. Run one optimization step first:

```bash
python -m medgemma_ecg.train \
  --config configs/rtx5080_16gb.yaml \
  --max-steps 1
```

Then launch the configured run:

```bash
python -m medgemma_ecg.train --config configs/rtx5080_16gb.yaml
```

Only the LoRA adapter and training state are saved in `outputs/`, which is ignored by Git.

## 6. Inference

```bash
python -m medgemma_ecg.infer \
  --model models/medgemma-1.5-4b-it \
  --adapter outputs/medgemma-ecg-qlora \
  --image /path/to/deidentified_ecg.png
```

## 7. Evaluate

Generate prediction JSONL from the untouched test manifest, then score it:

```bash
python -m medgemma_ecg.predict \
  --model models/medgemma-1.5-4b-it \
  --adapter outputs/medgemma-ecg-qlora \
  --manifest data/processed/ptbxl/test.jsonl \
  --output data/predictions/test.jsonl

python -m medgemma_ecg.evaluate \
  --references data/processed/ptbxl/test.jsonl \
  --predictions data/predictions/test.jsonl \
  --output outputs/test_metrics.json \
  --bootstrap-samples 1000 \
  --seed 42
```

The evaluator reports exact set accuracy, micro/macro/weighted precision, recall and
F1, plus per-label scores. Confidence intervals use a deterministic percentile
bootstrap that resamples patients rather than individual ECGs, preserving correlation
between records from the same patient. The default is 1,000 resamples at 95%
confidence. Intervals record how many resamples had a defined metric, so rare-label
results do not silently turn undefined sensitivity or precision into zero. Use
`--bootstrap-samples 0` only when a quick point-estimate check is needed. Duplicate or
mismatched sample IDs are rejected instead of being silently overwritten.

Do not tune on the test set. For a credible study, also report per-label
sensitivity/specificity, external-dataset performance, subgroup performance, and reader
comparison.

## Manifest schema

Each JSONL row contains:

```json
{
  "id": "ptbxl-00001",
  "patient_id": "123",
  "split": "train",
  "image": "data/processed/ptbxl/images/ptbxl-00001.png",
  "labels": ["NORM"],
  "target": {
    "rhythm": null,
    "rate_bpm": null,
    "axis": null,
    "intervals": null,
    "findings": [],
    "labels": ["NORM"],
    "impression": "source report or curated target",
    "uncertainty": null
  },
  "source": "PTB-XL",
  "record": "records100/00000/00001_lr"
}
```

Paths may be absolute or relative to the repository root. Never include direct identifiers.

## Dataset strategy

Start with PTB-XL for a reproducible baseline, then add separately licensed datasets through adapters that produce the same manifest schema. Keep every patient's records in exactly one split. Preserve source provenance and label ontology. Deduplicate near-identical ECGs and avoid report-derived leakage in prompts or filenames.

Useful sources:

- Dataset plan: [docs/DATASETS.md](docs/DATASETS.md)
- Experiment protocol: [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)
- PTB-XL: https://physionet.org/content/ptb-xl/
- MedGemma model card: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
- Official MedGemma repository: https://github.com/Google-Health/medgemma
- Official fine-tuning notebook: https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb

## License

Code in this repository is licensed under Apache-2.0. MedGemma weights are governed separately by Google's Health AI Developer Foundations terms. Every ECG dataset retains its own license and citation requirements.

LoRA adapters and merged checkpoints may qualify as Model Derivatives under those terms. Before distributing either, review the current agreement at https://developers.google.com/health-ai-developer-foundations/terms and include all required use restrictions, modification notices, agreement copy, and prescribed notice text. This repository does not distribute a model derivative.

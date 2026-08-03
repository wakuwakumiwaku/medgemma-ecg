# Dataset plan

The repository does not redistribute ECG datasets. Download each dataset from its official source, preserve its license and citation files, and keep it under an ignored local directory.

## Phase 1: PTB-XL baseline

Official source: https://physionet.org/content/ptb-xl/1.0.3/

PTB-XL 1.0.3 contains 21,799 ten-second 12-lead ECGs from 18,869 patients at 500 Hz, plus 100 Hz copies. It provides 71 SCP-ECG statements and patient-respecting recommended folds. Use folds 1 through 8 for training, fold 9 for validation, and fold 10 exactly once for final testing.

The implemented adapter:

- reads the WFDB waveform files
- renders a 3 by 4 lead layout plus a lead-II rhythm strip
- keeps positive diagnostic, rhythm, and form SCP codes
- includes the source report and heart-axis field when present
- emits the common JSONL manifest schema
- preserves `patient_id` for leakage validation

Always cite the dataset DOI and publication listed on the PhysioNet page.

## Phase 2: External validation

### Chapman/Shaoxing/Ningbo 12-lead database

Official source: https://physionet.org/content/ecg-arrhythmia/1.0.0/

This open-access dataset contains 45,152 patient ECGs at 500 Hz with physician-validated rhythm and condition labels. The files are licensed CC BY 4.0. Use it first as an untouched external test set. Build a label-mapping table instead of silently treating its SNOMED CT labels as PTB-XL SCP codes.

### MIMIC-IV-ECG

Official source: https://physionet.org/content/mimic-iv-ecg/1.0/

MIMIC-IV-ECG contains approximately 800,000 ten-second 12-lead ECGs across nearly 160,000 subjects at 500 Hz. Access to linked clinical modules or notes can have additional requirements. Verify the current access and license terms before use. Split by `subject_id`, not by waveform or encounter. Do not let reports or neighboring EHR events leak into an image-only benchmark prompt.

## Required controls

1. Keep every patient in exactly one split.
2. Hash raw waveforms to detect exact duplicates across sources and splits.
3. Detect near-duplicates after resampling, filtering, or image rendering.
4. Preserve raw source labels and map them into a versioned canonical ontology.
5. Record dataset version, download date, license, citation, preprocessing configuration, and exclusion counts.
6. Exclude direct identifiers and avoid committing any source or generated patient data.
7. Keep the final test and external-validation sets untouched until the training recipe is frozen.
8. Report performance by sex and age group where the license and metadata permit it, while avoiding small groups that create re-identification risk.
9. Manually audit a stratified sample of renderings for lead order, scale, clipping, missing leads, artifacts, and label consistency.
10. Treat machine-generated reports separately from cardiologist-validated reports.

## Manifest adapter contract

Every future adapter should produce one JSONL file per split with these stable fields:

- `id`: unique de-identified sample ID
- `patient_id`: grouping key used only for split validation
- `split`: `train`, `validation`, or `test`
- `image`: local ECG image path
- `labels`: canonical label strings
- `target`: structured interpretation target
- `source`: dataset name and optionally version
- `record`: source-relative waveform identifier

Dataset-specific fields may be added, but the common fields must retain the same meaning.

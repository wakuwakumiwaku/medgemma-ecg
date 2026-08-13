# Experiment protocol

## Research question

Can QLoRA adaptation of MedGemma 1.5 improve structured interpretation of standardized 12-lead ECG images over the unadapted base model on patient-disjoint and external datasets?

## Freeze before final testing

Freeze and commit:

- base model ID and revision
- dataset versions and checksums
- inclusion and exclusion rules
- waveform-to-image rendering parameters
- prompt and JSON schema
- label ontology and source mappings
- train/validation/test patient lists or reproducible split rules
- random seeds
- LoRA targets and hyperparameters
- response parser and metrics

## Minimum comparisons

1. Unadapted MedGemma 1.5 with the same prompt and decoding settings.
2. QLoRA-adapted MedGemma 1.5.
3. A conventional ECG waveform baseline appropriate to the label task.
4. If feasible, a text-only control using metadata without the ECG image.

The waveform baseline is important because a general-purpose image-language model may not preserve small morphology and interval details as effectively as a signal-specific model.

## Development sequence

1. Unit-test the renderer and manifest validator on synthetic data.
2. Visually audit real PTB-XL renderings before training.
3. Run one QLoRA optimization step and record peak VRAM.
4. Overfit a tiny training subset as a pipeline sanity check.
5. Run a small fixed hyperparameter sweep using training and validation only.
6. Freeze the selected recipe.
7. Evaluate once on PTB-XL fold 10.
8. Evaluate without further tuning on the external dataset.
9. Perform clinician review of a stratified error sample.

## Metrics

For multilabel generation:

- exact label-set accuracy
- Hamming loss
- micro, macro, and weighted precision, recall, and F1
- per-label sensitivity, specificity, positive predictive value, and negative predictive value
- deterministic percentile-bootstrap confidence intervals grouped by patient, with the resample count, confidence level, and random seed recorded
- invalid-JSON rate and missing-field rate

If calibrated label probabilities are added later:

- macro and per-label AUROC
- area under the precision-recall curve
- Brier score
- calibration curves and expected calibration error

For structured fields:

- rate: mean absolute error and clinically relevant tolerance accuracy
- intervals: mean absolute error with units and missingness reported
- axis: categorical agreement or circular error, depending on representation
- rhythm and findings: ontology-aware multilabel metrics
- impression: clinician rubric, not an embedding score alone

## Error analysis

Stratify errors by:

- label prevalence and co-occurrence
- heart rate and rhythm
- sex and age group where permitted
- device or site when available
- signal quality, clipping, baseline wander, and missing leads
- report provenance and human-validation status
- image resolution and rendering style

Review false negatives for time-critical patterns separately. A high aggregate score does not establish safe clinical performance.

## Reproducibility artifacts

Store code and small configuration files in Git. Keep all of the following outside Git and record checksums in private experiment metadata:

- model weights
- LoRA adapters
- raw ECG files
- rendered patient ECG images
- train/validation/test manifests containing patient grouping keys
- generated predictions
- TensorBoard or other run logs

# Dataset Changelog

This changelog tracks the dataset versions used by the Traffic Violation Detection & License Plate Recognition project.

## Current Dataset

| Field | Value |
|---|---|
| Dataset folder | `data/processed/ccpd_layer4` |
| Source dataset | CCPD2019 compact subset |
| Task | One-class license plate detection (`plate`) |
| Records | 8,598 |
| Positive samples | 8,000 |
| Negative samples | 598 |
| Train / Val / Test | 6,398 / 1,100 / 1,100 |
| OCR crops | 8,000 |
| Main manifest | `data/processed/ccpd_layer4/manifest.json` |
| Quality manifest | `data/processed/ccpd_layer4/manifest_quality.json` |
| Audit report | `outputs/reports/dataset_audit_report.md` |
| Duplicate report | `outputs/reports/dataset_duplicates_report.md` |

## Versions

### v0.4 - Dataset QA and JD Alignment

Status: implemented locally.

Added:

- Automated dataset audit script: `scripts/audit_plate_dataset.py`
- Integrity checks for image/label/crop consistency.
- YOLO bbox validation.
- Positive/negative sample consistency checks.
- Broken image/crop checks.
- Quality distribution analysis from brightness, blur, and plate-area metadata.
- Quality-flagged manifest: `manifest_quality.json`.
- Exact duplicate and perceptual-hash collision report.
- Edge-case contact sheets for dark, blur, small plate, weather, challenge, and negative samples.

Purpose:

- Provide evidence for dataset QA, metadata tracking, duplicate review, and edge-case handling.
- Align the project with dataset-focused AI intern requirements.

### v0.3 - Layer 4 Plate Dataset

Status: implemented.

Added:

- Compact CCPD subset at `data/processed/ccpd_layer4`.
- YOLO labels under `labels/{train,val,test}`.
- OCR crops under `ocr_crops/{train,val,test}`.
- OCR label text files under `ocr_labels`.
- Source provenance and split metadata in `manifest.json`.
- Dataset config `ccpd_plate.yaml` for Ultralytics training.

Purpose:

- Train and validate a one-class YOLO plate detector.
- Provide cropped plate images for OCR experiments and debugging.

### v0.2 - Split and Format Conversion

Status: implemented as part of `scripts/prepare_ccpd_layer4.py`.

Added:

- Stratified sampling from CCPD source folders.
- Train/val/test split.
- CCPD filename parsing.
- Conversion from CCPD bbox metadata to YOLO normalized bbox format.
- Negative samples from `ccpd_np`.

Purpose:

- Convert raw CCPD-style file naming into a standard object detection dataset.

### v0.1 - Raw Dataset Exploration

Status: completed before cleanup.

Added:

- Initial exploration of CCPD/CULane/UA-DETRAC and other traffic datasets.
- Decision to keep only the compact CCPD plate dataset needed for the current project.
- Removal of large unused datasets to reduce local storage usage.

Purpose:

- Avoid carrying large datasets that were not used by the final training pipeline.

## QA Rules

The current audit treats the following as blocking issues:

- Missing image referenced by manifest.
- Missing positive label file.
- Missing positive OCR crop.
- Orphan image/label/crop not referenced by manifest.
- Broken image or crop.
- YOLO label parse error.
- Positive sample with empty label.
- Negative sample with non-empty label.
- YOLO class id not equal to `0` (`plate`).
- YOLO bbox outside normalized range.
- YOLO bbox area outside guardrail thresholds.

The current quality thresholds are:

| Signal | Threshold |
|---|---|
| Dark image | `brightness < 80` |
| Bright image | `brightness > 180` |
| Heavy blur | `blurriness < 40` |
| Mild blur | `40 <= blurriness < 100` |
| Sharp | `blurriness >= 100` |
| Small plate | `plate_area_ratio < 0.01` |
| Large plate | `plate_area_ratio > 0.12` |

## Interview Positioning

Use this dataset story carefully:

- Claim: built a reproducible license plate dataset workflow with conversion, split, manifest, negative samples, OCR crops, QA, duplicate review, and contact sheets.
- Do not claim: built a 1,000-object Gameloft-style dataset.
- Do not claim: trained classification or segmentation models.
- Do not claim: OCR is fully solved. OCR still depends on plate crop quality and readable ground truth.

Strong short version:

```text
I built a license plate dataset processing workflow around CCPD2019:
stratified sampling, YOLO annotation conversion, train/val/test split,
negative samples, OCR crop generation, metadata manifest, automated QA,
duplicate review, and edge-case contact sheets for blur, low-light, weather,
and small-plate cases.
```

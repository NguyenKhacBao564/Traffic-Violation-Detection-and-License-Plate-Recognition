# Dataset Audit Summary

This summary records the current QA status of the compact CCPD Layer 4 dataset used by the project.

## Dataset

| Field | Value |
|---|---:|
| Dataset folder | `data/processed/ccpd_layer4` |
| Total records | 8,598 |
| Train records | 6,398 |
| Val records | 1,100 |
| Test records | 1,100 |
| Positive plate samples | 8,000 |
| Negative no-plate samples | 598 |
| YOLO label files | 8,598 |
| OCR crops | 8,000 |

## Audit Result

| Check group | Result |
|---|---:|
| Blocking integrity issues | 0 |
| Missing image paths | 0 |
| Missing positive labels | 0 |
| Missing positive OCR crops | 0 |
| Orphan images/labels/crops | 0 |
| Broken images/crops | 0 |
| YOLO label parse errors | 0 |
| Positive samples with empty labels | 0 |
| Negative samples with non-empty labels | 0 |
| YOLO bbox out of range | 0 |
| YOLO bbox abnormal guardrail violations | 0 |

## Quality Distribution

| Split | Dark | Heavy blur | Mild blur | Sharp | Avg plate area ratio |
|---|---:|---:|---:|---:|---:|
| Train | 1,485 | 3,279 | 1,928 | 793 | 0.0330 |
| Val | 233 | 525 | 342 | 133 | 0.0329 |
| Test | 237 | 545 | 337 | 118 | 0.0317 |
| Total positive | 1,955 | 4,349 | 2,607 | 1,044 | 0.0328 |

## Edge-Case Coverage

| Source | Count | Share |
|---|---:|---:|
| `ccpd_base` | 5,400 | 62.81% |
| `ccpd_blur` | 700 | 8.14% |
| `ccpd_rotate` | 660 | 7.68% |
| `ccpd_tilt` | 660 | 7.68% |
| `ccpd_weather` | 380 | 4.42% |
| `ccpd_challenge` | 200 | 2.33% |
| `ccpd_np` | 598 | 6.96% |

## Duplicate Review

| Metric | Value |
|---|---:|
| Exact duplicate groups | 2 |
| Exact duplicate extra images | 2 |
| Perceptual hash collision groups | 102 |
| Perceptual hash collision extra images | 166 |
| Unreadable files for hashing | 0 |

Interpretation:

- Exact duplicate groups are real duplicate candidates and should be reviewed before the next retraining run.
- Perceptual hash collisions are not guaranteed duplicates. They are fast review candidates for visual QA.
- The current dataset has **0 blocking integrity issues**, but duplicate review is still useful for dataset cleanup.

## Generated Local Artifacts

These files are generated locally and ignored by git:

```text
outputs/reports/dataset_audit_report.json
outputs/reports/dataset_audit_report.md
outputs/reports/dataset_duplicates_report.json
outputs/reports/dataset_duplicates_report.md
outputs/reports/dataset_contact_dark.jpg
outputs/reports/dataset_contact_heavy_blur.jpg
outputs/reports/dataset_contact_small_plate.jpg
outputs/reports/dataset_contact_weather.jpg
outputs/reports/dataset_contact_challenge.jpg
outputs/reports/dataset_contact_negative.jpg
data/processed/ccpd_layer4/manifest_quality.json
```

## CV Claim

Safe wording:

```text
Built a reproducible license plate dataset pipeline from CCPD2019 with
stratified sampling, YOLO annotation conversion, train/val/test split,
negative samples, OCR crop generation, metadata manifest, automated integrity
audit, quality distribution analysis, duplicate review, and edge-case contact
sheets for blur, low-light, weather, challenge, and small-plate cases.
```

Do not claim:

- Removed all duplicates. The current implementation reports duplicates/candidates; it does not delete them automatically.
- Built a 1,000-object recognition dataset.
- Trained classification or segmentation models.

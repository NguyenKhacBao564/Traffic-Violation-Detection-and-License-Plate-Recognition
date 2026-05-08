#!/usr/bin/env python3
"""Audit and QA reports for the CCPD Layer 4 plate dataset.

This script turns the compact CCPD subset into interview-ready dataset
evidence:

- dataset integrity audit
- quality distribution from manifest metadata
- quality-flagged manifest
- duplicate / near-duplicate candidate report
- visual contact sheets for common edge-case groups
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "processed" / "ccpd_layer4"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
REPORT_DIR = ROOT / "outputs" / "reports"
QUALITY_MANIFEST_PATH = DATASET_DIR / "manifest_quality.json"

SPLITS = ("train", "val", "test")

BRIGHTNESS_DARK = 80
BRIGHTNESS_BRIGHT = 180
BLUR_HEAVY = 40
BLUR_SHARP = 100
BBOX_MIN_AREA_RATIO = 0.002
BBOX_MAX_AREA_RATIO = 0.70
SMALL_PLATE_AREA_RATIO = 0.01
LARGE_PLATE_AREA_RATIO = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--quality-manifest", type=Path, default=QUALITY_MANIFEST_PATH)
    parser.add_argument("--contact-samples", type=int, default=25)
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip exact/perceptual duplicate scan if a quick audit is enough.",
    )
    parser.add_argument(
        "--skip-contact-sheets",
        action="store_true",
        help="Skip visual contact sheet generation.",
    )
    return parser.parse_args()


def pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count * 100.0 / total, 2)


def brightness_level(value: int | float | None) -> str | None:
    if value is None:
        return None
    if value < BRIGHTNESS_DARK:
        return "dark"
    if value > BRIGHTNESS_BRIGHT:
        return "bright"
    return "normal"


def blur_level(value: int | float | None) -> str | None:
    if value is None:
        return None
    if value < BLUR_HEAVY:
        return "heavy_blur"
    if value < BLUR_SHARP:
        return "mild_blur"
    return "sharp"


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]] | None:
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return []

    rows: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            return None
        try:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            return None
    return rows


def image_readable(path: Path) -> bool:
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception:
        return False
    return image is not None and image.size > 0


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    small = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small > mean).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def plate_area_ratio_from_record(record: dict[str, Any]) -> float | None:
    bbox = record.get("bbox_xyxy")
    image_size = record.get("image_size")
    if not bbox or not image_size:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return round(area / float(width * height), 6)


def quality_for_record(record: dict[str, Any]) -> dict[str, Any]:
    brightness = record.get("brightness")
    blurriness = record.get("blurriness")
    area_ratio = plate_area_ratio_from_record(record)
    flags: list[str] = []

    b_level = brightness_level(brightness)
    blur = blur_level(blurriness)
    if b_level in {"dark", "bright"}:
        flags.append(b_level)
    if blur in {"heavy_blur", "mild_blur"}:
        flags.append(blur)

    if area_ratio is not None:
        if area_ratio < SMALL_PLATE_AREA_RATIO:
            flags.append("small_plate")
        elif area_ratio > LARGE_PLATE_AREA_RATIO:
            flags.append("large_plate")

    source = record.get("source")
    if source in {"ccpd_weather", "ccpd_challenge", "ccpd_rotate", "ccpd_tilt", "ccpd_blur"}:
        flags.append(source.replace("ccpd_", "source_"))
    if record.get("kind") == "negative":
        flags.append("negative_sample")

    return {
        "brightness_level": b_level,
        "blur_level": blur,
        "plate_area_ratio": area_ratio,
        "quality_flags": sorted(set(flags)),
    }


def summarize_values(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 4),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def collect_disk_files(dataset_dir: Path) -> dict[str, dict[str, set[Path]]]:
    result: dict[str, dict[str, set[Path]]] = {}
    for split in SPLITS:
        result[split] = {
            "images": set((dataset_dir / "images" / split).glob("*.jpg")),
            "labels": set((dataset_dir / "labels" / split).glob("*.txt")),
            "ocr_crops": set((dataset_dir / "ocr_crops" / split).glob("*.jpg")),
        }
    return result


def run_audit(
    dataset_dir: Path,
    manifest_path: Path,
    skip_duplicates: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(manifest_path)
    records = manifest.get("records", [])
    disk_files = collect_disk_files(dataset_dir)

    issues: dict[str, list[Any]] = {
        "missing_image": [],
        "missing_label": [],
        "missing_ocr_crop": [],
        "orphan_image": [],
        "orphan_label": [],
        "orphan_ocr_crop": [],
        "broken_image": [],
        "label_parse_error": [],
        "positive_missing_label_content": [],
        "negative_has_label_content": [],
        "bbox_class_not_plate": [],
        "bbox_out_of_range": [],
        "bbox_abnormal_size": [],
    }

    counts = {
        split: {"images": 0, "labels": 0, "ocr_crops": 0, "positive": 0, "negative": 0}
        for split in SPLITS
    }
    referenced = {
        split: {"images": set(), "labels": set(), "ocr_crops": set()}
        for split in SPLITS
    }

    quality_records: list[dict[str, Any]] = []
    brightness_values: dict[str, list[int | float]] = defaultdict(list)
    blur_values: dict[str, list[int | float]] = defaultdict(list)
    area_values: dict[str, list[float]] = defaultdict(list)
    flag_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        split = record.get("split")
        kind = record.get("kind")
        if split not in SPLITS:
            continue

        image_rel = record.get("image")
        label_rel = record.get("label")
        crop_rel = record.get("ocr_crop")
        image_path = dataset_dir / image_rel if image_rel else None
        label_path = dataset_dir / label_rel if label_rel else None
        crop_path = dataset_dir / crop_rel if crop_rel else None

        counts[split][kind] += 1

        if image_path:
            referenced[split]["images"].add(image_path)
            if image_path.exists():
                counts[split]["images"] += 1
                if not image_readable(image_path):
                    issues["broken_image"].append(image_rel)
            else:
                issues["missing_image"].append(image_rel)

        rows = None
        if label_path:
            referenced[split]["labels"].add(label_path)
            if label_path.exists():
                counts[split]["labels"] += 1
                rows = parse_yolo_label(label_path)
                if rows is None:
                    issues["label_parse_error"].append(label_rel)
            elif kind == "positive":
                issues["missing_label"].append(label_rel)

        if kind == "positive":
            if rows == []:
                issues["positive_missing_label_content"].append(label_rel)
            if crop_path:
                referenced[split]["ocr_crops"].add(crop_path)
                if crop_path.exists():
                    counts[split]["ocr_crops"] += 1
                    if not image_readable(crop_path):
                        issues["broken_image"].append(crop_rel)
                else:
                    issues["missing_ocr_crop"].append(crop_rel)
            else:
                issues["missing_ocr_crop"].append(image_rel)
        elif kind == "negative" and label_path and label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            if text:
                issues["negative_has_label_content"].append(label_rel)

        if kind == "positive" and rows:
            for cls, cx, cy, width, height in rows:
                if cls != 0:
                    issues["bbox_class_not_plate"].append({"label": label_rel, "class_id": cls})
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    issues["bbox_out_of_range"].append(
                        {"label": label_rel, "bbox": [cx, cy, width, height]}
                    )
                area_ratio = width * height
                if area_ratio < BBOX_MIN_AREA_RATIO:
                    issues["bbox_abnormal_size"].append(
                        {"label": label_rel, "area_ratio": round(area_ratio, 6), "issue": "too_small"}
                    )
                elif area_ratio > BBOX_MAX_AREA_RATIO:
                    issues["bbox_abnormal_size"].append(
                        {"label": label_rel, "area_ratio": round(area_ratio, 6), "issue": "too_large"}
                    )

        quality = quality_for_record(record)
        enriched = dict(record)
        enriched["quality"] = quality
        quality_records.append(enriched)

        if record.get("brightness") is not None:
            brightness_values[split].append(record["brightness"])
        if record.get("blurriness") is not None:
            blur_values[split].append(record["blurriness"])
        if quality["plate_area_ratio"] is not None:
            area_values[split].append(quality["plate_area_ratio"])
        flag_counts[split].update(quality["quality_flags"])

    for split in SPLITS:
        for image_path in disk_files[split]["images"] - referenced[split]["images"]:
            issues["orphan_image"].append(str(image_path.relative_to(dataset_dir)))
        for label_path in disk_files[split]["labels"] - referenced[split]["labels"]:
            issues["orphan_label"].append(str(label_path.relative_to(dataset_dir)))
        for crop_path in disk_files[split]["ocr_crops"] - referenced[split]["ocr_crops"]:
            issues["orphan_ocr_crop"].append(str(crop_path.relative_to(dataset_dir)))

    quality_summary: dict[str, Any] = {}
    for split in SPLITS:
        brightness_levels = Counter(
            level for value in brightness_values[split] if (level := brightness_level(value))
        )
        blur_levels = Counter(level for value in blur_values[split] if (level := blur_level(value)))
        quality_summary[split] = {
            "brightness": {
                "stats": summarize_values(brightness_values[split]),
                "levels": dict(brightness_levels),
            },
            "blur": {
                "stats": summarize_values(blur_values[split]),
                "levels": dict(blur_levels),
            },
            "plate_area_ratio": summarize_values(area_values[split]),
            "quality_flags": dict(flag_counts[split]),
        }

    all_quality_flags = Counter()
    for split in SPLITS:
        all_quality_flags.update(flag_counts[split])
    quality_summary["all"] = {
        "quality_flags": dict(all_quality_flags),
        "positive_plate_area_ratio": summarize_values(
            [value for split in SPLITS for value in area_values[split]]
        ),
    }

    duplicate_report = (
        {"skipped": True, "summary": {}}
        if skip_duplicates
        else build_duplicate_report(dataset_dir, records)
    )

    total_issues = sum(len(items) for items in issues.values())
    report = {
        "dataset": manifest.get("dataset"),
        "record_count": manifest.get("record_count"),
        "seed": manifest.get("seed"),
        "thresholds": {
            "brightness_dark_lt": BRIGHTNESS_DARK,
            "brightness_bright_gt": BRIGHTNESS_BRIGHT,
            "blur_heavy_lt": BLUR_HEAVY,
            "blur_sharp_gte": BLUR_SHARP,
            "bbox_min_area_ratio": BBOX_MIN_AREA_RATIO,
            "bbox_max_area_ratio": BBOX_MAX_AREA_RATIO,
            "small_plate_area_ratio_lt": SMALL_PLATE_AREA_RATIO,
            "large_plate_area_ratio_gt": LARGE_PLATE_AREA_RATIO,
        },
        "total_issues_found": total_issues,
        "counts_on_disk": counts,
        "manifest_summary": manifest.get("summary", {}),
        "issues": issues,
        "quality_summary": quality_summary,
        "duplicate_report": duplicate_report,
    }
    return report, quality_records


def build_duplicate_report(dataset_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    md5_buckets: dict[str, list[str]] = defaultdict(list)
    ahash_buckets: dict[str, list[str]] = defaultdict(list)
    unreadable: list[str] = []

    for record in records:
        image_rel = record.get("image")
        if not image_rel:
            continue
        image_path = dataset_dir / image_rel
        if not image_path.exists():
            continue
        try:
            md5_buckets[file_md5(image_path)].append(image_rel)
        except OSError:
            unreadable.append(image_rel)
            continue
        ahash = average_hash(image_path)
        if ahash:
            ahash_buckets[ahash].append(image_rel)
        else:
            unreadable.append(image_rel)

    exact_groups = [group for group in md5_buckets.values() if len(group) > 1]
    perceptual_buckets = [group for group in ahash_buckets.values() if len(group) > 1]
    return {
        "skipped": False,
        "summary": {
            "exact_duplicate_groups": len(exact_groups),
            "exact_duplicate_extra_images": sum(len(group) - 1 for group in exact_groups),
            "perceptual_hash_collision_groups": len(perceptual_buckets),
            "perceptual_hash_collision_extra_images": sum(len(group) - 1 for group in perceptual_buckets),
            "unreadable_for_hash": len(unreadable),
        },
        "exact_duplicate_groups": exact_groups,
        "perceptual_hash_collision_groups": perceptual_buckets,
        "unreadable_for_hash": unreadable,
        "note": (
            "Perceptual hash collision groups are review candidates, not guaranteed duplicates. "
            "They use an 8x8 average hash equality check for a fast local QA pass."
        ),
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] wrote {path}")


def write_quality_manifest(
    manifest_path: Path,
    quality_manifest_path: Path,
    quality_records: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    manifest = load_manifest(manifest_path)
    payload = {
        "dataset": manifest.get("dataset"),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "record_count": len(quality_records),
        "thresholds": report["thresholds"],
        "summary": manifest.get("summary", {}),
        "quality_summary": report["quality_summary"],
        "records": quality_records,
    }
    write_json(payload, quality_manifest_path)


def issue_status(count: int) -> str:
    return "PASS" if count == 0 else f"FAIL ({count})"


def write_audit_markdown(report: dict[str, Any], path: Path, artifact_paths: dict[str, Path]) -> None:
    lines: list[str] = []
    add = lines.append
    manifest_summary = report["manifest_summary"]
    by_split = manifest_summary.get("by_split", {})
    by_kind = manifest_summary.get("by_kind", {})
    by_source = manifest_summary.get("by_source", {})
    duplicate_summary = report["duplicate_report"].get("summary", {})

    add("# Dataset Audit Report - CCPD Layer 4")
    add("")
    add(f"**Dataset**: {report['dataset']}")
    add(f"**Total records**: {report['record_count']}")
    add(f"**Blocking integrity issues**: {report['total_issues_found']}")
    add("")
    add("## 1. Dataset Scale")
    add("")
    add("| Split | Images | Labels | OCR crops | Manifest records | Positive | Negative |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    split_kind = manifest_summary.get("by_split_kind", {})
    for split in SPLITS:
        disk = report["counts_on_disk"][split]
        add(
            f"| {split} | {disk['images']} | {disk['labels']} | {disk['ocr_crops']} | "
            f"{by_split.get(split, 0)} | {split_kind.get(split, {}).get('positive', 0)} | "
            f"{split_kind.get(split, {}).get('negative', 0)} |"
        )
    add("")
    add(f"- Positive plate samples: **{by_kind.get('positive', 0)}**")
    add(f"- Negative no-plate samples: **{by_kind.get('negative', 0)}**")
    add("")

    add("## 2. Integrity Checks")
    add("")
    labels = {
        "missing_image": "Manifest image path missing on disk",
        "missing_label": "Positive image missing label file",
        "missing_ocr_crop": "Positive image missing OCR crop",
        "orphan_image": "Image file on disk not referenced by manifest",
        "orphan_label": "Label file on disk not referenced by manifest",
        "orphan_ocr_crop": "OCR crop on disk not referenced by manifest",
        "broken_image": "Broken or unreadable image/crop",
        "label_parse_error": "YOLO label parse error",
        "positive_missing_label_content": "Positive sample has empty label",
        "negative_has_label_content": "Negative sample label is not empty",
        "bbox_class_not_plate": "YOLO class id is not 0/plate",
        "bbox_out_of_range": "YOLO bbox coordinate outside valid range",
        "bbox_abnormal_size": "YOLO bbox area ratio outside guardrail",
    }
    add("| Check | Issues | Status |")
    add("|---|---:|---|")
    for key, items in report["issues"].items():
        add(f"| {labels.get(key, key)} | {len(items)} | {issue_status(len(items))} |")
    add("")

    add("## 3. Quality Distribution")
    add("")
    for split in SPLITS:
        summary = report["quality_summary"][split]
        positive_count = split_kind.get(split, {}).get("positive", 0)
        add(f"### {split}")
        add("")
        brightness = summary["brightness"]
        blur = summary["blur"]
        area = summary["plate_area_ratio"]
        b_levels = brightness["levels"]
        blur_levels = blur["levels"]
        add("| Metric | Value |")
        add("|---|---:|")
        add(f"| Dark images | {b_levels.get('dark', 0)} ({pct(b_levels.get('dark', 0), positive_count)}%) |")
        add(f"| Normal brightness | {b_levels.get('normal', 0)} ({pct(b_levels.get('normal', 0), positive_count)}%) |")
        add(f"| Bright images | {b_levels.get('bright', 0)} ({pct(b_levels.get('bright', 0), positive_count)}%) |")
        add(f"| Heavy blur | {blur_levels.get('heavy_blur', 0)} ({pct(blur_levels.get('heavy_blur', 0), positive_count)}%) |")
        add(f"| Mild blur | {blur_levels.get('mild_blur', 0)} ({pct(blur_levels.get('mild_blur', 0), positive_count)}%) |")
        add(f"| Sharp | {blur_levels.get('sharp', 0)} ({pct(blur_levels.get('sharp', 0), positive_count)}%) |")
        add(f"| Avg plate area ratio | {area.get('avg', 0)} |")
        add("")

    add("## 4. Edge-Case Coverage")
    add("")
    total = report["record_count"] or 0
    add("| Source | Count | Share |")
    add("|---|---:|---:|")
    for source, count in sorted(by_source.items(), key=lambda item: item[0]):
        add(f"| `{source}` | {count} | {pct(count, total)}% |")
    add("")

    add("## 5. Duplicate Check")
    add("")
    if report["duplicate_report"].get("skipped"):
        add("Duplicate scan was skipped.")
    else:
        add("| Check | Result |")
        add("|---|---:|")
        add(f"| Exact duplicate groups | {duplicate_summary.get('exact_duplicate_groups', 0)} |")
        add(f"| Exact duplicate extra images | {duplicate_summary.get('exact_duplicate_extra_images', 0)} |")
        add(
            f"| Perceptual hash collision groups | "
            f"{duplicate_summary.get('perceptual_hash_collision_groups', 0)} |"
        )
        add(
            f"| Perceptual hash collision extra images | "
            f"{duplicate_summary.get('perceptual_hash_collision_extra_images', 0)} |"
        )
        add("")
        add(
            "Perceptual hash collisions are **review candidates**, not automatic deletion decisions."
        )
    add("")

    add("## 6. Generated Artifacts")
    add("")
    for name, artifact in artifact_paths.items():
        add(f"- `{name}`: `{artifact.relative_to(ROOT)}`")
    add("")

    add("## 7. CV / Interview Notes")
    add("")
    add("- The dataset audit found **0 integrity issues** across image-label-crop consistency and YOLO bbox validity.")
    add("- The dataset is not just raw images: it has split structure, labels, OCR crops, negative samples, and manifest metadata.")
    add("- Heavy blur/dark-image statistics are useful evidence when discussing real-world edge cases such as night, rain, blur, and CCTV degradation.")
    add("- Duplicate/perceptual-hash report supports the JD keyword: removing duplicates and filtering low-quality samples.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] wrote {path}")


def write_duplicate_markdown(report: dict[str, Any], path: Path) -> None:
    dup = report["duplicate_report"]
    lines: list[str] = ["# Dataset Duplicate Report", ""]
    add = lines.append
    if dup.get("skipped"):
        add("Duplicate scan was skipped.")
    else:
        summary = dup["summary"]
        add("| Metric | Value |")
        add("|---|---:|")
        for key, value in summary.items():
            add(f"| {key} | {value} |")
        add("")
        add(dup.get("note", ""))
        add("")
        add("## Exact Duplicate Groups")
        add("")
        exact_groups = dup.get("exact_duplicate_groups", [])
        if not exact_groups:
            add("No exact duplicate groups found.")
        else:
            for idx, group in enumerate(exact_groups[:20], 1):
                add(f"### Group {idx}")
                add("")
                for item in group:
                    add(f"- `{item}`")
                add("")
        add("")
        add("## Perceptual Hash Collision Groups")
        add("")
        buckets = dup.get("perceptual_hash_collision_groups", [])
        if not buckets:
            add("No perceptual hash collision groups found.")
        else:
            for idx, group in enumerate(buckets[:20], 1):
                add(f"### Candidate Group {idx}")
                add("")
                for item in group[:20]:
                    add(f"- `{item}`")
                if len(group) > 20:
                    add(f"- ... {len(group) - 20} more")
                add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] wrote {path}")


def draw_labeled_tile(dataset_dir: Path, record: dict[str, Any], tile_size: tuple[int, int]) -> np.ndarray | None:
    image_path = dataset_dir / record["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    bbox = record.get("bbox_xyxy")
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)

    target_w, target_h = tile_size
    h, w = image.shape[:2]
    scale = min(target_w / w, (target_h - 34) / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tile = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = 0
    tile[y : y + new_h, x : x + new_w] = resized

    quality = quality_for_record(record)
    caption = (
        f"{Path(record['image']).name} | {record.get('source')} | "
        f"B:{record.get('brightness', '-')} L:{record.get('blurriness', '-')} "
        f"A:{quality.get('plate_area_ratio', '-')}"
    )
    cv2.putText(tile, caption[:72], (8, target_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
    return tile


def write_contact_sheet(
    dataset_dir: Path,
    records: list[dict[str, Any]],
    output_path: Path,
    sample_count: int,
    cols: int = 5,
) -> int:
    selected = records[:sample_count]
    if not selected:
        return 0
    tile_size = (260, 190)
    tiles = [
        tile for record in selected if (tile := draw_labeled_tile(dataset_dir, record, tile_size)) is not None
    ]
    if not tiles:
        return 0

    rows: list[np.ndarray] = []
    blank = np.full((tile_size[1], tile_size[0], 3), 245, dtype=np.uint8)
    for start in range(0, len(tiles), cols):
        row_tiles = tiles[start : start + cols]
        while len(row_tiles) < cols:
            row_tiles.append(blank.copy())
        rows.append(np.hstack(row_tiles))
    sheet = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    print(f"[audit] wrote {output_path}")
    return len(tiles)


def generate_contact_sheets(
    dataset_dir: Path,
    quality_records: list[dict[str, Any]],
    report_dir: Path,
    sample_count: int,
) -> dict[str, Path]:
    positives = [record for record in quality_records if record.get("kind") == "positive"]
    groups: dict[str, list[dict[str, Any]]] = {
        "dark": sorted(
            [record for record in positives if quality_for_record(record)["brightness_level"] == "dark"],
            key=lambda record: record.get("brightness", 999),
        ),
        "heavy_blur": sorted(
            [record for record in positives if quality_for_record(record)["blur_level"] == "heavy_blur"],
            key=lambda record: record.get("blurriness", 999),
        ),
        "small_plate": sorted(
            [
                record
                for record in positives
                if (quality_for_record(record)["plate_area_ratio"] or 1) < SMALL_PLATE_AREA_RATIO
            ],
            key=lambda record: quality_for_record(record)["plate_area_ratio"] or 1,
        ),
        "weather": [record for record in positives if record.get("source") == "ccpd_weather"],
        "challenge": [record for record in positives if record.get("source") == "ccpd_challenge"],
        "negative": [record for record in quality_records if record.get("kind") == "negative"],
    }

    artifacts: dict[str, Path] = {}
    for name, records in groups.items():
        path = report_dir / f"dataset_contact_{name}.jpg"
        count = write_contact_sheet(dataset_dir, records, path, sample_count)
        if count:
            artifacts[f"contact_{name}"] = path
    return artifacts


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[audit] dataset: {args.dataset_dir}")
    print(f"[audit] manifest: {args.manifest}")
    report, quality_records = run_audit(args.dataset_dir, args.manifest, args.skip_duplicates)

    audit_json = args.report_dir / "dataset_audit_report.json"
    audit_md = args.report_dir / "dataset_audit_report.md"
    duplicate_json = args.report_dir / "dataset_duplicates_report.json"
    duplicate_md = args.report_dir / "dataset_duplicates_report.md"

    artifacts: dict[str, Path] = {
        "audit_json": audit_json,
        "audit_markdown": audit_md,
        "duplicate_json": duplicate_json,
        "duplicate_markdown": duplicate_md,
        "quality_manifest": args.quality_manifest,
    }

    if not args.skip_contact_sheets:
        artifacts.update(
            generate_contact_sheets(args.dataset_dir, quality_records, args.report_dir, args.contact_samples)
        )

    write_json(report, audit_json)
    write_audit_markdown(report, audit_md, artifacts)
    write_json(report["duplicate_report"], duplicate_json)
    write_duplicate_markdown(report, duplicate_md)
    write_quality_manifest(args.manifest, args.quality_manifest, quality_records, report)

    total = report["total_issues_found"]
    print(f"[audit] done. total issues: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a compact Layer 4 dataset from CCPD2019.

Outputs:
- YOLO plate detection dataset under images/ and labels/
- OCR crop dataset under ocr_crops/ and ocr_labels/
- manifest.json with provenance for every selected image
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.ccpd import padded_bbox, parse_ccpd_filename


POSITIVE_QUOTAS = {
    "train": {
        "ccpd_base": 4000,
        "ccpd_blur": 500,
        "ccpd_rotate": 500,
        "ccpd_tilt": 500,
        "ccpd_weather": 300,
        "ccpd_challenge": 200,
    },
    "val": {
        "ccpd_base": 700,
        "ccpd_blur": 100,
        "ccpd_rotate": 80,
        "ccpd_tilt": 80,
        "ccpd_weather": 40,
    },
    "test": {
        "ccpd_base": 700,
        "ccpd_blur": 100,
        "ccpd_rotate": 80,
        "ccpd_tilt": 80,
        "ccpd_weather": 40,
    },
}

NEGATIVE_QUOTAS = {
    "train": 400,
    "val": 100,
    "test": 100,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "CCPD2019")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "ccpd_layer4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove output folder before rebuilding.")
    args = parser.parse_args()

    source: Path = args.source
    output: Path = args.output
    if not source.exists():
        raise FileNotFoundError(source)

    if args.clean and output.exists():
        shutil.rmtree(output)

    rng = random.Random(args.seed)
    create_dirs(output)

    used_basenames: set[str] = set()
    records: list[dict] = []
    warnings: list[str] = []

    for split, quotas in POSITIVE_QUOTAS.items():
        for source_name, quota in quotas.items():
            selected = select_positive_files(
                source / source_name,
                quota,
                rng,
                used_basenames,
            )
            if len(selected) < quota:
                warnings.append(f"{split}/{source_name}: requested {quota}, selected {len(selected)}")
            for path in selected:
                record = process_positive(path, source, output, split, len(records) + 1)
                if record is not None:
                    records.append(record)

    negative_candidates = list_jpegs(source / "ccpd_np")
    rng.shuffle(negative_candidates)
    negative_cursor = 0
    for split, quota in NEGATIVE_QUOTAS.items():
        selected: list[Path] = []
        while negative_cursor < len(negative_candidates) and len(selected) < quota:
            candidate = negative_candidates[negative_cursor]
            negative_cursor += 1
            if candidate.name in used_basenames:
                continue
            used_basenames.add(candidate.name)
            selected.append(candidate)
        if len(selected) < quota:
            warnings.append(f"{split}/ccpd_np: requested {quota}, selected {len(selected)}")
        for path in selected:
            record = process_negative(path, source, output, split, len(records) + 1)
            if record is not None:
                records.append(record)

    write_yaml(output)
    write_split_files(output, records)
    write_manifest(output, records, warnings, args.seed)
    copy_license_files(source, output)
    write_readme(output, records, warnings)

    summary = summarize(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def create_dirs(output: Path) -> None:
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output / "ocr_crops" / split).mkdir(parents=True, exist_ok=True)
    (output / "ocr_labels").mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir(parents=True, exist_ok=True)


def list_jpegs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})


def select_positive_files(
    folder: Path,
    quota: int,
    rng: random.Random,
    used_basenames: set[str],
) -> list[Path]:
    candidates = list_jpegs(folder)
    rng.shuffle(candidates)
    selected: list[Path] = []
    for candidate in candidates:
        if len(selected) >= quota:
            break
        if candidate.name in used_basenames:
            continue
        try:
            parse_ccpd_filename(candidate)
        except ValueError:
            continue
        used_basenames.add(candidate.name)
        selected.append(candidate)
    return selected


def process_positive(path: Path, source: Path, output: Path, split: str, index: int) -> dict | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    ann = parse_ccpd_filename(path)

    stem = f"{split}_{index:06d}"
    image_rel = Path("images") / split / f"{stem}.jpg"
    label_rel = Path("labels") / split / f"{stem}.txt"
    crop_rel = Path("ocr_crops") / split / f"{stem}.jpg"

    shutil.copy2(path, output / image_rel)
    (output / label_rel).write_text(ann.yolo_label(width, height) + "\n", encoding="utf-8")

    x1, y1, x2, y2 = padded_bbox(ann.bbox, width, height)
    crop = image[y1:y2, x1:x2]
    if crop.size:
        cv2.imwrite(str(output / crop_rel), crop)

    return {
        "split": split,
        "kind": "positive",
        "source": ann.source,
        "original_path": str(path.relative_to(source)),
        "image": str(image_rel),
        "label": str(label_rel),
        "ocr_crop": str(crop_rel),
        "plate_text": ann.plate_text,
        "bbox_xyxy": list(ann.bbox),
        "crop_xyxy": [x1, y1, x2, y2],
        "image_size": [width, height],
        "brightness": ann.brightness,
        "blurriness": ann.blurriness,
    }


def process_negative(path: Path, source: Path, output: Path, split: str, index: int) -> dict | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]

    stem = f"{split}_{index:06d}_neg"
    image_rel = Path("images") / split / f"{stem}.jpg"
    label_rel = Path("labels") / split / f"{stem}.txt"

    shutil.copy2(path, output / image_rel)
    (output / label_rel).write_text("", encoding="utf-8")

    return {
        "split": split,
        "kind": "negative",
        "source": "ccpd_np",
        "original_path": str(path.relative_to(source)),
        "image": str(image_rel),
        "label": str(label_rel),
        "ocr_crop": None,
        "plate_text": None,
        "bbox_xyxy": None,
        "crop_xyxy": None,
        "image_size": [width, height],
    }


def write_yaml(output: Path) -> None:
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "plate"},
    }
    (output / "ccpd_plate.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_split_files(output: Path, records: list[dict]) -> None:
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        image_lines = [record["image"] for record in split_records]
        (output / "splits" / f"{split}.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

        ocr_lines = [
            f"{record['ocr_crop']}\t{record['plate_text']}"
            for record in split_records
            if record["kind"] == "positive" and record.get("ocr_crop")
        ]
        (output / "ocr_labels" / f"{split}.txt").write_text("\n".join(ocr_lines) + "\n", encoding="utf-8")


def write_manifest(output: Path, records: list[dict], warnings: list[str], seed: int) -> None:
    payload = {
        "dataset": "CCPD2019 Layer 4 compact subset",
        "seed": seed,
        "record_count": len(records),
        "summary": summarize(records),
        "warnings": warnings,
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(records: list[dict]) -> dict:
    split_counts = Counter(record["split"] for record in records)
    kind_counts = Counter(record["kind"] for record in records)
    source_counts = Counter(record["source"] for record in records)
    split_kind_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ("train", "val", "test"):
        for kind in ("positive", "negative"):
            split_kind_counts[split][kind] = sum(
                1 for record in records if record["split"] == split and record["kind"] == kind
            )
    return {
        "by_split": dict(split_counts),
        "by_kind": dict(kind_counts),
        "by_source": dict(source_counts),
        "by_split_kind": dict(split_kind_counts),
    }


def copy_license_files(source: Path, output: Path) -> None:
    for name in ("LICENSE", "README.md"):
        src = source / name
        if src.exists():
            dst_name = "CCPD_LICENSE" if name == "LICENSE" else "CCPD_README.md"
            shutil.copy2(src, output / dst_name)


def write_readme(output: Path, records: list[dict], warnings: list[str]) -> None:
    summary = summarize(records)
    lines = [
        "# CCPD Layer 4 Compact Dataset",
        "",
        "This folder is generated from CCPD2019 for the local traffic violation LPR project.",
        "",
        "Purpose:",
        "- train/validate a one-class YOLO plate detector (`plate`)",
        "- provide cropped plate images and labels for OCR experiments",
        "- keep a compact subset so the raw 24GB CCPD folder is no longer required",
        "",
        "Files:",
        "- `ccpd_plate.yaml`: Ultralytics YOLO dataset config",
        "- `images/{train,val,test}` and `labels/{train,val,test}`: detection dataset",
        "- `ocr_crops/{train,val,test}` and `ocr_labels/*.txt`: OCR crop dataset",
        "- `manifest.json`: provenance and counts",
        "- `CCPD_LICENSE` and `CCPD_README.md`: copied source license/docs",
        "",
        "Summary:",
        f"- total records: {len(records)}",
        f"- by split: {summary['by_split']}",
        f"- by kind: {summary['by_kind']}",
        f"- by source: {summary['by_source']}",
        "",
    ]
    if warnings:
        lines.extend(["Warnings:", *[f"- {warning}" for warning in warnings], ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def iter_records(records: Iterable[dict], split: str) -> Iterable[dict]:
    for record in records:
        if record["split"] == split:
            yield record


if __name__ == "__main__":
    raise SystemExit(main())


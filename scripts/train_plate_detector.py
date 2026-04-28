#!/usr/bin/env python3
"""Train a one-class YOLO license plate detector on the compact CCPD subset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/ccpd_layer4/ccpd_plate.yaml"))
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--project", default="runs/plate_detector")
    parser.add_argument("--name", default="ccpd_yolov8n_layer4")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--export", type=Path, default=Path("models/plate_detector/ccpd_yolov8n_best.pt"))
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    model = YOLO(args.base_model)
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=2,
        exist_ok=True,
    )

    run_dir = Path(result.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best}")

    args.export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export)
    print(f"Best checkpoint: {best}")
    print(f"Exported checkpoint: {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


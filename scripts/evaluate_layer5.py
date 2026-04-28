#!/usr/bin/env python3
"""Layer 5 evaluation and review-template generator.

This script intentionally separates two kinds of results:
- pipeline/evidence readiness metrics, computed directly from Layer 4 outputs
- quality metrics, computed only from manually reviewed labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", type=Path, default=Path("outputs/events_layer4"))
    parser.add_argument("--layer4-report", type=Path, default=Path("outputs/reports/layer4_ocr_report.json"))
    parser.add_argument("--review", type=Path, default=Path("data/annotations/cam_01_layer5_review.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/layer5_evaluation_report.json"))
    args = parser.parse_args()

    events = load_events(args.events_dir)
    layer4_report = load_json(args.layer4_report, default={})
    review = load_or_create_review(args.review, events, layer4_report)
    report = evaluate(events, layer4_report, review)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summarize_for_console(report), ensure_ascii=False, indent=2))
    print(f"Review file: {args.review}")
    print(f"Report file: {args.output}")
    return 0


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(events_dir: Path) -> list[dict]:
    events: list[dict] = []
    for event_path in sorted(events_dir.glob("EVT_*/event.json")):
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["event_dir"] = str(event_path.parent)
        events.append(event)
    return events


def load_or_create_review(review_path: Path, events: list[dict], layer4_report: dict) -> dict:
    if review_path.exists():
        return json.loads(review_path.read_text(encoding="utf-8"))

    review_path.parent.mkdir(parents=True, exist_ok=True)
    review = {
        "camera_id": "CAM_01",
        "video": layer4_report.get("video", "data/raw_videos/clips/cam_01_clip_001.mp4"),
        "status": "needs_manual_review",
        "review_scope": (
            "configured_stop_line_rule: event is true when the target vehicle crosses "
            "the configured stop line while the configured traffic-light ROI is red. "
            "This is not a legal ruling because lane-specific turn permissions are not modeled."
        ),
        "instructions": [
            "Open each event frame/clip and set is_true_violation to true/false.",
            "Set light_state_verified after checking the traffic light in the frame/clip.",
            "If the plate is readable by eye, fill plate_text_gt.",
            "Set ocr_correct only when plate_text_gt is filled and the predicted plate exactly matches the reviewed text.",
            "Event Precision can be reported after all is_true_violation fields are filled.",
            "Plate Accuracy can be reported only for events with readable plate_text_gt.",
        ],
        "events": [review_event_template(event) for event in events],
    }
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    return review


def review_event_template(event: dict) -> dict:
    return {
        "event_id": event["event_id"],
        "predicted_state": event.get("state"),
        "predicted_track_id": event.get("track_id"),
        "frame_idx": event.get("frame_idx"),
        "predicted_light_state": event.get("light_state"),
        "light_state_verified": None,
        "is_true_violation": None,
        "plate_visible": None,
        "plate_readable": None,
        "plate_text_pred": event.get("plate_text"),
        "plate_text_gt": None,
        "ocr_correct": None,
        "evidence_image_path": event.get("evidence_image_path"),
        "plate_crop_path": event.get("plate_crop_path"),
        "evidence_clip_path": event.get("evidence_clip_path"),
        "review_notes": "",
    }


def evaluate(events: list[dict], layer4_report: dict, review: dict) -> dict:
    review_by_id = {item["event_id"]: item for item in review.get("events", [])}
    evidence_metrics = compute_evidence_metrics(events, layer4_report)
    quality_metrics = compute_quality_metrics(events, review_by_id)

    event_review_complete = quality_metrics["reviewed_event_count"] == len(events) and len(events) > 0
    if not event_review_complete:
        status = "pending_manual_review"
    elif quality_metrics["plate_reviewed_count"] == 0:
        status = "event_review_ready_plate_gt_pending"
    else:
        status = "ready"

    return {
        "layer": "layer_5_evaluation_portfolio_polish",
        "status": status,
        "event_source_count": len(events),
        "review_file_status": review.get("status", "unknown"),
        "review_scope": review.get("review_scope"),
        "evidence_metrics": evidence_metrics,
        "quality_metrics": quality_metrics,
        "events": [
            {
                "event_id": event["event_id"],
                "prediction": {
                    "state": event.get("state"),
                    "plate_text": event.get("plate_text"),
                    "plate_confidence": event.get("plate_confidence"),
                    "light_state": event.get("light_state"),
                    "frame_idx": event.get("frame_idx"),
                },
                "review": review_by_id.get(event["event_id"], {}),
            }
            for event in events
        ],
        "notes": [
            "Event precision is candidate precision under the configured stop-line rule.",
            "Event recall requires full timeline ground truth, not only predicted-event review.",
            "Plate accuracy is computed only on events with manually readable plate_text_gt.",
        ],
    }


def compute_evidence_metrics(events: list[dict], layer4_report: dict) -> dict:
    total = len(events)
    with_frame = sum(1 for event in events if path_exists(event.get("evidence_image_path")))
    with_plate = sum(1 for event in events if path_exists(event.get("plate_crop_path")))
    with_clip = sum(1 for event in events if path_exists(event.get("evidence_clip_path")))
    with_text = sum(1 for event in events if event.get("plate_text"))
    confirmed = sum(1 for event in events if event.get("state") == "confirmed")
    return {
        "total_events": total,
        "events_with_frame": with_frame,
        "events_with_plate_crop": with_plate,
        "events_with_clip": with_clip,
        "events_with_plate_text": with_text,
        "confirmed_events": confirmed,
        "frame_coverage": safe_ratio(with_frame, total),
        "plate_crop_coverage": safe_ratio(with_plate, total),
        "clip_coverage": safe_ratio(with_clip, total),
        "plate_text_coverage": safe_ratio(with_text, total),
        "processing_fps": layer4_report.get("processing_fps"),
        "tracker_backend": layer4_report.get("tracker_backend"),
        "plate_detector_backend": layer4_report.get("plate_detector_backend"),
        "ocr_backend": layer4_report.get("ocr_backend"),
    }


def compute_quality_metrics(events: list[dict], review_by_id: dict[str, dict]) -> dict:
    reviewed = [
        item
        for item in review_by_id.values()
        if item.get("is_true_violation") is not None
    ]
    true_positives = sum(1 for item in reviewed if item.get("is_true_violation") is True)
    false_positives = sum(1 for item in reviewed if item.get("is_true_violation") is False)

    plate_visible = [
        item
        for item in review_by_id.values()
        if item.get("plate_visible") is True
    ]
    plate_readable = [
        item
        for item in plate_visible
        if item.get("plate_text_gt")
    ]
    plate_reviewed = [
        item
        for item in plate_readable
        if item.get("ocr_correct") is not None
    ]
    ocr_correct = sum(1 for item in plate_reviewed if item.get("ocr_correct") is True)

    # Recall requires ground-truth violations that were not predicted; this
    # small candidate-only review file cannot know that without a full timeline annotation.
    event_precision = safe_ratio(true_positives, true_positives + false_positives)
    plate_accuracy = safe_ratio(ocr_correct, len(plate_reviewed))

    return {
        "reviewed_event_count": len(reviewed),
        "unreviewed_event_count": max(0, len(events) - len(reviewed)),
        "true_positive_events": true_positives,
        "false_positive_events": false_positives,
        "event_precision": event_precision,
        "event_recall": None,
        "event_recall_note": "Requires full timeline ground truth, not only predicted-event review.",
        "plate_visible_count": len(plate_visible),
        "plate_readable_count": len(plate_readable),
        "plate_gt_missing_count": max(0, len(plate_visible) - len(plate_readable)),
        "plate_reviewed_count": len(plate_reviewed),
        "ocr_correct_count": ocr_correct,
        "plate_accuracy": plate_accuracy,
        "plate_accuracy_note": (
            "Computed only for events with readable manual plate_text_gt and explicit ocr_correct."
        ),
    }


def path_exists(value: str | None) -> bool:
    return bool(value) and Path(value).exists()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def summarize_for_console(report: dict) -> dict:
    return {
        "status": report["status"],
        "evidence_metrics": report["evidence_metrics"],
        "quality_metrics": report["quality_metrics"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

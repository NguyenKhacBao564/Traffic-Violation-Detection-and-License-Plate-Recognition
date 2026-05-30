"""FastAPI wrapper for local traffic violation analysis and evidence review."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.event_reader import (
    ALLOWED_REVIEW_STATUSES,
    EventReadError,
    list_events,
    media_path,
    read_event,
    update_review,
)


ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    default_events_dir = os.getenv("DEFAULT_EVENTS_DIR", "outputs/events_api")
    default_max_frames = int(os.getenv("DEFAULT_MAX_FRAMES", "900"))
    default_ocr_backend = os.getenv("DEFAULT_OCR_BACKEND", "hyperlpr")
    enable_api_debug = _env_bool("ENABLE_API_DEBUG", False)


settings = Settings()
logging.basicConfig(
    level=logging.DEBUG if settings.enable_api_debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("traffic_violation_api")

app = FastAPI(
    title="Traffic Violation Evidence Review API",
    version="0.1.0",
    description="Local FastAPI wrapper around the offline traffic violation + ANPR pipeline.",
)


class AnalyzeVideoRequest(BaseModel):
    video_path: str
    camera_config: str = "configs/cameras/cam_01.json"
    events_dir: str = Field(default_factory=lambda: settings.default_events_dir)
    max_frames: int = Field(default_factory=lambda: settings.default_max_frames, ge=0)
    enable_ocr: bool = True
    ocr_backend: str = Field(default_factory=lambda: settings.default_ocr_backend)


class ReviewUpdate(BaseModel):
    review_status: str
    review_note: str | None = None


def _resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _report_summary(report_path: Path, events_dir: str, runtime_sec: float) -> dict:
    if report_path.exists():
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}

    events = list_events(events_dir, limit=10_000)
    processed_frames = report.get("processed_frames", 0)
    processing_fps = report.get("processing_fps")
    if processing_fps is None and runtime_sec > 0:
        processing_fps = processed_frames / runtime_sec

    return {
        "processed_frames": processed_frames,
        "violation_candidates": report.get("event_count", report.get("predicted_event_count", len(events))),
        "events_with_plate": report.get(
            "events_with_plate_crop",
            sum(1 for event in events if event.get("plate_path")),
        ),
        "ocr_events": report.get(
            "events_with_plate_text",
            sum(1 for event in events if event.get("plate_text")),
        ),
        "processing_fps": round(float(processing_fps or 0.0), 2),
        "total_runtime_seconds": round(runtime_sec, 2),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "traffic-violation-evidence-api"}


@app.post("/api/v1/videos/analyze")
def analyze_video(payload: AnalyzeVideoRequest) -> dict:
    video_path = _resolve_repo_path(payload.video_path)
    camera_config = _resolve_repo_path(payload.camera_config)
    events_dir = payload.events_dir
    events_path = _resolve_repo_path(events_dir)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"video_path not found: {payload.video_path}")
    if not camera_config.exists():
        raise HTTPException(status_code=404, detail=f"camera_config not found: {payload.camera_config}")

    job_id = f"local_sync_{int(time.time())}_{uuid4().hex[:8]}"
    layer3_report = ROOT / "outputs" / "reports" / f"{job_id}_layer3_report.json"
    layer4_report = ROOT / "outputs" / "reports" / f"{job_id}_layer4_report.json"
    cmd = [
        sys.executable,
        "src/main.py",
        "--video",
        payload.video_path,
        "--camera",
        payload.camera_config,
        "--events-dir",
        events_dir,
        "--layer3-report",
        str(layer3_report.relative_to(ROOT)),
        "--layer4-report",
        str(layer4_report.relative_to(ROOT)),
        "--no-debug-video",
    ]
    if payload.max_frames > 0:
        cmd.extend(["--max-frames", str(payload.max_frames)])
    if payload.enable_ocr:
        cmd.extend(["--enable-ocr", "--ocr-backend", payload.ocr_backend])

    logger.info(
        "starting_analysis video_path=%s camera_config=%s max_frames=%s enable_ocr=%s "
        "ocr_backend=%s events_dir=%s job_id=%s",
        payload.video_path,
        payload.camera_config,
        payload.max_frames,
        payload.enable_ocr,
        payload.ocr_backend,
        events_dir,
        job_id,
    )

    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    runtime_sec = time.time() - start

    report_path = layer4_report if payload.enable_ocr else layer3_report
    summary = _report_summary(report_path, events_dir, runtime_sec)
    logger.info(
        "completed_analysis job_id=%s returncode=%s processed_frames=%s violation_candidates=%s "
        "ocr_events=%s processing_fps=%s total_runtime_seconds=%.2f events_dir=%s",
        job_id,
        result.returncode,
        summary["processed_frames"],
        summary["violation_candidates"],
        summary["ocr_events"],
        summary["processing_fps"],
        runtime_sec,
        events_dir,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "job_id": job_id,
                "status": "failed",
                "stderr_tail": result.stderr[-4000:],
                "stdout_tail": result.stdout[-4000:],
            },
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "events_dir": str(events_path.relative_to(ROOT)) if events_path.is_relative_to(ROOT) else str(events_path),
        "summary": summary,
        "reports": {
            "layer3_report": str(layer3_report.relative_to(ROOT)),
            "layer4_report": str(layer4_report.relative_to(ROOT)) if payload.enable_ocr else None,
        },
    }


@app.get("/api/v1/events")
def get_events(
    events_dir: str = settings.default_events_dir,
    limit: int = Query(100, ge=0, le=10_000),
    plate_text: str | None = None,
    review_status: str | None = None,
) -> dict:
    if review_status and review_status not in ALLOWED_REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid review_status")
    events = list_events(events_dir, limit=limit, plate_text=plate_text, review_status=review_status)
    return {"events_dir": events_dir, "count": len(events), "events": events}


@app.get("/api/v1/events/{event_id}")
def get_event(event_id: str, events_dir: str = settings.default_events_dir) -> dict:
    try:
        event = read_event(events_dir, event_id)
    except EventReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if event is None:
        raise HTTPException(status_code=404, detail=f"event not found: {event_id}")
    return event


def _file_response(events_dir: str, event_id: str, media_type: str, media_type_header: str) -> FileResponse:
    try:
        path = media_path(events_dir, event_id, media_type)
    except EventReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"{media_type} not found for event: {event_id}")
    return FileResponse(path, media_type=media_type_header)


@app.get("/api/v1/events/{event_id}/frame")
def get_event_frame(
    event_id: str,
    events_dir: str = settings.default_events_dir,
) -> FileResponse:
    return _file_response(events_dir, event_id, "frame", "image/jpeg")


@app.get("/api/v1/events/{event_id}/plate")
def get_event_plate(
    event_id: str,
    events_dir: str = settings.default_events_dir,
) -> FileResponse:
    return _file_response(events_dir, event_id, "plate", "image/jpeg")


@app.get("/api/v1/events/{event_id}/clip")
def get_event_clip(
    event_id: str,
    events_dir: str = settings.default_events_dir,
) -> FileResponse:
    return _file_response(events_dir, event_id, "clip", "video/mp4")


@app.patch("/api/v1/events/{event_id}/review")
def patch_event_review(
    event_id: str,
    payload: ReviewUpdate,
    events_dir: str = settings.default_events_dir,
) -> dict:
    try:
        return update_review(
            events_dir,
            event_id,
            review_status=payload.review_status,
            review_note=payload.review_note,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"event not found: {event_id}") from exc
    except EventReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

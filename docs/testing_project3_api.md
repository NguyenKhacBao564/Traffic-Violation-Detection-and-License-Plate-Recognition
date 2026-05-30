# Testing The Local Evidence Review API

This project keeps the original offline CLI pipeline. The FastAPI layer is a lightweight wrapper for local analysis and evidence review.

## Run The API

```bash
python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Optional environment defaults:

```bash
export DEFAULT_EVENTS_DIR=outputs/events_api
export DEFAULT_MAX_FRAMES=900
export DEFAULT_OCR_BACKEND=hyperlpr
export ENABLE_API_DEBUG=false
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "ok",
  "service": "traffic-violation-evidence-api"
}
```

## Analyze A Video

This call runs the existing CLI pipeline synchronously through the API.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/videos/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "data/raw_videos/clips/cam_01_clip_001.mp4",
    "camera_config": "configs/cameras/cam_01.json",
    "events_dir": "outputs/events_api",
    "max_frames": 900,
    "enable_ocr": true,
    "ocr_backend": "hyperlpr"
  }'
```

Example response shape:

```json
{
  "job_id": "local_sync_...",
  "status": "completed",
  "events_dir": "outputs/events_api",
  "summary": {
    "processed_frames": 900,
    "violation_candidates": 4,
    "events_with_plate": 3,
    "ocr_events": 2,
    "processing_fps": 18.7,
    "total_runtime_seconds": 48.1
  },
  "reports": {
    "layer3_report": "outputs/reports/local_sync_..._layer3_report.json",
    "layer4_report": "outputs/reports/local_sync_..._layer4_report.json"
  }
}
```

## List Events

```bash
curl "http://127.0.0.1:8000/api/v1/events?events_dir=outputs/events_api&limit=20"
```

Filter by plate text:

```bash
curl "http://127.0.0.1:8000/api/v1/events?events_dir=outputs/events_api&plate_text=京"
```

Filter by review status:

```bash
curl "http://127.0.0.1:8000/api/v1/events?events_dir=outputs/events_api&review_status=confirmed"
```

## Get One Event

```bash
curl "http://127.0.0.1:8000/api/v1/events/EVT_0001?events_dir=outputs/events_api"
```

## Get Evidence Files

Frame image:

```bash
curl -o frame.jpg \
  "http://127.0.0.1:8000/api/v1/events/EVT_0001/frame?events_dir=outputs/events_api"
```

Plate crop:

```bash
curl -o plate.jpg \
  "http://127.0.0.1:8000/api/v1/events/EVT_0001/plate?events_dir=outputs/events_api"
```

Clip, if present:

```bash
curl -o clip.mp4 \
  "http://127.0.0.1:8000/api/v1/events/EVT_0001/clip?events_dir=outputs/events_api"
```

## Update Review Status

Allowed statuses:

- `pending`
- `confirmed`
- `rejected`
- `uncertain`

```bash
curl -X PATCH \
  "http://127.0.0.1:8000/api/v1/events/EVT_0001/review?events_dir=outputs/events_api" \
  -H "Content-Type: application/json" \
  -d '{
    "review_status": "confirmed",
    "review_note": "Clear stop-line crossing during red light"
  }'
```

The API writes a transparent `review.json` inside the event folder instead of requiring a database.

## Notes

- This API is intentionally local and unauthenticated.
- It does not add a database, queue, Celery, Kafka, or RabbitMQ.
- It does not retrain or modify model weights.
- Generated videos, event folders, frames, clips, snapshots, and model weights should remain ignored by git.

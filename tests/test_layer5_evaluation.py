import importlib.util
from pathlib import Path


def load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_layer5.py"
    spec = importlib.util.spec_from_file_location("evaluate_layer5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_metrics_wait_for_manual_review():
    mod = load_eval_module()
    events = [{"event_id": "EVT_0001"}]
    review = {
        "EVT_0001": {
            "event_id": "EVT_0001",
            "is_true_violation": None,
            "plate_visible": None,
            "plate_readable": None,
            "ocr_correct": None,
        }
    }

    metrics = mod.compute_quality_metrics(events, review)

    assert metrics["reviewed_event_count"] == 0
    assert metrics["event_precision"] is None
    assert metrics["plate_visible_count"] == 0
    assert metrics["plate_gt_missing_count"] == 0
    assert metrics["plate_accuracy"] is None


def test_quality_metrics_compute_precision_and_plate_accuracy():
    mod = load_eval_module()
    events = [{"event_id": "EVT_0001"}, {"event_id": "EVT_0002"}]
    review = {
        "EVT_0001": {
            "event_id": "EVT_0001",
            "is_true_violation": True,
            "plate_visible": True,
            "plate_readable": True,
            "plate_text_gt": "京A12345",
            "ocr_correct": True,
        },
        "EVT_0002": {
            "event_id": "EVT_0002",
            "is_true_violation": False,
            "plate_visible": True,
            "plate_readable": True,
            "plate_text_gt": "京B54321",
            "ocr_correct": False,
        },
    }

    metrics = mod.compute_quality_metrics(events, review)

    assert metrics["reviewed_event_count"] == 2
    assert metrics["event_precision"] == 0.5
    assert metrics["plate_visible_count"] == 2
    assert metrics["plate_readable_count"] == 2
    assert metrics["plate_accuracy"] == 0.5


def test_quality_metrics_keeps_plate_accuracy_pending_without_gt():
    mod = load_eval_module()
    events = [{"event_id": "EVT_0001"}]
    review = {
        "EVT_0001": {
            "event_id": "EVT_0001",
            "is_true_violation": True,
            "plate_visible": True,
            "plate_readable": False,
            "plate_text_gt": None,
            "ocr_correct": None,
        }
    }

    metrics = mod.compute_quality_metrics(events, review)

    assert metrics["reviewed_event_count"] == 1
    assert metrics["event_precision"] == 1.0
    assert metrics["plate_visible_count"] == 1
    assert metrics["plate_readable_count"] == 0
    assert metrics["plate_gt_missing_count"] == 1
    assert metrics["plate_accuracy"] is None

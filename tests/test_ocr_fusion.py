"""Tests for plate/fusion.py"""
from src.plate.fusion import OCRFusion


def test_single_reading():
    """Single OCR reading should return that text."""
    fusion = OCRFusion(vote_frames=3)
    fusion.add_reading("51H-12345", 0.9)
    result = fusion.fuse()
    assert result.final_text == "51H-12345"
    assert result.confidence > 0.0


def test_majority_vote_wins():
    """Text appearing most should win."""
    fusion = OCRFusion(vote_frames=5)
    for _ in range(3):
        fusion.add_reading("51H-12345", 0.9)
    fusion.add_reading("51H-1234S", 0.8)  # wrong
    fusion.add_reading("51H-1234S", 0.8)  # wrong
    result = fusion.fuse()
    assert result.final_text == "51H-12345"


def test_empty_readings_returns_empty():
    """No readings should return empty result."""
    fusion = OCRFusion()
    result = fusion.fuse()
    assert result.final_text == ""


def test_short_readings_ignored():
    """Readings shorter than 3 chars should be ignored."""
    fusion = OCRFusion()
    fusion.add_reading("AB", 0.5)  # too short
    fusion.add_reading("", 0.9)    # empty
    result = fusion.fuse()
    assert result.final_text == ""

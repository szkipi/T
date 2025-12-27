"""Unit tests for BaseBallTracker and related classes."""

import numpy as np
import pytest

from tennis30.src.core.tracking.ball.base_tracker import BaseBallTracker, DummyBallTracker


class TestBaseBallTracker:
    """Tests for BaseBallTracker abstract class."""

    def test_weight_initialization(self):
        """Test that weight is correctly initialized from config."""
        config = {"weight": 0.3, "threshold": 0.5}
        tracker = DummyBallTracker(config)

        assert tracker.weight == 0.3
        assert tracker.threshold == 0.5

    def test_default_weight(self):
        """Test default weight when not specified in config."""
        config = {}
        tracker = DummyBallTracker(config)

        assert tracker.weight == 1.0

    def test_set_weight_valid(self):
        """Test setting a valid weight."""
        tracker = DummyBallTracker({})
        tracker.set_weight(0.5)

        assert tracker.weight == 0.5

    def test_set_weight_invalid(self):
        """Test that invalid weights raise ValueError."""
        tracker = DummyBallTracker({})

        with pytest.raises(ValueError, match="Weight must be in"):
            tracker.set_weight(1.5)

        with pytest.raises(ValueError, match="Weight must be in"):
            tracker.set_weight(-0.1)

    def test_is_valid_detection_above_threshold(self):
        """Test detection validation with confidence above threshold."""
        config = {"threshold": 0.5}
        tracker = DummyBallTracker(config)

        detection = {"confidence": 0.7}
        assert tracker.is_valid_detection(detection) is True

    def test_is_valid_detection_below_threshold(self):
        """Test detection validation with confidence below threshold."""
        config = {"threshold": 0.5}
        tracker = DummyBallTracker(config)

        detection = {"confidence": 0.3}
        assert tracker.is_valid_detection(detection) is False

    def test_is_valid_detection_exactly_at_threshold(self):
        """Test detection validation at exact threshold."""
        config = {"threshold": 0.5}
        tracker = DummyBallTracker(config)

        detection = {"confidence": 0.5}
        assert tracker.is_valid_detection(detection) is True


class TestDummyBallTracker:
    """Tests for DummyBallTracker implementation."""

    def test_detect_returns_fixed_position(self):
        """Test that detect returns the configured fixed position."""
        fixed_pos = (100, 200)
        fixed_conf = 0.8
        tracker = DummyBallTracker(
            {},
            fixed_position=fixed_pos,
            fixed_confidence=fixed_conf,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.detect(frame)

        assert result["position"] == fixed_pos
        assert result["confidence"] == fixed_conf
        assert result["bbox"] is None

    def test_detect_batch_returns_same_for_all_frames(self):
        """Test that detect_batch returns consistent results."""
        fixed_pos = (150, 250)
        tracker = DummyBallTracker({}, fixed_position=fixed_pos)

        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        results = tracker.detect_batch(frames)

        assert len(results) == 5
        for result in results:
            assert result["position"] == fixed_pos

    def test_load_model_is_noop(self):
        """Test that load_model does nothing for dummy tracker."""
        tracker = DummyBallTracker({})
        # Should not raise any exceptions
        tracker.load_model("fake_checkpoint.pth")

    def test_repr(self):
        """Test string representation."""
        tracker = DummyBallTracker({"weight": 0.25, "threshold": 0.4})
        repr_str = repr(tracker)

        assert "DummyBallTracker" in repr_str
        assert "weight=0.25" in repr_str
        assert "threshold=0.4" in repr_str


class TestTrackerComparison:
    """Tests for comparing multiple trackers."""

    def test_multiple_trackers_different_weights(self):
        """Test ensemble scenario with multiple trackers."""
        trackers = [
            DummyBallTracker({"weight": 0.30}, fixed_position=(100, 200)),
            DummyBallTracker({"weight": 0.25}, fixed_position=(102, 198)),
            DummyBallTracker({"weight": 0.15}, fixed_position=(98, 201)),
        ]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Collect detections
        detections = [tracker.detect(frame) for tracker in trackers]

        # Verify each tracker maintains its weight
        assert trackers[0].get_weight() == 0.30
        assert trackers[1].get_weight() == 0.25
        assert trackers[2].get_weight() == 0.15

        # Verify positions are different
        positions = [d["position"] for d in detections]
        assert len(set(positions)) == 3  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

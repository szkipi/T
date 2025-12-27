"""Kalman filter for ball trajectory smoothing.

Implements an 8-dimensional state space Kalman filter for tracking
tennis ball position with velocity estimation.
"""

from typing import Optional, Tuple

import numpy as np


class KalmanFilter:
    """8D Kalman filter for ball tracking.

    State vector: [x, y, aspect_ratio, height, vx, vy, va, vh]
    - x, y: Position
    - aspect_ratio: Bounding box aspect ratio (width/height)
    - height: Bounding box height
    - vx, vy: Velocity in x, y
    - va, vh: Velocity of aspect ratio and height

    This is adapted from the SORT tracking algorithm and SAMURAI.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
    ) -> None:
        """Initialize Kalman filter.

        Args:
            initial_state: Initial state vector (8,). If None, will be set on first update.
            process_noise_scale: Scale factor for process noise covariance
            measurement_noise_scale: Scale factor for measurement noise covariance
        """
        self.ndim = 8  # State dimensionality
        self.dt = 1.0  # Time step (1 frame)

        # Motion model: constant velocity
        self._motion_mat = np.eye(self.ndim)
        for i in range(4):
            self._motion_mat[i, i + 4] = self.dt

        # Measurement model: we observe [x, y, a, h]
        self._measurement_mat = np.eye(4, self.ndim)

        # Process noise covariance
        self._process_noise = np.eye(self.ndim) * process_noise_scale
        self._process_noise[4:, 4:] *= 10.0  # Higher uncertainty in velocities

        # Measurement noise covariance
        self._measurement_noise = np.eye(4) * measurement_noise_scale

        # State and covariance
        self.mean: Optional[np.ndarray] = initial_state
        self.covariance: Optional[np.ndarray] = None

        if initial_state is not None:
            self.covariance = np.eye(self.ndim) * 10.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize filter state from first measurement.

        Args:
            measurement: Initial measurement [x, y, aspect_ratio, height]

        Returns:
            Tuple of (mean, covariance)
        """
        mean = np.zeros(self.ndim)
        mean[:4] = measurement
        mean[4:] = 0  # Zero initial velocity

        # Initial covariance
        covariance = np.eye(self.ndim)
        covariance[:4, :4] *= 10.0  # Moderate uncertainty in position
        covariance[4:, 4:] *= 100.0  # High uncertainty in velocity

        self.mean = mean
        self.covariance = covariance

        return mean, covariance

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state.

        Returns:
            Tuple of (predicted_mean, predicted_covariance)

        Raises:
            RuntimeError: If filter not initialized
        """
        if self.mean is None or self.covariance is None:
            raise RuntimeError("Filter not initialized. Call initiate() first.")

        # Predict state
        mean = self._motion_mat @ self.mean

        # Predict covariance
        covariance = (
            self._motion_mat @ self.covariance @ self._motion_mat.T
            + self._process_noise
        )

        self.mean = mean
        self.covariance = covariance

        return mean, covariance

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update filter with measurement.

        Args:
            measurement: Measurement vector [x, y, aspect_ratio, height]

        Returns:
            Tuple of (updated_mean, updated_covariance)

        Raises:
            RuntimeError: If filter not initialized
        """
        if self.mean is None or self.covariance is None:
            raise RuntimeError("Filter not initialized. Call initiate() first.")

        # Predicted measurement
        predicted_measurement = self._measurement_mat @ self.mean

        # Innovation (measurement residual)
        innovation = measurement - predicted_measurement

        # Innovation covariance
        innovation_cov = (
            self._measurement_mat @ self.covariance @ self._measurement_mat.T
            + self._measurement_noise
        )

        # Kalman gain
        kalman_gain = self.covariance @ self._measurement_mat.T @ np.linalg.inv(
            innovation_cov
        )

        # Update state
        self.mean = self.mean + kalman_gain @ innovation

        # Update covariance
        self.covariance = (
            np.eye(self.ndim) - kalman_gain @ self._measurement_mat
        ) @ self.covariance

        return self.mean, self.covariance

    def predict_future(self, n_steps: int = 1) -> np.ndarray:
        """Predict future states.

        Args:
            n_steps: Number of time steps to predict

        Returns:
            Array of predicted positions (n_steps, 2) with [x, y]

        Raises:
            RuntimeError: If filter not initialized
        """
        if self.mean is None:
            raise RuntimeError("Filter not initialized")

        predictions = []
        mean = self.mean.copy()

        for _ in range(n_steps):
            mean = self._motion_mat @ mean
            predictions.append(mean[:2])  # Extract x, y

        return np.array(predictions)

    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get current position estimate.

        Returns:
            (x, y) position or None if not initialized
        """
        if self.mean is None:
            return None
        return (float(self.mean[0]), float(self.mean[1]))

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Get current velocity estimate.

        Returns:
            (vx, vy) velocity or None if not initialized
        """
        if self.mean is None:
            return None
        return (float(self.mean[4]), float(self.mean[5]))

    def reset(self) -> None:
        """Reset filter state."""
        self.mean = None
        self.covariance = None

    def __repr__(self) -> str:
        """String representation."""
        pos = self.get_position()
        vel = self.get_velocity()
        return f"KalmanFilter(position={pos}, velocity={vel})"


class BidirectionalKalmanSmoother:
    """Bidirectional Kalman smoothing for improved accuracy.

    Runs Kalman filter forward and backward, then combines the results
    for smoother, more accurate trajectory.
    """

    def __init__(
        self,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
    ) -> None:
        """Initialize smoother.

        Args:
            process_noise_scale: Process noise scale
            measurement_noise_scale: Measurement noise scale
        """
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale

    def smooth(self, measurements: np.ndarray) -> np.ndarray:
        """Apply bidirectional smoothing to measurements.

        Args:
            measurements: Array of measurements (N, 4) with [x, y, aspect, height]
                         Use NaN for missing measurements

        Returns:
            Smoothed positions (N, 2) with [x, y]
        """
        n_frames = len(measurements)

        # Forward pass
        forward_positions = self._forward_pass(measurements)

        # Backward pass
        backward_positions = self._backward_pass(measurements)

        # Combine (weighted average)
        # Give more weight to estimates from the direction with less future/past frames
        smoothed = np.zeros((n_frames, 2))

        for i in range(n_frames):
            # Weight based on distance from start/end
            weight_forward = (i + 1) / n_frames
            weight_backward = 1.0 - weight_forward

            # Handle NaN values
            if not np.isnan(forward_positions[i]).any():
                smoothed[i] += forward_positions[i] * weight_forward

            if not np.isnan(backward_positions[i]).any():
                smoothed[i] += backward_positions[i] * weight_backward

        return smoothed

    def _forward_pass(self, measurements: np.ndarray) -> np.ndarray:
        """Run Kalman filter forward."""
        kf = KalmanFilter(
            process_noise_scale=self.process_noise_scale,
            measurement_noise_scale=self.measurement_noise_scale,
        )

        positions = []
        initialized = False

        for measurement in measurements:
            # Skip NaN measurements for initialization
            if not initialized:
                if not np.isnan(measurement).any():
                    kf.initiate(measurement)
                    initialized = True
                    positions.append(measurement[:2])
                else:
                    positions.append(np.array([np.nan, np.nan]))
                continue

            # Predict
            kf.predict()

            # Update if measurement available
            if not np.isnan(measurement).any():
                kf.update(measurement)

            # Record position
            pos = kf.get_position()
            positions.append(np.array(pos) if pos else np.array([np.nan, np.nan]))

        return np.array(positions)

    def _backward_pass(self, measurements: np.ndarray) -> np.ndarray:
        """Run Kalman filter backward."""
        # Reverse measurements
        measurements_rev = measurements[::-1]

        # Run forward on reversed
        positions_rev = self._forward_pass(measurements_rev)

        # Reverse back
        return positions_rev[::-1]

"""Tennis physics engine for trajectory validation and prediction.

Implements physics-based validation including gravity, air drag,
bounce detection, and trajectory fitting.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


class TennisPhysicsEngine:
    """Physics engine for tennis ball trajectory analysis.

    Validates and corrects ball trajectories using physics constraints:
    - Gravity: 9.81 m/s²
    - Air drag: Quadratic drag force
    - Bounce detection and damping
    - Trajectory fitting (parabolic)

    Attributes:
        gravity: Gravitational acceleration (m/s²)
        drag_coefficient: Drag coefficient for tennis ball
        air_density: Air density (kg/m³)
        ball_mass: Ball mass (kg)
        ball_radius: Ball radius (m)
        bounce_damping: Coefficient of restitution for bounces
    """

    def __init__(self, config: Dict) -> None:
        """Initialize physics engine.

        Args:
            config: Configuration dictionary with physics parameters
        """
        self.gravity = config.get("gravity", 9.81)  # m/s²
        self.drag_coefficient = config.get("drag_coefficient", 0.55)
        self.air_density = config.get("air_density", 1.225)  # kg/m³
        self.ball_mass = config.get("ball_properties", {}).get("mass", 0.058)  # kg
        self.ball_radius = config.get("ball_properties", {}).get("radius", 0.033)  # m
        self.bounce_damping = config.get("ball_properties", {}).get(
            "restitution", 0.75
        )

        # Maximum realistic ball velocity (m/s)
        max_velocity_kmh = config.get("max_ball_velocity", 180)
        self.max_velocity = max_velocity_kmh / 3.6  # Convert to m/s

    def validate_trajectory(
        self,
        positions_3d: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        fps: float = 30.0,
    ) -> Dict:
        """Validate trajectory against physics constraints.

        Args:
            positions_3d: Ball positions in 3D (N, 3) [x, y, z]
            timestamps: Optional timestamps. If None, assumes constant FPS
            fps: Frames per second (used if timestamps is None)

        Returns:
            Validation result dictionary:
                - is_valid: Overall validity
                - velocity_valid: Velocity within bounds
                - acceleration_valid: Acceleration reasonable
                - outlier_indices: Indices of outlier points
                - bounces: Detected bounce indices
        """
        if timestamps is None:
            timestamps = np.arange(len(positions_3d)) / fps

        # Compute velocities
        velocities = self._compute_velocities(positions_3d, timestamps)

        # Check velocity magnitude
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        velocity_valid = np.all(velocity_magnitudes <= self.max_velocity)

        # Compute accelerations
        accelerations = self._compute_accelerations(velocities, timestamps[:-1])
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

        # Reasonable acceleration (allowing for impacts)
        max_acceleration = 5 * self.gravity  # 5g
        acceleration_valid = np.all(acceleration_magnitudes <= max_acceleration)

        # Detect outliers using velocity changes
        outliers = self._detect_outliers(velocity_magnitudes)

        # Detect bounces
        bounces = self._detect_bounces(positions_3d[:, 2], velocity_magnitudes)

        return {
            "is_valid": velocity_valid and acceleration_valid,
            "velocity_valid": velocity_valid,
            "acceleration_valid": acceleration_valid,
            "outlier_indices": outliers,
            "bounces": bounces,
            "max_velocity": float(np.max(velocity_magnitudes)),
            "max_acceleration": float(np.max(acceleration_magnitudes)),
        }

    def fit_trajectory(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        fps: float = 30.0,
    ) -> Dict:
        """Fit parabolic trajectory to positions.

        Args:
            positions: Ball positions (N, 2 or 3)
            timestamps: Optional timestamps
            fps: Frames per second

        Returns:
            Trajectory parameters:
                - coefficients: Polynomial coefficients
                - fitted_positions: Fitted positions
                - residuals: Fitting residuals
                - rmse: Root mean squared error
        """
        if timestamps is None:
            timestamps = np.arange(len(positions)) / fps

        # Remove NaN values
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_positions = positions[valid_mask]
        valid_timestamps = timestamps[valid_mask]

        if len(valid_positions) < 3:
            return {
                "coefficients": None,
                "fitted_positions": np.full_like(positions, np.nan),
                "residuals": np.full(len(positions), np.nan),
                "rmse": np.nan,
            }

        # Fit polynomial (2nd degree for parabolic trajectory)
        # For each dimension
        coefficients = []
        fitted = np.zeros_like(valid_positions)

        for dim in range(valid_positions.shape[1]):
            # Fit y = at² + bt + c
            coef = np.polyfit(valid_timestamps, valid_positions[:, dim], deg=2)
            coefficients.append(coef)
            fitted[:, dim] = np.polyval(coef, valid_timestamps)

        # Compute residuals
        residuals_valid = np.linalg.norm(valid_positions - fitted, axis=1)
        rmse = np.sqrt(np.mean(residuals_valid**2))

        # Reconstruct full array with NaNs
        fitted_positions = np.full_like(positions, np.nan)
        fitted_positions[valid_mask] = fitted

        residuals = np.full(len(positions), np.nan)
        residuals[valid_mask] = residuals_valid

        return {
            "coefficients": coefficients,
            "fitted_positions": fitted_positions,
            "residuals": residuals,
            "rmse": float(rmse),
        }

    def predict_trajectory(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        n_steps: int,
        dt: float = 1.0 / 30,
    ) -> np.ndarray:
        """Predict future trajectory with physics.

        Args:
            initial_position: Initial 3D position [x, y, z]
            initial_velocity: Initial 3D velocity [vx, vy, vz]
            n_steps: Number of time steps to predict
            dt: Time step (seconds)

        Returns:
            Predicted positions (n_steps, 3)
        """
        positions = [initial_position.copy()]
        velocity = initial_velocity.copy()
        position = initial_position.copy()

        for _ in range(n_steps - 1):
            # Compute drag force
            speed = np.linalg.norm(velocity)
            if speed > 0:
                drag_force = self._compute_drag_force(velocity)
                drag_acceleration = drag_force / self.ball_mass
            else:
                drag_acceleration = np.zeros(3)

            # Total acceleration (gravity + drag)
            acceleration = np.array([0, 0, -self.gravity]) + drag_acceleration

            # Update velocity and position (Euler integration)
            velocity = velocity + acceleration * dt
            position = position + velocity * dt

            # Check for ground bounce
            if position[2] < 0:
                position[2] = 0
                velocity[2] = -velocity[2] * self.bounce_damping
                velocity[:2] *= 0.9  # Friction

            positions.append(position.copy())

        return np.array(positions)

    def _compute_drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """Compute drag force vector.

        F_drag = -0.5 * ρ * C_d * A * v²

        Args:
            velocity: Velocity vector (3,)

        Returns:
            Drag force vector (3,)
        """
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.zeros(3)

        # Cross-sectional area
        area = np.pi * self.ball_radius**2

        # Magnitude of drag force
        drag_magnitude = 0.5 * self.air_density * self.drag_coefficient * area * speed**2

        # Direction opposite to velocity
        drag_force = -drag_magnitude * (velocity / speed)

        return drag_force

    def _compute_velocities(
        self, positions: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """Compute velocities from positions.

        Args:
            positions: Positions (N, 3)
            timestamps: Timestamps (N,)

        Returns:
            Velocities (N-1, 3)
        """
        dt = np.diff(timestamps)
        dpos = np.diff(positions, axis=0)

        # Avoid division by zero
        dt = np.maximum(dt, 1e-6)

        velocities = dpos / dt[:, np.newaxis]
        return velocities

    def _compute_accelerations(
        self, velocities: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """Compute accelerations from velocities.

        Args:
            velocities: Velocities (N, 3)
            timestamps: Timestamps (N,)

        Returns:
            Accelerations (N-1, 3)
        """
        dt = np.diff(timestamps)
        dvel = np.diff(velocities, axis=0)

        dt = np.maximum(dt, 1e-6)

        accelerations = dvel / dt[:, np.newaxis]
        return accelerations

    def _detect_outliers(
        self, velocity_magnitudes: np.ndarray, threshold_std: float = 3.0
    ) -> List[int]:
        """Detect outliers in velocity using standard deviation.

        Args:
            velocity_magnitudes: Velocity magnitudes (N,)
            threshold_std: Threshold in standard deviations

        Returns:
            List of outlier indices
        """
        mean_vel = np.mean(velocity_magnitudes)
        std_vel = np.std(velocity_magnitudes)

        outliers = np.where(
            np.abs(velocity_magnitudes - mean_vel) > threshold_std * std_vel
        )[0]

        return outliers.tolist()

    def _detect_bounces(
        self, z_positions: np.ndarray, velocities: np.ndarray
    ) -> List[int]:
        """Detect bounces from z-position and velocity.

        Args:
            z_positions: Height values (N,)
            velocities: Velocity magnitudes (N-1,)

        Returns:
            List of bounce frame indices
        """
        # Find local minima in z-position (near ground)
        # that are followed by velocity increases
        peaks, _ = find_peaks(-z_positions, height=-0.2)  # Within 20cm of ground

        bounces = []
        for peak in peaks:
            # Check if velocity increases after this point
            if peak < len(velocities) - 1:
                if velocities[peak + 1] > velocities[peak] * 1.1:  # 10% increase
                    bounces.append(int(peak))

        return bounces

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TennisPhysicsEngine("
            f"gravity={self.gravity}, "
            f"drag_coefficient={self.drag_coefficient}, "
            f"max_velocity={self.max_velocity:.1f} m/s)"
        )

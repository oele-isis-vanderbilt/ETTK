from filterpy.kalman import KalmanFilter
import numpy as np

class RotationVectorKalmanFilter:
    def __init__(self):
        # Define Kalman filter
        self.kf = KalmanFilter(dim_x=3, dim_z=3)

        # State Transition matrix - Identity for rotation vectors
        self.kf.F = np.eye(3)

        # Measurement matrix - Identity for rotation vectors
        self.kf.H = np.eye(3)

        # Process noise covariance - Adjust based on your needs
        self.kf.Q = 1e-4 * np.eye(3)

        # Measurement noise covariance - Adjust based on your needs
        self.kf.R = 1e-2 * np.eye(3)

        # Error covariance - Adjust based on your needs
        self.kf.P *= 1e-1

    def process(self, rvec: np.ndarray) -> np.ndarray:
        # Predict
        self.kf.predict()

        # Update
        self.kf.update(rvec)

        return self.kf.x

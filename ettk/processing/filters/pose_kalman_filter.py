from filterpy.kalman import KalmanFilter
import numpy as np

class PoseKalmanFilter:
    def __init__(self):
        # Define Kalman filter
        # 6 for [rotation vector, translation]
        self.kf = KalmanFilter(dim_x=6, dim_z=6)

        # State Transition matrix - Identity for pose (rotation + translation)
        self.kf.F = np.eye(6)

        # Measurement matrix - Identity for pose
        self.kf.H = np.eye(6)

        # Process noise covariance - Adjust based on your needs
        self.kf.Q = np.diag([1e-4] * 3 + [1e-3] * 3)  # Assuming translation might have a bit more noise

        # Measurement noise covariance - Adjust based on your needs
        self.kf.R = np.diag([1e-2] * 3 + [1e-1] * 3)  # Assuming translation measurements might be less precise

        # Error covariance - Adjust based on your needs
        self.kf.P *= 1e-1

    def process(self, rvec: np.ndarray, tvec: np.ndarray):
        # Convert to numpy arrays for concatenation
        rvec = np.array(rvec).flatten()
        tvec = np.array(tvec).flatten()

        # Concatenate to form the measurement vector
        z = np.concatenate((rvec, tvec))

        # Predict
        self.kf.predict()

        # Update
        self.kf.update(z)

        # Separate the state into rotation and translation for return
        return self.kf.x[:3], self.kf.x[3:]

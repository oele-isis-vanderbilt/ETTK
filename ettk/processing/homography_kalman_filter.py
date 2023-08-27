
from filterpy.kalman import KalmanFilter
import numpy as np

class HomographyKalmanFilter:
    def __init__(self):
        # Define Kalman filter
        # 9 for the homography matrix
        self.kf = KalmanFilter(dim_x=9, dim_z=9)

        # State Transition matrix - Identity for the homography matrix
        self.kf.F = np.eye(9)

        # Measurement matrix - Identity for the homography matrix
        self.kf.H = np.eye(9)

        # Process noise covariance - Adjust based on your needs
        self.kf.Q = np.eye(9) * 1e-5

        # Measurement noise covariance - Adjust based on your needs
        self.kf.R = np.eye(9) * 1e-2

        # Error covariance - Adjust based on your needs
        self.kf.P *= 1000

    def process(self, H_new: np.ndarray):
        # Convert to numpy arrays for flattening
        H_new_measurement = np.array(H_new).flatten()

        # Predict
        self.kf.predict()

        # Update
        self.kf.update(H_new_measurement)

        # Return the estimated homography
        return self.kf.x.reshape(3, 3)

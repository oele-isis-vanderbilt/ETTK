from filterpy.kalman import KalmanFilter
import numpy as np

class PoseKalmanFilter:
    def __init__(self, scale: float = 0.3):

        # Saving parameters
        self.scale = scale 

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

    def compute_uncertainty(self, z: np.ndarray):

        # Calculate the prior pose
        prior_pose = np.concatenate((self.kf.x[:3], self.kf.x[3:]))

        # Compute the difference between the prior and post-processed poses
        pose_difference = np.linalg.norm(prior_pose - z)

        # Compute the uncertainty
        uncertainty = 2 * (1 / (1 + np.exp(-self.scale * pose_difference)) - 0.5)

        # Ensure the value is bounded between 0 and 1
        uncertainty = max(0, min(1, uncertainty))

        return uncertainty

    def process(self, rvec: np.ndarray, tvec: np.ndarray):
        # Convert to numpy arrays for concatenation
        rvec = np.array(rvec).flatten()
        tvec = np.array(tvec).flatten()

        # Concatenate to form the measurement vector
        z = np.concatenate((rvec, tvec))
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(z)

        # Predict
        self.kf.predict()

        # Update
        self.kf.update(z)

        # Separate the state into rotation and translation for return
        return self.kf.x[:3], self.kf.x[3:], uncertainty

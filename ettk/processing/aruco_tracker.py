from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import logging
import math

from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

from .filters import RotationVectorKalmanFilter

logger = logging.getLogger('ettk')

MATRIX_COEFFICIENTS = np.array(
    [
        [910.5968017578125, 0, 958.43426513671875],
        [0, 910.20758056640625, 511.6611328125],
        [0, 0, 1],
    ]
)
DISTORTION_COEFFICIENTS = np.array(
    [
        -0.055919282138347626,
        0.079781122505664825,
        -0.048538044095039368,
        -0.00014426070265471935,
        0.00044536130735650659,
    ]
)

@dataclass
class ArucoResult:
    corners: np.ndarray = field(default_factory=lambda: np.empty((0,4))) # (M,4)
    ids: np.ndarray = field(default_factory=lambda: np.empty((0,1))) # (N,1)
    rvec: np.ndarray = field(default_factory=lambda: np.empty((0,3,1))) # (N,3,1)
    tvec: np.ndarray = field(default_factory=lambda: np.empty((0,3,1))) # (N,3,1)


@dataclass
class ArucoEntry:
    id: int
    rvec: np.ndarray # (3,1)
    tvec: np.ndarray # (3,1)
    ttl: int = 30


def euler_distance(euler1, euler2, seq='xyz'):
    """
    Compute the angular distance between two rotations represented by Euler angles.

    Parameters:
    - euler1, euler2: The two sets of Euler angles.
    - seq: The Euler sequence (default is 'zyx' for roll, pitch, yaw).

    Returns:
    - Angular distance between the two rotations.
    """

    # Convert Euler angles to quaternions
    q1 = R.from_euler(seq, euler1).as_quat()
    q2 = R.from_euler(seq, euler2).as_quat()

    # Compute the dot product
    d = np.dot(q1, q2)

    # Compute the angle difference
    theta = 2 * np.arccos(np.abs(d))

    return np.degrees(theta)



class ArucoTracker:
    def __init__(
        self, 
        matrix_coefficients: Optional[np.ndarray] = None, 
        distortion_coefficients: Optional[np.ndarray] = None,
        aruco_omit: Optional[List[int]] = []
    ):

        # Save parameters
        if isinstance(matrix_coefficients, np.ndarray):
            global MATRIX_COEFFICIENTS
            MATRIX_COEFFICIENTS = matrix_coefficients
        if isinstance(distortion_coefficients, np.ndarray):
            global DISTORTION_COEFFICIENTS
            DISTORTION_COEFFICIENTS = distortion_coefficients

        self.aruco_omit = aruco_omit

        # Aruco initialization
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )

        # State variables
        self.aruco_database: Dict[int, ArucoEntry] = {}
        self.aruco_filters: Dict[int, RotationVectorKalmanFilter] = {}

    def step(self, frame: np.ndarray, repair: bool = True) -> ArucoResult:

        # Getting frame's markers
        corners, ids, _ = self._aruco_detector.detectMarkers(frame)

        if not len(corners):
            return ArucoResult()

        if self.aruco_omit:
            valid_indices = [i for i, marker_id in enumerate(ids) if marker_id[0] not in self.aruco_omit]
            corners = [corners[i] for i in valid_indices]
            ids = ids[valid_indices]
        
        if not len(corners):
            return ArucoResult()

        # Compute RT
        rs = []
        ts = []
        # ms = []
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers

                # Define 3D coordinates for the marker's corners (assuming marker side length is 0.02)
                marker_length = 0.02
                obj_points = np.array([
                    [-marker_length / 2, -marker_length / 2, 0],
                    [ marker_length / 2, -marker_length / 2, 0],
                    [ marker_length / 2,  marker_length / 2, 0],
                    [-marker_length / 2,  marker_length / 2, 0]
                ])
                
                # Image points are the detected corners
                img_points = corners[i][0].astype(np.float32)
                
                # Solve for pose using PnP
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)
                
                if success:
                    rs.append(rvec)
                    ts.append(tvec)
                    # ms.append(img_points) # If you want the 2D corners, not sure what markerPoints was intended for in the original

        if len(rs) > 0:
            rs = np.stack(rs)
            ts = np.stack(ts)
            # ms = np.stack(ms)
        else:
            rs = np.empty((0,1,1,3))
            ts = np.empty((0,1,1,3))
            # ms = np.array([])

        # Type safety
        if ids is None:
            ids = np.empty((0,1))

        # Construct result
        result = ArucoResult(corners=np.stack(corners), ids=ids, rvec=rs, tvec=ts)

        # Repair 
        if repair:
            result = self.repair(result)

        return result

    def repair(self, result: ArucoResult) -> ArucoResult:

        if len(result.ids) == 0:
            return result

        for i in range(len(result.ids)):

            # Check if Z axis of marker is pointing away from camera
            id = result.ids[i, 0]

            if id in self.aruco_database:
                prior = self.aruco_database[id].rvec
                self.aruco_database[id].ttl -= 1
                distance = euler_distance(prior.squeeze(), result.rvec[i].squeeze())
            else:
                prior = np.zeros((3,1))
                distance = 0

            # Apply flip correction
            if (np.abs(distance-45)/45) < 0.2: # or (np.abs(distance-90)/90) < 0.2 or (np.abs(distance-135)/135) < 0.2 or (np.abs(distance-180)/180) < 0.2:
                result.rvec[i] = prior

            # Apply filtering
            if id not in self.aruco_filters:
                self.aruco_filters[id] = RotationVectorKalmanFilter()
            result.rvec[i] = self.aruco_filters[id].process(result.rvec[i].astype(np.float32))

            # Update entry
            self.aruco_database[result.ids[i, 0]] = ArucoEntry(
                id=result.ids[i, 0],
                rvec=result.rvec[i],
                tvec=result.tvec[i],
            )

        return result

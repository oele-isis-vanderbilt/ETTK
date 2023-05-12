# Built-in Imports
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Third-party
import imutils
from dataclasses import dataclass
import numpy as np
import cv2
import numpy as np

# Internal Imports
from .. import utils
from .template_database import TemplateDatabase

import pdb

# Constants
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

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

L = 0.23
H = 0.15
MS = 0.04


@dataclass
class ArucoResult:
    corners: np.ndarray  # (M,4)
    ids: np.ndarray  # (N,1)
    rvec: np.ndarray  # (N,1,1,3)
    tvec: np.ndarray  # (N,1,1,3)


@dataclass
class MonitorResult:
    corners: List


@dataclass
class PlanarResult:
    frame: np.ndarray  # (H,W,3)
    aruco: ArucoResult
    monitor: MonitorResult


class ComputerTracker:
    def __init__(self):

        self.grid = {
            0: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            1: np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            2: np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            # 3: np.array([
            #     [0,1,0,-H-MS/2],
            #     [1,0,0,-L],
            #     [0,0,1,0],
            #     [0,0,0,1]
            # ]),
            # 3: np.array([
            #     [  -0.042328,     0.97418,    -0.22175,      -H],
            #     [    0.85745,     -0.0785,    -0.50854,      -L],
            #     [   -0.51282,    -0.21166,    -0.83199,      0     ],
            #     [          0,           0,           0,           1]]),
            3: np.array(
                [
                    [0.042328, 0.97418, 0.22175, -H],
                    [-0.85745, -0.0785, 0.50854, -L],
                    [0.51282, -0.21166, 0.83199, 0],
                    [0, 0, 0, 1],
                ]
            ),
            4: np.array(
                [[-1, 0, 0, +L + MS / 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            5: np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            6: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            7: np.array(
                [[1, 0, 0, 0], [0, -1, 0, -MS / 2], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
        }

        self.pts = np.array([[0, 0, 0], [L, 0, 0], [0, H, 0], [L, H, 0]])

    def step(self, aruco_data: ArucoResult) -> MonitorResult:
        surface_points = {"corners": []}

        # Project 3D point onto 2D image
        ids = aruco_data.ids
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers

                if ids[i][0] not in self.grid.keys():
                    continue

                # Debugging only!
                if ids[i][0] != 3:
                    continue

                # Compute 3d points
                r = aruco_data.rvec[i]
                t = aruco_data.tvec[i]
                pdb.set_trace()
                point_3d = (
                    np.array([[0, 0, 0, 1], [L, 0, 0, 1], [L, H, 0, 1], [0, H, 0, 1]])
                    .astype(np.float32)
                    .T
                )

                # Apply the offset and direction
                rt = self.grid[ids[i][0]]
                # import pdb; pdb.set_trace()
                point_3d = rt @ point_3d
                point_3d = (point_3d[:3] / point_3d[-1]).T

                # Get the 2d point
                points_2d, _ = cv2.projectPoints(
                    point_3d.reshape((1, 4, 3)),
                    r,
                    t,
                    MATRIX_COEFFICIENTS,
                    DISTORTION_COEFFICIENTS,
                )
                points_2d = np.rint(points_2d).astype(np.float32)
                points_2d = points_2d.reshape((1, 4, 2))
                surface_points["corners"].append(points_2d)

        surface_points["corners"] = tuple(surface_points["corners"])

        # Create the results
        return MonitorResult(**surface_points)


class ArucoTracker:
    def __init__(self):

        # Aruco initialization
        # self.use_aruco_markers = use_aruco_markers
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )

    def step(self, frame: np.ndarray) -> ArucoResult:

        # Getting frame's markers
        corners, ids, _ = self._aruco_detector.detectMarkers(frame)

        # Compute RT
        rs = []
        ts = []
        ms = []
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], 0.02, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS
                )
                rs.append(rvec)
                ts.append(tvec)
                ms.append(markerPoints)

        rs = np.stack(rs)
        ts = np.stack(ts)
        ms = np.stack(ms)

        # Type safety
        if ids is None:
            ids = np.array([])

        return ArucoResult(corners=corners, ids=ids, rvec=rs, tvec=ts)


class PlanarTracker:
    def __init__(
        self,
    ):

        # Create aruco tracker
        self.aruco_tracker = ArucoTracker()

        # Create computer tracker
        self.computer_tracker = ComputerTracker()

    def step(self, frame: np.ndarray) -> PlanarResult:

        # Increase the brightness
        frame = utils.increase_brightness(frame)

        # First, let's get the aruco markers
        aruco_data = self.aruco_tracker.step(frame)
        monitor_data = self.computer_tracker.step(aruco_data)

        return PlanarResult(frame=frame, aruco=aruco_data, monitor=monitor_data)

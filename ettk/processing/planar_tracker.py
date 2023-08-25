# Built-in Imports
from typing import Dict, List, Optional, Tuple
import logging

# Third-party
import imutils
from dataclasses import dataclass, field
import numpy as np
import cv2
import numpy as np

# Internal Imports
from .. import utils
from .aruco_tracker import ArucoTracker, ArucoResult
from .template_database import TemplateDatabase
from .filters import PoseKalmanFilter

import pdb
logger = logging.getLogger('ettk')

# Constants
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
class ArucoConfig:
    id: int
    size: float
    offset_rvec: np.ndarray
    offset_tvec: np.ndarray


@dataclass
class SurfaceConfig:
    id: str
    aruco_config: Dict[int, ArucoConfig]
    height: float
    width: float
    scale: Tuple[float] = (1.0, 1.0)


@dataclass
class Hypothesis: # by an aruco
    id: int
    rvec: np.ndarray # (3,1)
    tvec: np.ndarray # (3,1)


@dataclass
class SurfaceEntry:
    id: str
    rvec: np.ndarray # (3,1)
    tvec: np.ndarray # (3,1)
    corners: np.ndarray # (4,2)
    hypotheses: List[Hypothesis] = field(default_factory=list)


@dataclass
class PlanarResult:
    aruco: ArucoResult
    surfaces: Dict[str, SurfaceEntry]


class PlanarTracker:

    def __init__(self, surface_configs: List[SurfaceConfig], aruco_tracker: Optional[ArucoTracker] = None):

        # Process parameters
        if aruco_tracker:
            self.aruco_tracker = aruco_tracker
        else:
            self.aruco_tracker = ArucoTracker()
        self.surface_configs = surface_configs

        # Surface filters
        self.surface_filters: Dict[str, PoseKalmanFilter] = {}

    def step(self, frame: np.ndarray):
        
        # First, obtain the aruco results
        aruco_results = self.aruco_tracker.step(frame)

        # Create empty planar container
        planar_results = PlanarResult(aruco=aruco_results, surfaces={})

        # For each surface, try to compute the surface pose
        for surface_config in self.surface_configs:

            # Extract the aruco markers pertaining to this surface
            ids = aruco_results.ids[:,0].tolist()
            surface_arucos = [a for a in ids if a in surface_config.aruco_config]
            if not surface_arucos:
                continue

            # Compute surface pose
            rvecs = []
            tvecs = []
            hypotheses = []
            for a in surface_arucos:
                # Extract
                aruco_config = surface_config.aruco_config[a]
                a_rvec = aruco_results.rvec[ids.index(a)]
                a_tvec = aruco_results.tvec[ids.index(a)]
               
                # Compute hypothesis
                rvec = a_rvec - aruco_config.offset_rvec.reshape((3, 1))
                scaled_offset = aruco_config.offset_tvec * np.array([*surface_config.scale, 1])
                tvec = a_tvec - scaled_offset.reshape((3, 1)) 
               
                # Make hypothesis
                hypothesis = Hypothesis(a, rvec, tvec)
                hypotheses.append(hypothesis)
                rvecs.append(rvec)
                tvecs.append(tvec)

            # Compute average pose
            rvec = np.mean(np.stack(rvecs), axis=0)
            tvec = np.mean(np.stack(tvecs), axis=0)

            # Apply Kalman filter
            if surface_config.id not in self.surface_filters:
                self.surface_filters[surface_config.id] = PoseKalmanFilter()
            rvec, tvec = self.surface_filters[surface_config.id].process(rvec, tvec)

            # Compute corners
            corners3D = np.array([
                [0, 0, 0],
                [ surface_config.width * surface_config.scale[0], 0, 0],
                [ surface_config.width * surface_config.scale[0], surface_config.height * surface_config.scale[1], 0],
                [ 0, surface_config.height * surface_config.scale[1], 0]
            ])
            corners2D, _ = cv2.projectPoints(corners3D, rvec, tvec, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)

            # Save the surface entry
            surface_entry = SurfaceEntry(
                id=surface_config.id,
                rvec=rvec,
                tvec=tvec,
                corners=corners2D,
                hypotheses=hypotheses
            )

            # Save the surface entry
            planar_results.surfaces[surface_config.id] = surface_entry

        return planar_results

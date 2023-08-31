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
from .aruco_tracker import ArucoTracker
from .homography_refiner import HomographyRefiner
from .filters import PoseKalmanFilter
from ..types import PlanarResult, Hypothesis, SurfaceEntry, WeightConfig, HomographyResult, SurfaceConfig, ArucoConfig

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


def rotation_vector_to_quaternion(rot_vec):
    theta = np.linalg.norm(rot_vec)
    if theta < 1e-10:
        return np.array([1, 0, 0, 0])
    axis = rot_vec / theta
    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = axis * np.sin(theta / 2)
    return q


def quaternion_to_rotation_vector(q):
    angle = 2 * np.arccos(q[0])
    if abs(q[0]) < 1 - 1e-10:
        axis = q[1:] / np.sin(angle / 2)
    else:
        # Angle is close to 0 or 180 degrees, arbitrary axis
        axis = np.array([1, 0, 0])
    rot_vec = angle * axis
    return rot_vec


def average_quaternion(quaternions):
    # Normalize quaternions first
    quaternions = [q/np.linalg.norm(q) for q in quaternions]

    # Start with the first quaternion
    avg_q = quaternions[0]

    for q in quaternions[1:]:
        if np.dot(avg_q, q) < 0:
            q = -q
        avg_q += q

    # Normalize to get the final average quaternion
    avg_q /= np.linalg.norm(avg_q)
    return avg_q
    

def weighted_fusion(original_rvec, original_tvec, homography_rvec, homography_tvec, weight=0.5) -> Tuple[np.ndarray, np.ndarray]:

    # Convert rvec to quaternion for fusion
    original_quat = cv2.Rodrigues(original_rvec)[0]
    homography_quat = cv2.Rodrigues(homography_rvec)[0]
    
    fused_quat = weight * original_quat + (1 - weight) * homography_quat
    fused_rvec = cv2.Rodrigues(fused_quat)[0]
    
    fused_tvec = weight * original_tvec + (1 - weight) * homography_tvec
    
    return fused_rvec, fused_tvec


class PlanarTracker:

    def __init__(self, 
            surface_configs: List[SurfaceConfig], 
            aruco_tracker: Optional[ArucoTracker] = None,
            weight_config: Optional[WeightConfig] = None,
        ):

        # Process parameters
        if aruco_tracker:
            self.aruco_tracker = aruco_tracker
        else:
            self.aruco_tracker = ArucoTracker()
        if weight_config:
            self.weight_config = weight_config
        else:
            self.weight_config = WeightConfig()
        self.surface_configs = surface_configs

        # Surface filters
        self.surface_filters: Dict[str, PoseKalmanFilter] = {}

        # Homography refiner
        surfaces = {s.id: s for s in surface_configs}
        self.refiner = HomographyRefiner(surfaces)

    def step(self, frame: np.ndarray):
        
        # First, obtain the aruco results
        aruco_results = self.aruco_tracker.step(frame)

        # Create empty planar container
        planar_results = PlanarResult(aruco=aruco_results, surfaces={})

        # Process the frame once for lines
        self.refiner.process_frame(frame)

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
            # combined_rvec = np.median(np.stack(rvecs), axis=0)
            quats = [rotation_vector_to_quaternion(r.squeeze()) for r in rvecs]
            combined_quat = average_quaternion(quats)
            combined_rvec = np.expand_dims(quaternion_to_rotation_vector(combined_quat), axis=1)
            combined_tvec = np.median(np.stack(tvecs), axis=0)

            # Also compute pose if enough points
            if len(hypotheses) >= 3:
                
                obj_points = []
                img_points = []
                for a in surface_arucos:
                    # Extract
                    aruco_config = surface_config.aruco_config[a]
                    obj_points.append(aruco_config.offset_tvec * np.array([*surface_config.scale, 1]))

                    # Compute 2D points
                    corners = aruco_results.corners[ids.index(a)]
                    point = np.mean(corners, axis=1)[0]
                    img_points.append(point)

                # Stack
                obj_points = np.stack(obj_points)
                img_points = np.stack(img_points)
                    
                # Solve for pose using PnP
                try:
                    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS)
                except:
                    success = False

                if success:
                    combined_rvec = self.weight_config.aruco*combined_rvec + self.weight_config.surface*rvec
                    combined_tvec = self.weight_config.aruco*combined_tvec + self.weight_config.surface*tvec
 
            # Perform homography
            if surface_config.template is not None:
                homography_results = self.refiner.find_homography(surface_config.id)
                if homography_results is not None and homography_results.success:
                    h_rvec = homography_results.rvec
                    h_tvec = homography_results.tvec
                    rvec = rvec * (1 - self.weight_config.homo) + h_rvec * self.weight_config.homo
                    tvec = tvec * (1 - self.weight_config.homo) + h_tvec * self.weight_config.homo
            else:
                homography_results = None
            
            # Apply Kalman filter
            if surface_config.id not in self.surface_filters:
                self.surface_filters[surface_config.id] = PoseKalmanFilter()
            rvec, tvec, uncertainty = self.surface_filters[surface_config.id].process(combined_rvec, combined_tvec)

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
                uncertainty=uncertainty,
                hypotheses=hypotheses,
                homography=homography_results,
                config=surface_config
            )

            # Save the surface entry
            planar_results.surfaces[surface_config.id] = surface_entry

        return planar_results

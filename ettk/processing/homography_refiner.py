from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import logging

import numpy as np
import cv2

from .filters import HomographyKalmanFilter
from ..types import HomographyResult, TemplateEntry, HomographyConfig, SurfaceConfig

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


def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(dot_product / norms))

class HomographyRefiner:

    def __init__(self, surfaces: Dict[str, SurfaceConfig], config: Optional[HomographyConfig] = None):

        # Process inputs
        if not config:
            self.config = HomographyConfig()
        else:
            self.config = config
        self.surfaces = surfaces

        # Create tools
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.filter = HomographyKalmanFilter()

        # Preprocess the templates
        self.templates: Dict[str, TemplateEntry] = {}
        for name, surface in self.surfaces.items():

            if surface.template is None:
                continue
            
            gray = cv2.cvtColor(surface.template, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)

            self.templates[name] = TemplateEntry(
                name=name,
                template=surface.template,
                kp=kp,
                des=des
            )

        # Containers
        self.gray: Optional[np.ndarray] = None
        self.kp: Optional[np.ndarray] = None
        self.des: Optional[np.ndarray] = None

    def warp_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp a list of points using a homography matrix H."""
        # Convert points to homogeneous coordinates
        points_homogeneous = np.array([points[:, 0], points[:, 1], np.ones(points.shape[0])])
        
        # Warp using the homography matrix
        warped_homogeneous = np.dot(H, points_homogeneous)
        
        # Convert back to inhomogeneous coordinates
        warped = np.zeros_like(points)
        warped[:, 0] = warped_homogeneous[0, :] / warped_homogeneous[2, :]
        warped[:, 1] = warped_homogeneous[1, :] / warped_homogeneous[2, :]
        
        return np.expand_dims(warped, axis=1)

    def step(self, frame: np.ndarray, template_name: str) -> Optional[HomographyResult]:
        self.process_frame(frame)
        return self.find_homography(template_name)

    def process_frame(self, frame: np.ndarray):
        
        # Convert images to grayscale
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the keypoints and descriptors with ORB
        self.kp, self.des = self.orb.detectAndCompute(self.gray, None)

    def find_homography(self, template_name: str) -> Optional[HomographyResult]:
        assert self.kp is not None and self.des is not None, "Process the frame before running ``find_homography``"

        # Get the template
        template = self.templates[template_name]

        # Use BFMatcher to find the best matches
        matches = self.bf.match(template.des, self.des)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < self.config.min_matches:
            return None

        # Extract the coordinates of the matched keypoints
        template_pts = np.float32([template.kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts = np.float32([self.kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute the homography matrix
        M, mask = cv2.findHomography(template_pts, pts, cv2.RANSAC, self.config.ransac_threshold)

        num_inliers = np.sum(mask)
        if num_inliers < self.config.min_inliers:
            return None

        # Compute the warped points
        # Define the four corners of the template image (assuming img1 is the template)
        h, w = template.template.shape[:2]
        template_aspect_ratio = w / h
        corners_template = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners_template, M)

        # Compute aspect ratio
        width = np.linalg.norm(warped[0] - warped[1])
        height = np.linalg.norm(warped[1] - warped[2])
        aspect_ratio = width / height
        # logger.debug(f"Aspect ratio: {aspect_ratio}")
        if (aspect_ratio > 1) or (aspect_ratio < 0.15):
            return None

        # Compute angle between edges
        fail = False
        for i in range(4):
            edge1 = warped[i] - warped[(i+1)%4]
            edge2 = warped[(i+1)%4] - warped[(i+2)%4]
            angle = angle_between(edge1[0], edge2[0])
            if abs(90 - angle) > self.config.angle_threshold:
                fail = True
                break

        if fail:
            return None

        # Apply filtering
        M = self.filter.process(M)

        # Compute RT from H
        surface = self.surfaces[template_name]
        w, h = surface.width, surface.height
        w_r, h_r = surface.scale
        obj_pts = np.array([
            [0, 0, 0], 
            [w*w_r, 0, 0], 
            [w*w_r, h*h_r, 0], 
            [0, h*h_r, 0]
        ]).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_pts, warped, MATRIX_COEFFICIENTS, DISTORTION_COEFFICIENTS, flags=cv2.SOLVEPNP_IPPE)
        
        if not success:
            rvec = np.empty((3,1))
            tvec = np.empty((3,1))

        # Create entry
        H = HomographyResult(
            name=template_name,
            H=M,
            corners=warped,
            success=success,
            rvec=rvec,
            tvec=tvec,
            size=(w,h)
        )

        return H

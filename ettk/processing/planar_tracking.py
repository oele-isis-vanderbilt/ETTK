# Built-in Imports
from typing import Any, Optional, Tuple, List, Union, Dict
import time
import collections
import logging

logger = logging.getLogger(__name__)

# Third-party
import imutils
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

FIX_RADIUS = 10
FIX_COLOR = (0, 0, 255)
FIX_THICKNESS = 3


class PlanarResult:
    def __init__(
        self,
        img: np.ndarray,
        success: bool,
        template_data: Dict[str, Any],
        M: np.ndarray,
        corners: np.ndarray,
        tracked_points: np.ndarray,
        delay: Union[int, float],
    ):

        # Save inputs
        self.img = img
        self.success = success
        self.template_data = template_data
        self.M = M
        self.corners = corners
        self.tracked_points = tracked_points
        self.delay = delay

    def project_fix(self, fix: Tuple[int, int]):

        fix_pt = np.float32([[fix[0], fix[1]]]).reshape(-1, 1, 2)
        fix_dst = (
            cv2.perspectiveTransform(fix_pt, np.linalg.inv(self.M))
            .flatten()
            .astype(np.int32)
        )

        return fix_dst

    def draw_fix(self, fix: Tuple[int, int], img: np.ndarray = None):

        # Draw eye-tracking into the original video frame
        draw_frame = cv2.circle(img.copy(), fix, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

        return draw_frame

    def draw_surface_outline(self, img: np.ndarray = None):

        # Draw paper outline
        draw_frame = utils.draw_homography_outline(
            img.copy(), self.corners, color=(0, 255, 0)
        )

        return draw_frame

    def draw_tracked_points(self, img: np.ndarray = None):

        # Draw the tracked points
        draw_frame = utils.draw_pts(img, self.tracked_points)
        draw_frame = utils.draw_text(
            draw_frame, f"{(1/self.delay):.2f}", location=(0, 50), color=(0, 0, 255)
        )

        return draw_frame

    def render(self, fix: Tuple[int, int]) -> np.ndarray:

        draw_frame = self.draw_fix(fix, self.img)

        # Select the template that was detected
        if self.success:
            template = self.template_data["template"]
            id = self.template_data["id"]
            draw_frame = utils.draw_text(
                draw_frame,
                str(id),
                location=(0, 100),
                color=(0, 255, 0),
            )
        else:
            template = np.zeros_like(draw_frame)
            draw_frame = utils.draw_text(
                draw_frame, str(None), location=(0, 100), color=(0, 255, 0)
            )

        # Draw other detection-related information
        draw_frame = self.draw_surface_outline(draw_frame)
        draw_frame = self.draw_tracked_points(draw_frame)

        # Apply homography to fixation and draw it on the page
        if type(self.M) != type(None):
            fix_dst = self.project_fix(fix)
            draw_template = self.draw_fix(fix_dst, template.copy())
        else:
            draw_template = template.copy()

        # Combine frames
        vis_frame = utils.combine_frames(draw_template, draw_frame)

        return vis_frame


class PlanarTracker:

    last_seen_counter = 0
    step_id = 0

    previous_frame = None
    delay_deque = collections.deque(maxlen=100)

    def __init__(
        self,
        feature_extractor: Any = cv2.ORB_create(),
        matcher: Any = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
        alpha: float = 0.5,
        homography_every_frame: int = 5,
        max_corner_movement: float = 50,
        object_memory_limit: int = 5,
        use_aruco_markers: bool = True,
    ):

        # Feature Matching parameters
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.alpha = alpha
        self.max_corner_movement = max_corner_movement
        self.homography_every_frame = homography_every_frame
        self.object_memory_limit = object_memory_limit

        # Aruco initialization
        self.use_aruco_markers = use_aruco_markers
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )

        # Initialize template database
        self.template_database = TemplateDatabase(
            feature_extractor=feature_extractor,
            aruco_dict=self._aruco_dict,
            aruco_params=self._aruco_params,
            use_aruco_markers=use_aruco_markers,
        )

        # Initialize the tracker
        self.initialize_tracker()

    def initialize_tracker(self):

        # Reseting object detection
        self.M = None
        self.template = None
        self.found_template_id = None
        self.src_tracked_points = np.empty((0, 1, 2))
        self.dst_tracked_points = np.empty((0, 1, 2))
        self.corners = np.empty((0, 1, 2))

    def register_templates(self, templates: List[np.ndarray]) -> List:

        # Compute the hash for the new templates
        generated_hashes = []
        for template in templates:

            # Add the template to the database
            template_hash, success = self.template_database.add(template)

            generated_hashes.append(template_hash)

        return generated_hashes

    def mix_homography(self, M):

        # Mix resulting matrix
        if not isinstance(self.M, type(None)):
            self.M = (1 - self.alpha) * self.M + (self.alpha) * M
        else:
            self.M = M

    def check_if_homography_matrix_valid(
        self,
        M: np.ndarray,
        template_corners: np.ndarray,
        max_corner_movement_check: bool = False,
    ):

        # Obvious check
        if type(M) == type(None):
            return False

        # Compute corners with the acquired corners
        corners = np.array(
            cv2.perspectiveTransform(template_corners, M), dtype=np.int32
        ).reshape((4, 2))

        # Check if there is a previous M
        if max_corner_movement_check and not isinstance(self.M, type(None)):
            prev_corners = np.array(
                cv2.perspectiveTransform(template_corners, self.M), dtype=np.int32
            ).reshape((4, 2))

            # Determine total distance
            total_d = np.sqrt(np.power(prev_corners - corners, 2))
            if (total_d > self.max_corner_movement).any():
                return False

        # Decompose corners
        tl, bl, br, tr = corners

        # Check if the destination points are valid (top above bottom, left is left of right)
        if (
            tl[0] < tr[0]
            and tl[0] < br[0]
            and bl[0] < br[0]
            and bl[0] < tr[0]
            and tl[1] < bl[1]
            and tl[1] < br[1]
            and tr[1] < bl[1]
            and tr[1] < br[1]
        ):

            # Check that the rectangle has a decent size area
            x, y = corners[:, 0], corners[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            if area <= 1000:
                return False

            # Check the diagonal
            diagonal = np.linalg.norm(tl - br)
            antidiagonal = np.linalg.norm(bl - tr)
            diff = (diagonal - antidiagonal) / diagonal
            if diff > 0.2:
                return False

            # Check that the points have at least certain distance from each other
            d_from_tl = np.linalg.norm(np.abs((corners - corners[0])[1:]), axis=1)
            if (d_from_tl < 100).any():
                return False

            return True

        else:
            return False

    def perform_homography(self, template_id: int):

        # Extracting the template corners
        template_corners = self.template_database[template_id]["template_corners"]

        # Obtain the keypoints
        kpts1 = self.template_database[template_id]["kpts"]
        descs1 = self.template_database[template_id]["descs"]

        # Extract frame data
        kpts2 = self.frame_data["kpts"]
        descs2 = self.frame_data["descs"]

        # Match between keypoints
        matches = self.matcher.match(descs1, descs2)
        dmatches = sorted(matches, key=lambda x: x.distance)

        # If not enough matches stop
        if len(dmatches) < 4:
            return None, kpts1, self.frame_data["kpts"], dmatches

        # extract the matched keypoints
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Now fine-tuning the homography matrix
        if self.check_if_homography_matrix_valid(M, template_corners):

            # Update the last seen counter
            self.success_tracking = True
            self.found_template_id = template_id
            self.mix_homography(M)

        else:
            self.last_seen_counter += 1

        # Then compute the new points
        if not isinstance(self.M, type(None)):
            self.corners = np.array(
                cv2.perspectiveTransform(template_corners, self.M), dtype=np.int32
            )
        else:
            self.corners = np.empty((0, 1, 2))

        # Obtain the locations of the tracked points
        if not isinstance(self.M, type(None)):
            self.src_tracked_points = np.float32(
                [kpts1[m.queryIdx].pt for m in dmatches]
            ).reshape(-1, 1, 2)
            self.dst_tracked_points = np.array(
                cv2.perspectiveTransform(self.src_tracked_points, self.M),
                dtype=np.int32,
            ).reshape((-1, 1, 2))

    def identify_template_with_aruco(self):

        # Getting frame's markers
        corners, ids, _ = self._aruco_detector.detectMarkers(self.frame)

        # Type safety
        if ids is None:
            ids = np.array([])

        self.frame_data.update({"aruco": {"corners": corners, "ids": ids}})

        # Use the aruco markers to find the correct template
        max_inter = -1
        max_template_id = -1
        for template_id in self.template_database:

            inter = len(
                np.intersect1d(
                    self.template_database[template_id]["aruco"]["ids"].flatten(),
                    self.frame_data["aruco"]["ids"].flatten(),
                )
            )

            if inter > max_inter:
                max_template_id = template_id
                max_inter = inter

        if max_inter >= 1 and max_template_id != -1:
            self.found_template_id = max_template_id
            self.success_tracking = True

    def initial_estimation(self):

        # Resize image to reduce computational cost
        # small_frame = imutils.resize(self.frame, width=1500)
        ratio = 0.75
        h, w = self.frame.shape[:2]
        sh, sw = int(h / ratio), int(w / ratio)
        small_frame = imutils.resize(self.frame, width=sw)
        rh, rw = h / sh, w / sw

        # First, compute the new frame's kpts and descriptors
        # (instead of repeating within the next for loop
        kpts, descs = self.feature_extractor.detectAndCompute(small_frame, None)
        self.frame_data = {"kpts": kpts, "descs": descs}

        # Apply the resizing to the keypoints
        for i in range(len(kpts)):
            kpts[i].pt = (kpts[i].pt[0] * rh, kpts[i].pt[1] * rw)

        # If no object is found, look throughout the database
        if isinstance(self.found_template_id, type(None)):

            for template_id in self.template_database:
                self.perform_homography(template_id)
                if self.success_tracking:
                    break
        else:
            self.perform_homography(self.found_template_id)

    def optical_flow_tracking(self):

        # Convert image to grey
        current_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Apply optical flow
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(
            self.previous_frame,
            current_frame,
            self.dst_tracked_points.astype(np.float32),
            None,
            **LK_PARAMS,
        )
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(
            current_frame, self.previous_frame, p1, None, **LK_PARAMS
        )

        # Determine which points were tracked well
        d = abs(self.dst_tracked_points - p0r).reshape(-1, 2).max(-1)
        good = d < 0.2

        # Update the tracked points
        self.dst_tracked_points = p1

        # Compute a new homography
        self.src_tracked_points = self.src_tracked_points[good]
        self.dst_tracked_points = self.dst_tracked_points[good]

        if self.dst_tracked_points.shape[0] >= 4:
            M, mask = cv2.findHomography(
                self.src_tracked_points, self.dst_tracked_points, cv2.RANSAC, 5.0
            )

            # Extracting the template corners
            template_corners = self.template_database[self.found_template_id][
                "template_corners"
            ]

            # Only use the generated M if it is reasonable
            if self.check_if_homography_matrix_valid(
                M, template_corners, max_corner_movement_check=True
            ):
                self.M = (1 - self.alpha) * self.M + (self.alpha) * M
                self.success_tracking = True

    def step(
        self, frame: np.ndarray, templates: Optional[List[np.ndarray]] = None
    ) -> PlanarResult:

        # Store information to be used in other methods
        self.frame = frame
        if isinstance(templates, list):
            self.register_templates(templates)

        # Start timing
        # tic = time.perf_counter()
        tic = time.time()

        # Tracking if the step has been successful
        self.success_tracking = False

        # Every once in a while try using homography
        if self.step_id % self.homography_every_frame == 0:
            self.initial_estimation()

        # Use optical flow tracking to handle movements
        elif self.dst_tracked_points.shape[0] != 0:
            self.optical_flow_tracking()

        # Update the counter if no success tracking
        if self.success_tracking:
            self.last_seen_counter = 0
        else:
            self.last_seen_counter += 1

        # If the object is not found in a long time, raise Flag
        if self.last_seen_counter > self.object_memory_limit:
            self.object_found = False
            self.initialize_tracker()

        # Update step id
        self.step_id += 1
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the time it takes to take step
        # toc = time.perf_counter()
        toc = time.time()
        self.delay_deque.append(toc - tic)
        # fps = np.average(
        #     self.delay_deque, axis=0, weights=[1 for x in range(len(self.delay_deque))]
        # )
        delay = sum(self.delay_deque) / len(self.delay_deque)

        # Extract the latest detected template data, if successful
        if self.success_tracking:
            template_data = self.template_database[self.found_template_id]
        else:
            template_data = []

        return PlanarResult(
            img=frame,
            success=self.success_tracking,
            template_data=template_data,
            M=self.M,
            corners=self.corners,
            tracked_points=self.dst_tracked_points,
            delay=delay,
        )

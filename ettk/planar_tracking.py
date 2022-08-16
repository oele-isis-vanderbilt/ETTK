# Built-in Imports
from typing import Any, Optional, Tuple, List
import time
import collections

# Third-party
import numpy as np
import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)

import pdb

# Constants
LK_PARAMS = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class PlanarTracker():

    last_seen_counter = 0
    step_id = 0

    previous_frame = None
    previous_template = None
    previous_template_data = None

    fps_deque = collections.deque(maxlen=100)

    def __init__(
            self, 
            feature_extractor:Any=cv2.ORB_create(), 
            matcher:Any=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
            alpha:float=0.2,
            homography_every_frame:int=5,
            max_corner_movement:float=50,
            object_memory_limit:int=5
        ):

        # Feature Matching parameters
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.alpha = alpha
        self.max_corner_movement = max_corner_movement
        self.homography_every_frame = homography_every_frame
        self.object_memory_limit = object_memory_limit

        # Initialize the tracker
        self.initialize_tracker()

    def initialize_tracker(self):

        # Reseting object detection
        self.M = None
        self.object_found = False
        self.src_tracked_points = np.empty((0,1,2))
        self.dst_tracked_points = np.empty((0,1,2))
        self.corners = np.empty((0,1,2))

    def check_if_homography_matrix_valid(
            self, 
            M:np.ndarray, 
            template_corners:np.ndarray,
            max_corner_movement_check:bool=False
        ):

        # Obvious check
        if type(M) == type(None):
            return False
        
        # Compute corners with the acquired corners
        corners = np.array(cv2.perspectiveTransform(template_corners, M), dtype=np.int32).reshape((4,2))

        # Check if there is a previous M
        if max_corner_movement_check and type(self.M) != type(None):
            prev_corners = np.array(cv2.perspectiveTransform(template_corners, self.M), dtype=np.int32).reshape((4,2))

            # Determine total distance
            total_d = np.sqrt(np.power(prev_corners - corners, 2))
            if (total_d > self.max_corner_movement).any():
                return False

        # Decompose corners
        tl, bl, br, tr = corners

        # Check if the destination points are valid (top above bottom, left is left of right)
        if tl[0] < tr[0] and tl[0] < br[0] and\
            bl[0] < br[0] and bl[0] < tr[0] and\
            tl[1] < bl[1] and tl[1] < br[1] and\
            tr[1] < bl[1] and tr[1] < br[1]:

            # Check that the rectangle has a decent size area
            x, y = corners[:,0], corners[:,1]
            area = 0.5*np.abs(np.dot(x, np.roll(y,1)) - np.dot(y,np.roll(x,1)))
            if area <= 1000:
                return False
            
            # Check the diagonal
            diagonal = np.linalg.norm(tl-br)
            antidiagonal = np.linalg.norm(bl-tr)
            diff = (diagonal-antidiagonal)/diagonal
            if diff > 0.2:
                return False

            # Check that the points have at least certain distance from each other
            d_from_tl = np.linalg.norm(np.abs((corners - corners[0])[1:]), axis=1)
            if (d_from_tl < 100).any():
                return False

            return True
        
        else:
            return False

    def perform_homography(self, template:np.ndarray, frame:np.ndarray):
        
        # find the keypoints and descriptors with SIFT
        if type(self.previous_template) == type(None) or (self.previous_template != template).all():
            self.previous_template_data = self.feature_extractor.detectAndCompute(template,None)
            # self.previous_template_data = self.feature_extractor.compute(template,None)
            self.previous_template = template

        # Obtain the keypoints
        kpts1, descs1 = self.previous_template_data
        kpts2, descs2 = self.feature_extractor.detectAndCompute(frame,None)
        # kpts2, descs2 = self.feature_extractor.compute(frame,None)

        # Match between keypoints
        matches = self.matcher.match(descs1, descs2)
        dmatches = sorted(matches, key = lambda x:x.distance) 

        # If not enough matches stop
        if len(dmatches) < 4:
            return None, kpts1, kpts2, dmatches
        
        # extract the matched keypoints
        src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return M, kpts1, kpts2, dmatches
    
    def initial_estimation(self, template:np.ndarray, frame:np.ndarray):
        
        # Perform homography
        M, kpts1, kpts2, dmatches = self.perform_homography(template, frame)

        # Obtain size of the template
        h, w = template.shape[:2]

        # First get the destinatin points
        template_corners = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Check if we should stop
        if type(M) != type(None):
            # Fine-tune estimation
            self.M, self.corners = self._fine_tune_homography(M, template_corners)

        # Obtain the locations of the tracked points
        if type(self.M) != type(None):
            self.src_tracked_points = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
            self.dst_tracked_points = np.array(cv2.perspectiveTransform(self.src_tracked_points, self.M), dtype=np.int32).reshape((-1,1,2))
    
    def _fine_tune_homography(self, M:np.ndarray, template_corners:np.ndarray):

        if self.check_if_homography_matrix_valid(M, template_corners):
           
            # Update the last seen counter
            self.last_seen_counter = 0
            self.object_found = True
            
            # Mix resulting matrix
            if type(self.M) != type(None):
                self.M = (1-self.alpha)*self.M + (self.alpha)*M
            else:
                self.M = M
        else:
            self.last_seen_counter += 1

        # Then compute the new points
        if type(self.M) != type(None):
            corners = np.array(cv2.perspectiveTransform(template_corners, self.M), dtype=np.int32)
        else:
            corners = np.empty((0,1,2))

        return self.M, corners

    def optical_flow_tracking(self, template:np.ndarray, frame:np.ndarray):
        
        # Obtain size of the template
        h, w = template.shape[:2]

        # First get the destinatin points
        template_corners = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Convert image to grey
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply optical flow
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(
            self.previous_frame, 
            current_frame, 
            self.dst_tracked_points.astype(np.float32), 
            None, 
            **LK_PARAMS
        )
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(
            current_frame, 
            self.previous_frame, 
            p1, 
            None, 
            **LK_PARAMS
        )

        # Determine which points were tracked well
        d = abs(self.dst_tracked_points-p0r).reshape(-1, 2).max(-1)
        good = d < 0.2

        # Update the tracked points
        self.dst_tracked_points = p1

        # Compute a new homography
        self.src_tracked_points = self.src_tracked_points[good]
        self.dst_tracked_points = self.dst_tracked_points[good]

        if self.dst_tracked_points.shape[0] >= 4:
            M, mask = cv2.findHomography(
                self.src_tracked_points, 
                self.dst_tracked_points, 
                cv2.RANSAC, 
                5.0
            )

            # Only use the generated M if it is reasonable
            if self.check_if_homography_matrix_valid(M, template_corners, max_corner_movement_check=True):
                self.M = (1-self.alpha)*self.M + (self.alpha)*M
    
    def step(self, template:np.ndarray, frame:np.ndarray) -> dict:

        # Start timing
        tic = time.time()

        # Every once in a while try using homography
        if self.step_id % self.homography_every_frame == 0:# or self.dst_tracked_points.shape[0] == 0:

            # Take initial estimation
            self.initial_estimation(template, frame)

        # Else, just use optical flow tracking to handle movements
        elif self.dst_tracked_points.shape[0] != 0:
            self.optical_flow_tracking(template, frame)

        # If the object is not found in a long time, raise Flag
        logger.debug(self.last_seen_counter)
        if self.last_seen_counter > self.object_memory_limit:
            self.object_found = False
            self.initialize_tracker()

        # Update step id
        self.step_id += 1
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the time it takes to take step
        toc = time.time()
        self.fps_deque.append(1/(toc-tic))
        fps = np.average(self.fps_deque, axis=0, weights=[1 for x in range(len(self.fps_deque))])

        return {
            'object_found': self.object_found,
            'M': self.M, 
            'corners': self.corners, 
            'tracked_points': self.dst_tracked_points,
            'fps': fps
        }


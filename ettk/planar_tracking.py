# Built-in Imports
from typing import Any, Optional, Tuple, List
import time
import collections
import logging
logger = logging.getLogger(__name__)

# Third-party
import numpy as np
import cv2
import numpy as np

# Internal Imports
from . import utils

import pdb

# Constants
LK_PARAMS = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class PlanarTracker():

    last_seen_counter = 0
    step_id = 0

    template_database = {}
    previous_frame = None
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
        self.template = None
        self.found_template_id = None
        self.src_tracked_points = np.empty((0,1,2))
        self.dst_tracked_points = np.empty((0,1,2))
        self.corners = np.empty((0,1,2))

    def check_if_homography_matrix_valid(
            self, 
            M:np.ndarray, 
            template_corners:np.ndarray,
            max_corner_movement_check:bool=False,
        ):

        # Obvious check
        if type(M) == type(None):
            return False

        # Compute corners with the acquired corners
        corners = np.array(cv2.perspectiveTransform(template_corners, M), dtype=np.int32).reshape((4,2))

        # Check if there is a previous M
        if max_corner_movement_check and not isinstance(self.M, type(None)):
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

    def perform_homography(self, template_id:int):
       
        # Obtain the keypoints
        kpts1 = self.template_database[template_id]['kpts']
        descs1 = self.template_database[template_id]['descs']
        kpts2, descs2 = self.feature_extractor.detectAndCompute(self.frame,None)

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
       
        # Extracting the template corners
        template_corners = self.template_database[template_id]['template_corners']
                   
        # Now fine-tuning the homography matrix
        if self.check_if_homography_matrix_valid(M, template_corners):
           
            # Update the last seen counter
            self.last_seen_counter = 0
            self.found_template_id = template_id
            
            # Mix resulting matrix
            if not isinstance(self.M, type(None)):
                self.M = (1-self.alpha)*self.M + (self.alpha)*M
            else:
                self.M = M
        else:
            self.last_seen_counter += 1

        # Then compute the new points
        if not isinstance(self.M, type(None)):
            self.corners = np.array(cv2.perspectiveTransform(template_corners, self.M), dtype=np.int32)
        else:
            self.corners = np.empty((0,1,2))
        
        # Obtain the locations of the tracked points
        if not isinstance(self.M, type(None)):
            self.src_tracked_points = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
            self.dst_tracked_points = np.array(cv2.perspectiveTransform(self.src_tracked_points, self.M), dtype=np.int32).reshape((-1,1,2))
    
    def initial_estimation(self):
        
        # If no object is found, look throughout the database
        if isinstance(self.found_template_id, type(None)):
            for template_id in self.template_database.keys():
                self.perform_homography(template_id)
                if not isinstance(self.found_template_id, type(None)):
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

            # Extracting the template corners
            template_corners = self.template_database[self.found_template_id]['template_corners']

            # Only use the generated M if it is reasonable
            if self.check_if_homography_matrix_valid(M, template_corners, max_corner_movement_check=True):
                self.M = (1-self.alpha)*self.M + (self.alpha)*M

    def register_templates(self, templates:List[np.ndarray]) -> List:
        
        # Compute the hash for the new templates
        generated_hashes = []
        for template in templates:

            # Compute template's id
            template_hash = utils.dhash(template)

            # Check if the template has been added before
            if template_hash in self.template_database:
                continue
            
            # Compute additional template information
            kpts, descs = self.feature_extractor.detectAndCompute(template, None)
            h, w = template.shape[:2]
            template_corners = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

            # Store template into database
            self.template_database[template_hash] = {
                'template': template,
                'kpts': kpts,
                'descs': descs,
                'template_corners': template_corners
            }
            generated_hashes.append(template_hash)

        return generated_hashes
    
    def step(self, frame:np.ndarray, templates:Optional[List[np.ndarray]]=None) -> dict:

        # Store information to be used in other methods
        self.frame = frame
        if isinstance(templates, list):
            self.register_templates(templates)

        # Start timing
        tic = time.perf_counter()

        # Every once in a while try using homography
        if self.step_id % self.homography_every_frame == 0:

            # Take initial estimation
            self.initial_estimation()

        # Else, just use optical flow tracking to handle movements
        elif self.dst_tracked_points.shape[0] != 0:
            self.optical_flow_tracking()

        # If the object is not found in a long time, raise Flag
        if self.last_seen_counter > self.object_memory_limit:
            self.object_found = False
            self.initialize_tracker()

        # Update step id
        self.step_id += 1
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the time it takes to take step
        toc = time.perf_counter()
        self.fps_deque.append(1/(toc-tic))
        fps = np.average(self.fps_deque, axis=0, weights=[1 for x in range(len(self.fps_deque))])

        return {
            'template_id': self.found_template_id,
            'M': self.M, 
            'corners': self.corners, 
            'tracked_points': self.dst_tracked_points,
            'fps': fps
        }


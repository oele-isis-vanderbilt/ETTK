# Built-in Imports
from typing import Any

# Third-party
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import logging
logger = logging.getLogger(__name__)

import pdb

# Constants
MIN_MATCH_COUNT = 10

class PlanerTracker():

    M = None

    def __init__(
            self, 
            feature_extractor:Any=cv2.ORB_create(), 
            matcher:Any=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
            alpha:float=0.1
        ):

        # Store input parameters
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.alpha = alpha
        # self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def perform_homography(self, template:np.ndarray, frame:np.ndarray):
        
        # find the keypoints and descriptors with SIFT
        kpts1, descs1 = self.feature_extractor.detectAndCompute(template,None)
        kpts2, descs2 = self.feature_extractor.detectAndCompute(frame,None)

        # Match between keypoints
        matches = self.matcher.match(descs1, descs2)
        dmatches = sorted(matches, key = lambda x:x.distance) 
        
        # extract the matched keypoints
        src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return M, kpts1, kpts2, dmatches

    @staticmethod
    def draw_homography_outline(img, dst):
        
        if type(dst) != type(None):
            # draw found regions
            return cv2.polylines(img, [dst], True, (0,0,255), 3, cv2.LINE_AA)
        else:
            return img

    def step(self, template:np.ndarray, frame:np.ndarray):

        # Perform homography
        M, kpts1, kpts2, dmatches = self.perform_homography(template, frame)

        # Obtain size of the template
        h, w = template.shape[:2]

        # First get the destinatin points
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = np.array(cv2.perspectiveTransform(pts,M), dtype=np.int32).reshape((4,2))
        top_left, bottom_left, bottom_right, top_right = dst.tolist()

        # Check if the destination points are valid
        # x verification
        if top_left[0] < top_right[0] and top_left[0] < bottom_right[0] and\
            bottom_left[0] < bottom_right[0] and bottom_left[0] < top_right[0] and\
            top_left[1] < bottom_left[1] and top_left[1] < bottom_right[1] and\
            top_right[1] < bottom_left[1] and top_right[1] < bottom_right[1]:

            # Mix resulting matrix
            if type(self.M) != type(None):
                self.M = (1-self.alpha)*self.M + (self.alpha)*M
            else:
                self.M = M

        # Then compute the new points
        if type(self.M) != type(None):
            dst = np.array(cv2.perspectiveTransform(pts,self.M), dtype=np.int32)
        else:
            dst = None

        return M, dst, kpts1, kpts2, dmatches

# Built-in Imports
from typing import Any, Optional, Tuple, List

# Third-party
import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)

import pdb

# Constants
MIN_MATCH_COUNT = 10

class PlanerTracker():

    # Initial homography matrix
    M = None

    def __init__(
            self, 
            feature_extractor:Any=cv2.ORB_create(), 
            matcher:Any=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
            alpha:float=0.1,
            kernel_size:int=5,
            low_threshold:int=50,
            high_threshold:int=150,
            hough_border:int=50
        ):

        # Store input parameters
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hough_border = hough_border
    
    @staticmethod
    def draw_homography_outline(img:np.ndarray, dst:np.ndarray) -> np.ndarray:
        
        if type(dst) != type(None):
            # draw found regions
            return cv2.polylines(img, [dst], True, (0,0,255), 3, cv2.LINE_AA)
        else:
            return img

    @staticmethod
    def draw_hough_lines(img:np.ndarray, lines:list) -> np.ndarray:

        # Make copy to safely draw
        draw_img = img.copy()

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(draw_img, (x1, y1), (x2, y2), (255,0,0), )

        return draw_img

    @staticmethod
    def draw_contours(img:np.ndarray, cnts:list, color=(0,255,0)) -> np.ndarray:
        
        # Make copy to safely draw
        draw_img = img.copy()

        # For each contour, draw it!
        for c in cnts:
            cv2.drawContours(draw_img,[c], 0, color, 3)

        return draw_img

    @staticmethod
    def draw_rects(img:np.ndarray, rects:List[tuple]) -> np.ndarray:
        
        # Make copy to safely draw
        draw_img = img.copy()

        for rect in rects:
            x,y,w,h = rect
            cv2.rectangle(draw_img, (x,y), (x+w, y+h), (0,0,255), 2)

        return draw_img

    @staticmethod
    def draw_pts(img:np.ndarray, pts:np.ndarray, color:tuple=(255,0,0)) -> np.ndarray:
        
        # Make copy to safely draw
        draw_img = img.copy()

        for pt in pts.astype(np.int32):
            cv2.circle(draw_img, pt, 3, color, 2)

        return draw_img

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

    def _fine_tune_homography(self, M:np.ndarray, template:np.ndarray) -> Optional[np.ndarray]:
        
        # Obtain size of the template
        h, w = template.shape[:2]

        # First get the destinatin points
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = np.array(cv2.perspectiveTransform(pts,M), dtype=np.int32).reshape((4,2))
        tl, bl, br, tr = dst.tolist()

        # Check if the destination points are valid
        # x verification
        if tl[0] < tr[0] and tl[0] < br[0] and\
            bl[0] < br[0] and bl[0] < tr[0] and\
            tl[1] < bl[1] and tl[1] < br[1] and\
            tr[1] < bl[1] and tr[1] < br[1]:

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

        return dst

    def perform_hough_line_prediction(self, frame:np.ndarray) -> Tuple[np.ndarray, list]:
        
        # Convert frame to grey and apply Guassian blur
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey_frame = cv2.GaussianBlur(grey_frame, (self.kernel_size, self.kernel_size), 0)

        # Apply Canny line detection
        edges = cv2.Canny(grey_frame, self.low_threshold, self.high_threshold)

        # Then predict the lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 100, 10)

        return edges, lines

    def perform_contour_prediction(self, frame:np.ndarray):

        # Convert frame to grey and apply Guassian blur
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey_frame = cv2.GaussianBlur(grey_frame, (self.kernel_size, self.kernel_size), 0)

        # Apply Canny line detection
        edges = cv2.Canny(grey_frame, self.low_threshold, self.high_threshold)

        # Get contours
        cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        return cnts

    def select_best_contour_based_on_pts(self, cnts, pts):

        # Initial values
        max_count = -1
        max_cnt = None

        for cnt in cnts:
            
            # Get the convex hull of the contour
            count = 0

            # Get the number of count 
            for pt in pts:
                if cv2.pointPolygonTest(cnt, pt, False) == 1:
                    count += 1

            # Keep the max
            if count >= max_count:
                max_count = count
                max_cnt = cnt

        return max_cnt, max_count
    
    def step(self, template:np.ndarray, frame:np.ndarray) -> dict:

        # Perform homography
        M, kpts1, kpts2, dmatches = self.perform_homography(template, frame)

        # Perform hough line prediction
        # edges, lines = self.perform_hough_line_prediction(frame)

        # Refine homography
        dst = self._fine_tune_homography(M, template)

        # Refine through predicted lines
        # dst = self._fine_tune_hough(lines, template)

        return {
            'M': self.M, 
            'dst': dst, 
            'homography': {
                'kpts1': kpts1, 
                'kpts2': kpts2, 
                'dmatches': dmatches
            },
            'hough_lines': {
            }
        }

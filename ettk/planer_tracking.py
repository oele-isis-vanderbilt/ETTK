# Built-in Imports
from typing import Any, Optional, Tuple, List
import time
import collections

# Third-party
import numpy as np
import cv2
import numpy as np
import sympy as sy
import sympy.geometry as gm

import logging
logger = logging.getLogger(__name__)

import pdb

# Constants
MIN_MATCH_COUNT = 10
LK_PARAMS = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class PlanerTracker():

    # Initial homography matrix
    M = None
    src_tracked_points = np.empty((0,1,2))
    dst_tracked_points = np.empty((0,1,2))
    corners = np.empty((0,1,2))
    step_id = 0
    previous_frame = None
    previous_template = None
    previous_template_data = None
    fps_deque = collections.deque(maxlen=100)

    def __init__(
            self, 
            feature_extractor:Any=cv2.ORB_create(), 
            matcher:Any=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
            alpha:float=0.7,
            kernel_size:int=5,
            low_threshold:int=20,
            high_threshold:int=200,
            hough_border:int=50,
            min_threshold:float=50,
            beta:float=0.6,
            max_corner_movement:float=50
        ):

        # Feature Matching parameters
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.alpha = alpha
        self.max_corner_movement = max_corner_movement

        # Hough Line parameters
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hough_border = hough_border
        self.min_threshold = min_threshold
        self.beta = beta
    
    @staticmethod
    def get_intersection(l1, l2):
        # https://stackoverflow.com/a/64853478/13231446
        line1=gm.Line(gm.Point(l1[0],l1[1]),gm.Point(l1[2],l1[3])) #Line1
        line2=gm.Line(gm.Point(l2[0],l2[1]),gm.Point(l2[2],l2[3])) #Line2

         #These are two infinite lines defined by two points on the line
        i = line1.intersection(line2)
        return np.float32([i[0].evalf().x, i[0].evalf().y])

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
        tl, bl, br, tr = corners.tolist()

        # Check if the destination points are valid (top above bottom, left is left of right)
        if tl[0] < tr[0] and tl[0] < br[0] and\
            bl[0] < br[0] and bl[0] < tr[0] and\
            tl[1] < bl[1] and tl[1] < br[1] and\
            tr[1] < bl[1] and tr[1] < br[1]:


            # Check that the rectangle has a decent size area
            x, y = corners[:,0], corners[:,1]
            area = 0.5*np.abs(np.dot(x, np.roll(y,1)) - np.dot(y,np.roll(x,1)))
            if area <= 100:
                return False
            else:
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
        # matches = self.matcher.knnMatch(np.float32(descs1), np.float32(descs2), k=2)

        # Lowe's ratio test
        # ratio_thresh = 0.7

        # "Good" matches
        # good_matches = []

        # Filter matches
        # for m, n in matches:
        #     if m.distance < ratio_thresh * n.distance:
        #         good_matches.append(m)

        # If not enough matches stop
        if len(dmatches) < 4:
            return None, kpts1, kpts2, dmatches
        
        # extract the matched keypoints
        src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return M, kpts1, kpts2, dmatches
    
    def perform_hough_line_prediction(self, frame:np.ndarray) -> Tuple[np.ndarray, list]:
        
        # Convert frame to grey and apply Guassian blur
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # grey_frame = cv2.GaussianBlur(grey_frame, (self.kernel_size, self.kernel_size), 0)
        grey_frame = cv2.bilateralFilter(grey_frame, 5, 75, 75)

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

    def _fine_tune_homography(self, M:np.ndarray, template_corners:np.ndarray):

        if self.check_if_homography_matrix_valid(M, template_corners):
            
            # Mix resulting matrix
            if type(self.M) != type(None):
                self.M = (1-self.alpha)*self.M + (self.alpha)*M
            else:
                self.M = M

        # Then compute the new points
        if type(self.M) != type(None):
            corners = np.array(cv2.perspectiveTransform(template_corners, self.M), dtype=np.int32)
        else:
            corners = np.empty((0,1,2))

        return self.M, corners

    def _fine_tune_hough(
            self, 
            template:np.ndarray,
            lines:list
        ):
        
        # Obtain size of the template
        h, w = template.shape[:2]

        # First get the destinatin points
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = np.array(cv2.perspectiveTransform(pts,self.M), dtype=np.int32).reshape((4,2))

        # Filter the lines
        line_clusters = self.assign_lines_to_outline_lines(dst, lines)

        # Obtain the lines predictions
        outline_preds = self.predict_outline(line_clusters)

        # Estimate paper corners
        corners_preds = self.predict_paper_corners(outline_preds, dst)

        # Obtain a new homography matrix with the predicted corners
        M, mask = cv2.findHomography(pts, np.expand_dims(corners_preds, axis=1), cv2.RANSAC, 5.0)

        # Mix resulting matrix
        if type(self.M) != type(None):
            self.M = (1-self.beta)*self.M + (self.beta)*M
        else:
            self.M = M

        return line_clusters, outline_preds, corners_preds

    def assign_lines_to_outline_lines(self, pts, lines):
        """
         pt0 --- line3 --- pt3
          |                 |
          |                 |
        line0             line2
          |                 |
          |                 |
         pt1 --- line1 --- pt2

        """
        li = lines[:,0,:]
        lpt1 = np.expand_dims(li[:,:2], axis=1)
        lpt2 = np.expand_dims(li[:,2:], axis=1)

        # Get the distances between the line and the points
        ds = np.abs(np.cross(lpt2-lpt1, lpt1-pts))/np.linalg.norm(lpt2-lpt1, axis=2)

        # Determine which outline line does the collected line matches to
        # First apply threshold
        close_to_pts = ds < self.min_threshold

        # Given which points they are closest, select which line they are likely to fall to
        which_line = np.zeros((lines.shape[0],)) - 1
        which_line[(close_to_pts == np.array([True, True, False, False])).all(axis=1)] = 0
        which_line[(close_to_pts == np.array([False, True, True, False])).all(axis=1)] = 1
        which_line[(close_to_pts == np.array([False, False, True, True])).all(axis=1)] = 2
        which_line[(close_to_pts == np.array([True, False, False, True])).all(axis=1)] = 3

        # Select the lines
        line_clusters = [
            lines[which_line==0],
            lines[which_line==1],
            lines[which_line==2],
            lines[which_line==3]
        ]

        return line_clusters

    def predict_outline(self, assigned_lines):

        outline_preds = []
        for i in range(len(assigned_lines)):
           
            # Select the cluster of lines
            lines = assigned_lines[i]

            # If no lines, put empty
            if lines.shape[0] == 0:
                outline_pred = np.empty((0,1,4))
            else:
                # Get the average
                outline_pred = np.expand_dims(np.average(lines, axis=0), axis=0)

            outline_preds.append(outline_pred)

        return outline_preds

    def predict_paper_corners(self, outline_preds, pts):
        """
        (0,0)               +
         pt0 --- line3 --- pt3
          |                 |
          |                 |
        line0             line2
          |                 |
          |                 |
         pt1 --- line1 --- pt2
        +
        """

        # Mapping between pts and lines
        pt_to_line_map = [ # [Vertical, Horizontle]
            [0,3],
            [0,1],
            [2,1],
            [2,3],
        ]

        # Mapping in how lines and points can assisted construct other points
        pt_assist_map = [
            [{'sign': '+', 'adj_pt_id': 1}, {'sign': '+', 'adj_pt_id': 3}],
            [{'sign': '-', 'adj_pt_id': 0}, {'sign': '+', 'adj_pt_id': 2}],
            [{'sign': '-', 'adj_pt_id': 3}, {'sign': '-', 'adj_pt_id': 1}],
            [{'sign': '+', 'adj_pt_id': 2}, {'sign': '-', 'adj_pt_id': 0}],
        ]

        # Resulting pts
        estimated_pts = []

        # First see which intersections we can get
        for pt_id, pt in enumerate(pts):

            # Get the line segments that contribute to the point
            pt_lines = pt_to_line_map[pt_id]
            l1, l2 = outline_preds[pt_lines[0]], outline_preds[pt_lines[1]]

            # If no line segments are predicted!
            if l1.shape[0] == 0 or l2.shape[0] == 0:
                estimated_pts.append(pt)

            # If we can reconstruct the line
            else:
                # Find the line intersections
                intersection_pt = self.get_intersection(l1[0,0,:], l2[0,0,:])
                estimated_pts.append(intersection_pt)

        # # Refine pts by using existing segments
        # for pt_id, pt in enumerate(pts):
            
        #     # Get the line segments that contribute to the point
        #     pt_lines = pt_to_line_map[pt_id]
        #     l1, l2 = outline_preds[pt_lines[0]], outline_preds[pt_lines[1]]

        #     # Deal with scenarios where we only had 1 line segments
        #     # Determine supporting pt
        #     if l1.shape[0] != 0 and l2.shape[0] == 0: # vertical
        #         sup_info = pt_assist_map[pt_id][0]
        #         sup_line = l1
        #     elif l1.shape[0] == 0 and l2.shape[0] != 0: # horizontle
        #         sup_info = pt_assist_map[pt_id][1]
        #         sup_line = l2

        #     # Then use the supporting information to compute new point
        #     # First, find the slope
        #     pdb.set_trace()
        #     x1, y1, x2, y2 = sup_line
        #     m = (y2-y1)/(x2-x1)


        # Stack the resulting pts
        estimated_pts = np.stack(estimated_pts)

        return estimated_pts

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

    def refinement_process(self, template:np.ndarray, frame:np.ndarray) -> dict:
        
        # Perform hough line prediction
        edges, lines = self.perform_hough_line_prediction(frame)

        # Refine homography
        dst = self._fine_tune_homography(M, template)

        # Refine through predicted lines
        if type(self.M) != type(None):
            line_clusters, outline_preds, corners_pred = self._fine_tune_hough(template, lines)
        else:
            line_clusters = []
            outline_preds = []
            corners_pred = np.empty((0,2))

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
        good = d < 0.5

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
        if self.step_id % 1 == 0 or self.dst_tracked_points.shape[0] == 0:

            # Take initial estimation
            self.initial_estimation(template, frame)

        # Else, just use optical flow tracking to handle movements
        # else:
        #     self.optical_flow_tracking(template, frame)

        # # Refinement process
        # refine = self.refinement_process(template, frame, initial)

        # Update step id
        self.step_id += 1
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the time it takes to take step
        toc = time.time()
        self.fps_deque.append(1/(toc-tic))
        fps = np.average(self.fps_deque, axis=0, weights=[1 for x in range(len(self.fps_deque))])

        return {
            'M': self.M, 
            'corners': self.corners, 
            'tracked_points': self.dst_tracked_points,
            'fps': fps
        }


# Built-in Imports
from typing import Any, Optional, Tuple, List

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
            hough_border:int=50,
            min_threshold:float=50,
            beta:float=0.6
        ):

        # Feature Matching parameters
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.alpha = alpha

        # Hough Line parameters
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hough_border = hough_border
        self.min_threshold = min_threshold
        self.beta = beta
    
    @staticmethod
    def draw_homography_outline(img:np.ndarray, dst:np.ndarray) -> np.ndarray:
        
        if type(dst) != type(None):
            # draw found regions
            return cv2.polylines(img, [dst], True, (0,0,255), 3, cv2.LINE_AA)
        else:
            return img

    @staticmethod
    def draw_hough_lines(img:np.ndarray, lines:list, color:tuple=(255,0,0)) -> np.ndarray:

        # Make copy to safely draw
        draw_img = img.copy()

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

        return draw_img

    @staticmethod
    def draw_contours(img:np.ndarray, cnts:list, color:tuple=(0,255,0)) -> np.ndarray:
        
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
    def draw_pts(
            img:np.ndarray, 
            pts:np.ndarray, 
            color:tuple=(255,0,0), 
            radius:int=2
        ) -> np.ndarray:
        
        # Make copy to safely draw
        draw_img = img.copy()

        for pt in pts.astype(np.int32):
            cv2.circle(draw_img, pt, 3, color, radius)

        return draw_img

    @staticmethod
    def get_intersection(l1, l2):
        # https://stackoverflow.com/a/64853478/13231446
        line1=gm.Line(gm.Point(l1[0],l1[1]),gm.Point(l1[2],l1[3])) #Line1
        line2=gm.Line(gm.Point(l2[0],l2[1]),gm.Point(l2[2],l2[3])) #Line2

         #These are two infinite lines defined by two points on the line
        i = line1.intersection(line2)
        return np.float32([i[0].evalf().x, i[0].evalf().y])

    def perform_homography(self, template:np.ndarray, frame:np.ndarray):
        
        # find the keypoints and descriptors with SIFT
        kpts1, descs1 = self.feature_extractor.detectAndCompute(template,None)
        kpts2, descs2 = self.feature_extractor.detectAndCompute(frame,None)

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

    def _fine_tune_homography(self, M:np.ndarray, template:np.ndarray) -> Optional[np.ndarray]:
        
        # Obtain size of the template
        h, w = template.shape[:2]

        # First get the destinatin points
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Check if we should stop
        if type(M) == type(None):
            dst = np.array(cv2.perspectiveTransform(pts,self.M), dtype=np.int32).reshape((4,2))
            return dst
        
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
        assigned_lines = self.assign_lines_to_outline_lines(dst, lines)

        # Obtain the lines predictions
        outline_preds = self.predict_outline(assigned_lines)

        # Estimate paper corners
        corners_preds = self.predict_paper_corners(outline_preds, dst)

        # Obtain a new homography matrix with the predicted corners
        M, mask = cv2.findHomography(pts, np.expand_dims(corners_preds, axis=1), cv2.RANSAC, 5.0)

        # Mix resulting matrix
        if type(self.M) != type(None):
            self.M = (1-self.beta)*self.M + (self.beta)*M
        else:
            self.M = M

        return outline_preds, corners_preds

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
        lines0 = lines[which_line==0]
        lines1 = lines[which_line==1]
        lines2 = lines[which_line==2]
        lines3 = lines[which_line==3]

        return lines0, lines1, lines2, lines3

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
            if l1.shape[0] != 0 or l2.shape[0] != 0:
                estimated_pts.append(pt)

            # If we can reconstruct the line
            else:
                # Find the line intersections
                intersection_pt = self.get_intersection(l1[0,0,:], l2[0,0,:])
                estimated_pts.append(intersection_pt)

        # Refine pts by using existing segments
        for pt_id, pt in enumerate(pts):
            
            # Get the line segments that contribute to the point
            pt_lines = pt_to_line_map[pt_id]
            l1, l2 = outline_preds[pt_lines[0]], outline_preds[pt_lines[1]]

            # Deal with scenarios where we only had 1 line segments
            # Determine supporting pt
            if l1.shape[0] != 0 and l2.shape[0] == 0: # vertical
                sup_info = pt_assist_map[pt_id][0]
                sup_line = l1
            elif l1.shape[0] == 0 and l2.shape[0] != 0: # horizontle
                sup_info = pt_assist_map[pt_id][1]
                sup_line = l2

            # Then use the supporting information to compute new point
            # First, find the slope
            pdb.set_trace()
            x1, y1, x2, y2 = sup_line
            m = (y2-y1)/(x2-x1)


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
    
    def step(self, template:np.ndarray, frame:np.ndarray) -> dict:

        # Perform homography
        M, kpts1, kpts2, dmatches = self.perform_homography(template, frame)

        # Perform hough line prediction
        edges, lines = self.perform_hough_line_prediction(frame)

        # Refine homography
        dst = self._fine_tune_homography(M, template)

        # Refine through predicted lines
        if type(self.M) != type(None):
            outline_preds, corners_pred = self._fine_tune_hough(template, lines)
        else:
            outline_preds = []
            corners_pred = np.empty((0,2))

        return {
            'M': self.M, 
            'dst': dst, 
            'homography': {
                'kpts1': kpts1, 
                'kpts2': kpts2, 
                'dmatches': dmatches
            },
            'hough_lines': {
                'edges': edges,
                'lines': lines,
                'outline_preds': outline_preds,
                'corners_pred': corners_pred
            }
        }

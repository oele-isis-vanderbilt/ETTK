# Built-in Imports
import os
import sys
import pathlib
import ast
import pdb
import logging

logger = logging.getLogger(__name__)

# Third-party Imports
import cv2
import pytest
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pandas as pd

# Internal Imports
import ettk

# CONSTANTS
CWD = pathlib.Path(os.path.abspath(__file__)).parent

# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec1_v4'
# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec2_v4'
# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec3_v4'
# TEST_IMAGE_PATH = CWD/'data'/'resources'/'paper_v4'/'UnwrappingthePast-PRINT-1.png'

# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec4_v4'
# TEST_IMAGE_PATH = CWD/'data'/'resources'/'paper_v4'/'UnwrappingthePast-PRINT-3.png'

# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec1_v5'
# TEST_IMAGE_PATH = CWD/'data'/'resources'/'paper_v5'/'UnwrappingthePast-PRINT-1.png'

# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_rec1_v1'
# TEST_IMAGE_PATH = CWD/'data'/'resources'/'computer'/'computer_screenshot.png'

TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_rec1_v2'
# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_rec1_v3'
TEST_IMAGE_PATH = CWD/'data'/'resources'/'computer'/'computer_screenshot_large_text.png'

VIDEO_START_INDEX = 1500
# VIDEO_START_INDEX = 0

# TRIM_MARGIN_X = 80
# TRIM_MARGIN_Y_TOP = 100
# TRIM_MARGIN_Y_BOTTOM = 150

TRIM_MARGIN_X = 1
TRIM_MARGIN_Y_TOP = 1
TRIM_MARGIN_Y_BOTTOM = 1

BLACK_MARGIN_SIZE = 50

FIX_RADIUS = 10
FIX_COLOR = (0, 0, 255)
FIX_THICKNESS = 3

assert TEST_TOBII_REC_PATH.exists() 
assert TEST_IMAGE_PATH.exists()

@pytest.fixture
def cap():
    
    # Load the video and get a single frame
    video_path = TEST_TOBII_REC_PATH/'scenevideo.mp4'
    assert video_path.exists()

    cap = cv2.VideoCapture(str(video_path), 0)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert length > VIDEO_START_INDEX+1
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    return cap

@pytest.fixture
def template():
    # Load the image
    template = cv2.imread(str(TEST_IMAGE_PATH), 0)

    # Might need to trim the margins for now
    template = template[TRIM_MARGIN_Y_TOP:-TRIM_MARGIN_Y_BOTTOM, TRIM_MARGIN_X:-TRIM_MARGIN_X]
 
    # Put the padding
    black_margin_template = cv2.copyMakeBorder(
        template, 
        BLACK_MARGIN_SIZE,
        BLACK_MARGIN_SIZE,
        BLACK_MARGIN_SIZE,
        BLACK_MARGIN_SIZE,
        cv2.BORDER_CONSTANT,
        value=[0,0,0]
    )
    
    # Get the size of the new template
    h, w = black_margin_template.shape[:2]

    # Draw the circles
    cv2.circle(black_margin_template, (BLACK_MARGIN_SIZE//2+1, BLACK_MARGIN_SIZE//2+1), 3, (255,255,255), 15)
    cv2.circle(black_margin_template, (w-BLACK_MARGIN_SIZE//2+1, BLACK_MARGIN_SIZE//2+1), 3, (255,255,255), 15)
    cv2.circle(black_margin_template, (BLACK_MARGIN_SIZE//2+1, h-BLACK_MARGIN_SIZE//2+1), 3, (255,255,255), 15)
    cv2.circle(black_margin_template, (w-BLACK_MARGIN_SIZE//2+1, h-BLACK_MARGIN_SIZE//2+1), 3, (255,255,255), 15)

    # cv2.imshow('template', black_margin_template)
    # cv2.waitKey(0)
    
    return black_margin_template

@pytest.fixture
def tracker():

    FLANN_INDEX_KDTREE = 1 
    index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 5)

    search_params = dict(checks = 50) 
    
    # Create tracker
    tracker = ettk.PlanarTracker(
        # feature_extractor=cv2.xfeatures2d.FREAK_create()
        feature_extractor=cv2.AKAZE_create(),
        alpha=0.5
        # feature_extractor=cv2.BRISK_create(),
        # matcher=cv2.FlannBasedMatcher(index_params, search_params)
    )
    
    return tracker

def test_speed_video(cap, template):
    
    # Load the video and get a single frame
    ret, frame = cap.read()
 
    # Get the size of the video
    h, w, _ = frame.shape

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    # Set the starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Feature-point detector
    # feature_detector = cv2.KAZE_create() # Use AKAZE
    feature_detector=cv2.BRISK_create(1000)

    # Then perform homography
    while(True):

        ret, frame = cap.read()
        if ret:

            # Make a copy to draw
            draw_frame = frame.copy()

            # Keypoint (kp) detection and calculate descriptors (des)
            # kp, des = feature_detector.detectAndCompute(frame, None)
            kp, des = feature_detector.compute(frame, None)
            
            # Create a visual representation with everything
            cv2.imshow('output', draw_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # break

    # Closing the video
    cv2.destroyAllWindows()

def test_step_video(template, cap, tracker):

    # Load the video and get a single frame
    ret, frame = cap.read()
 
    # Get the size of the video
    h, w, _ = frame.shape

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    # Set the starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Then perform homography
    while(True):

        ret, frame = cap.read()
        if ret:
            
            # Input frame
            # frame = imutils.resize(frame, width=1500)

            # Make a copy to draw
            draw_frame = frame.copy()
 
            # Apply homography
            result = tracker.step(template, frame)
            
            # Draw paper outline
            draw_frame = ettk.utils.draw_homography_outline(draw_frame, result['corners'], color=(0,255,0))

            # Draw the tracked points
            draw_frame = ettk.utils.draw_pts(draw_frame, result['tracked_points'])
            draw_frame = ettk.utils.draw_text(draw_frame, f"{result['fps']:.2f}", color=(0,0,255))
            
            # Create a visual representation with everything
            cv2.imshow('output', draw_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # break

    # Closing the video
    cv2.destroyAllWindows()

def test_step_video_with_eye_tracking(template, cap, tracker):
    
    # Load the video and get a single frame
    ret, frame = cap.read()
    
    # Load other eye-tracking information
    gaze_df = ettk.utils.tobii.load_gaze_data(TEST_TOBII_REC_PATH)
    # gaze_df = pd.DataFrame({'timestamp':[]})
    
     # Determine fixation timestamp setup information
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    # Set the starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Creating counter to track video index
    video_index_counter = 0

    # Then perform homography
    while(True):

        # Get video
        ret, frame = cap.read()

        if ret:
            
            # Input frame
            frame = imutils.resize(frame, width=1500)

            # Get the size of the video
            h, w, _ = frame.shape
        
            # Get fixation
            current_time = (VIDEO_START_INDEX+video_index_counter) * (1/fps)
            try:
                raw_fix = gaze_df[gaze_df['timestamp'] > current_time].reset_index().iloc[0]['gaze2d']
            except IndexError:
                raw_fix = [0, 0]

            if isinstance(raw_fix, str):
                raw_fix = ast.literal_eval(raw_fix)
            
            fix = (int(raw_fix[0]*w), int(raw_fix[1]*h))
              
            # Draw eye-tracking into the original video frame
            draw_frame = cv2.circle(frame.copy(), fix, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

            # Apply homography
            result = tracker.step(template, frame)
            
            # Draw paper outline
            draw_frame = ettk.utils.draw_homography_outline(draw_frame, result['corners'], color=(0,255,0))

            # Draw the tracked points
            draw_frame = ettk.utils.draw_pts(draw_frame, result['tracked_points'])
            draw_frame = ettk.utils.draw_text(draw_frame, f"{result['fps']:.2f}", color=(0,0,255))

            # Apply homography to fixation and draw it on the page
            if type(result['M']) != type(None):
                fix_pt = np.float32([ [fix[0], fix[1]] ]).reshape(-1,1,2)
                fix_dst = cv2.perspectiveTransform(fix_pt, np.linalg.inv(result['M'])).flatten().astype(np.int32)
                draw_template = cv2.circle(template.copy(), fix_dst, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)
            else:
                draw_template = template.copy()

            # Combine frames
            vis_frame = ettk.utils.combine_frames(draw_template, draw_frame)
            cv2.imshow('output', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # Updated counter
        video_index_counter += 1

    # Closing the video
    cv2.destroyAllWindows()

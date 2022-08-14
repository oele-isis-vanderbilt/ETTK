# Built-in Imports
import os
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

# Internal Imports
import ettk

# CONSTANTS
CWD = pathlib.Path(os.path.abspath(__file__)).parent

# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec1_v4'
# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec2_v4'
# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec3_v4'
# TEST_IMAGE_PATH = CWD/'data'/'resources'/'paper_v4'/'UnwrappingthePast-PRINT-1.png'

TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec4_v4'
TEST_IMAGE_PATH = CWD/'data'/'resources'/'paper_v4'/'UnwrappingthePast-PRINT-3.png'

# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_rec1_v1'
# TEST_IMAGE_PATH = CWD/'data'/'resources'/'computer'/'computer_screenshot.png'

# TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_rec1_v2'
# TEST_IMAGE_PATH = CWD/'data'/'resources'/'computer'/'computer_screenshot_large_text.png'

VIDEO_START_INDEX = 1000

TRIM_MARGIN_X = 80
TRIM_MARGIN_Y_TOP = 100
TRIM_MARGIN_Y_BOTTOM = 150

# TRIM_MARGIN_X = 1
# TRIM_MARGIN_Y_TOP = 1
# TRIM_MARGIN_Y_BOTTOM = 1

FIX_RADIUS = 15
FIX_COLOR = (0, 0, 255)
FIX_THICKNESS = 5

assert TEST_TOBII_REC_PATH.exists() 
assert TEST_IMAGE_PATH.exists()

@pytest.fixture
def cap():
    
    # Load the video and get a single frame
    cap = cv2.VideoCapture(str(TEST_TOBII_REC_PATH/'scenevideo.mp4'), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    return cap

@pytest.fixture
def template():
    # Load the image
    template = cv2.imread(str(TEST_IMAGE_PATH), 0)

    # Might need to trim the margins for now
    template = template[TRIM_MARGIN_Y_TOP:-TRIM_MARGIN_Y_BOTTOM, TRIM_MARGIN_X:-TRIM_MARGIN_X]

    return template

def test_step_single_frame(template, cap):
    
    ret, frame = cap.read()
    draw_frame = frame.copy()
 
    # Make sure the frame is valid
    assert ret and isinstance(frame, np.ndarray)

    # Create planer tracker
    planer_tracker = ettk.PlanerTracker()

    # Apply step
    result = planer_tracker.step(template, frame)

    # Draw paper outline
    draw_frame = ettk.utils.draw_homography_outline(draw_frame, result['dst'], color=(0,255,0))

    colors = [
        (255,255,255),
        (255,0,0),
        (255,0,255),
        (0,0,255),
        (0,255,0)
    ]

    # Extract the result content
    lines = result['hough_lines']['lines']
    line_clusters = result['hough_lines']['line_clusters']
    outline = result['hough_lines']['outline_preds']
    corners = result['hough_lines']['corners_pred']
    edges = result['hough_lines']['edges']
    
    # Draw lines
    draw_frame = ettk.utils.draw_hough_lines(draw_frame, lines, color=colors[0], thickness=2)

    # Draw the line clusters
    for cluster_lines, color in zip(line_clusters, colors[1:]):
        draw_frame = ettk.utils.draw_hough_lines(draw_frame, cluster_lines, color=color, thickness=3)

    # Draw the outline
    for line, color in zip(outline, colors[1:]):
        draw_frame = ettk.utils.draw_hough_lines(draw_frame, line, color=color, thickness=5)

    # Draw the corners
    draw_frame = ettk.utils.draw_pts(draw_frame, corners, color=(0,0,255), radius=10)
    
    # draw match lines
    draw_frame = cv2.drawMatches(
        template, 
        result['homography']['kpts1'], 
        draw_frame, 
        result['homography']['kpts2'], 
        result['homography']['dmatches'][:20],
        None, 
        flags=2
    )

    # Create a visual representation with everything
    visual_output = ettk.utils.combine_frames(draw_frame, edges)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', visual_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_initial_estimate_video(template, cap):

    # Load the video and get a single frame
    ret, frame = cap.read()

    # Create tracker
    planer_tracker = ettk.PlanerTracker()
    
    # Get the size of the video
    h, w, _ = frame.shape

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    # Set the starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Then perform homography
    while(True):

        ret, frame = cap.read()
        if ret:

            # Make a copy to draw
            draw_frame = frame.copy()
            
            # Apply homography
            M, corners = planer_tracker.initial_estimation(template, frame)

            # Draw paper outline
            draw_frame = ettk.utils.draw_homography_outline(draw_frame, corners, color=(0,255,0))
            
            cv2.imshow('output', draw_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Closing the video
    cv2.destroyAllWindows()

def test_step_video(template, cap):
    
    # Load the video and get a single frame
    ret, frame = cap.read()

    # Create tracker
    planer_tracker = ettk.PlanerTracker()
    
    # Get the size of the video
    h, w, _ = frame.shape

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    # Set the starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Define the colors for the lines
    colors = [
        (255,255,255),
        (255,0,0),
        (255,0,255),
        (0,0,255),
        (0,255,0)
    ]

    # Then perform homography
    while(True):

        ret, frame = cap.read()
        if ret:

            # Make a copy to draw
            draw_frame = frame.copy()
            
            # Apply homography
            result = planer_tracker.step(template, frame)
            
            # Draw paper outline
            draw_frame = ettk.utils.draw_homography_outline(draw_frame, result['dst'], color=(0,255,0))
            
            # Extract the result content
            lines = result['hough_lines']['lines']
            line_clusters = result['hough_lines']['line_clusters']
            outline = result['hough_lines']['outline_preds']
            corners = result['hough_lines']['corners_pred']
            edges = result['hough_lines']['edges']
            
            # Draw lines
            draw_frame = ettk.utils.draw_hough_lines(draw_frame, lines, color=colors[0], thickness=2)

            # Draw the line clusters
            for cluster_lines, color in zip(line_clusters, colors[1:]):
                draw_frame = ettk.utils.draw_hough_lines(draw_frame, cluster_lines, color=color, thickness=3)

            # Draw the outline
            for line, color in zip(outline, colors[1:]):
                draw_frame = ettk.utils.draw_hough_lines(draw_frame, line, color=color, thickness=5)

            # Draw the corners
            draw_frame = ettk.utils.draw_pts(draw_frame, corners, color=(0,0,255), radius=10)
            
            # draw match lines
            draw_frame = cv2.drawMatches(
                template, 
                result['homography']['kpts1'], 
                draw_frame, 
                result['homography']['kpts2'], 
                result['homography']['dmatches'][:20],
                None, 
                flags=2
            )

            # Create a visual representation with everything
            visual_output = ettk.utils.combine_frames(draw_frame, edges)
            cv2.imshow('output', visual_output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # break

    # Closing the video
    cv2.destroyAllWindows()

def test_step_video_with_eye_tracking(template, cap):
    
    # Load the video and get a single frame
    ret, frame = cap.read()
    
    # Load other eye-tracking information
    gaze_df = ettk.utils.tobii.load_gaze_data(TEST_TOBII_REC_PATH)
    
    # Create tracker
    planer_tracker = ettk.PlanerTracker()

    # Get the size of the video
    h, w, _ = frame.shape
    
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
            result = planer_tracker.step(template, frame)

            if type(result['M']) == type(None):
                continue
            
            # Draw paper outline
            draw_frame = ettk.utils.draw_homography_outline(draw_frame, result['dst'])

            # Apply homography to fixation and draw it on the page
            fix_pt = np.float32([ [fix[0], fix[1]] ]).reshape(-1,1,2)
            fix_dst = cv2.perspectiveTransform(fix_pt, np.linalg.inv(result['M'])).flatten().astype(np.int32)

            draw_template = cv2.circle(template.copy(), fix_dst, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

            # draw match lines
            output = cv2.drawMatches(
                draw_template, 
                result['homography']['kpts1'], 
                draw_frame, 
                result['homography']['kpts2'], 
                result['homography']['dmatches'][:20],
                None, 
                flags=2
            )
            new_output = output.astype(np.uint8)
            cv2.imshow('output', new_output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # Updated counter
        video_index_counter += 1

    # Closing the video
    cv2.destroyAllWindows()

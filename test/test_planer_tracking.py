# Built-in Imports
import os
import pathlib
import ast
import pdb
import logging

logger = logging.getLogger(__name__)

# Third-party Imports
import pytest
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Internal Imports
import ettk
import mtr

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

assert TEST_TOBII_REC_PATH.exists() and TEST_IMAGE_PATH.exists()

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


def test_single_hough_line_frame(cap):
    
    ret, frame = cap.read()
 
    # Make sure the frame is valid
    assert ret and isinstance(frame, np.ndarray)

    # Create planer tracker
    planer_tracker = ettk.PlanerTracker()

    # Apply hough lines prediction
    edges, lines = planer_tracker.perform_hough_line_prediction(frame)

    # Draw lines
    line_img = planer_tracker.draw_hough_lines(frame, lines)

    # Combine images
    output = ettk.utils.combine_frames(edges, line_img)

    # draw match lines
    plt.imshow(output); plt.show()

def test_single_contour_frame(template, cap):
    
    ret, frame = cap.read()
 
    # Make sure the frame is valid
    assert ret and isinstance(frame, np.ndarray)

    # Create planer tracker
    planer_tracker = ettk.PlanerTracker()
        
    # Perform homography
    M, kpts1, kpts2, dmatches = planer_tracker.perform_homography(template, frame)

    # Find contours
    cnts = planer_tracker.perform_contour_prediction(frame)
        
    # Then see if we can select the right contour based on dmatches
    dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,2)
    best_cnt, npts = planer_tracker.select_best_contour_based_on_pts(cnts, dst_pts)

    assert type(best_cnt) != type(None)
    logger.info(f'best_cnt: {best_cnt}, npts: {npts}')

    # Draw contours
    output = planer_tracker.draw_contours(frame, cnts)
    output = planer_tracker.draw_pts(output, dst_pts)
    output = planer_tracker.draw_contours(output, [best_cnt], color=(0,0,255))
    
    # draw match lines
    plt.imshow(output); plt.show()

def test_single_homography_frame(template, cap):
    
    ret, frame = cap.read()
 
    # Make sure the frame is valid
    assert ret and isinstance(frame, np.ndarray)

    # Create planer tracker
    planer_tracker = ettk.PlanerTracker()

    # Apply homography
    result = planer_tracker.step(template, frame)
    
    # Draw paper outline
    frame = planer_tracker.draw_homography_outline(frame, result['dst'])

    # draw match lines
    output = cv2.drawMatches(
        template, 
        result['homography']['kpts1'], 
        frame, 
        result['homography']['kpts2'], 
        result['homography']['dmatches'][:20],
        None, 
        flags=2
    )
    plt.imshow(output); plt.show()

def test_complete_video_homography_frame(template, cap):
    
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
            # Apply homography
            result = planer_tracker.step(template, frame)
            
            # Draw paper outline
            frame = planer_tracker.draw_homography_outline(frame, result['dst'])

            # draw match lines
            output = cv2.drawMatches(
                template, 
                result['homography']['kpts1'], 
                frame, 
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

        # break

    # Closing the video
    cv2.destroyAllWindows()

def test_eye_tracking_with_homography_single_frame(template, cap):

    # Load the video and get a single frame
    ret, frame = cap.read()

    # Load other eye-tracking information
    gaze_df = ettk.utils.tobii.load_gaze_data(TEST_TOBII_REC_PATH)
 
    # Make sure the frame is valid
    assert ret and isinstance(frame, np.ndarray)

    # Create tracker
    planer_tracker = ettk.PlanerTracker()

    # Determine the fixation associated with the frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_time = VIDEO_START_INDEX * (1/fps)
    raw_fix = ast.literal_eval(gaze_df[gaze_df['timestamp'] > current_time].reset_index().iloc[0]['gaze2d'])
    
    # Transform proportional fixation to pixel fixation
    h, w = frame.shape[:2]
    fix = (int(raw_fix[0]*w), int(raw_fix[1]*h))

    # Draw eye-tracking into the original video frame
    frame = cv2.circle(frame, fix, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

    # Apply homography
    result = planer_tracker.step(template, frame)
    
    # Draw paper outline
    frame = planer_tracker.draw_homography_outline(frame, result['dst'])

    # Draw the projected eye-gaze to the template
    # Apply homography to fixation and draw it on the page
    fix_pt = np.float32([ [fix[0], fix[1]] ]).reshape(-1,1,2)
    fix_dst = cv2.perspectiveTransform(fix_pt, np.linalg.inv(result['M'])).flatten().astype(np.int32)
    template = cv2.circle(template, fix_dst, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

    # draw match lines
    output = cv2.drawMatches(
        template, 
        result['homography']['kpts1'], 
        frame, 
        result['homography']['kpts2'], 
        result['homography']['dmatches'][:20],
        None, 
        flags=2
    )
    plt.imshow(output); plt.show()

def test_eye_tracking_with_complete_video_homography_frame(template, cap):
    
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

        if ret:
            # Apply homography
            result = planer_tracker.step(template, frame)

            if type(result['M']) == type(None):
                continue
            
            # Draw paper outline
            draw_frame = planer_tracker.draw_homography_outline(draw_frame, result['dst'])

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

def test_DL_approach(template, cap):

    # Create a model
    nyu_net = mtr.TrainedNet(
        dataset='nyu', 
        tasks=['seg', 'normals'],
        device=torch.device('cpu')
    )
    nyu_net.eval()

    # Get cmap
    cmap = mtr.get_cmap('nyu')

    # Then perform homography
    while(True):

        # Get video
        ret, img = cap.read()
        img = mtr.match_size_img(img)

        # Prepare the image
        prep_img = mtr.prepare_img(img)

        # forward propagate
        seg, depth, norm = nyu_net(prep_img)

        # Clean outputs
        c_seg = mtr.clean_seg(seg, img, 'nyu')
        c_depth = mtr.clean_depth(depth, img)
        c_norm = mtr.clean_norm(norm, img)
        
        # Constructing visualization method
        vis_seg = cmap[c_seg.argmax(axis=2) + 1].astype(np.uint8)
        visual = ettk.utils.combine_frames(img, vis_seg)
        cv2.imshow('visual', visual)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

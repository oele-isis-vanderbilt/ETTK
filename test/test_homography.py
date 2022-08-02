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
TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec1_v4'
TEST_IMAGE_PATH = CWD/'data'/'resources'/'paper_v4'/'UnwrappingthePast-PRINT-1.png'
VIDEO_START_INDEX = 1000

FIX_RADIUS = 15
FIX_COLOR = (0, 0, 255)
FIX_THICKNESS = 5

assert TEST_TOBII_REC_PATH.exists() and TEST_IMAGE_PATH.exists()

def test_single_homography_frame():

    # Load the video and get a single frame
    cap = cv2.VideoCapture(str(TEST_TOBII_REC_PATH/'scenevideo.mp4'), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)
    ret, img2 = cap.read()
    img1 = cv2.imread(str(TEST_IMAGE_PATH), 0)
    
    # Make sure the frame is valid
    assert ret and isinstance(img2, np.ndarray)

    # Initiate SIFT detector
    # feature_extractor = cv2.SIFT_create()
    feature_extractor = cv2.ORB_create()

    # Apply homography
    M, dst, kpts1, kpts2, dmatches = ettk.perform_homography(feature_extractor, img1, img2)
    
    # Draw paper outline
    img2 = ettk.draw_homography_outline(img2, dst)

    # draw match lines
    output = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20], None, flags=2)
    plt.imshow(output); plt.show()

def test_complete_video_homography_frame():
    
    # Load the video and get a single frame
    cap = cv2.VideoCapture(str(TEST_TOBII_REC_PATH/'scenevideo.mp4'), 0)
    ret, img2 = cap.read()
    img1 = cv2.imread(str(TEST_IMAGE_PATH), 0)
    
    # Initiate SIFT detector
    # feature_extractor = cv2.SIFT_create()
    feature_extractor = cv2.ORB_create()

    # Get the size of the video
    h, w, _ = img2.shape

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    # Set the starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Then perform homography
    while(True):
        ret, img2 = cap.read()
        if ret:
            # Apply homography
            M, dst, kpts1, kpts2, dmatches = ettk.perform_homography(feature_extractor, img1, img2)
            
            # Draw paper outline
            img2 = ettk.draw_homography_outline(img2, dst)

            # draw match lines
            output = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20], None, flags=2)
            new_output = output.astype(np.uint8)
            cv2.imshow('output', new_output)
            cv2.waitKey(1)
        else:
            break

        # break

    # Closing the video
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_eye_tracking_with_homography_single_frame():

    # Load the video and get a single frame
    cap = cv2.VideoCapture(str(TEST_TOBII_REC_PATH/'scenevideo.mp4'), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)
    ret, img2 = cap.read()
    img1 = cv2.imread(str(TEST_IMAGE_PATH), 0)

    # Load other eye-tracking information
    gaze_df = ettk.utils.tobii.load_gaze_data(TEST_TOBII_REC_PATH)
 
    # Make sure the frame is valid
    assert ret and isinstance(img2, np.ndarray)

    # Initiate SIFT detector
    # feature_extractor = cv2.SIFT_create()
    feature_extractor = cv2.ORB_create()

    # Determine the fixation associated with the frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_time = VIDEO_START_INDEX * (1/fps)
    fix = ast.literal_eval(gaze_df[gaze_df['timestamp'] > current_time].reset_index().iloc[0]['gaze2d'])
    h, w, _ = img2.shape

    # Draw eye-tracking into the original video frame
    img2 = cv2.circle(img2, (int(fix[0]*w), int(fix[1]*h)), FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

    # Apply homography
    M, dst, kpts1, kpts2, dmatches = ettk.perform_homography(feature_extractor, img1, img2)
    
    # Draw paper outline
    img2 = ettk.draw_homography_outline(img2, dst)

    # draw match lines
    output = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20], None, flags=2)
    plt.imshow(output); plt.show()
    logger.debug(fix)

def test_eye_tracking_with_complete_video_homography_frame():
    
    # Load the video and get a single frame
    cap = cv2.VideoCapture(str(TEST_TOBII_REC_PATH/'scenevideo.mp4'), 0)
    ret, img2 = cap.read()
    img1 = cv2.imread(str(TEST_IMAGE_PATH), 0)
    
    # Load other eye-tracking information
    gaze_df = ettk.utils.tobii.load_gaze_data(TEST_TOBII_REC_PATH)
    
    # Initiate SIFT detector
    # feature_extractor = cv2.SIFT_create()
    feature_extractor = cv2.ORB_create()

    # Get the size of the video
    h, w, _ = img2.shape
    
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
        ret, img2 = cap.read()

        # Get fixation
        current_time = (VIDEO_START_INDEX+video_index_counter) * (1/fps)
        raw_fix = ast.literal_eval(gaze_df[gaze_df['timestamp'] > current_time].reset_index().iloc[0]['gaze2d'])
        
        fix = (int(raw_fix[0]*w), int(raw_fix[1]*h))
        
        # Draw eye-tracking into the original video frame
        img2 = cv2.circle(img2, fix, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

        if ret:
            # Apply homography
            M, dst, kpts1, kpts2, dmatches = ettk.perform_homography(feature_extractor, img1, img2)
            logger.info(f'rect dst: {dst}')
            
            # Draw paper outline
            img2 = ettk.draw_homography_outline(img2, dst)

            # Apply homography to fixation and draw it on the page
            fix_pt = np.float32([ [fix[0], fix[1]] ]).reshape(-1,1,2)
            # fix_pt = dst[0].reshape(-1,1,2).astype(np.float32)
            fix_dst = cv2.perspectiveTransform(fix_pt, np.linalg.inv(M)).flatten().astype(np.int32)

            logger.info(f'fix: {fix}, fix_pt: {fix_pt}, fix_dst: {fix_dst}')
            draw_img1 = cv2.circle(img1.copy(), fix_dst, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

            # draw match lines
            output = cv2.drawMatches(draw_img1, kpts1, img2, kpts2, dmatches[:20], None, flags=2)
            new_output = output.astype(np.uint8)
            cv2.imshow('output', new_output)
            cv2.waitKey(1)
        else:
            break

        # Updated counter
        video_index_counter += 1

        # break

    # Closing the video
    # writer.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Built-in Imports
import os
import sys
import pathlib
import ast
import time
import pdb
import logging
from typing import Literal, List
from dataclasses import asdict

# Third-party Imports
import imutils
import cv2
import pytest
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

import ettk
from scipy.spatial.transform import Rotation as R
import pytest
import cv2

from .conftest import rec_data
from .conftest import VIDEO_TOBII_REC_PATH, OUTPUT_DIR
from .surface_configs import (
    unwrap1_config, 
    unwrap2_config, 
    unwrap3_config, 
    suffrage1_config, 
    suffrage2_config, 
    suffrage3_config, 
    mooca1_config,
    mooca2_config,
    mooca3_config,
    mooca4_config,
    mooca5_config,
    mooca6_config,
    mooca7_config,
    mooca8_config,
    mooca9_config,
    mooca10_config,
    monitor_config
)

logger = logging.getLogger('ettk')

MATRIX_COEFFICIENTS = np.array(
    [
        [910.5968017578125, 0, 958.43426513671875],
        [0, 910.20758056640625, 511.6611328125],
        [0, 0, 1],
    ]
)
DISTORTION_COEFFICIENTS = np.array(
    [
        -0.055919282138347626,
        0.079781122505664825,
        -0.048538044095039368,
        -0.00014426070265471935,
        0.00044536130735650659,
    ]
)

# Monitor RT
MONITOR_RT = np.array(
    [
        [0.96188002, 0.05182879, 0.26851556, 0.23665886],
        [-0.00466573, 0.98484384, -0.17338061, -0.24785235],
        [-0.27343201, 0.16551852, 0.94754343, -6.43561886],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

def test_planar_tracking(rec_data):

    # Get original video
    cap, gaze = rec_data
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(str(OUTPUT_DIR/'planar_tracking.avi'), fourcc, fps, (1920, 1080))

    # Surface Configs
    surface_configs = {
        'unwrap1': unwrap1_config,
        'unwrap2': unwrap2_config,
        'unwrap3': unwrap3_config,
        'suffrage1': suffrage1_config,
        'suffrage2': suffrage2_config,
        'suffrage3': suffrage3_config,
        'mooca1': mooca1_config,
        'mooca2': mooca2_config,
        'mooca3': mooca3_config,
        'mooca4': mooca4_config,
        'mooca5': mooca5_config,
        'mooca6': mooca6_config,
        'mooca7': mooca7_config,
        'mooca8': mooca8_config,
        'mooca9': mooca9_config,
        'mooca10': mooca10_config,
        'monitor': monitor_config
    }

    # Tracker
    # aruco_tracker = ettk.ArucoTracker(aruco_omit=[5, 36, 37, 0, 1, 2, 3, 4, 5, 6])
    aruco_tracker = ettk.ArucoTracker(aruco_omit=[5, 2, 1, 4, 6, 36, 37])
    planar_tracker = ettk.PlanarTracker(
        surface_configs=list(surface_configs.values()), 
        aruco_tracker=aruco_tracker
    )

    while True: 
        try:
            ret, frame = cap.read()
        except Exception as e:
            break

        if ret:

            # Checking FPS
            tic = time.perf_counter()

            # Processing
            planar_results = planar_tracker.step(frame)
            draw = frame.copy()

            # Get the fixation
            current_time = current_frame_index / fps
            fix = ettk.utils.tobii.get_absolute_fix(gaze, current_time)

            # Draw
            draw = ettk.utils.vis.draw_fix(fix, draw)
            draw = ettk.utils.vis.draw_aruco_markers(draw, **asdict(planar_results.aruco), with_ids=True)
            for surface in planar_results.surfaces.values():
                draw = ettk.utils.draw_axis(draw, surface.rvec, surface.tvec)

                # Debugging
                # for hypothesis in surface.hypotheses:
                #     draw = ettk.utils.draw_axis(draw, hypothesis.rvec, hypothesis.tvec)
            
                draw = ettk.utils.vis.draw_surface_corners(draw, surface.corners)

                # If homo, draw it
                # if surface.homography is not None:
                #     corners = surface.homography.corners
                #     draw = ettk.utils.vis.draw_surface_corners(draw, corners)

            # Obtain the XY of the fixation
            fix_result = ettk.utils.surface_map_points(planar_results, fix)
            if fix_result:
                surface_config = surface_configs[fix_result.surface_id]
                s_h, s_w = surface_config.height, surface_config.width
                pt = fix_result.pt
                RATIO = 50
                s_h *= RATIO
                s_w *= RATIO
                pt = pt * RATIO
                img = np.zeros((int(s_h), int(s_w), 3))
                draw_surface = ettk.utils.vis.draw_fix((pt[0], pt[1]), img)
                cv2.imshow('surface', draw_surface)

            # Checking FPS
            toc = time.perf_counter()
            fps_performance = 1 / (toc - tic)

            # Draw FPS
            draw = cv2.putText(draw, f"FPS: {fps_performance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Testing
            # draw = ettk.utils.vis.draw_lines(draw, surface.lines)

            cv2.imshow('frame', imutils.resize(draw, width=1920))
            key = cv2.waitKey(1)
            writer.write(draw)

            if key & 0xFF == ord("q"):
                break
        else:
            break

        # Update
        current_frame_index += 1

    writer.release()
    cv2.destroyAllWindows()


def test_planar_tracking_step_by_gaze(rec_data):

    # Get original video
    cap, gaze = rec_data
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_time = current_frame_index / fps
    gaze = gaze[gaze.timestamp >= current_time].reset_index(drop=True)
    w, h = 1920, 1080

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(str(OUTPUT_DIR/'planar_tracking.avi'), fourcc, fps, (1920, 1080))

    # Surface Configs
    surface_configs = {
        'unwrap1': unwrap1_config,
        'unwrap2': unwrap2_config,
        'unwrap3': unwrap3_config,
        'suffrage1': suffrage1_config,
        'suffrage2': suffrage2_config,
        'suffrage3': suffrage3_config,
        'mooca1': mooca1_config,
        'mooca2': mooca2_config,
        'mooca3': mooca3_config,
        'mooca4': mooca4_config,
        'mooca5': mooca5_config,
        'mooca6': mooca6_config,
        'mooca7': mooca7_config,
        'mooca8': mooca8_config,
        'mooca9': mooca9_config,
        'mooca10': mooca10_config,
        'monitor': monitor_config
    }

    # Tracker
    aruco_tracker = ettk.ArucoTracker(aruco_omit=[5, 2, 1, 4, 6, 36, 37])
    planar_tracker = ettk.PlanarTracker(
        surface_configs=list(surface_configs.values()), 
        aruco_tracker=aruco_tracker
    )

    # Keep track of the PlanarResults
    planar_results = None

    for i, row in gaze.iterrows():
        
        # Get the fixation
        try:
            raw_fix = row["gaze2d"]
        except IndexError:
            raw_fix = [0, 0]

        if isinstance(raw_fix, str):
            raw_fix = ast.literal_eval(raw_fix)

        fix = (int(raw_fix[0] * w), int(raw_fix[1] * h))
        # import pdb; pdb.set_trace()

        # First, check if we need to update our planar results
        timestamp = row.timestamp
        expected_frame_index = int(timestamp * fps)

        # If we expected a new frame, then perform image processing
        while (current_frame_index <= expected_frame_index or current_frame_index == 0):

            try:
                ret, frame = cap.read()
            except Exception as e:
                break
           
            # Keep track of clocks
            delta = expected_frame_index - current_frame_index
            # logger.debug(f"{expected_frame_index} - {current_frame_index} = {delta}")
            current_frame_index += 1
            # import pdb; pdb.set_trace()
            
            if ret:

                # Checking FPS
                tic = time.perf_counter()

                # Processing
                if delta <= 0:
                    planar_results = planar_tracker.step(frame)

                # Draw
                frame = ettk.utils.vis.draw_fix(fix, frame)
                frame = ettk.utils.vis.draw_aruco_markers(frame , **asdict(planar_results.aruco), with_ids=True)
                for surface in planar_results.surfaces.values():
                    frame = ettk.utils.draw_axis(frame , surface.rvec, surface.tvec)

                    # Debugging
                    # for hypothesis in surface.hypotheses:
                    #     draw = ettk.utils.draw_axis(draw, hypothesis.rvec, hypothesis.tvec)
                
                    frame = ettk.utils.vis.draw_surface_corners(frame , surface.corners)

                    # If homo, draw it
                    # if surface.homography is not None:
                    #     corners = surface.homography.corners
                    #     draw = ettk.utils.vis.draw_surface_corners(draw, corners)

                # Checking FPS
                toc = time.perf_counter()
                fps_performance = 1 / (toc - tic)

                # Draw FPS
                frame = cv2.putText(frame , f"FPS: {fps_performance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Testing
                # frame = ettk.utils.vis.draw_lines(frame , surface.lines)

                cv2.imshow('frame', imutils.resize(frame , width=1920))
                key = cv2.waitKey(1)
                writer.write(frame )

                if key & 0xFF == ord("q"):
                    sys.exit()
            else:
                break

        # Obtain the XY of the fixation
        if planar_results:
            fix_result = ettk.utils.surface_map_points(planar_results, fix)
            if fix_result:
                surface_config = surface_configs[fix_result.surface_id]
                pt = fix_result.pt

                # Get drawing surface
                if isinstance(surface_config.template, np.ndarray):
                    img = surface_config.template
                    s_h, s_w = img.shape[:2]
                    pt *= 20
                else:
                    s_h, s_w = surface_config.height, surface_config.width
                    RATIO = 50
                    s_h *= RATIO
                    s_w *= RATIO
                    pt = pt * RATIO
                    img = np.zeros((int(s_h), int(s_w), 3))

                draw_surface = ettk.utils.vis.draw_fix((pt[0], pt[1]), img)
                cv2.imshow('surface', draw_surface)

    writer.release()
    cv2.destroyAllWindows()

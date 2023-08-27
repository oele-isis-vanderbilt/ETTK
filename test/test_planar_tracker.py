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
from .conftest import VIDEO_TOBII_REC_PATH
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

    # Tracker
    # aruco_tracker = ettk.ArucoTracker(aruco_omit=[5, 36, 37, 0, 1, 2, 3, 4, 5, 6])
    aruco_tracker = ettk.ArucoTracker(aruco_omit=[5, 2, 1])
    planar_tracker = ettk.PlanarTracker(
        surface_configs=[
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
        ], 
        aruco_tracker=aruco_tracker
    )

    cap, gaze = rec_data

    while True: 
        ret, frame = cap.read()

        # Checking FPS
        tic = time.perf_counter()

        # Processing
        planar_results = planar_tracker.step(frame)
        draw = frame.copy()
        draw = ettk.utils.vis.draw_aruco_markers(draw, **asdict(planar_results.aruco), with_ids=True)
        for surface in planar_results.surfaces.values():
            draw = ettk.utils.draw_axis(draw, surface.rvec, surface.tvec)

            # Debugging
            # for hypothesis in surface.hypotheses:
            #     draw = ettk.utils.draw_axis(draw, hypothesis.rvec, hypothesis.tvec)
        
            draw = ettk.utils.vis.draw_surface_corners(draw, surface.corners)

        # Checking FPS
        toc = time.perf_counter()
        fps = 1 / (toc - tic)

        # Draw FPS
        draw = cv2.putText(draw, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Testing
        # draw = ettk.utils.vis.draw_lines(draw, surface.lines)

        if ret:
            cv2.imshow('frame', imutils.resize(draw, width=1920))
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("s"):
                cv2.imwrite(f"output_{video_index_counter}.png", output)

# Built-in Imports
import os
import sys
import pathlib
import ast
import pdb
import logging
from typing import Literal, List
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Third-party Imports
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

# Page Aruco Size
PAGE_ARUCO_SIZE = 0.297
PAGE_HEIGHT_SIZE = 29
PAGE_WIDTH_SIZE = 21.5
W_SCALE = 1/105
H_SCALE = 1/110
R_CORR = 0.3

# Grid
x_grid = [1.8, 20.0]
y_grid = [4.6, 13.5, 24.0]

def test_planar_tracking(rec_data):

    # Create configuration for UnwrappingOfThePast
    unwrap1_config = ettk.SurfaceConfig(
        id='unwrap1',
        aruco_config={
            8: ettk.ArucoConfig(
                8,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            9: ettk.ArucoConfig(
                9,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            10: ettk.ArucoConfig(
                10,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            11: ettk.ArucoConfig(
                11,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            12: ettk.ArucoConfig(
                12,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            13: ettk.ArucoConfig(
                13,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
        },
        height=PAGE_HEIGHT_SIZE,
        width=PAGE_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )
    unwrap2_config = ettk.SurfaceConfig(
        id='unwrap2',
        aruco_config={
            14: ettk.ArucoConfig(
                14,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            15: ettk.ArucoConfig(
                15,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            16: ettk.ArucoConfig(
                16,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            17: ettk.ArucoConfig(
                17,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            18: ettk.ArucoConfig(
                18,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            19: ettk.ArucoConfig(
                19,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
        },
        height=PAGE_HEIGHT_SIZE,
        width=PAGE_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )
    unwrap3_config = ettk.SurfaceConfig(
        id='unwrap3',
        aruco_config={
            20: ettk.ArucoConfig(
                20,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            21: ettk.ArucoConfig(
                21,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            22: ettk.ArucoConfig(
                22,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            23: ettk.ArucoConfig(
                23,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            24: ettk.ArucoConfig(
                24,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            25: ettk.ArucoConfig(
                25,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
        },
        height=PAGE_HEIGHT_SIZE,
        width=PAGE_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )
    
    suffrage1_config = ettk.SurfaceConfig(
        id='suffrage1',
        aruco_config={
            26: ettk.ArucoConfig(
                26,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            27: ettk.ArucoConfig(
                27,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            28: ettk.ArucoConfig(
                28,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            29: ettk.ArucoConfig(
                29,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            30: ettk.ArucoConfig(
                30,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            31: ettk.ArucoConfig(
                31,
                PAGE_ARUCO_SIZE,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
        },
        height=PAGE_HEIGHT_SIZE,
        width=PAGE_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )

    # Tracker
    aruco_tracker = ettk.ArucoTracker(aruco_omit=[5])
    planar_tracker = ettk.PlanarTracker(
        surface_configs=[
            unwrap1_config,
            unwrap2_config,
            unwrap3_config,
            suffrage1_config
        ], 
        aruco_tracker=aruco_tracker
    )

    cap, gaze = rec_data

    while True: 
        ret, frame = cap.read()

        # Processing
        planar_results = planar_tracker.step(frame)
        draw = ettk.utils.vis.draw_aruco_markers(frame, **asdict(planar_results.aruco), with_ids=True)
        for surface in planar_results.surfaces.values():
            draw = ettk.utils.draw_axis(draw, surface.rvec, surface.tvec)

            # Debugging
            for hypothesis in surface.hypotheses:
                draw = ettk.utils.draw_axis(draw, hypothesis.rvec, hypothesis.tvec)

            draw = ettk.utils.vis.draw_surface_corners(draw, surface.corners)

        if ret:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("s"):
                cv2.imwrite(f"output_{video_index_counter}.png", output)

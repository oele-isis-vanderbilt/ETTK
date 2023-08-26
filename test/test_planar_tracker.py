# Built-in Imports
import os
import sys
import pathlib
import ast
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

# Page Aruco Size
PAGE_HEIGHT_SIZE = 27
PAGE_WIDTH_SIZE = 21.5
MONITOR_HEIGHT_SIZE = 19
MONITOR_WIDTH_SIZE = 29
M_ARUCO_SIZE = 2.5
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
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            9: ettk.ArucoConfig(
                9,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            10: ettk.ArucoConfig(
                10,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            11: ettk.ArucoConfig(
                11,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            12: ettk.ArucoConfig(
                12,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            13: ettk.ArucoConfig(
                13,
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
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            15: ettk.ArucoConfig(
                15,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            16: ettk.ArucoConfig(
                16,
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            17: ettk.ArucoConfig(
                17,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            18: ettk.ArucoConfig(
                18,
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            19: ettk.ArucoConfig(
                19,
                np.array([0, 0, 0]),
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
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            21: ettk.ArucoConfig(
                21,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            22: ettk.ArucoConfig(
                22,
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            23: ettk.ArucoConfig(
                23,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            24: ettk.ArucoConfig(
                24,
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            25: ettk.ArucoConfig(
                25,
                np.array([0, 0, 0]),
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
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            27: ettk.ArucoConfig(
                27,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            28: ettk.ArucoConfig(
                28,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            29: ettk.ArucoConfig(
                29,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            30: ettk.ArucoConfig(
                30,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            31: ettk.ArucoConfig(
                31,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
        },
        height=PAGE_HEIGHT_SIZE,
        width=PAGE_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )
    suffrage2_config = ettk.SurfaceConfig(
        id='suffrage2',
        aruco_config={
            32: ettk.ArucoConfig(
                32,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            33: ettk.ArucoConfig(
                33,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            34: ettk.ArucoConfig(
                34,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            35: ettk.ArucoConfig(
                35,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            36: ettk.ArucoConfig(
                36,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            37: ettk.ArucoConfig(
                37,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
        },
        height=PAGE_HEIGHT_SIZE,
        width=PAGE_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )
    suffrage3_config = ettk.SurfaceConfig(
        id='suffrage3',
        aruco_config={
            38: ettk.ArucoConfig(
                38,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[0], 0])
            ),
            39: ettk.ArucoConfig(
                39,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            40: ettk.ArucoConfig(
                40,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            41: ettk.ArucoConfig(
                41,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            42: ettk.ArucoConfig(
                42,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            43: ettk.ArucoConfig(
                43,
                np.array([R_CORR, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
        },
        height=PAGE_HEIGHT_SIZE,
        width=PAGE_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )

    monitor_config = ettk.SurfaceConfig(
        id='monitor',
        aruco_config={
            0: ettk.ArucoConfig(
                0,
                np.array([0, 0, 0]),
                np.array([0, 0, 0])
            ),
            1: ettk.ArucoConfig(
                1,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[0], 0])
            ),
            2: ettk.ArucoConfig(
                2,
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[1], 0])
            ),
            3: ettk.ArucoConfig(
                3,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[1], 0])
            ),
            4: ettk.ArucoConfig(
                4,
                np.array([0, 0, 0]),
                np.array([x_grid[0], y_grid[2], 0])
            ),
            5: ettk.ArucoConfig(
                5,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
            6: ettk.ArucoConfig(
                6,
                np.array([0, 0, 0]),
                np.array([x_grid[1], y_grid[2], 0])
            ),
            7: ettk.ArucoConfig(
                7,
                np.array([0, -0.03, -0.05]),
                np.array([2.3, -M_ARUCO_SIZE/2, 0])
            ),
        },
        height=MONITOR_HEIGHT_SIZE,
        width=MONITOR_WIDTH_SIZE,
        scale=(W_SCALE, H_SCALE)
    )

    # Tracker
    aruco_tracker = ettk.ArucoTracker(aruco_omit=[5, 36, 37, 0, 1, 2, 3, 4, 5, 6])
    planar_tracker = ettk.PlanarTracker(
        surface_configs=[
            unwrap1_config,
            unwrap2_config,
            unwrap3_config,
            suffrage1_config,
            suffrage2_config,
            suffrage3_config,
            monitor_config
        ], 
        aruco_tracker=aruco_tracker
    )

    cap, gaze = rec_data

    while True: 
        ret, frame = cap.read()

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

        # Testing
        # draw = ettk.utils.vis.draw_lines(draw, surface.lines)

        if ret:
            cv2.imshow('frame', imutils.resize(draw, width=1000))
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("s"):
                cv2.imwrite(f"output_{video_index_counter}.png", output)

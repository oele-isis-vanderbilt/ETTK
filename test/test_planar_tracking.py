# Built-in Imports
from typing import Literal, List
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
from scipy.spatial.transform import Rotation as R

# Internal Imports
import ettk

# CONSTANTS
CWD = pathlib.Path(os.path.abspath(__file__)).parent

# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v4_rec1'
# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v4_rec2'
# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v4_rec3'
# PAPER_TEMPLATE = CWD/'data'/'resources'/'paper_v4'/'UnwrappingthePast-PRINT-1.png'

# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v4_rec4'
# PAPER_TEMPLATE = CWD/'data'/'resources'/'paper_v4'/'UnwrappingthePast-PRINT-3.png'

# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v5_rec1'
# PAPER_TEMPLATE = CWD/'data'/'resources'/'paper_v5'/'UnwrappingthePast-PRINT-1.png'

# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v4_rec5_multi_paper'
# PAPER_TEMPLATE = CWD/'data'/'resources'/'paper_v4'

# PAPER_TOBII_REC_PATH = CWD / "data" / "recordings" / "tobii_paper_v6_rec5_multi_paper"
# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v6_rec4'
# PAPER_TEMPLATE = CWD/'data'/'resources'/'paper_v6'/'test_paper_template_1.png'
# PAPER_TEMPLATE = CWD / "data" / "resources" / "paper_v6"

# COMPUTER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_v1_rec1'
# COMPUTER_TEMPLATE = CWD/'data'/'resources'/'computer'/'computer_screenshot.png'

# COMPUTER_TOBII_REC_PATH = CWD / "data" / "recordings" / "tobii_computer_v2_rec1"
# COMPUTER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_v3_rec1'
# COMPUTER_TEMPLATE = (
#     CWD / "data" / "resources" / "computer" / "computer_screenshot_large_text.png"
# )
VIDEO_TOBII_REC_PATH = (
    CWD
    / "data"
    / "recordings"
    / "220506_chimerapy-2023_04_18_09_37_35-7152"
    / "tg3"
    / "20230418T142941Z"
)

# VIDEO_START_INDEX = 1000
# VIDEO_START_INDEX = 2200
VIDEO_START_INDEX = 50000

# TRIM_MARGIN_X = 80
# TRIM_MARGIN_Y_TOP = 100
# TRIM_MARGIN_Y_BOTTOM = 150

BLACK_MARGIN_SIZE = 50

assert VIDEO_TOBII_REC_PATH.exists()

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


def get_rec_data(path):

    # Load the video and get a single frame
    video_path = path / "scenevideo.mp4"
    assert video_path.exists()

    cap = cv2.VideoCapture(str(video_path), 0)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert length > VIDEO_START_INDEX + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Load other eye-tracking information
    gaze = ettk.utils.tobii.load_gaze_data(path)
    return cap, gaze


@pytest.fixture
def tracker():
    tracker = ettk.PlanarTracker()
    return tracker


@pytest.fixture
def rec_data():
    return get_rec_data(VIDEO_TOBII_REC_PATH)


@pytest.fixture
def monitor_test_image(rec_data):

    # Decompose data
    cap, _ = rec_data

    # Set the starting point
    calibrate_index = 50185
    cap.set(cv2.CAP_PROP_POS_FRAMES, calibrate_index)

    # Get video
    _, frame = cap.read()

    return frame


def test_3d_project_with_only_aruco(monitor_test_image):

    pts_3d = (
        0.02
        * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32).T
    )

    aruco_r = np.array([[[-1.8736, -1.7129, 0.6151]]])
    aruco_t = np.array([[[0.13669, 0.080151, 0.25044]]])[0]

    cv2.drawFrameAxes(
        monitor_test_image,
        MATRIX_COEFFICIENTS,
        DISTORTION_COEFFICIENTS,
        aruco_r,
        aruco_t,
        0.01,
    )  # Draw Axis
    output = monitor_test_image.copy()

    cv2.imshow("output", output)
    key = cv2.waitKey(0)


def test_3d_project_aruco_and_monitor(monitor_test_image):

    pts_3d = (
        0.02
        * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32).T
    )

    aruco_r = np.array([[[-1.8736, -1.7129, 0.6151]]])
    aruco_t = np.array([[[0.13669, 0.080151, 0.25044]]])[0]

    monitor_r = (
        R.from_matrix(MONITOR_RT[:3, :3]).as_rotvec().reshape((1, 1, 3))
    )  # * np.array([[[0, 0, np.pi]]])
    monitor_t = np.array([[[0.02, -0.02, 0.25044]]])
    # monitor_t = MONITOR_RT[:3,-1].reshape((1,1,3))

    # Determine delta
    m_r = R.from_rotvec(monitor_r.squeeze())
    a_r = R.from_rotvec(aruco_r.squeeze())
    delta = m_r * a_r.inv()
    recon_m_r = delta * a_r
    recon_monitor_r = recon_m_r.as_rotvec().reshape((1, 1, 3))
    import pdb

    pdb.set_trace()

    cv2.drawFrameAxes(
        monitor_test_image,
        MATRIX_COEFFICIENTS,
        DISTORTION_COEFFICIENTS,
        aruco_r,
        aruco_t,
        0.01,
    )  # Draw Axis
    cv2.drawFrameAxes(
        monitor_test_image,
        MATRIX_COEFFICIENTS,
        DISTORTION_COEFFICIENTS,
        recon_monitor_r,
        monitor_t,
        0.01,
    )  # Draw Axis
    output = monitor_test_image.copy()

    cv2.imshow("output", output)
    key = cv2.waitKey(0)


def test_monitor_alignment(monitor_test_image, tracker):

    # Apply homography
    result = tracker.step(monitor_test_image)
    output = ettk.utils.render((0, 0), result)

    # Put frame ID
    output = cv2.putText(
        output,
        f"frame: {video_index_counter}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("output", output)
    key = cv2.waitKey(1)

    for i, id in enumerate(result.aruco.ids[:, 0]):
        r = R.from_rotvec(result.aruco.rvec[i, 0, 0])
        t = np.expand_dims(result.aruco.tvec[i, 0, 0], axis=1)
        rt = np.vstack((np.hstack((r.as_matrix(), t)), np.array([0, 0, 0, 1])))
        delta = MONITOR_RT @ np.linalg.inv(rt)
        delta_r = R.from_matrix(delta[:3, :3])
        diff = R.from_rotvec([0, np.pi, 0])
        pdb.set_trace()

    # Closing the video
    cv2.destroyAllWindows()


def test_step_video_with_eye_tracking(rec_data, tracker):

    # Decompose data
    cap, gaze_logs = rec_data

    # Determine fixation timestamp setup information
    fps = cap.get(cv2.CAP_PROP_FPS)

    # # Creating video writer to save generated output
    # writer = cv2.VideoWriter(
    #     str(CWD/'output'/f"{exp_type}_video.avi"),
    #     cv2.VideoWriter_fourcc(*'DIVX'),
    #     fps=fps,
    #     frameSize=[w,h]
    # )

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    # Set the starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_START_INDEX)

    # Creating counter to track video index
    video_index_counter = VIDEO_START_INDEX

    # Then perform homography
    while True:

        # Get video
        ret, frame = cap.read()

        if ret:

            # Get fixation
            current_time = (VIDEO_START_INDEX + video_index_counter) * (1 / fps)
            fix = ettk.utils.tobii.get_absolute_fix(gaze_logs, current_time)

            # Apply homography
            result = tracker.step(frame)
            output = ettk.utils.render(fix, result)

            # Put frame ID
            output = cv2.putText(
                output,
                f"frame: {video_index_counter}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("output", output)

            key = cv2.waitKey(0)

            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("s"):
                cv2.imwrite(f"output_{video_index_counter}.png", output)
        else:
            break

        # Updated counter
        video_index_counter += 1

    # Closing the video
    cv2.destroyAllWindows()

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
import imutils
import pandas as pd
from pytest_lazyfixture import lazy_fixture

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


def get_templates(
    templates_path, trim_type: Literal["paper", "computer"]
) -> List[np.ndarray]:

    if templates_path.is_dir():
        templates_filepaths = list(templates_path.iterdir())
    else:
        templates_filepaths = [templates_path]

    # Load multiple templates inside a directory
    templates: List[np.ndarray] = []
    for id, template_filepath in enumerate(templates_filepaths):

        template = cv2.imread(str(template_filepath), 0)

        if trim_type == "paper":
            TRIM_MARGIN_X = 40
            TRIM_MARGIN_Y_TOP = 1
            TRIM_MARGIN_Y_BOTTOM = 250
        elif trim_type == "computer":
            TRIM_MARGIN_X = 1
            TRIM_MARGIN_Y_TOP = 1
            TRIM_MARGIN_Y_BOTTOM = 1
        else:
            raise Exception

        # Might need to trim the margins for now
        template = template[
            TRIM_MARGIN_Y_TOP:-TRIM_MARGIN_Y_BOTTOM, TRIM_MARGIN_X:-TRIM_MARGIN_X
        ]

        # Put the padding
        template = cv2.copyMakeBorder(
            template,
            BLACK_MARGIN_SIZE,
            BLACK_MARGIN_SIZE,
            BLACK_MARGIN_SIZE,
            BLACK_MARGIN_SIZE,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        templates.append(template)
        # cv2.imshow(f'template-{id}', template)

    # cv2.waitKey(0)

    return templates


@pytest.fixture
def tracker():
    tracker = ettk.PlanarTracker()
    return tracker


@pytest.fixture
def rec_data():
    return get_rec_data(VIDEO_TOBII_REC_PATH)


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
    video_index_counter = 0

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
            output = result.render(fix)
            cv2.imshow("output", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

        # Updated counter
        video_index_counter += 1

    # Closing the video
    cv2.destroyAllWindows()

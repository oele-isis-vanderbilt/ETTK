# Built-in Imports
from typing import Literal
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

PAPER_TOBII_REC_PATH = CWD / "data" / "recordings" / "tobii_paper_v6_rec5_multi_paper"
# PAPER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_v6_rec4'
# PAPER_TEMPLATE = CWD/'data'/'resources'/'paper_v6'/'test_paper_template_1.png'
PAPER_TEMPLATE = CWD / "data" / "resources" / "paper_v6"

# COMPUTER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_v1_rec1'
# COMPUTER_TEMPLATE = CWD/'data'/'resources'/'computer'/'computer_screenshot.png'

COMPUTER_TOBII_REC_PATH = CWD / "data" / "recordings" / "tobii_computer_v2_rec1"
# COMPUTER_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_computer_v3_rec1'
COMPUTER_TEMPLATE = (
    CWD / "data" / "resources" / "computer" / "computer_screenshot_large_text.png"
)

# VIDEO_START_INDEX = 1000
# VIDEO_START_INDEX = 2200
VIDEO_START_INDEX = 0

# TRIM_MARGIN_X = 80
# TRIM_MARGIN_Y_TOP = 100
# TRIM_MARGIN_Y_BOTTOM = 150

BLACK_MARGIN_SIZE = 50

FIX_RADIUS = 10
FIX_COLOR = (0, 0, 255)
FIX_THICKNESS = 3

assert PAPER_TOBII_REC_PATH.exists() and PAPER_TEMPLATE.exists()
assert COMPUTER_TOBII_REC_PATH.exists() and COMPUTER_TEMPLATE.exists()


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


@pytest.fixture()
def paper_rec_data():
    return get_rec_data(PAPER_TOBII_REC_PATH)


@pytest.fixture()
def computer_rec_data():
    return get_rec_data(COMPUTER_TOBII_REC_PATH)


def get_templates(templates_path, trim_type: Literal["paper", "computer"]):

    if templates_path.is_dir():
        templates_filepaths = list(templates_path.iterdir())
    else:
        templates_filepaths = [templates_path]

    # Load multiple templates inside a directory
    templates = []
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
def paper_tracker():
    return ettk.PlanarTracker()


@pytest.fixture
def computer_tracker():
    return ettk.PlanarTracker(use_aruco_markers=False)


def test_template_database():

    # Create the templates
    paper_templates = get_templates(PAPER_TEMPLATE, "paper")
    template_database = ettk.TemplateDatabase()

    # Add the templates
    for template in paper_templates:
        template_database.add(template)

    assert len(template_database) == len(paper_templates)


@pytest.mark.parametrize(
    "templates,rec_data,tracker,exp_type",
    [
        pytest.param(
            get_templates(COMPUTER_TEMPLATE, "computer"),
            lazy_fixture("computer_rec_data"),
            lazy_fixture("computer_tracker"),
            "computer",
            id="computer",
        ),
        pytest.param(
            get_templates(PAPER_TEMPLATE, "paper"),
            lazy_fixture("paper_rec_data"),
            lazy_fixture("paper_tracker"),
            "paper",
            id="paper",
        ),
    ],
)
def test_step_video_with_eye_tracking(templates, rec_data, tracker, exp_type):

    # Decompose data
    cap, gaze = rec_data

    # Register the templates
    templates_ids = tracker.register_templates(templates)

    # Load the video and get a single frame
    ret, frame = cap.read()
    h, w = frame.shape[:2]

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

            # Input frame
            # frame = imutils.resize(frame, width=1500)

            # Get the size of the video
            h, w, _ = frame.shape

            # Get fixation
            current_time = (VIDEO_START_INDEX + video_index_counter) * (1 / fps)
            try:
                raw_fix = (
                    gaze[gaze["timestamp"] > current_time]
                    .reset_index()
                    .iloc[0]["gaze2d"]
                )
            except IndexError:
                raw_fix = [0, 0]

            if isinstance(raw_fix, str):
                raw_fix = ast.literal_eval(raw_fix)

            fix = (int(raw_fix[0] * w), int(raw_fix[1] * h))

            # Draw eye-tracking into the original video frame
            draw_frame = cv2.circle(
                frame.copy(), fix, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS
            )

            # Apply homography
            result = tracker.step(frame)

            # Select the template that was detected
            if not isinstance(result["template_id"], type(None)):
                template = templates[templates_ids.index(result["template_id"])]
            else:
                template = np.zeros_like(draw_frame)

            # Draw template id
            if result["template_id"] is not None:
                draw_frame = ettk.utils.draw_text(
                    draw_frame,
                    str(templates_ids.index(result["template_id"])),
                    location=(0, 100),
                    color=(0, 255, 0),
                )
            else:
                draw_frame = ettk.utils.draw_text(
                    draw_frame, str(None), location=(0, 100), color=(0, 255, 0)
                )

            # Draw paper outline
            draw_frame = ettk.utils.draw_homography_outline(
                draw_frame, result["corners"], color=(0, 255, 0)
            )

            # Draw the tracked points
            draw_frame = ettk.utils.draw_pts(draw_frame, result["tracked_points"])
            draw_frame = ettk.utils.draw_text(
                draw_frame, f"{result['fps']:.2f}", location=(0, 50), color=(0, 0, 255)
            )

            # Apply homography to fixation and draw it on the page
            if type(result["M"]) != type(None):
                fix_pt = np.float32([[fix[0], fix[1]]]).reshape(-1, 1, 2)
                fix_dst = (
                    cv2.perspectiveTransform(fix_pt, np.linalg.inv(result["M"]))
                    .flatten()
                    .astype(np.int32)
                )
                draw_template = cv2.circle(
                    template.copy(), fix_dst, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS
                )
            else:
                draw_template = template.copy()

            # Combine frames
            vis_frame = ettk.utils.combine_frames(draw_template, draw_frame)
            cv2.imshow("output", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

        # Updated counter
        video_index_counter += 1

    # Closing the video
    cv2.destroyAllWindows()

import logging
import pathlib
import os

import pytest
import cv2
import numpy as np
import ettk

# CONSTANTS
CWD = pathlib.Path(os.path.abspath(__file__)).parent
PAGES_DIR = CWD / 'config' / 'pages'
DATA_DIR = CWD / 'data'
OUTPUT_DIR = CWD / 'output'

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
# VIDEO_TOBII_REC_PATH = (
#     CWD
#     / "data"
#     / "220473_chimerapy-2023_04_25_11_08_41-6228"
#     / "tg3"
#     / "20230425T161158Z"
# )
# VIDEO_START_INDEX = 0 # MONITOR
# VIDEO_START_INDEX = 17000 # SUFFRAGE

# VIDEO_TOBII_REC_PATH = (
#     CWD
#     / "data"
#     / '220322_chimerapy-2023_04_25_11_49_50-8942'
#     / 'tg3'
#     / '20230425T170919Z'
# )
VIDEO_START_INDEX = 0
# VIDEO_START_INDEX = 5000 # paper
# VIDEO_START_INDEX = 10000 # monitor
# VIDEO_START_INDEX = 37800 # MOOCA

VIDEO_TOBII_REC_PATH = (
    CWD
    / "data"
    / '220244_chimerapy-2023_04_13_12_55_01-7733'
    / 'tg3'
    / '20230413T151408Z'
)
# VIDEO_START_INDEX = 0
VIDEO_START_INDEX = 1000 # paper

# VIDEO_TOBII_REC_PATH = (
#     CWD
#     / "data"
#     / '220965_chimerapy-2023_04_25_12_52_17-1557'
#     / 'tg3'
#     / '20230425T180505Z'
# )
# VIDEO_START_INDEX = 0 # monitor
# VIDEO_START_INDEX = 17500 # paper
# VIDEO_START_INDEX = 32500 # MOOCA

# TRIM_MARGIN_X = 80
# TRIM_MARGIN_Y_TOP = 100
# TRIM_MARGIN_Y_BOTTOM = 150

BLACK_MARGIN_SIZE = 50

assert VIDEO_TOBII_REC_PATH.exists()


# Constants
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

disable_loggers = [
    "matplotlib",
    "chardet.charsetprober",
    "matplotlib.font_manager",
    "PIL.PngImagePlugin",
]


def pytest_configure():
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False


@pytest.fixture
def rec_data():
    return get_rec_data(VIDEO_TOBII_REC_PATH)


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
    # gaze = None
    return cap, gaze

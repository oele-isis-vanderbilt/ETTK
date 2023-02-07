import pathlib
import logging
import os

logger = logging.getLogger(__name__)

import pytest
import pdb

import torch
import numpy as np
import cv2

from ettk import SegTracker


CWD = pathlib.Path(os.path.abspath(__file__)).parent
TEST_TOBII_REC_PATH = CWD / "data" / "recordings" / "tobii_paper_v4_rec4"


@pytest.fixture
def cap():
    video_path = TEST_TOBII_REC_PATH / "scenevideo.mp4"
    assert video_path.exists()
    cap = cv2.VideoCapture(str(video_path), 0)
    return cap

@pytest.fixture
def tracker():
    return SegTracker(str(CWD/"model"/"yolov8s-seg.pt"))

def test_yolo_seg_on_single_frame(cap, tracker):

    # Get test image
    ret, frame = cap.read()

    # Test the model
    results = tracker.step(frame)
    logger.debug(results)

    # cv2.waitKey(0)
    # assert isinstance(output_img, np.ndarray)

def test_yolo_seg_on_sample_frame(tracker):

    img = cv2.imread(str(CWD/'data'/'samples'/'zidane.jpg'))

    results = tracker.step(img)

    cv2.imshow('output', img)
    cv2.waitKey(0)

def test_yolo_seg_on_video(cap, model):

    while True:
        ret, frame = cap.read()

        if ret:

            # Apply the model
            results = model(frame)
            results.show()
            # output = results.render()[0]
            # cv2.imshow("output", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

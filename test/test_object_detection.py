import pathlib
import logging
import os

logger = logging.getLogger(__name__)

import pytest
import pdb

import torch
import numpy as np
import cv2

CWD = pathlib.Path(os.path.abspath(__file__)).parent
TEST_TOBII_REC_PATH = CWD/'data'/'recordings'/'tobii_paper_rec4_v4'

@pytest.fixture
def cap():
    video_path = TEST_TOBII_REC_PATH/'scenevideo.mp4'
    assert video_path.exists()
    cap = cv2.VideoCapture(str(video_path), 0)
    return cap

@pytest.fixture
def model():
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

def test_yolo_on_single_frame(cap, model):
 
    # Get test image
    ret, frame = cap.read()

    # Test the model
    results = model(frame)
    pdb.set_trace()
    output_img = results.render()[0]
    assert isinstance(output_img, np.ndarray)

def test_yolo_on_video(cap, model):

    while(True):
        ret, frame = cap.read()
        
        if ret:

            # Apply the model
            results = model(frame)
            output = results.render()[0]
            cv2.imshow('output', output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


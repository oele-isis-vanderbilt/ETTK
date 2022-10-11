# Third-party Imports
import pytest
import cv2

# Internal Imports
import ettk

# Constants
TG3_IP_ADDR = "10.0.0.204"
TG3_NAME = "TG03B-080201035331"


@pytest.fixture
def tg3():
    return ettk.TG3(TG3_IP_ADDR, TG3_NAME)


def test_connecting_to_tg3(tg3):
    tg3.connect()


def test_showing_video_stream(tg3):
    tg3.connect()
    tg3.setup_stream()

    for i in range(400):
        ret, frame = tg3.get()
        cv2.imshow("tg3 scene camera", frame)
        cv2.waitKey(1)

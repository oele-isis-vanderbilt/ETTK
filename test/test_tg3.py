# Built-in Imports
import pdb
import logging

# Third-party Imports
import pytest
import cv2
import av

# Internal Imports
import ettk

logger = logging.getLogger("ettk")

# Constants
TG3_IP_ADDR = "10.0.0.204"
TG3_NAME = "TG03B-080201035331"


@pytest.fixture
def tg3():
    tg3_controller = ettk.TG3(TG3_IP_ADDR, TG3_NAME)
    yield tg3_controller
    tg3_controller.close()


@pytest.fixture
def rtsp_container():
    url = f"rtsp://{TG3_IP_ADDR}:8554/live/all"
    container = av.open(
        url,
        "r",
        options={
            "rtsp_transport": "tcp",
            "stimeout": "5000000",
            "max_delay": "5000000",
        },
    )
    return container


def test_connecting_to_tg3(tg3):
    tg3.connect()


def test_showing_video_stream(tg3):
    tg3.connect()
    tg3.setup_stream()

    for i in range(100):
        ret, frame = tg3.get()
        if ret:
            cv2.imshow("tg3 scene camera", frame)
            cv2.waitKey(1)


def test_pyav_scene_video(rtsp_container):

    # https://github.com/PyAV-Org/PyAV/issues/567
    # https://github.com/PyAV-Org/PyAV/discussions/860
    # https://stackoverflow.com/questions/67594208/pyav-libav-ffmpeg-what-happens-when-frame-from-live-source-are-not-processed

    # Pull the streams
    streams = rtsp_container.streams
    scene_video_stream = streams.video[0]

    video_demux = iter(rtsp_container.demux(scene_video_stream))

    counter = 0
    for packet in video_demux:
        for frame in packet.decode():
            safe_frame = frame.to_ndarray(format="rgb24")
            safe_frame = cv2.cvtColor(safe_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("tg3 scene camera", safe_frame)
            cv2.waitKey(1)

        if counter > 100:
            break

        counter += 1


def test_pyav_video_and_gaze(rtsp_container):

    # https://github.com/PyAV-Org/PyAV/issues/567
    # https://github.com/PyAV-Org/PyAV/discussions/860
    # https://stackoverflow.com/questions/67594208/pyav-libav-ffmpeg-what-happens-when-frame-from-live-source-are-not-processed

    # Pull the streams
    streams = rtsp_container.streams
    gaze_data_stream = streams.data[2]
    gaze_demux = iter(rtsp_container.demux(gaze_data_stream))

    counter = 0
    for packet in gaze_demux:
        for gaze in packet.decode():
            logger.debug(f"Gaze: {gaze}")

        if counter > 100:
            break

        counter += 1

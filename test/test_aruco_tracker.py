from dataclasses import asdict

import pytest

import cv2
import ettk

from .conftest import rec_data

def test_aruco_detection(rec_data):

    # Tracker
    aruco_tracker = ettk.ArucoTracker(aruco_omit=[5])

    cap, gaze = rec_data

    while True: 
        ret, frame = cap.read()

        # Processing
        a_results = aruco_tracker.step(frame)
        draw = ettk.utils.vis.draw_aruco_markers(frame, **asdict(a_results), with_ids=True)

        if ret:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("s"):
                cv2.imwrite(f"output_{video_index_counter}.png", output)

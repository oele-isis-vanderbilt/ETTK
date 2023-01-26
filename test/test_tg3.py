# Built-in Imports
import pathlib
import os
import time
import pdb

# Third-party Imports
import cv2

# Internal Imports
import ettk

# Constants
CWD = pathlib.Path(os.path.abspath(__file__)).parent
TG3_NAME = "TG03B-080201035331"


def test_tg3_main():

    tg3 = ettk.TG3Node(name="tg3", tg3_name=TG3_NAME, debug="step")

    tg3.main(30 * 2)

    tg3.shutdown()
    cv2.destroyAllWindows()


def test_tg3_stream():

    tg3 = ettk.TG3Node(name="tg3", tg3_name=TG3_NAME)
    tg3.config("", 0, CWD / "runs", [], [], None, False)

    tg3.start()
    time.sleep(30)

    tg3.shutdown()
    tg3.join()
    cv2.destroyAllWindows()


import ast
import cv2
import numpy as np
import ettk

def test_gaze_data(rec_data):
    
    # Get original video
    cap, gaze = rec_data
    w = 1920
    h = 1080

    for _, row in gaze.iterrows():
        bg = np.zeros((h,w, 3))

        # import pdb; pdb.set_trace()
        try:
            raw_fix = row["gaze2d"]
        except IndexError:
            raw_fix = [0, 0]

        if isinstance(raw_fix, str):
            raw_fix = ast.literal_eval(raw_fix)

        fix = (int(raw_fix[0] * w), int(raw_fix[1] * h))

        draw = ettk.utils.vis.draw_fix(fix, bg)
        cv2.imshow('frame', draw)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

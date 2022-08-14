
#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def anorm2(a):
    return (a*a).sum(-1)
def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            dst = cv.cornerHarris(frame_gray,2,3,0.04)
            #result is dilated for marking the corners, not important
            dst = cv.dilate(dst,None)
            # Threshold for an optimal value, it may vary depending on the image.
            vis[dst>0.01*dst.max()]=[0,0,255]

            cv.imshow('vis', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import logging
logger = logging.getLogger(__name__)

import pdb

MIN_MATCH_COUNT = 10

class Tracker():

    def __init__(self, alpha=0.1):
        self.M = None
        self.alpha = alpha

    def step(self, M, h, w):

        # First get the destinatin points
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = np.array(cv2.perspectiveTransform(pts,M), dtype=np.int32).reshape((4,2))
        top_left, bottom_left, bottom_right, top_right = dst.tolist()

        # Check if the destination points are valid
        # x verification
        if top_left[0] < top_right[0] and top_left[0] < bottom_right[0] and\
            bottom_left[0] < bottom_right[0] and bottom_left[0] < top_right[0] and\
            top_left[1] < bottom_left[1] and top_left[1] < bottom_right[1] and\
            top_right[1] < bottom_left[1] and top_right[1] < bottom_right[1]:

            # print("Valid")

            # Take average between matrix
            if type(self.M) != type(None):
                self.M = (1-self.alpha)*self.M + (self.alpha)*M
            else:
                self.M = M

        # Then compute the new points
        if type(self.M) != type(None):
            dst = np.array(cv2.perspectiveTransform(pts,self.M), dtype=np.int32)
        else:
            dst = None

        return dst

tracker = Tracker()

def perform_homography(feature_extractor, img1, img2):
    
    # find the keypoints and descriptors with SIFT
    kpts1, descs1 = feature_extractor.detectAndCompute(img1,None)
    kpts2, descs2 = feature_extractor.detectAndCompute(img2,None)
    
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # matches = flann.knnMatch(des1,des2,k=2)

    # match descriptors and sort them in the order of their distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key = lambda x:x.distance) 
    
    # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    # if len(good)<MIN_MATCH_COUNT:
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()

    #     h,w = img1.shape
    #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,M)

    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
    # else:
    #     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    #     matchesMask = None

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None,
    #                matchesMask = matchesMask, # draw only inliers
    #                flags = 2)

    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    # plt.imshow(img3, 'gray'),plt.show()

    # extract the matched keypoints
    src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

    # find homography matrix and do perspective transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    dst = tracker.step(M, *img1.shape[0:2])

    return tracker.M, dst, kpts1, kpts2, dmatches

def draw_homography_outline(img, dst):
    
    if type(dst) != type(None):
        # draw found regions
        return cv2.polylines(img, [dst], True, (0,0,255), 3, cv2.LINE_AA)
    else:
        return img

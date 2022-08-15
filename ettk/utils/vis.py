from typing import List
import pdb

# Third-party Imports
import numpy as np
import cv2

def ensure_rgb(img):
    
    # Convert all grey images to rgb images
    if len(img.shape) == 2:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        rgb_img = img
    else:
        raise RuntimeError(f"Unexpected number of channels: {img.shape}")

    return rgb_img

def combine_frames(img1, img2):

    # Ensure the input images have the same number of channels
    safe_img1 = ensure_rgb(img1)
    safe_img2 = ensure_rgb(img2)
    
    h1, w1 = safe_img1.shape[:2]
    h2, w2 = safe_img2.shape[:2]

    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

    #combine 2 images
    vis[:h1, :w1,:3] = safe_img1
    vis[:h2, w1:w1+w2,:3] = safe_img2

    return vis

def draw_text(img:np.ndarray, text:str, color:tuple=(255,0,0), location:tuple=(50,50)) -> np.ndarray:
    return cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def draw_homography_outline(img:np.ndarray, dst:np.ndarray, color:tuple=(255,0,0)) -> np.ndarray:

    # pdb.set_trace()
    if type(dst) != type(None):
        # draw found regions
        return cv2.polylines(img, [dst], True, color, 3, cv2.LINE_AA)
    else:
        return img

def draw_hough_lines(img:np.ndarray, lines:list, color:tuple=(255,0,0), thickness:int=3) -> np.ndarray:

    # Make copy to safely draw
    draw_img = img.copy()

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return draw_img

def draw_contours(img:np.ndarray, cnts:list, color:tuple=(0,255,0)) -> np.ndarray:
    
    # Make copy to safely draw
    draw_img = img.copy()

    # For each contour, draw it!
    for c in cnts:
        cv2.drawContours(draw_img,[c], 0, color, 3)

    return draw_img

def draw_rects(img:np.ndarray, rects:List[tuple]) -> np.ndarray:
    
    # Make copy to safely draw
    draw_img = img.copy()

    for rect in rects:
        x,y,w,h = rect
        cv2.rectangle(draw_img, (x,y), (x+w, y+h), (0,0,255), 2)

    return draw_img

def draw_pts(
        img:np.ndarray, 
        pts:np.ndarray, 
        color:tuple=(255,0,0), 
        radius:int=2
    ) -> np.ndarray:

    if type(pts) == type(None):
        return img
    elif len(pts.shape) == 3:
        pts = pts[:,0,:]
    
    # Make copy to safely draw
    draw_img = img.copy()

    for pt in pts.astype(np.int32):
        cv2.circle(draw_img, pt, 3, color, radius)

    return draw_img

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

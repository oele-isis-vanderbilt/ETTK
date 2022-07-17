# Built-in Imports
import os

# Third-party Imports
import numpy as np
import cv2

def load_calib(filepath):
    """
    Loads the calibration of the camera
    Parameters
    ----------
    filepath (str): The file path to the camera file

    Returns
    -------
    K (ndarray): Intrinsic parameters
    P (ndarray): Projection matrix
    """
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P

def load_poses(filepath):
    """
    Loads the GT poses

    Parameters
    ----------
    filepath (str): The file path to the poses file

    Returns
    -------
    poses (ndarray): The GT poses
    """
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def load_images(filepath):
    """
    Loads the images

    Parameters
    ----------
    filepath (str): The file path to image dir

    Returns
    -------
    images (list): grayscale images
    """
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

import numpy as np


def homo2cart(pts: np.ndarray) -> np.ndarray:
    return pts[:-1] / pts[-1]


def cart2homo(pts: np.ndarray) -> np.ndarray:
    return np.vstack((pts, np.ones(pts.shape[1])))


def project_points(pts: np.ndarray, rt: np.ndarray, k: np.ndarray):
    s1 = rt @ cart2homo(pts)
    s2 = k @ s1
    return homo2cart(s2).T

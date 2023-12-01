import pdb
import logging
from typing import List, Tuple, Optional

# Third-party Imports
import numpy as np
import cv2

logger = logging.getLogger('ettk')

# Constants
FIX_RADIUS = 10
FIX_COLOR = (0, 0, 255)
FIX_THICKNESS = 3

MATRIX_COEFFICIENTS = np.array(
    [
        [910.5968017578125, 0, 958.43426513671875],
        [0, 910.20758056640625, 511.6611328125],
        [0, 0, 1],
    ]
)
DISTORTION_COEFFICIENTS = np.array(
    [
        -0.055919282138347626,
        0.079781122505664825,
        -0.048538044095039368,
        -0.00014426070265471935,
        0.00044536130735650659,
    ]
)


def render(fix: Tuple[int, int], planar_result: "PlanarResult") -> np.ndarray:

    draw_frame = draw_fix(fix, planar_result.frame)

    a = planar_result.aruco
    draw_frame = draw_aruco_markers(
        draw_frame,
        corners=a.corners,
        ids=a.ids,
        rvec=a.rvec,
        tvec=a.tvec,
        with_ids=True,
    )

    m = planar_result.monitor
    draw_frame = draw_surface_corners(draw_frame, corners=m.corners)

    return draw_frame


def project_fix(self, fix: Tuple[int, int]):

    fix_pt = np.float32([[fix[0], fix[1]]]).reshape(-1, 1, 2)
    fix_dst = (
        cv2.perspectiveTransform(fix_pt, np.linalg.inv(self.M))
        .flatten()
        .astype(np.int32)
    )

    return fix_dst


def draw_fix(fix: Tuple[int, int], img: np.ndarray = None):

    # Draw eye-tracking into the original video frame
    fix = (int(fix[0]), int(fix[1]))
    draw_frame = cv2.circle(img.copy(), fix, FIX_RADIUS, FIX_COLOR, FIX_THICKNESS)

    return draw_frame


def draw_lines(img: np.ndarray, lines: List[tuple]) -> np.ndarray:

    if type(lines) != type(None):
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img

def draw_aruco_markers(
    img: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    with_ids: bool = True,
):

    cv2.aruco.drawDetectedMarkers(img, corners)  # Draw A square around the markers

    if np.all(ids is not None):  # If there are markers found by detector
        for i in range(0, len(ids)):  # Iterate in markers

            cv2.drawFrameAxes(
                img,
                MATRIX_COEFFICIENTS,
                DISTORTION_COEFFICIENTS,
                rvec[i],
                tvec[i],
                0.01,
            )  # Draw Axis

            # Draw Ids
            pt = tuple((int(corners[i][0][0][0]), int(corners[i][0][0][1])))
            cv2.putText(
                img,
                f"{ids[i]}",
                pt,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    return img

def draw_axis(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:

    cv2.drawFrameAxes(
        img,
        MATRIX_COEFFICIENTS,
        DISTORTION_COEFFICIENTS,
        rvec,
        tvec,
        0.01,
    )  # Draw Axis

    return img


def draw_surface_corners(img: np.ndarray, corners: np.ndarray):

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]

    # Draw paper outline
    for i in [(0, 1), (1, 2), (2, 3), (3, 0)]:

        s = corners[i[0]].squeeze()
        e = corners[i[1]].squeeze()
        s_point = (int(s[0]), int(s[1]))
        e_point = (int(e[0]), int(e[1]))
        try:
            cv2.line(img, s_point, e_point, colors[i[0]], 2)
        except:
            continue

    return img


def draw_tracked_points(img: np.ndarray = None):

    # Draw the tracked points
    draw_frame = draw_pts(img, tracked_points)
    draw_frame = draw_text(
        draw_frame, f"{(1/delay):.2f}", location=(0, 50), color=(0, 0, 255)
    )

    return draw_frame


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

    # create empty matrix
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    # combine 2 images
    vis[:h1, :w1, :3] = safe_img1
    vis[:h2, w1 : w1 + w2, :3] = safe_img2

    return vis


def draw_text(
    img: np.ndarray, text: str, color: tuple = (255, 0, 0), location: tuple = (50, 50)
) -> np.ndarray:
    return cv2.putText(
        img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
    )


def draw_homography_outline(
    img: np.ndarray, dst: np.ndarray, color: tuple = (255, 0, 0)
) -> np.ndarray:

    # pdb.set_trace()
    if type(dst) != type(None):
        # draw found regions
        return cv2.polylines(img, [dst], True, color, 3, cv2.LINE_AA)
    else:
        return img


def draw_hough_lines(
    img: np.ndarray, lines: list, color: tuple = (255, 0, 0), thickness: int = 3
) -> np.ndarray:

    # Make copy to safely draw
    draw_img = img.copy()

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return draw_img


def draw_contours(
    img: np.ndarray, cnts: list, color: tuple = (0, 255, 0)
) -> np.ndarray:

    # Make copy to safely draw
    draw_img = img.copy()

    # For each contour, draw it!
    for c in cnts:
        cv2.drawContours(draw_img, [c], 0, color, 3)

    return draw_img


def draw_rects(img: np.ndarray, rects: List[tuple]) -> np.ndarray:

    # Make copy to safely draw
    draw_img = img.copy()

    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return draw_img


def draw_pts(
    img: np.ndarray, pts: np.ndarray, color: tuple = (255, 0, 0), radius: int = 2
) -> np.ndarray:

    if type(pts) == type(None):
        return img
    elif len(pts.shape) == 3:
        pts = pts[:, 0, :]

    # Make copy to safely draw
    draw_img = img.copy()

    for pt in pts.astype(np.int32):
        cv2.circle(draw_img, pt, 3, color, radius)

    return draw_img

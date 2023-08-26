from dataclasses import dataclass, field
import logging

import numpy as np
import cv2

logger = logging.getLogger('ettk')


@dataclass
class HoughResult:
    lines: np.ndarray = field(default_factory=lambda: np.empty((0,1,2))) # (N,1,2)


class HoughRefiner:

    def __init__(self, threshold_angle=0.1, threshold_distance=50):

        # Parameters
        self.threshold_angle = threshold_angle
        self.threshold_distance = threshold_distance

        # Container
        self.lines = np.empty((0,1,2))
    
    def predict_hough_lines(self, frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur
        blurred = cv2.GaussianBlur(gray, (3,3), 0)

        # Detect edges
        edged = cv2.Canny(blurred, 50, 150)

        # Detect lines
        lines = cv2.HoughLines(edged, 1, np.pi / 180, 50, None, 0, 0)

        # Return
        return lines

    def process_frame(self, frame: np.ndarray):
        
        # Use hough lines to improve pose
        self.lines = self.predict_hough_lines(frame)

    def from_corners_to_anchors(self, corners: np.ndarray):
        
        # Calculate anchor lines from corner coordinates
        anchor_lines = []
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]  # Next corner in sequence

            # Calculate the direction of the normal (perpendicular) to the line
            dx = y2 - y1
            dy = -(x2 - x1)

            # Calculate theta using the direction of the normal
            theta = np.arctan2(dy, dx)

            # Calculate rho using the formula: rho = x*cos(theta) + y*sin(theta)
            rho = x1 * np.cos(theta) + y1 * np.sin(theta)
            
            anchor_lines.append((rho, theta))
        
        return anchor_lines

    def rho_theta_to_slope_intercept(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        
        # Handle nearly vertical lines
        if np.isclose(b, 0):
            m = float('inf')
            b = rho / a
        else:
            m = -a / b
            b = rho / b
        
        return m, b

    def refine(self, surface_entry: 'SurfaceEntry') -> 'SurfaceEntry':
    
        if type(self.lines) == type(None):
            return surface_entry

        # Detect which lines are close to the surface outline
        anchor_lines = np.array(self.from_corners_to_anchors(surface_entry.corners.squeeze()))
        # anchor_lines = np.expand_dims(np.stack(anchor_lines), axis=1)
        # surface_entry.lines = anchor_lines

        # Ensure self.lines and anchor_lines are numpy arrays
        lines_np = np.array(self.lines).squeeze()  # Assuming shape (N,1,2) -> (N,2)
        anchor_lines_np = np.array(anchor_lines)   # Assuming shape (M,2)
        
        # Convert anchor lines to slope-intercept form
        anchor_slopes_intercepts = np.array([self.rho_theta_to_slope_intercept(rho, theta) for rho, theta in anchor_lines_np])

        # Convert detected lines to slope-intercept form
        detected_slopes_intercepts = np.array([self.rho_theta_to_slope_intercept(rho, theta) for rho, theta in lines_np])

        # Calculate slope and intercept differences using broadcasting
        m_diffs = np.abs(detected_slopes_intercepts[:, 0:1] - anchor_slopes_intercepts[:, 0])  # Shape: (N, M)
        b_diffs = np.abs(detected_slopes_intercepts[:, 1:2] - anchor_slopes_intercepts[:, 1])  # Shape: (N, M)

        # Set high threshold for slope to handle nearly vertical lines
        m_conditions = np.where(m_diffs != float('inf'), m_diffs < self.threshold_angle, True)
        b_conditions = b_diffs < self.threshold_distance

        # Combine conditions
        combined_conditions = np.logical_and(m_conditions, b_conditions)

        # Check for each line if there's any anchor line that is similar
        any_similarities = np.any(combined_conditions, axis=1)

        # Filter the lines
        filtered_lines = lines_np[any_similarities]

        # Update surface entry
        if filtered_lines.any():
            surface_entry.lines = np.expand_dims(filtered_lines, axis=1)

        return surface_entry

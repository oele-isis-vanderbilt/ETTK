# References:
# https://github.com/uoip/monoVO-python/blob/master/visual_odometry.py
# https://github.com/niconielsen32/VisualSLAM

# Built-in Imports
from typing import Tuple
import os

# Third-party Imports
import numpy as np
import cv2

class VisualOdometry():
    def __init__(self, K, P, initial_pose=np.eye(4)):
        """
        Create Visual Odometry Algorithm object for tracking camera pose.

        Args:
            K (np.ndarray): Intrinsic parameters
            P (np.ndarray): Project matrix
        """
        # Saving input parameters
        self.K, self.P = K, P

        # Create feature extractor and matcher
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, 
            table_number=6, 
            key_size=12, 
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        # Initialization values
        self.first_step = True
        self.previous_image = np.array([])
        self.current_pose = initial_pose

    @staticmethod
    def _form_transf(R:np.ndarray, t:np.ndarray) -> np.ndarray:
        """Makes a transformation matrix from the given rotation matrix and translation vector.

        Args:
            R (ndarray): The rotation matrix
            t (list): The translation vector

        Returns:
            T (ndarray): The transformation matrix

        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def get_matches(
            self, 
            previous_image:np.ndarray, 
            image: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and compute keypoints and descriptors between images.

        Args:
            previous_image (np.ndarray): image from previous step.
            image (np.ndarray): new image.

        Returns:
            q1 (ndarray): The keypoints matches position in i-1'th image.
            q2 (ndarray): The keypoints matches position in i'th image.

        """
        # Get keypoints
        keypoints1, descriptors1 = self.orb.detectAndCompute(previous_image, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(image, None)

        # Get the matches
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
       
        # Determine good matches
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        # Convert the matches to NumPy arrays
        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])

        return q1, q2

    def get_pose(self, q1, q2):
        """Calculates the transformation matrix.

        Args:
            q1 (ndarray): The good keypoints matches position in i-1'th image
            q2 (ndarray): The good keypoints matches position in i'th image

        Returns:
            ndarray: The transformation matrix

        """
        essential, mask = cv2.findEssentialMat(q1, q2, self.K)

        R, t = self.decomp_essential_mat(essential, q1, q2)

        return self._form_transf(R,t)

    def decomp_essential_mat(self, E, q1, q2):
        """Decompose the Essential matrix.

        Args:
            E (ndarray): Essential matrix
            q1 (ndarray): The good keypoints matches position in i-1'th image
            q2 (ndarray): The good keypoints matches position in i'th image

        Returns:
            right_pair (list): Contains the rotation matrix and translation vector

        """
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)
    
    def process_image(self, image:np.ndarray) -> np.ndarray:
        """Process a monocular image.

        Args:
            image (np.ndarray): 

        Returns:
            np.ndarray: current pose within frame

        """
        # If initial step, just skip and setup values
        if self.first_step:
            self.first_step = False
        else:
            # Get keypoints
            q1, q2 = self.get_matches(self.previous_image, image)

            # Find the transformation between keypoints
            transf = self.get_pose(q1, q2)

            # Then compute new pose
            self.current_pose = np.matmul(self.current_pose, np.linalg.inv(transf))

        # Update previous image
        self.previous_image = image

        # Return the pose
        return self.current_pose

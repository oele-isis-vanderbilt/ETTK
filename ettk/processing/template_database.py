import numpy as np
import cv2

from .. import utils

import pdb


class TemplateDatabase:
    def __init__(
        self,
        feature_extractor=cv2.AKAZE_create(),
        aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        aruco_params=cv2.aruco.DetectorParameters(),
        use_aruco_markers=False,
    ):

        self.use_aruco_markers = use_aruco_markers
        self.feature_extractor = feature_extractor
        self._aruco_dict = aruco_dict
        self._aruco_params = aruco_params
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )
        self.id_counter = 0

        self.data = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, name: str):
        return self.data[name]

    def __iter__(self):
        self._data_iter = iter(self.data)
        return self._data_iter

    def __next__(self):
        return next(self._data_iter)

    def add(self, template: np.ndarray):

        # Compute template's id
        template_hash = utils.dhash(template)

        # Check if the template has been added before
        if template_hash in self.data:
            return template_hash, False

        # Compute additional template information
        kpts, descs = self.feature_extractor.detectAndCompute(template, None)
        h, w = template.shape[:2]
        template_corners = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)

        # Store template into database
        self.data[template_hash] = {
            "id": self.id_counter,
            "template": template,
            "kpts": kpts,
            "descs": descs,
            "template_corners": template_corners,
        }
        self.id_counter += 1

        # Add aruco if requested
        if self.use_aruco_markers:
            corners, ids, _ = self._aruco_detector.detectMarkers(template)
            self.data[template_hash].update({"aruco": {"corners": corners, "ids": ids}})

        return template_hash, True

    def remove(self, hash: int):
        if hash in self.data:
            del self.data[hash]
        else:
            raise IndexError

    def clear(self):
        del self.data
        self.data = {}

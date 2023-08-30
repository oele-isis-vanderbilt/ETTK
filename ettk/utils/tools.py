from typing import Tuple, Optional

import cv2
import numpy as np

from ..types import PlanarResult, FixInSurfaceResult


def dhash(img, hash_size=32):

    # resize the input img, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(img, (hash_size + 1, hash_size))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # convert the difference img to a hash
    return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])


def surface_map_points(planar_results: PlanarResult, fix: Tuple[int, int]) -> Optional[FixInSurfaceResult]:

    if not planar_results.surfaces:
        return None

    for surface in planar_results.surfaces.values():

        # First, check if the fixation is within the surface
        if not cv2.pointPolygonTest(surface.corners.astype(np.float32), fix, False) >= 0:
            continue
        
        # fix = surface.corners[2].squeeze()
        # import pdb; pdb.set_trace()
        
        # Compute the homography matrix
        w, h = surface.config.width, surface.config.height
        template_pts = np.array([
            [0, 0], 
            [w, 0], 
            [w, h], 
            [0, h]
        ]).astype(np.float32)
        pts = surface.corners
        M, _ = cv2.findHomography(template_pts, pts, cv2.RANSAC, 5.0)

        # For each surface, we need to get the homography matrix
        fix_pt = np.float32([[fix[0], fix[1]]]).reshape(-1, 1, 2)
        fix_dst = (
            cv2.perspectiveTransform(fix_pt, np.linalg.inv(M))
            .flatten()
            .astype(np.int32)
        )

        # Create container
        fix_result = FixInSurfaceResult(
            surface_id=surface.id,
            pt=fix_dst
        )

        return fix_result

    return None

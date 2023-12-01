from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np

########################################################################
## Aruco
########################################################################

@dataclass
class ArucoResult:
    corners: np.ndarray = field(default_factory=lambda: np.empty((0,4))) # (M,4,2)
    ids: np.ndarray = field(default_factory=lambda: np.empty((0,1))) # (N,1)
    rvec: np.ndarray = field(default_factory=lambda: np.empty((0,3,1))) # (N,3,1)
    tvec: np.ndarray = field(default_factory=lambda: np.empty((0,3,1))) # (N,3,1)


@dataclass
class ArucoEntry:
    id: int
    rvec: np.ndarray # (3,1)
    tvec: np.ndarray # (3,1)
    corners: np.ndarray # (4,2)
    counts: int = 0


########################################################################
## Homography
########################################################################

@dataclass
class TemplateEntry:
    name: str
    template: np.ndarray
    kp: np.ndarray
    des: np.ndarray


@dataclass
class HomographyConfig:
    min_matches: int = 10
    min_inliers: int = 10
    ransac_threshold: float = 5.0
    ransac_max_trials: int = 1000
    aspect_ratio_threshold: float = 0.3
    angle_threshold: float = 20.0


@dataclass
class HomographyResult:
    name: str
    H: np.ndarray
    corners: np.ndarray # (4,2)
    success: bool
    rvec: np.ndarray
    tvec: np.ndarray
    size: Tuple[int, int]

########################################################################
## Surface Config
########################################################################

@dataclass
class ArucoConfig:
    id: int
    offset_rvec: np.ndarray
    offset_tvec: np.ndarray


@dataclass
class SurfaceConfig:
    id: str
    aruco_config: Dict[int, ArucoConfig]
    height: float
    width: float
    scale: Tuple[float] = (1.0, 1.0)
    template: Optional[np.ndarray] = None # (H,W,3)

########################################################################
## Planar
########################################################################

@dataclass
class Hypothesis: # by an aruco
    id: int
    rvec: np.ndarray # (3,1)
    tvec: np.ndarray # (3,1)


@dataclass
class SurfaceEntry:
    id: str
    rvec: np.ndarray # (3,1)
    tvec: np.ndarray # (3,1)
    corners: np.ndarray # (4,2)
    uncertainty: float
    config: SurfaceConfig
    hypotheses: List[Hypothesis] = field(default_factory=list)
    lines: np.ndarray = field(default_factory=lambda:np.empty((0,1,2))) # (N,1,2)
    homography: Optional[HomographyResult] = None


@dataclass
class PlanarResult:
    aruco: ArucoResult = field(default_factory=ArucoResult)
    surfaces: Dict[str, SurfaceEntry] = field(default_factory=dict)


@dataclass
class WeightConfig:
    aruco: float = 0.2
    surface: float = 0.8
    homo: float = 0.95


@dataclass
class FixInSurfaceResult:
    surface_id: str
    pt: np.ndarray # (2,)
    rel_pt: np.ndarray # (2,)
    uncertainty: float


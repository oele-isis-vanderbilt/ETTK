from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np

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

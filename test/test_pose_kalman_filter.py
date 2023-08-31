import logging

import numpy as np
from ettk.processing.filters.pose_kalman_filter import PoseKalmanFilter

logger = logging.getLogger('ettk')

def test_kalman_filter():
    
    filter = PoseKalmanFilter()
    r = np.array([0, 0, 0])
    t = np.array([0, 0, 0])

    # Initial
    filter.process(r, t)

    # Same pose, no uncertainty
    n_r, n_t, uncertainty = filter.process(r, t)
    logger.info(f'Uncertainty: {uncertainty}')
    assert np.allclose(uncertainty, 0)
    
    # Now let's pass a slightly different pose
    n_r, n_t, uncertainty = filter.process(r + 0.1, t + 0.1)
    logger.info(f'Uncertainty: {uncertainty}')
    assert uncertainty > 0 and uncertainty < 0.5

    # Now let's pass a very different pose
    n_r, n_t, uncertainty = filter.process(r + 1, t + 1)
    logger.info(f'Uncertainty: {uncertainty}')
    assert uncertainty > 0.25 and uncertainty < 0.75
    
    # Extreme case
    n_r, n_t, uncertainty = filter.process(r + 10, t + 10)
    logger.info(f'Uncertainty: {uncertainty}')
    assert uncertainty > 0.75 and uncertainty < 1

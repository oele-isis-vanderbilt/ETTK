# References:
# https://makersportal.com/blog/2019/11/11/raspberry-pi-python-accelerometer-gyroscope-magnetometer
# https://stackoverflow.com/questions/47210512/using-pykalman-on-raw-acceleration-data-to-calculate-position
# https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
# https://pykalman.github.io
# https://ahrs.readthedocs.io/en/latest/filters/ekf.html
# https://thepoorengineer.com/en/ekf-impl/

# Third-party
from ahrs.filters import EKF
import numpy as np

class InertialOdometry():
    
    def __init__(self):
        ...

    def update_imu(gyr:np.ndarray, acc:np.ndarray, mag:np.ndarray, dt:float):
        ...


import os
import sys

# Appending file's directory to the PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Providing access to package API
from .visual_odometry import VisualOdometry
from . import utils

import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}

# Setup the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# from .tg3 import TG3Node
from . import utils
from .processing import (
    ArucoTracker, 
    PlanarTracker, 
    SurfaceConfig, 
    ArucoConfig,
    HomographyRefiner,
    HomographyConfig,
)

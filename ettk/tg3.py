# Built-in Imports
import time
import sys
import json
import socket
import logging
import pdb
import http.client

# Third-party Imports
import cv2
import netifaces

logger = logging.getLogger("ettk")


class TG3:
    def __init__(self, ip_addr: str, device_name: str):
        self.ip_addr = ip_addr
        self.device_name = device_name

    def connect(self):

        # Make the connect
        self.conn = http.client.HTTPConnection(self.ip_addr)

        # Check that the connection is valid
        self.conn.request("GET", "/rest/system.recording-unit-serial")
        res = self.conn.getresponse()

        obtain_device_name = res.read().decode().replace('"', "")
        logger.debug(f"Connected to: {obtain_device_name}")
        assert obtain_device_name == self.device_name, "Incorrect device"

    def setup_stream(self):
        rstp_path = f"rtsp://{self.ip_addr}:8554/live/all"
        logger.debug(f"Setup stream to {rstp_path}")
        self.vcap = cv2.VideoCapture(rstp_path)
        logger.debug("Stream setup complete")

    def get(self):
        return self.vcap.read()

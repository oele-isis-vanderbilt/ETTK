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

logger = logging.getLogger("ettk")


class TG3:
    def __init__(self, ip_addr: str, device_name: str):

        self.ip_addr = ip_addr
        self.device_name = device_name
        self.vcap = None
        self.connected = False

    def connect(self):

        # Make the connect
        self.conn = http.client.HTTPConnection(self.ip_addr)

        # Check that the connection is valid
        self.conn.request("GET", "/rest/system.recording-unit-serial")
        res = self.conn.getresponse()

        obtain_device_name = res.read().decode().replace('"', "")
        logger.debug(f"Connected to: {obtain_device_name}")
        assert obtain_device_name == self.device_name, "Incorrect device"
        self.connected = True

    def setup_stream(self):
        rstp_path = f"rtsp://{self.ip_addr}:8554/live/all"
        # rstp_path = f"rtspsrc location=rtsp://{self.ip_addr}:8554/live/all payload=96"
        # rstp_path = f"rtspsrc location=rtsp://{self.ip_addr}:8554/live/all latency=300 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        # rstp_path = f"rtspsrc location=rtsp://{self.ip_addr}:8554/live/all latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink"
        logger.debug(f"Setup stream to {rstp_path}")
        self.vcap = cv2.VideoCapture(rstp_path)
        # self.vcap = cv2.VideoCapture(rstp_path, cv2.CAP_GSTREAMER)
        logger.debug("Stream setup complete")

    def get(self):
        return self.vcap.read()

    def close(self):

        if isinstance(self.vcap, cv2.VideoCapture):
            self.vcap.release()

        if self.connected:
            self.conn.close()

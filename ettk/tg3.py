# Built-in Imports
from typing import Optional, Literal
import asyncio
import time
import os
import pdb

# Third-party Imports
import chimerapy as cp
import g3pylib as g3
from g3pylib import connect_to_glasses
import cv2
import imutils


class TG3Node(cp.Node):
    def __init__(
        self,
        name: str,
        tg3_name: str,
        debug: Optional[Literal["step", "stream"]] = None,
    ):
        super().__init__(name, debug)
        self.tg3_name = tg3_name

    async def async_main(self, max_steps: Optional[int] = None):

        # Having step counter
        i = 0

        async with connect_to_glasses.with_hostname(
            self.tg3_name, using_zeroconf=True, using_ip=True
        ) as tg3:
            async with tg3.stream_rtsp(scene_camera=True, gaze=True) as streams:
                async with streams.gaze.decode() as gaze_stream, streams.scene_camera.decode() as scene_stream:
                    while self.running.value:
                        frame, frame_timestamp = await scene_stream.get()
                        gaze, gaze_timestamp = await gaze_stream.get()
                        while gaze_timestamp is None or frame_timestamp is None:
                            if frame_timestamp is None:
                                frame, frame_timestamp = await scene_stream.get()
                            if gaze_timestamp is None:
                                gaze, gaze_timestamp = await gaze_stream.get()
                        while gaze_timestamp < frame_timestamp:
                            gaze, gaze_timestamp = await gaze_stream.get()
                            while gaze_timestamp is None:
                                gaze, gaze_timestamp = await gaze_stream.get()

                        # self.logger.info(f"Frame: {frame_timestamp}, Gaze: {gaze_timestamp}")
                        img = frame.to_ndarray(format="bgr24")

                        if "gaze2d" in gaze:
                            gaze2d = gaze["gaze2d"]
                            h, w = img.shape[:2]
                            fix = (int(gaze2d[0] * w), int(gaze2d[1] * h))
                            img = cv2.circle(img.copy(), fix, 10, (0, 0, 255), 3)

                        i += 1
                        # if hasattr(self, 'out_queue'):
                        #     self.out_queue.put({"step_id": i, "data": img})

                        cv2.imshow("Video", imutils.resize(img, width=400))  # type: ignore
                        cv2.waitKey(1)  # type: ignore

                        # Save data
                        # if hasattr(self, 'save_queue'):
                        #     self.save_video(name="scene", data=img, fps=20)
                        #     if gaze:
                        #         # Add timestamp
                        #         gaze['timestamp'] = gaze_timestamp
                        #         self.save_tabular(name="gaze", data=gaze)

                        # # Determine if we need to break
                        if max_steps and i > max_steps:
                            break

    def main(self, max_steps: Optional[int] = None):
        asyncio.run(self.async_main(max_steps))

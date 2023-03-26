# Built-in Imports
from typing import Optional, Literal
import asyncio
import time
import os
import pdb

# Third-party Imports
import chimerapy as cp
import cv2
import imutils


class TG3Node(cp.Node):
    def __init__(
        self,
        name: str,
        tg3_name: str,
        url: str,
        debug: Optional[Literal["step", "stream"]] = None,
    ):
        super().__init__(name, debug)
        self.tg3_name = tg3_name
        self.url = url

    async def async_main(
        self, max_steps: Optional[int] = None, debug: Optional[bool] = None
    ):
        import g3pylib as g3
        from g3pylib import connect_to_glasses

        # Having step counter
        i = 0

        # url = "10.0.0.178"

        # async with connect_to_glasses.with_hostname(
        #     self.tg3_name, using_zeroconf=True, using_ip=True
        # ) as tg3:
        async with connect_to_glasses.with_url(
            f"ws://{self.url}/websocket",
            f"rtsp://{self.url}:8554/live/all",
            f"http://{self.url}:80",
        ) as tg3:

            await tg3.recorder.start()

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

                        frame = frame.to_ndarray(format="bgr24")
                        i += 1

                        # Save data
                        self.save_video(name="scene", data=frame, fps=20)
                        if "gaze2d" in gaze:
                            self.save_tabular(name="gaze", data=gaze)
                            # If given gaze data
                            gaze2d = gaze["gaze2d"]

                            # Convert rational (x,y) to pixel location (x,y)
                            h, w = frame.shape[:2]
                            fix = (int(gaze2d[0] * w), int(gaze2d[1] * h))

                            # Draw gaze
                            frame = cv2.circle(frame, fix, 10, (0, 0, 255), 3)

                        # Visualize data for debugging
                        # cv2.imshow("scene", imutils.resize(frame, width=900))
                        # cv2.waitKey(1)

                        # Create data chunk
                        data_chunk = cp.DataChunk()
                        data_chunk.add("scene", frame, "image")
                        data_chunk.add("gaze", gaze)
                        if self.publisher:
                            self.publisher.publish(data_chunk)

                        # # Determine if we need to break
                        if max_steps and i > max_steps:
                            break

            await tg3.recorder.stop()
            self.logger.debug(tg3.recordings.children)

    def main(self, max_steps: Optional[int] = None, debug: Optional[bool] = None):
        asyncio.run(self.async_main(max_steps, debug))

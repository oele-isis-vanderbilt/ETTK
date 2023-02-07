import pathlib
from typing import Union
import logging

from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)

# Reference: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
COCO_ORIGINAL_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# COCO_CLASSES = [0,39,59,62,63,64,65,66,67,73]
# COCO_CLASSES_RENAMES = ['person', 'bottle', 'bed', 'monitor', 'laptop', 'mouse', 'thermometer', 'keyboard', 'phone/IV pump', 'paper']

class SegTracker:

    def __init__(self, weight_path: Union[str, pathlib.Path]):

        if isinstance(weight_path, str):
            weight_path = pathlib.Path(weight_path)

        self.weight_path = weight_path

        # Create the YOLOv5 model
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # self.model = yolov5.load(str(self.weight_path))
        self.model = YOLO(str(self.weight_path))
        logger.debug(self.model.__dict__)

    def step(self, img: np.ndarray):
            
        # Apply the model
        results = self.model(img)

        return results

    def render(self, results):

        for r in results:

            # Conver to numpy
            r = r.cpu().numpy()
            boxes = r.boxes.xyxy
            for box in boxes:
                logger.debug(box[0])
                box = box.astype(int)
                img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,0,0), 5)

from typing import Tuple, Union
import copy

import numpy as np
import torch

# Reference: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
COCO_ORIGINAL_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# COCO_CLASSES = [0,39,59,62,63,64,65,66,67,73]
# COCO_CLASSES_RENAMES = ['person', 'bottle', 'bed', 'monitor', 'laptop', 'mouse', 'thermometer', 'keyboard', 'phone/IV pump', 'paper']

class ObjectTracker:

    def __init__(self, class_map):

        # Select the classes we are interested
        self.interested_classes_idx = []
        self.renamed_classes = copy.deepcopy(COCO_ORIGINAL_NAMES)

        for o_class, n_class in class_map.items():

            # Obtain the index of the object in the original list of classes
            class_index = COCO_ORIGINAL_NAMES.index(o_class)
            self.interested_classes_idx.append(class_index)

            # Updating the name
            self.renamed_classes[class_index] = n_class

        # Create the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes = self.interested_classes_idx

    def step(self, img: np.ndarray, fix: Tuple[Union[int,float], Union[int,float]]):
            
        # Apply the model
        results = self.model(img)

        # Change the name 
        results.names = self.renamed_classes

        # Find the object that the fixation matches to
        df = results.pandas().xyxy[0]
        hit_boxes = df[(df['xmin'] < fix[0]) & (df['xmax'] > fix[0]) & (df['ymin'] < fix[1]) & (df['ymax'] > fix[1])]

        # Find the AOI
        if len(hit_boxes) == 0:
            aoi = None
        elif len(hit_boxes) == 1:
            aoi = hit_boxes['name'].tolist()[0]
        else:
            # Compute centroid
            hit_boxes['x_center'] = (hit_boxes['xmin'] + hit_boxes['xmax'])/2
            hit_boxes['y_center'] = (hit_boxes['ymin'] + hit_boxes['ymax'])/2

            # Compute the euclidean distance
            hit_boxes['e_distance'] = np.sqrt((fix[0] - hit_boxes['x_center'])** 2 + (fix[1] - hit_boxes['y_center'])**2)
            closes_box = hit_boxes[hit_boxes['e_distance'] == hit_boxes['e_distance'].min()]
            aoi = closes_box['name'].tolist()[0]

        # Get the rendered image
        render = results.render()[0]

        return {'aoi': aoi, 'render': render}

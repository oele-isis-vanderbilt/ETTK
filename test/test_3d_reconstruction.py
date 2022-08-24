import os
import pathlib

import pytest
import yaml
import cv2

# import torch
import numpy as np

# import pyslam
# import neuralrecon as nr
import ettk

# get the file's location
CWD = pathlib.Path(os.path.abspath(__file__)).parent
TEST_VIDEO = CWD / "data" / "tobii_rec1_v3" / "scenevideo.mp4"
NR_CONFIG_FILE = CWD / "config" / "demo.yaml"
NR_MODEL_WEIGHTS = CWD / "config" / "model_000047.ckpt"

# assert TEST_VIDEO.exists() and NR_CONFIG_FILE.exists() and NR_MODEL_WEIGHTS.exists()

# @pytest.fixture
# def cam():

#     # Load camera configurations
#     with open(str(CWD/'config'/'tobii_camera.yaml'), 'r') as f:
#         config = yaml.safe_load(f)

#     # Create Camera object
#     cam = pyslam.PinholeCamera(**config['Camera'])

#     return cam

# @pytest.fixture
# def slam(cam):

#     # Selected parameters for tracker
#     num_features=2000
#     tracker_type = pyslam.features.FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn
#     tracker_config = pyslam.features.FeatureTrackerConfigs.TEST
#     tracker_config['num_features'] = num_features
#     tracker_config['tracker_type'] = tracker_type
#     feature_tracker = pyslam.features.feature_tracker_factory(**tracker_config)

#     # create SLAM object
#     slam = pyslam.Slam(cam, feature_tracker)

#     yield slam
#     slam.quit()

# @pytest.fixture
# def nr_model():

#     # Apply configurations
#     cfg = nr.DF_CF.clone()
#     cfg.merge_from_file(NR_CONFIG_FILE)

#     # Creating model
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#     model = nr.NeuralRecon(cfg).to(device).eval()
#     model = torch.nn.DataParallel(model, device_ids=[0])

#     # use the latest checkpoint file
#     state_dict = torch.load(NR_MODEL_WEIGHTS, map_location=device)
#     model.load_state_dict(state_dict['model'], strict=False)
#     epoch_idx = state_dict['epoch']

#     return model


@pytest.mark.skip(reason="dependency issues.")
def test_tobii_data(cam, slam, nr_model):

    # Load the video that we are interested in
    video = cv2.VideoCapture(str(TEST_VIDEO))
    fps = video.get(cv2.CAP_PROP_FPS)
    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the transforms
    transform = [
        nr.datasets.transforms.ResizeImage((640, 480)),
        nr.datasets.transforms.ToTensor(),
        nr.datasets.transforms.RandomTransformSpace(
            nr.DF_CF.MODEL.N_VOX,
            nr.DF_CF.MODEL.VOXEL_SIZE,
            random_rotation=False,
            random_translation=False,
            paddingXY=0,
            paddingZ=0,
            max_epoch=nr.DF_CF.TRAIN.EPOCHS,
        ),
        nr.datasets.transforms.IntrinsicsPoseToProjection(nr.DF_CF.TEST.N_VIEWS, 4),
    ]
    transforms = nr.datasets.transforms.Compose(transform)

    # The mesh that contains the scene
    save_mesh_scene = nr.utils.SaveScene(nr.DF_CF)

    # Iterate
    for img_id in range(nb_frames):

        # Get the frame
        ret, img = video.read()

        # Compute the timestamp
        timestamp = img_id * (1 / fps)

        # Get the pose
        slam.track(img, img_id, timestamp)  # main SLAM function
        pose = slam.tracking.predicted_pose

        # Feed the pose, intrinsics, and image to NeuralRecon
        items = {
            "imgs": torch.from_numpy(img).unsqueeze(0),
            "intrinsics": torch.from_numpy(cam.K).unsqueeze(0),
            "extrinsics": torch.from_numpy(pose).unsqueeze(0),
            "scene": "TEST",  # Name of the scene
            "fragment": str(img_id),
            "epoch": [0],
            "vol_origin": torch.from_numpy(np.array([0, 0, 0])),
        }
        # sample = transforms(items)

        save_scene = img_id >= nb_frames - 1

        nr_model.forward(sample, save_scene)

        # Break condition
        if not ret:
            break

        # Updated counter
        img_id += 1

        break

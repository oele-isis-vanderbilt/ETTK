# Built-in library
import os
import pathlib

# Third-party library
import numpy as np
import pytest
import tqdm

# Internal Library
import et3d

def test_kitti_dataset():

    # Loading data
    data_dir = 'tests/data/KITTI_sequence_2'  # Try KITTI_sequence_2 too
    K, P = et3d.utils.load_calib(os.path.join(data_dir, 'calib.txt'))
    gt_poses = et3d.utils.load_poses(os.path.join(data_dir, 'poses.txt'))
    images = et3d.utils.load_images(os.path.join(data_dir, 'image_l'))

    # Creating visual odometry object
    vo = et3d.VisualOdometry(K, P)

    # play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm.tqdm(gt_poses, unit="pose")):

        # Get the image
        image = images[i]

        # Process the image
        vo.process_image(image)

        print ("\nGround truth pose:\n" + str(gt_pose))
        print ("\n Current pose:\n" + str(cur_pose))
        print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

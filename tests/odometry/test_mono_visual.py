# Built-in library
import os
import pathlib

# Third-party library
import numpy as np
import pytest
import tqdm
import matplotlib.pyplot as plt

# Internal Library
import et3d

def test_kitti_dataset():

    # Loading data
    data_dir = 'tests/data/KITTI_sequence_1'  # Try KITTI_sequence_2 too
    K, P = et3d.utils.load_calib(os.path.join(data_dir, 'calib.txt'))
    gt_poses = et3d.utils.load_poses(os.path.join(data_dir, 'poses.txt'))
    images = et3d.utils.load_images(os.path.join(data_dir, 'image_l'))

    # Creating visual odometry object
    vo = et3d.VisualOdometry(K, P, gt_poses[0])

    # Storing data
    gt_path = {'x':[], 'y': [], 'z': []}
    es_path = {'x':[], 'y': [], 'z': []} # estimated path

    # Iterating over the video
    for i, gt_pose in enumerate(tqdm.tqdm(gt_poses, unit="pose")):

        # Get the image
        image = images[i]

        # Process the image
        current_pose = vo.process_image(image)

        print("\nGround truth pose:\n" + str(gt_pose))
        print("\n Current pose:\n" + str(current_pose))
        print("The current pose used x,y: \n" + str(current_pose[0,3]) + "   " + str(current_pose[2,3]) )
        gt_path['x'].append(gt_pose[0,3]); gt_path['z'].append(gt_pose[1,3]); gt_path['y'].append(gt_pose[2,3])
        es_path['x'].append(current_pose[0,3]); es_path['z'].append(current_pose[1,3]); es_path['y'].append(current_pose[2,3])

    # Creating 3D figure of the movement path
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    for path, color, label in zip([gt_path, es_path], ['red', 'green'], ['gt','pred']):
        ax.plot(path['x'], path['y'], path['z'], color, label=label)
        ax.legend()

    plt.show()


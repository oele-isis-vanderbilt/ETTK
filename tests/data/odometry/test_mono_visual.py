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
    vo = et3d.VisualOdometry(data_dir)

    # play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm.tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
    et3d.lib.visualization.plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''

import numpy as np
import os
from typing import List


fdir = os.path.dirname(__file__)

# for coco 17 dataset
coco_names = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle"  # 16
]

coco_skeleton =  [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6]
]

coco_to_unified = [
    0,  # nose
    5,  # left shoulder
    7,  # left elbow
    9,  # left wrist
    11, # left hip
    13, # left knee
    15, # left ankle
    6,
    8,
    10,
    12,
    14,
    16
]

coco_to_unified_np = np.array(coco_to_unified)

def _get_coco_skeleton_name() -> List[List[str]]:
    result: List[List[str]] = []
    for node in coco_skeleton:
        result.append([coco_names[node[0]], coco_names[node[1]]])
    return result


coco_skeleton_name = _get_coco_skeleton_name()

# some convert tools for coco dataset
# get the code from https://github.com/zju3dv/mvpose
def coco2shelf3D(coco_pose: np.ndarray) -> np.ndarray:
    """
    transform coco order(our method output) 3d pose to shelf dataset order with interpolation
    :param coco_pose: np.array with shape 3x17
    :return: 3D pose in shelf order with shape 14x3
    """
    coco_pose = coco_pose.T
    shelf_pose = np.zeros((14, 3))
    coco2shelf = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
    shelf_pose[0: 12] += coco_pose[coco2shelf]
    neck = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
    head_bottom = (neck + coco_pose[0]) / 2  # nose and head center
    head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear
    # head_top = coco_pose[0] + (coco_pose[0] - head_bottom)
    head_top = head_bottom + (head_center - head_bottom) * 2
    # shelf_pose[12] += head_bottom
    # shelf_pose[13] += head_top
    shelf_pose[12] = (shelf_pose[8] + shelf_pose[9]) / 2  # Use middle of shoulder to init
    shelf_pose[13] = coco_pose[0]  # use nose to init
    shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [0.75, 0.75, 1.5] )
    shelf_pose[12] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array ( [1. / 2., 1. / 2., 1. / 2.] )
    # shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [0.5, 0.5, 1.5] )
    # shelf_pose[12] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [1.0 / 3, 1.0 / 3, 1.0 / 3] )
    return shelf_pose


# get the code from https://github.com/zju3dv/mvpose
def coco2panoptic(coco_pose: np.ndarray) -> np.ndarray:
    """
    :param coco_pose: 17x3 MS COCO17 order keypoints
    :return: 15x3 old style panoptic order keypoints
    """
    panoptic_pose = np.zeros(coco_pose.shape[:-2] + (15, 3))
    map_array = np.array ( [5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16] )
    panoptic_pose[3:] += coco_pose[map_array]
    panoptic_pose[2] += (coco_pose[11] + coco_pose[12]) / 2  # Take middle of two hips as BodyCenter
    mid_shoulder = (coco_pose[5] + coco_pose[6]) / 2  # Use middle of shoulder to init
    nose = coco_pose[0]  # use nose to init
    head_top = mid_shoulder + (nose - mid_shoulder) * np.array ( [0.4, 1.75, 0.4] )
    neck = mid_shoulder + (nose - mid_shoulder) * np.array ( [.3, .5, .3] )
    panoptic_pose[0] += neck
    panoptic_pose[1] = head_top
    return panoptic_pose


cmu_to_coco_joint_label = {
    1: 0,  # nose
    3: 5,  # shouder_l
    4: 7,  # elbow_l
    5: 9,  # wrist_l
    6: 11,  # hip_l
    7: 13,  # knee_l
    8: 15,  # ankle_l
    9: 6,  # shoulder_r
    10: 8,  # elbow_r
    11: 10,  # wrist_r
    12: 12,  # hip_r
    13: 14,  # knee_r
    14: 16,  # ankle_r
    15: 2,  # eye_r
    16: 1,  # eye_l
    17: 4,  # ear_r
    18: 3,  # ear_l
}

# Note: the paper:
# Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image
# use 14 subset joint for compute 2d loss..
# Note: nose, left eye, right eye, left ear, right ear contains head rotation info..


def get_coco_subset(coco_2d_joints: np.ndarray):
    """
    Note: maybe we can compute head location by average nose, left eye, right eye, left ear, right ear.
    However, the root position is unknown..
    input shape : (..., 17, 2)
    output shape : (..., 13, 2)
    """
    avg_pos: np.ndarray = np.mean(coco_2d_joints[..., 0:5, :], axis=-1)
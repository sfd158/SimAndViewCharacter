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
from scipy.spatial.transform import Rotation
import subprocess
from typing import List

from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib import MotionHelper
from VclSimuBackend.Common.MathHelper import MathHelper

file_dir = os.path.dirname(__file__)
xml_ske_dir = os.path.join(file_dir, "../../../CharacterData/StdHuman/StdHuman_New.bvh")

pfnn_dir = r"D:\downloads\PFNN-master\data\animations"
output_dir = r"G:\retarget-pfnn-mocap"


def retarget_pfnn_data(mocap_dir: str):
    save_fname = os.path.join(output_dir, mocap_dir[:-4] + "-mocap-100.bvh")
    xml_ske: MotionData = BVHLoader.load(xml_ske_dir)
    pfnn_motion = BVHLoader.load(os.path.join(pfnn_dir, mocap_dir), insert_T_pose=True)

    sub_motion: MotionData = pfnn_motion.sub_sequence(0, 1, copy=True)
    pfnn_motion: MotionData = pfnn_motion.sub_sequence(1, None)

    # compute the rotation offset.
    ret_joint_rotation: np.ndarray = sub_motion.joint_rotation[0]
    def compute_left():
        left_knee = pfnn_motion.joint_names.index("LeftLeg")
        left_knee_parent = pfnn_motion.joint_parents_idx[left_knee]
        left_knee_offset: np.ndarray = pfnn_motion.joint_offsets[left_knee].copy()
        unit_left_knee_offset: np.ndarray = left_knee_offset / np.linalg.norm(left_knee_offset)
        delta_quat = MathHelper.quat_between(unit_left_knee_offset, np.array([0.0, -1.0, 0.0]))
        ret_joint_rotation[left_knee_parent] = delta_quat

    # compute the rotation offset.
    def compute_right():
        right_knee = pfnn_motion.joint_names.index("RightLeg")
        right_knee_parent = pfnn_motion.joint_parents_idx[right_knee]
        right_knee_offset: np.ndarray = pfnn_motion.joint_offsets[right_knee].copy()
        unit_right_knee_offset: np.ndarray = right_knee_offset / np.linalg.norm(right_knee_offset)
        delta_quat = MathHelper.quat_between(unit_right_knee_offset, np.array([0.0, -1.0, 0.0]))
        ret_joint_rotation[right_knee_parent] = delta_quat

    compute_left()
    compute_right()

    pfnn_motion: MotionData = pfnn_motion.reconfig_reference_pose(ret_joint_rotation, False, False)

    joint_map = {
        "RootJoint": "Hips",
        "pelvis_lowerback": ["LowerBack", "Spine"],
        "lowerback_torso": "Spine1",
        "torso_head": ["Neck", "Neck1", "Head"],
        "torso_head_end": "Head_end",
        "rTorso_Clavicle": "RightShoulder",
        "rShoulder": "RightArm",
        "rElbow": "RightForeArm",
        "rWrist": "RightHand",
        "rWrist_end": "RightHandIndex1_end",
        "lTorso_Clavicle": "LeftShoulder",
        "lShoulder": "LeftArm",
        "lElbow": "LeftForeArm",
        "lWrist": "LeftHand",
        "lWrist_end": "LeftHandIndex1_end",
        "rHip": ["RHipJoint", "RightUpLeg"],
        "rKnee": "RightLeg",
        "rAnkle": "RightFoot",
        "rToeJoint": "RightToeBase",
        "rToeJoint_end": "RightToeBase_end",
        "lHip": ["LHipJoint", "LeftUpLeg"],
        "lKnee": "LeftLeg",
        "lAnkle": "LeftFoot",
        "lToeJoint": "LeftToeBase",
        "lToeJoint_end": "LeftToeBase_end"
    }

    # test get count..
    # sum_val = 0
    # for key, value in joint_map.items():
    #     if isinstance(value, str):
    #         sum_val += 1
    #     elif isinstance(value, List):
    #         sum_val += len(value)
    # print(sum_val, len(pfnn_motion.joint_names))

    retarget_mocap: MotionData = pfnn_motion.retarget(joint_map)
    # # Calc Children
    children = [[] for _ in range(retarget_mocap.num_joints)]
    for i, p in enumerate(retarget_mocap._skeleton_joint_parents[1:]):
        children[p].append(i + 1)

    # Calc End Site
    retarget_mocap._end_sites = []
    for i, child in enumerate(children):
        if len(child) == 0:
            retarget_mocap.end_sites.append(i)
    #
    # # Scale Retarget Mocap
    # # Assume y axis is up
    retar_root_height = retarget_mocap.joint_position[0, 0, 1] - np.min(retarget_mocap.joint_position[0, :, 1])
    xml_root_height = xml_ske.joint_position[0, 0, 1] - np.min(xml_ske.joint_position[0, :, 1])
    retarget_mocap._joint_translation *= xml_root_height / retar_root_height
    #
    # # joint order in new_mocap is different from joint order in xml
    # # Modify joint offset, and recompute
    for xml_idx, xml_name in enumerate(xml_ske.joint_names):
        retar_idx = retarget_mocap.joint_names.index(xml_name)
        retarget_mocap._skeleton_joint_offsets[retar_idx] = xml_ske.joint_offsets[xml_idx].copy()
    #
    retarget_mocap.recompute_joint_global_info()
    retarget_mocap = retarget_mocap.resample(100)

    BVHLoader.save(retarget_mocap, save_fname)
    print(f"save to {save_fname}")
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", "test.bvh"])


for pfnn_fname_ in os.listdir(pfnn_dir):
    if pfnn_fname_.endswith(".bvh"):
        # pfnn_fname = os.path.join(pfnn_dir, pfnn_fname_)
        retarget_pfnn_data(pfnn_fname_)

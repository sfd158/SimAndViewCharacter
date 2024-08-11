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
# from scipy.spatial.transform import Rotation

from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.Common.MathHelper import MathHelper

file_dir = os.path.dirname(__file__)

xml_ske_dir = os.path.join(file_dir, "../../../CharacterData/StdHuman/StdHuman_New.bvh")
xml_ske: MotionData = BVHLoader.load(xml_ske_dir)
lafan1 = r"D:\downloads\ubisoft-laforge-animation-dataset\lafan1\lafan1"
lafan1_output = os.path.abspath(os.path.join(file_dir, "../../../CharacterData/lafan"))


def retarget_lafan1(bvh_fname: str):
    motion: MotionData = BVHLoader.load(os.path.join(lafan1, bvh_fname))
    motion.joint_offsets[np.abs(motion.joint_offsets) < 1e-3] = 0
    motion.recompute_joint_global_info()

    rotation_0 = motion.joint_rotation[0].copy()

    for index in range(motion.num_joints):
        print(f"{index}, {motion.joint_names[index]}, {rotation_0[index]}")

    Spine = motion.joint_names.index("Spine")
    Spine1 = motion.joint_names.index("Spine1")
    Spine2 = motion.joint_names.index("Spine2")
    Neck = motion.joint_names.index("Neck")
    Head = motion.joint_names.index("Head")

    LeftLeg = motion.joint_names.index("LeftLeg")
    RightLeg = motion.joint_names.index("RightLeg")

    LeftArm = motion.joint_names.index("LeftArm")
    RightArm = motion.joint_names.index("RightArm")

    LeftHand = motion.joint_names.index("LeftHand")
    RightHand = motion.joint_names.index("RightHand")

    RightForeArm = motion.joint_names.index("RightForeArm")
    LeftForeArm = motion.joint_names.index("LeftForeArm")

    LeftToe = motion.joint_names.index("LeftToe")
    RightToe = motion.joint_names.index("RightToe")

    rotation_0[Spine] = MathHelper.unit_quat()
    rotation_0[Spine1] = MathHelper.unit_quat()
    rotation_0[Spine2] = MathHelper.unit_quat()
    rotation_0[Neck] = MathHelper.unit_quat()
    rotation_0[Head] = MathHelper.unit_quat()
    rotation_0[LeftLeg] = MathHelper.unit_quat()
    rotation_0[RightLeg] = MathHelper.unit_quat()
    rotation_0[LeftArm] = MathHelper.unit_quat()
    rotation_0[RightArm] = MathHelper.unit_quat()

    rotation_0[LeftHand] = MathHelper.unit_quat()
    rotation_0[RightHand] = MathHelper.unit_quat()
    rotation_0[RightForeArm] = MathHelper.unit_quat()
    rotation_0[LeftForeArm] = MathHelper.unit_quat()

    rotation_0[LeftToe] = MathHelper.unit_quat()
    rotation_0[RightToe] = MathHelper.unit_quat()

    motion.reconfig_reference_pose(motion.joint_rotation[0].copy(), False, False)

    joint_map = {
        "RootJoint": "Hips",  # OK
        "pelvis_lowerback": ["Spine"],
        "lowerback_torso": ["Spine1", "Spine2"],
        "torso_head": ["Neck", "Head"],
        "torso_head_end": "Head_end",
        "rTorso_Clavicle": "RightShoulder",
        "rShoulder": "RightArm",
        "rElbow": "RightForeArm",
        "rWrist": "RightHand",
        "rWrist_end": "RightHand_end",
        "lTorso_Clavicle": "LeftShoulder",
        "lShoulder": "LeftArm",
        "lElbow": "LeftForeArm",
        "lWrist": "LeftHand",
        "lWrist_end": "LeftHand_end",
        "rHip": "RightUpLeg",
        "rKnee": "RightLeg",
        "rAnkle": "RightFoot",
        "rToeJoint": "RightToe",
        "rToeJoint_end": "RightToe_end",
        "lHip": "LeftUpLeg",  # OK
        "lKnee": "LeftLeg",  # OK
        "lAnkle": "LeftFoot",  # OK
        "lToeJoint": "LeftToe",  # OK
        "lToeJoint_end": "LeftToe_end"  # OK
    }

    retarget_mocap: MotionData = motion.retarget(joint_map)
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

    output_fname = os.path.join(lafan1_output, bvh_fname)
    BVHLoader.save(retarget_mocap, output_fname)
    print(f"save retargeting result to {bvh_fname}")


def main():
    for fname in os.listdir(lafan1):
        if fname.endswith(".bvh"):
            retarget_lafan1(fname)


if __name__ == "__main__":
    main()

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

# 0019_AdvanceBollywoodDance001.bvh


import numpy as np
import os
from scipy.spatial.transform import Rotation
import subprocess

from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib import MotionHelper
from VclSimuBackend.Common.MathHelper import MathHelper


def main():
    file_dir = os.path.dirname(__file__)
    xml_ske_dir = os.path.join(file_dir, "../../../CharacterData/StdHuman/StdHuman_New.bvh")

    # The first frame is zero pose.
    # TODO: 0007_Cartwheel001 motion has bug.. arm length doesn't match..
    mocap_dir = os.path.abspath(os.path.join(file_dir, "../../../CharacterData/sfu/0019_AdvanceBollywoodDance001.bvh"))

    output_dir = mocap_dir[:-4] + "-mocap-100.bvh"

    xml_ske: MotionData = BVHLoader.load(xml_ske_dir)
    mocap: MotionData = BVHLoader.load(mocap_dir)  # already T Pose
    t_pose_rot = mocap.joint_rotation[-1].copy()
    l_hip_index = mocap.joint_names.index("LeftUpLeg")
    r_hip_index = mocap.joint_names.index("RightUpLeg")
    print(t_pose_rot[l_hip_index])
    print(t_pose_rot[r_hip_index])
    t_pose_rot[l_hip_index] = MathHelper.unit_quat()
    t_pose_rot[r_hip_index] = MathHelper.unit_quat()
    mocap.reconfig_reference_pose(t_pose_rot, False, False)
    print(mocap.joint_names)

    # # Contains End Site
    joint_map = {
        "RootJoint": ["Hips"],  # OK
        "pelvis_lowerback": ["Spine", "Spine1"],  # OK
        "lowerback_torso": ["Spine2", "Spine3"],  # OK
        "torso_head": ["Neck", "Head"],  # OK
        "torso_head_end": "Head_end",  # OK
        "rTorso_Clavicle": "RightShoulder",  # OK
        "rShoulder": "RightArm",  # OK
        "rElbow": "RightForeArm",  # OK
        "rWrist": "RightHand",  # OK
        "rWrist_end": "RightHandEnd",
        "lTorso_Clavicle": "LeftShoulder",
        "lShoulder": "LeftArm",
        "lElbow": "LeftForeArm",
        "lWrist": "LeftHand",
        "lWrist_end": "LeftHandEnd",
        "rHip": "RightUpLeg",  # OK
        "rKnee": "RightLeg",  # OK
        "rAnkle": "RightFoot",  # OK
        "rToeJoint": "RightToeBase",  # OK
        "rToeJoint_end": "RightToeBase_end",  # OK
        "lHip": "LeftUpLeg",  # OK
        "lKnee": "LeftLeg",  # OK
        "lAnkle": "LeftFoot",  # OK
        "lToeJoint": "LeftToeBase",  # OK
        "lToeJoint_end": "LeftToeBase_end"  # OK
    }

    retarget_mocap: MotionData = mocap.retarget(joint_map)

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
    retarget_mocap._joint_translation[:, 0, :] *= xml_root_height / retar_root_height
    retarget_mocap._skeleton_joint_offsets *= xml_root_height / retar_root_height
    #
    # # joint order in new_mocap is different from joint order in xml
    # # Modify joint offset, and recompute
    for xml_idx, xml_name in enumerate(xml_ske.joint_names):
        retar_idx = retarget_mocap.joint_names.index(xml_name)
        retarget_mocap._skeleton_joint_offsets[retar_idx] = xml_ske.joint_offsets[xml_idx].copy()

    # Save to bvh file
    # retarget_mocap.joint_translation[:, 0, 1] -= 0.5 * np.min(retarget_mocap.joint_position[:, :, 1])
    # retarget_mocap = retarget_mocap.resample(100)
    # print(retarget_mocap.fps)
    BVHLoader.save(retarget_mocap, output_dir)
    print("Retargeting Success to %s" % output_dir)
    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", output_dir])


if __name__ == "__main__":
    main()

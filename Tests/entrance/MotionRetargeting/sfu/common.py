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

from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib import MotionHelper
from VclSimuBackend.Common.MathHelper import MathHelper


def main():
    file_dir = os.path.dirname(__file__)
    xml_ske_dir = os.path.join(file_dir, "../../../CharacterData/StdHuman/StdHuman_New.bvh")

    # The first frame is zero pose.
    # TODO: 0007_Cartwheel001 motion has bug.. arm length doesn't match..
    mocap_dir = os.path.abspath(os.path.join(file_dir, "../../../CharacterData/sfu/0007_Cartwheel001.bvh"))

    output_dir = mocap_dir[:-4] + "-mocap-100.bvh"

    xml_ske: MotionData = BVHLoader.load(xml_ske_dir)
    mocap: MotionData = BVHLoader.load(mocap_dir)  # already T Pose
    # print(mocap.joint_names())

    # # Contains End Site
    joint_map = {
        "RootJoint": "Hips",
        "pelvis_lowerback": "Spine",
        "lowerback_torso": "Spine1",
        "torso_head": ["Neck", "Head"],
        "torso_head_end": "Head_end",
        "rTorso_Clavicle": "RightShoulder",
        "rShoulder": "RightArm",
        "rElbow": "RightForeArm",
        "rWrist": "RightHand",
        "rWrist_end": "RightHandThumb",
        "lTorso_Clavicle": "LeftShoulder",
        "lShoulder": "LeftArm",
        "lElbow": "LeftForeArm",
        "lWrist": "LeftHand",
        "lWrist_end": "LeftHandThumb",
        "rHip": "RightUpLeg",
        "rKnee": "RightLeg",
        "rAnkle": "RightFoot",
        "rToeJoint": "RightToeBase",
        "rToeJoint_end": "RightToeBase_end",
        "lHip": "LeftUpLeg",
        "lKnee": "LeftLeg",
        "lAnkle": "LeftFoot",
        "lToeJoint": "LeftToeBase",
        "lToeJoint_end": "LeftToeBase_end"
    }

    # initial ankle angle
    def calc_init_delta_ankle_angle():
        tpose_mocap = mocap.sub_sequence(0, 1, copy=True)
        tpose_mocap.joint_rotation[0] = Rotation.identity(tpose_mocap.joint_rotation.shape[1]).as_quat()
        tpose_mocap.recompute_joint_global_info()

        def calc_angle(pos, l_knee_idx, l_ankle_idx, l_toe_idx):
            vec0 = pos[0, l_ankle_idx, :] - pos[0, l_knee_idx, :]
            vec1 = pos[0, l_ankle_idx, :] - pos[0, l_toe_idx, :]
            vec0 = vec0 / np.linalg.norm(vec0)
            vec1 = vec1 / np.linalg.norm(vec1)
            angle = np.arccos(np.dot(vec0, vec1))
            return angle

        names = tpose_mocap.joint_names
        angle_mocap = calc_angle(tpose_mocap.joint_position, names.index("LeftLeg"), names.index("LeftFoot"), names.index("LeftToeBase"))

        names = xml_ske.joint_names
        angle_xml = calc_angle(xml_ske.joint_position, names.index("lKnee"), names.index("lAnkle"), names.index("lToeJoint"))
        delta_angle = angle_mocap - angle_xml
        print(delta_angle)
        return -delta_angle

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
    retarget_mocap._joint_translation *= xml_root_height / retar_root_height
    #
    # # joint order in new_mocap is different from joint order in xml
    # # Modify joint offset, and recompute
    for xml_idx, xml_name in enumerate(xml_ske.joint_names):
        retar_idx = retarget_mocap.joint_names.index(xml_name)
        retarget_mocap._skeleton_joint_offsets[retar_idx] = xml_ske.joint_offsets[xml_idx].copy()
    #
    retarget_mocap.recompute_joint_global_info()

    # add delta angle to retarget_mocap
    def add_delta_angle_to_ankle():
        delta_angle = calc_init_delta_ankle_angle()
        ankle_rot: Rotation = Rotation.from_rotvec(delta_angle * np.array([1.0, 0.0, 0.0]))
        l_ankle_idx = retarget_mocap.joint_names.index("lAnkle")
        r_ankle_idx = retarget_mocap.joint_names.index("rAnkle")
        retarget_mocap.joint_rotation[:, l_ankle_idx, :] = (ankle_rot * Rotation(retarget_mocap.joint_rotation[:, l_ankle_idx, :])).as_quat()
        retarget_mocap.joint_rotation[:, r_ankle_idx, :] = (ankle_rot * Rotation(retarget_mocap.joint_rotation[:, r_ankle_idx, :])).as_quat()

        # we should also modify toe rotation
        def clip_toe_angle(toe_idx: int):
            q_x, q_yz = MathHelper.x_decompose(retarget_mocap.joint_rotation[:, toe_idx, :])
            q_x = MathHelper.flip_quat_by_w(q_x)
            angle = np.arcsin(q_x[:, 0])
            angle[angle > 0] = 0
            retarget_mocap.joint_rotation[:, toe_idx, :] = Rotation.from_rotvec(angle[:, None] @ np.array([[1, 0, 0]])).as_quat()

        clip_toe_angle(retarget_mocap.joint_names.index("lToeJoint"))
        clip_toe_angle(retarget_mocap.joint_names.index("rToeJoint"))

        retarget_mocap.recompute_joint_global_info()

        # We need to do IK here.


    # add_delta_angle_to_ankle()
    # add delta angle to ankle only works for walking motion..
    # Save to bvh file
    # retarget_mocap.joint_translation[:, 0, 1] -= 0.5 * np.min(retarget_mocap.joint_position[:, :, 1])
    retarget_mocap = retarget_mocap.resample(100)
    print(retarget_mocap.fps)
    BVHLoader.save(retarget_mocap, output_dir)
    print("Retargeting Success to %s" % output_dir)
    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", output_dir])


if __name__ == "__main__":
    main()

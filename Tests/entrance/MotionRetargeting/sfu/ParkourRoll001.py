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

from VclSimuBackend.pymotionlib import BVHLoader, MotionData
from VclSimuBackend.pymotionlib import MotionHelper
from VclSimuBackend.Common.MathHelper import MathHelper


def main():
    xml_ske_dir = "../../../CharacterData/StdHuman/StdHuman_New.bvh"

    # The first frame is zero pose.
    mocap_dir = "../../../CharacterData/sfu/0017_ParkourRoll001.bvh"

    output_dir = mocap_dir[:-4] + "-mocap.bvh"

    xml_ske = BVHLoader.load(xml_ske_dir)
    mocap = BVHLoader.load(mocap_dir)  # already T Pose

    # # Contains End Site
    joint_map = {"RootJoint": "Hips",
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
    #
    retarget_mocap = mocap.retarget(joint_map)
    #
    # # Calc Children
    children = [[] for _ in range(retarget_mocap._num_joints)]
    for i, p in enumerate(retarget_mocap._skeleton_joint_parents[1:]):
        children[p].append(i + 1)

    # Calc End Site
    retarget_mocap._end_sites = []
    for i, child in enumerate(children):
        if len(child) == 0:
            retarget_mocap._end_sites.append(i)
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
    # # recompute global info
    retarget_mocap.recompute_joint_global_info()
    lankle = retarget_mocap.joint_names.index("lAnkle")
    lankle_q, _ = MathHelper.x_decompose(retarget_mocap.joint_orientation[:, lankle, :])
    retarget_mocap._joint_orientation[:, lankle, :] = lankle_q

    rankle = retarget_mocap.joint_names.index("rAnkle")
    rankle_q, _ = MathHelper.x_decompose(retarget_mocap.joint_orientation[:, rankle, :])
    retarget_mocap._joint_orientation[:, rankle, :] = rankle_q

    retarget_mocap.compute_joint_local_info(retarget_mocap._joint_position, retarget_mocap._joint_orientation)
    retarget_mocap.recompute_joint_global_info()

    ltoe = retarget_mocap.joint_names.index("lToeJoint")
    ltoe_q, _ = MathHelper.x_decompose(retarget_mocap.joint_orientation[:, ltoe, :])
    retarget_mocap._joint_orientation[:, ltoe, :] = ltoe_q

    rtoe = retarget_mocap.joint_names.index("rToeJoint")
    rtoe_q, _ = MathHelper.x_decompose(retarget_mocap.joint_orientation[:, rtoe, :])
    retarget_mocap._joint_orientation[:, rtoe, :] = rtoe_q

    # rankle = retarget_mocap.joint_names.index("rAnkle")
    #
    # # Knee's rotation should only have one component..
    # left_knee = retarget_mocap.joint_names.index("lKnee")
    # right_knee = retarget_mocap.joint_names.index("rKnee")
    # lknee_q, _ = MathHelper.x_decompose(retarget_mocap._joint_orientation[:, left_knee, :])
    # rknee_q, _ = MathHelper.x_decompose(retarget_mocap._joint_orientation[:, right_knee, :])
    #
    # retarget_mocap._joint_orientation[:, left_knee, :] = lknee_q
    # retarget_mocap._joint_orientation[:, right_knee, :] = rknee_q
    #
    retarget_mocap.compute_joint_local_info(retarget_mocap._joint_position, retarget_mocap._joint_orientation)
    retarget_mocap.recompute_joint_global_info()
    #
    # # Root Position in retarget_mocap should match root position in xml
    # MotionHelper.adjust_root_height(retarget_mocap)

    # Save to bvh file
    BVHLoader.save(retarget_mocap, output_dir)
    print("Retargeting Success to %s" % output_dir)


if __name__ == "__main__":
    main()

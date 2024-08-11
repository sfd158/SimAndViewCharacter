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


from VclSimuBackend.pymotionlib import BVHLoader, MotionData
from VclSimuBackend.pymotionlib import MotionHelper
from VclSimuBackend.Common.MathHelper import MathHelper


def main():
    file_dir = os.path.dirname(__file__)
    xml_ske_dir = os.path.join(file_dir, "../../CharacterData/StdHuman/StdHuman_New.bvh")

    # The first frame is zero pose.
    # TODO: 0007_Cartwheel001 motion has bug.. arm length doesn't match..
    mocap_dir = os.path.abspath( os.path.join(file_dir, "../../CharacterData/WalkF.bvh"))
    output_dir = mocap_dir[:-4] + "-mocap.bvh"
    # "../../CharacterData/WalkF.bvh"
    xml_ske: MotionData = BVHLoader.load(xml_ske_dir)
    mocap: MotionData = BVHLoader.load(mocap_dir)  # already T Pose

    # # Contains End Site
    joint_map = {"RootJoint": "root",  # OK
                 "pelvis_lowerback": "pelvis_lowerback",  # OK
                 "lowerback_torso": "lowerback_torso",  # OK
                 "torso_head": "torso_head",  # OK
                 "torso_head_end": "torso_head_end",  # OK
                 "rTorso_Clavicle": "rClavicle",  # OK
                 "rShoulder": "rShoulder",  # OK
                 "rElbow": "rElbow",  # OK
                 "rWrist": "rWrist",  # OK
                 "rWrist_end": "rLowerKnuckle_end",  # OK
                 "lTorso_Clavicle": "lClavicle",  # OK
                 "lShoulder": "lShoulder",  # OK
                 "lElbow": "lElbow",  # OK
                 "lWrist": "lWrist",  # OK
                 "lWrist_end": "lLowerKnuckle_end",  # OK
                 "rHip": "rHip",  # OK
                 "rKnee": "rKnee",  # OK
                 "rAnkle": "rAnkle",  # OK
                 "rToeJoint": "rToeJoint",  # OK
                 "rToeJoint_end": "rToeJoint_end",  # OK
                 "lHip": "lHip",  # OK
                 "lKnee": "lKnee",  # OK
                 "lAnkle": "lAnkle",  # OK
                 "lToeJoint": "lToeJoint",  # OK
                 "lToeJoint_end": "lToeJoint_end"  # OK
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

    retarget_mocap = mocap.retarget(joint_map)

    # # Calc Children
    children = [[] for _ in range(retarget_mocap.num_joints)]
    for i, p in enumerate(retarget_mocap._skeleton_joint_parents[1:]):
        children[p].append(i + 1)

    # Calc End Site
    retarget_mocap._end_sites = []
    for i, child in enumerate(children):
        if len(child) == 0:
            retarget_mocap.end_sites.append(i)

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
    retarget_mocap.recompute_joint_global_info()
    retarget_mocap.resample(120)

    # add delta angle to ankle only works for walking motion..
    # Save to bvh file
    BVHLoader.save(retarget_mocap, output_dir)
    print("Retargeting Success to %s" % output_dir)


if __name__ == "__main__":
    main()

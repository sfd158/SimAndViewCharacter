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

from typing import Optional
import numpy as np
import os
from scipy.spatial.transform import Rotation
import subprocess

from VclSimuBackend.pymotionlib import BVHLoader, MotionData
from VclSimuBackend.pymotionlib import MotionHelper
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.Utils.IK.TwoLinkIK import two_link_ik


def dist(bvh: MotionData, idx_a: int, idx_b: int) -> float:
    ra: np.ndarray = Rotation(bvh.joint_rotation[idx_a], copy=False).as_rotvec()
    rb: np.ndarray = Rotation(bvh.joint_rotation[idx_b], copy=False).as_rotvec()
    delta = np.linalg.norm(ra - rb).item()
    return delta


def get_last_pair(bvh):
    best_val = 1000.0
    pa, pb = 0, 0
    for i in range(10, bvh.num_frames):
        for j in range(i + 100, bvh.num_frames, 10):
            d = dist(bvh, i, j)
            if d < best_val:
                best_val = d
                pa, pb = i, j
                print(i, best_val, pa, pb, (pb - pa) % 12)
    return pa, pb


# make foot flatten
def flatten_foot(mocap: MotionData, index: str):
    mocap_heel_index = mocap.joint_names.index(f"{index}Foot")
    mocap_toe_index = mocap.joint_names.index(f"{index}ToeBase")
    toe_rotvec = Rotation(mocap.joint_rotation[:, mocap_toe_index, :]).as_rotvec()
    mocap.joint_rotation[:, mocap_heel_index, :] = (Rotation(mocap.joint_rotation[:, mocap_heel_index, :]) *
        Rotation.from_rotvec(0.3 * toe_rotvec)).as_quat()
    mocap.joint_rotation[:, mocap_toe_index, :] = Rotation.from_rotvec(0.7 * toe_rotvec).as_quat()


def modify_knee(mocap: MotionData, index: str):
    knee_index: int = mocap.joint_names.index(f"{index}LowLeg")
    mocap.joint_rotation[:, knee_index, :], _ = MathHelper.x_decompose(mocap.joint_rotation[:, knee_index, :])



def save_foot_position(mocap: MotionData, index: str):
    foot_index: int = mocap.joint_names.index(f"{index}Foot")
    return mocap.joint_position[:, foot_index, :].copy()


def refine_foot_process(mocap: MotionData, index: str, saved_foot_pos: np.ndarray):
    """
    When heel and toe rotation is modified, foot may sliding on the ground.
    We should apply two link IK on leg-knee-toe_end joints,
    to make sure that end joint position is not moved.
    """

    num_frame: int = mocap.num_frames
    leg_index: int = mocap.joint_names.index(f"{index}UpLeg")
    knee_index: int = mocap.joint_names.index(f"{index}LowLeg")

    # in human 3.6 data, the foot joint is at heel, rather than ankle.
    foot_index: int = mocap.joint_names.index(f"{index}Foot")
    input_target_global: np.ndarray = saved_foot_pos - mocap.joint_position[:, leg_index, :]

    input_target_global: np.ndarray = mocap.joint_position[:, foot_index, :] - mocap.joint_position[:, leg_index, :]
    leg_rotate: np.ndarray = mocap.joint_orientation[:, leg_index, :].copy()
    knee_rotate: np.ndarray = mocap.joint_rotation[:, knee_index, :].copy()

    # (frame, 2, 4)
    input_rotate: np.ndarray = np.concatenate([leg_rotate[:, None, :], knee_rotate[:, None, :]], axis=1)

    knee_offset: np.ndarray = np.tile(mocap.joint_offsets[knee_index], (num_frame, 1))
    # (frame, 3)
    knee_to_end_global: np.ndarray = mocap.joint_position[:, foot_index, :] - mocap.joint_position[:, knee_index, :]
    knee_to_end_local: np.ndarray = Rotation(mocap.joint_orientation[:, knee_index, :]).inv().apply(knee_to_end_global)

    # (frame, 2, 3)
    input_offset: np.ndarray = np.concatenate([knee_offset[:, None, :], knee_to_end_local[:, None, :]], axis=1)

    # run two line IK
    output_rotate: np.ndarray = two_link_ik(input_rotate, input_offset, input_target_global, np.array([1.0, 0.0, 0.0]))
    # apply output rotate to motion data..
    leg_parent_index: int = mocap.joint_parents_idx[leg_index]
    mocap.joint_rotation[:, leg_index, :] = (Rotation(mocap.joint_rotation[:, leg_parent_index, :]).inv() * \
                                            Rotation(output_rotate[:, 0, :])).as_quat()
    mocap.joint_rotation[:, knee_index, :] = output_rotate[:, 1, :]


def main(mocap_dir: Optional[str] = None, output_dir: Optional[str] = None, visualize: bool = True):
    file_dir = os.path.dirname(__file__)
    xml_ske_dir = os.path.join(file_dir, "../../CharacterData/StdHuman/StdHuman_New.bvh")

    if mocap_dir is None:
        mocap_dir = os.path.join(file_dir, "../../CharacterData/Human3.6/S1/Phoning.bvh")

    if output_dir is None:
        output_dir = mocap_dir[:-4] + "-mocap-100.bvh"

    xml_ske: MotionData = BVHLoader.load(xml_ske_dir)
    mocap: MotionData = BVHLoader.load(mocap_dir)

    # smaller head rotation...
    mocap_head_index = mocap.joint_names.index("Head")
    mocap.joint_rotation[:, mocap_head_index, :] = Rotation.from_rotvec(
        0.6 * Rotation(mocap.joint_rotation[:, mocap_head_index, :]).as_rotvec()).as_quat()

    saved_lfoot = save_foot_position(mocap, "Left")
    saved_rfoot = save_foot_position(mocap, "Right")
    flatten_foot(mocap, "Left")
    flatten_foot(mocap, "Right")
    modify_knee(mocap, "Left")
    modify_knee(mocap, "Right")
    mocap.recompute_joint_global_info()

    refine_foot_process(mocap, "Left", saved_lfoot)
    refine_foot_process(mocap, "Right", saved_rfoot)

    # # Contains End Site
    joint_map = {"RootJoint": "Hips",  # OK
                 "pelvis_lowerback": "Spine",  # OK
                 "lowerback_torso": "Spine1",  # OK
                 "torso_head": ["Neck", "Head"],  # OK
                 "torso_head_end": "Head_end",  # OK
                 "rTorso_Clavicle": "RightShoulder",  # OK
                 "rShoulder": "RightUpArm",  # OK
                 "rElbow": "RightForeArm",  # OK
                 "rWrist": "RightHand",  # OK
                 "rWrist_end": "RightHandThumb",  # OK
                 "lTorso_Clavicle": "LeftShoulder",  # OK
                 "lShoulder": "LeftUpArm",  # OK
                 "lElbow": "LeftForeArm",  # OK
                 "lWrist": "LeftHand",  # OK
                 "lWrist_end": "LeftHandThumb",  # OK
                 "rHip": "RightUpLeg",  # OK
                 "rKnee": "RightLowLeg",  # OK
                 "rAnkle": "RightFoot",  # OK
                 "rToeJoint": "RightToeBase",  # OK
                 "rToeJoint_end": "RightToeBase_end",  # OK
                 "lHip": "LeftUpLeg",  # OK
                 "lKnee": "LeftLowLeg",  # OK
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
    retarget_mocap._joint_translation *= xml_root_height / retar_root_height

    # # joint order in new_mocap is different from joint order in xml
    # # Modify joint offset, and recompute
    for xml_idx, xml_name in enumerate(xml_ske.joint_names):
        retar_idx = retarget_mocap.joint_names.index(xml_name)
        retarget_mocap._skeleton_joint_offsets[retar_idx] = xml_ske.joint_offsets[xml_idx].copy()

    retarget_mocap.recompute_joint_global_info()
    retarget_mocap = retarget_mocap.resample(100)

    # Save to bvh file
    BVHLoader.save(retarget_mocap, output_dir)
    print("Retargeting Success to %s" % output_dir)

    # short_retarget_mocap = retarget_mocap.sub_sequence(0, 1600)
    # BVHLoader.save(short_retarget_mocap, output_dir[:-4] + "-short.bvh")
    if visualize:
        subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", output_dir])

    return retarget_mocap


if __name__ == "__main__":
    main()

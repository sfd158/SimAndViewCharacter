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

"""
Test for tracking foot with multiple soft joints
However, it doesn't work..
"""
import numpy as np
from scipy.spatial.transform import Rotation
from ..Common.MathHelper import MathHelper
from ..pymotionlib.MotionData import MotionData
from ..ODESim.ODECharacter import ODECharacter, BodyInfoState
from ..ODESim.Saver.CharacterToBVH import CharacterTOBVH
from ..ODESim.BVHToTarget import BVHToTargetBase, TargetPose
from ..ODESim.TargetPose import SetTargetToCharacter


def compute_soft_mocap(soft_foot_mocap: MotionData, stdhuman_mocap: MotionData):
    result = soft_foot_mocap.sub_sequence(copy=True)
    result._num_frames = stdhuman_mocap._num_frames
    result._fps = stdhuman_mocap._fps
    result._joint_position = None
    result._joint_orientation = None
    result._joint_rotation = MathHelper.unit_quat_arr((stdhuman_mocap.num_frames, result.num_joints, 4))
    result._joint_translation = np.zeros((stdhuman_mocap.num_frames, result.num_joints, 3))
    result._joint_translation[:, 0, :] = stdhuman_mocap.joint_translation[:, 0, :]
    for std_idx, std_name in enumerate(stdhuman_mocap.joint_names):
        soft_idx = soft_foot_mocap.joint_names.index(std_name)
        result._joint_rotation[:, soft_idx, :] = stdhuman_mocap.joint_rotation[:, std_idx, :]
    result.recompute_joint_global_info()
    return result


def get_zero_frame_state(stdhuman_mocap: MotionData, character: ODECharacter) -> BodyInfoState:
    to_bvh = CharacterTOBVH(character, stdhuman_mocap.fps)
    to_bvh.bvh_hierarchy_no_root()
    soft_foot_mocap: MotionData = to_bvh.bvh_append_no_root()
    stdhuman_mocap: MotionData = stdhuman_mocap.sub_sequence(0, 1)
    require_motion: MotionData = compute_soft_mocap(soft_foot_mocap, stdhuman_mocap)
    bvh_to_target = BVHToTargetBase(require_motion, stdhuman_mocap.fps, character)
    target: TargetPose = bvh_to_target.only_init_global_target()
    bvh_set: SetTargetToCharacter = SetTargetToCharacter(character, target)
    body_state: BodyInfoState = bvh_set.set_character_byframe(0)
    return body_state


def remove_single_joint(bvh: MotionData, remove_name: str):
    remove_index = bvh.joint_names.index(remove_name)
    remain_index = list(range(bvh.num_joints))
    remain_index.remove(remove_index)

    bvh._num_joints -= 1
    parent_name_dict = {name: bvh.joint_names[bvh.joint_parents_idx[index]] for index, name in enumerate(bvh.joint_names)}
    parent_name_dict["RootJoint"] = None
    bvh._joint_translation = bvh._joint_translation[:, remain_index, :]
    bvh._joint_rotation = bvh._joint_rotation[:, remain_index, :]

    bvh._skeleton_joint_offsets = bvh._skeleton_joint_offsets[remain_index, :]
    bvh._skeleton_joints = [name for index, name in enumerate(bvh.joint_names) if index in remain_index]

    new_parent_dict = {key: value for key, value in parent_name_dict.items() if key in bvh._skeleton_joints}
    for name in bvh._skeleton_joints:
        if new_parent_dict[name] == remove_name:
            new_parent_dict[name] = parent_name_dict[remove_name]

    result = []
    for index, name in enumerate(bvh._skeleton_joints):
        if name == "RootJoint":
            result.append(-1)
        else:
            parent_name = new_parent_dict[name]
            parent_index = bvh._skeleton_joints.index(parent_name)
            result.append(parent_index)

    # remove end joints.
    bvh._skeleton_joint_parents = result
    bvh._end_sites = [node if node < remove_index else node - 1 for node in bvh._end_sites]


def modify_toe_end(bvh: MotionData, prefix: str):
    global_ltoe_end = bvh.joint_position[0, bvh.joint_names.index(prefix + "ToeJoint_end"), :]
    global_lankle = bvh.joint_position[0, bvh.joint_names.index(prefix + "Ankle"), :]
    global_lankle_rot_inv = Rotation(bvh.joint_orientation[0, bvh.joint_names.index(prefix + "Ankle"), :]).inv()
    local_ltoe = global_lankle_rot_inv.apply(global_ltoe_end - global_lankle)
    bvh._skeleton_joint_offsets[bvh.joint_names.index(prefix + "ToeJoint_end")] = local_ltoe


def generate_ref_motion(bvh: MotionData, character: ODECharacter):
    """
    bvh: reference motion with same hierarchy as std-human
    character: Character with modified foot
    """
    bvh = bvh.sub_sequence(copy=True)
    # modify original end pos..
    # remove ltoe and rtoe joint in bvh
    modify_toe_end(bvh, "l")
    modify_toe_end(bvh, "r")

    bvh._joint_position = None
    bvh._joint_orientation = None

    remove_single_joint(bvh, "lToeJoint")
    remove_single_joint(bvh, "rToeJoint")
    bvh.joint_names[bvh.joint_names.index("lToeJoint_end")] = "lAnkle_end"
    bvh.joint_names[bvh.joint_names.index("rToeJoint_end")] = "rAnkle_end"
    bvh.recompute_joint_global_info()

    # retarget to current human
    to_bvh = CharacterTOBVH(character, bvh.fps)
    to_bvh.bvh_hierarchy_no_root()
    to_bvh.append_no_root_to_buffer()
    soft_foot_mocap: MotionData = to_bvh.to_file(None)
    result = compute_soft_mocap(soft_foot_mocap, bvh)

    return bvh, result

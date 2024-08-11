"""
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
Build motion dataset from std-human..
"""
import copy
import os
import numpy as np
import subprocess
from typing import List, Optional, Dict, Set
from scipy.spatial.transform import Rotation
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH, BVHLoader
from VclSimuBackend.Common.MathHelper import MathHelper, RotateType

cma_sample_window: Dict[str, float] = {
    "L_Hip_and_Pelvis": 0.4,
    "R_Hip_and_Pelvis": 0.4,
    "Spine1_and_Pelvis": 0.2,
    "L_Knee_and_L_Hip": 0.2,
    "R_Knee_and_R_Hip": 0.2,
    "Spine2_and_Spine1": 0.2,
    "L_Ankle_and_L_Knee": 0.4,
    "R_Ankle_and_R_Knee": 0.4,
    "Spine3_and_Spine2": 0.15,
    "L_Foot_and_L_Ankle": 0.35,
    "R_Foot_and_R_Ankle": 0.35,
    "Neck_and_Spine3": 0.15,
    "L_Collar_and_Spine3": 0.15,
    "R_Collar_and_Spine3": 0.15,
    "Head_and_Neck": 0.15,
    "L_Shoulder_and_L_Collar": 0.2,
    "R_Shoulder_and_R_Collar": 0.2,
    "L_Elbow_and_L_Shoulder": 0.1,
    "R_Elbow_and_R_Shoulder": 0.1,
    "L_Wrist_and_L_Elbow": 0.05,
    "R_Wrist_and_R_Elbow": 0.05
}

def build_cma_cov(joint_names: List[str], action_type: RotateType):
    if action_type == RotateType.AxisAngle:
        result = np.zeros((len(joint_names), 3))
        for jindex, jname in enumerate(joint_names):
            result[jindex, :] = cma_sample_window[jname]
        return result.flatten()
    else:
        raise NotImplementedError


smpl_body_mirror_dict = {
    "Pelvis": "Pelvis",
    "L_Hip": "R_Hip",
    "R_Hip": "L_Hip",
    "Spine1": "Spine1",
    "L_Knee": "R_Knee",
    "R_Knee": "L_Knee",
    "Spine2": "Spine2",
    "L_Ankle": "R_Ankle",
    "R_Ankle": "L_Ankle",
    "Spine3": "Spine3",
    "L_Foot": "R_Foot",
    "R_Foot": "L_Foot",
    "Neck": "Neck",
    "L_Collar": "R_Collar",
    "R_Collar": "L_Collar",
    "Head": "Head",
    "L_Shoulder": "R_Shoulder",
    "R_Shoulder": "L_Shoulder",
    "L_Elbow": "R_Elbow",
    "R_Elbow": "L_Elbow",
    "L_Wrist": "R_Wrist",
    "R_Wrist": "L_Wrist"
}

smpl_mirror_dict = {
    "RootJoint": "RootJoint",
    "L_Hip_and_Pelvis": "R_Hip_and_Pelvis",
    "R_Hip_and_Pelvis": "L_Hip_and_Pelvis",
    "Spine1_and_Pelvis": "Spine1_and_Pelvis",
    "L_Knee_and_L_Hip": "R_Knee_and_R_Hip",
    "R_Knee_and_R_Hip": "L_Knee_and_L_Hip",
    "Spine2_and_Spine1": "Spine2_and_Spine1",
    "L_Ankle_and_L_Knee": "R_Ankle_and_R_Knee",
    "R_Ankle_and_R_Knee": "L_Ankle_and_L_Knee",
    "Spine3_and_Spine2": "Spine3_and_Spine2",
    "L_Foot_and_L_Ankle": "R_Foot_and_R_Ankle",
    "R_Foot_and_R_Ankle": "L_Foot_and_L_Ankle",
    "Neck_and_Spine3": "Neck_and_Spine3",
    "L_Collar_and_Spine3": "R_Collar_and_Spine3",
    "R_Collar_and_Spine3": "L_Collar_and_Spine3",
    "Head_and_Neck": "Head_and_Neck",
    "L_Shoulder_and_L_Collar": "R_Shoulder_and_R_Collar",
    "R_Shoulder_and_R_Collar": "L_Shoulder_and_L_Collar",
    "L_Elbow_and_L_Shoulder": "R_Elbow_and_R_Shoulder",
    "R_Elbow_and_R_Shoulder": "L_Elbow_and_L_Shoulder",
    "L_Wrist_and_L_Elbow": "R_Wrist_and_R_Elbow",
    "R_Wrist_and_R_Elbow": "L_Wrist_and_L_Elbow"
}

smpl_joint_name_list: List[str] = list(smpl_mirror_dict.keys())
smpl_joint_name_set: Set[str] = set(smpl_joint_name_list)
smpl_joint_name_dict: Dict[str, int] = {node: index for index, node in enumerate(smpl_joint_name_list)}

smpl_end_mirror_dict = {
    "Head_and_Neck_end": "Head_and_Neck_end",
    "L_Foot_and_L_Ankle_end": "R_Foot_and_R_Ankle_end",
    "R_Foot_and_R_Ankle_end": "L_Foot_and_L_Ankle_end",
    "L_Wrist_and_L_Elbow_end": "R_Wrist_and_R_Elbow_end",
    "R_Wrist_and_R_Elbow_end": "L_Wrist_and_L_Elbow_end"
}


smpl_joint_to_stdhuman = {
    "RootJoint": "RootJoint",
    "L_Hip_and_Pelvis": "lHip",
    "R_Hip_and_Pelvis": "rHip",
    "Spine1_and_Pelvis": "pelvis_lowerback",
    "L_Knee_and_L_Hip": "lKnee",
    "R_Knee_and_R_Hip": "rKnee",
    "Spine2_and_Spine1": "lowerback_torso",
    "L_Ankle_and_L_Knee": "lAnkle",
    "R_Ankle_and_R_Knee": "rAnkle",
    "Spine3_and_Spine2": "half_parent",
    "L_Foot_and_L_Ankle": "lToeJoint",
    "R_Foot_and_R_Ankle": "rToeJoint",
    "Neck_and_Spine3": "half_child",
    "L_Collar_and_Spine3": "lTorso_Clavicle",
    "R_Collar_and_Spine3": "rTorso_Clavicle",
    "Head_and_Neck": "torso_head",
    "L_Shoulder_and_L_Collar": "lShoulder", 
    "R_Shoulder_and_R_Collar": "rShoulder",
    "L_Elbow_and_L_Shoulder": "lElbow",
    "R_Elbow_and_R_Shoulder": "rElbow",
    "L_Wrist_and_L_Elbow": "lWrist",
    "R_Wrist_and_R_Elbow": "rWrist"
}

def _build_stdhuman_to_smpl_map() -> Dict[str, str]:
    return {value: key for key, value in smpl_joint_to_stdhuman.items() if "half" not in value}


stdhuman_joint_to_smpl_map: Dict[str, str] = _build_stdhuman_to_smpl_map()


def build_smpl_mirror_index(bvh_joint_names: List[str]) -> List[int]:
    name_dict = {node: index for index, node in enumerate(bvh_joint_names)}
    result: List[int] = []
    for joint_index, joint_name in enumerate(bvh_joint_names):
        if "end" in joint_name:
            mirror_index = name_dict[smpl_end_mirror_dict[joint_name]]
        else:
            mirror_index = name_dict[smpl_mirror_dict[joint_name]]
        result.append(mirror_index)
    return result


def build_smpl_body_mirror_index(body_names: List[str]) -> np.ndarray:
    name_dict = {node: index for index, node in enumerate(body_names)}
    result: np.ndarray = np.zeros(len(body_names), dtype=np.int32)
    for body_index, body_name in enumerate(body_names):
        result[body_index] = name_dict[smpl_body_mirror_dict[body_name]]
    return result


def stdhuman_to_smpl(smpl_fname: str, bvh_fname: str, save_dir: str, is_mirror: bool = False):
    if smpl_fname.endswith(".pickle") or smpl_fname.endswith(".json"):
        scene = JsonSceneLoader().load_from_file(smpl_fname)
        ratio = scene.character0.height / 1.6
        hierarchy = CharacterTOBVH(scene.character0).build_hierarchy().append_no_root_to_buffer().to_file(None)
    elif smpl_fname.endswith(".bvh"):
        hierarchy = BVHLoader.load(smpl_fname)
        ratio = hierarchy.joint_position[0, 0, 1] / 0.931
    else:
        raise ValueError

    bvh = BVHLoader.load(bvh_fname)
    if is_mirror:
        bvh = bvh.flip(np.array([1.0, 0.0, 0.0]))

    # we should refine the end site here..
    num_frames: int = bvh._num_frames
    nj: int = hierarchy.num_joints
    fps = bvh._fps

    result = hierarchy.sub_sequence(copy=True)
    result._num_frames = num_frames
    result._fps = fps
    result._joint_position = np.zeros((num_frames, nj, 3))
    result._joint_translation = result.joint_position.copy()
    result._joint_orientation = np.zeros((num_frames, nj, 4))
    result._joint_orientation[:, :, 3] = 1
    result._joint_rotation = result.joint_orientation.copy()

    result_idx = {node: index for index, node in enumerate(result.joint_names)}
    children = [[] for _ in range(result._num_joints)]
    for i, p in enumerate(result._skeleton_joint_parents[1:]):
        children[p].append(i + 1)

    stdhuman_idx = {node: index for index, node in enumerate(bvh.joint_names)}

    joint_map = copy.deepcopy(smpl_joint_to_stdhuman)
    for key, value in joint_map.items():
        key_idx = result_idx[key]
        value_idx = stdhuman_idx.get(value, 233)
        if value == "half_child":
            continue
        if value == "half_parent":
            parent_idx = result.joint_parents_idx[key_idx]
            rotvec = Rotation(result.joint_rotation[:, parent_idx, :]).as_rotvec()
            half_rot: np.ndarray = Rotation.from_rotvec(0.5 * rotvec).as_quat()
            result.joint_rotation[:, parent_idx, :] = half_rot.copy()
            result.joint_rotation[:, key_idx, :] = half_rot.copy()
        else:
            result.joint_rotation[:, key_idx, :] = bvh.joint_rotation[:, value_idx, :].copy()
    # handle half case..
    for key, value in joint_map.items():
        if value != "half_child":
            continue
        key_idx = result_idx[key]
        child_index = [idx_ for idx_, node in enumerate(result.joint_parents_idx) if node == key_idx][0]
        # print(result.joint_names[key_idx], result.joint_names[child_index])
        rotvec = Rotation(result.joint_rotation[:, child_index, :]).as_rotvec()
        half_rot = Rotation.from_rotvec(0.5 * rotvec).as_quat()
        result.joint_rotation[:, child_index, :] = half_rot.copy()
        result.joint_rotation[:, key_idx, :] = half_rot.copy()
    # here we should adjust hip rotation, considering the offset.
    if True:
        for lr in ["L", "R"]:  # This have bug..
            adjust_name_list = [f"{lr}_Hip_and_Pelvis", f"{lr}_Knee_and_{lr}_Hip"]
            for adjust_name in adjust_name_list:
                adjust_index = result_idx[adjust_name]
                adjust_child = [idx_ for idx_, node in enumerate(result.joint_parents_idx) if node == adjust_index][0]
                # print(result.joint_names[adjust_child])
                stdhuman_child: int = stdhuman_idx[joint_map[hierarchy.joint_names[adjust_child]]]
                res_offset: np.ndarray = hierarchy.joint_offsets[adjust_child]
                res_offset /= np.linalg.norm(res_offset)
                stdhuman_offset: np.ndarray = bvh.joint_offsets[stdhuman_child]
                stdhuman_offset /= np.linalg.norm(stdhuman_offset)
                delta_quat = Rotation(MathHelper.quat_between(res_offset, stdhuman_offset)).inv()
                result.joint_rotation[:, adjust_index, :] = (Rotation(result.joint_rotation[:, adjust_index, :]) * delta_quat).as_quat()

    # here we should modify the ankle joint
    for lr in ["L", "R"]:
        ankle_name = f"{lr}_Ankle_and_{lr}_Knee"
        adjust_index = result_idx[ankle_name]  # adjust the ankle.
        # delta_quat = Rotation.from_rotvec(np.array([-0.225, 0.0, 0.0]))  # This works for human3.6, but is too large for sfu mocap data.
        delta_quat = Rotation.from_rotvec(np.array([-0.14, 0.0, 0.0]))  # This works for sfu mocap data.
        ankle_quat = (Rotation(result.joint_rotation[:, adjust_index, :]) * delta_quat).as_quat()
        # here we should decompose this rotation.
        quat_z, quat_xy = MathHelper.z_decompose(ankle_quat)
        quat_z = Rotation.from_rotvec(0.5 * Rotation(quat_z).as_rotvec())
        ankle_quat = (quat_z * Rotation(quat_xy)).as_quat()
        result.joint_rotation[:, adjust_index, :] = ankle_quat
    
    # here we should make the toe rotation smaller.
    for lr in ["L", "R"]:
        toe_name = f"{lr}_Foot_and_{lr}_Ankle"
        toe_index = result_idx[toe_name]
        result.joint_rotation[:, toe_index, :] = Rotation.from_rotvec(
            np.clip(Rotation(result.joint_rotation[:, toe_index, :]).as_rotvec(), -0.09, 0.09)).as_quat()

    # scale root translation.
    result.joint_translation[:, 0, :] = ratio * bvh.joint_translation[:, 0, :]
    result.recompute_joint_global_info()
    # Maybe we need to run fabr IK to finetune this motion..
    # make sure end site is similar..

    prefix = os.path.split(bvh_fname)[1].split(".")[0]
    prefix = prefix.replace("-mocap-100", "")
    output_fname: str = os.path.join(save_dir, f"{prefix}-{is_mirror}.bvh")
    result = result.resample(20)
    BVHLoader.save(result, output_fname)
    return output_fname
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", output_fname])


if __name__ == "__main__":
    stdhuman_to_smpl("Smpl-world.json", "Tests/CharacterData/WalkF-mocap-100.bvh", None)

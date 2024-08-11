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

import copy
import subprocess
from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Any, Dict, Optional, Tuple, Union

from ..Common.MathHelper import MathHelper
from ..Common.SmoothOperator import smooth_operator, GaussianBase, ButterWorthBase, SmoothMode
from ..pymotionlib.MotionData import MotionData
from ..pymotionlib import BVHLoader

import MotionUtils

class SliceChangeMode(IntEnum):
    front = 0
    behind = 1


class MergeMode(IntEnum):
    only_root = 0
    all_body = 1


def smooth_motion_data(
    bvh: MotionData,
    smooth_type: Union[GaussianBase, ButterWorthBase],
    test_out_fname: Optional[str] = None,
    smooth_position: bool = True,
    smooth_rotation: bool = True,
) -> MotionData:
    # smooth bvh joint rotation, position and recompute...

    # filter joint rotations as rotvec dosen't work.
    # because a[i] * a[i+1] < 0 may occurs.
    # we can not let a[i] := -a[i] directly.

    # filter joint rotations as 3x3 rotation matrix for filter is not plausible...
    # in butter worth filter, for walking motion with sample 0.12 KHz, cut off 5 Hz, the result will not change too much..

    # print(f"Smooth motion data")
    if smooth_rotation:
        for j_idx in range(bvh.num_joints):
            vec6d: np.ndarray = MathHelper.quat_to_vec6d(bvh.joint_rotation[:, j_idx, :]).reshape((-1, 6))
            vec6d_new: np.ndarray = smooth_operator(vec6d, smooth_type)
            bvh._joint_rotation[:, j_idx, :] = MathHelper.flip_quat_by_dot(MathHelper.vec6d_to_quat(vec6d_new.reshape((-1, 3, 2))))

    # filter root global position
    if smooth_position:
        bvh.joint_translation[:, 0, :] = smooth_operator(bvh.joint_translation[:, 0, :], smooth_type)

    bvh.recompute_joint_global_info()

    if test_out_fname:
        BVHLoader.save(bvh, test_out_fname)
        print(f"Save to {test_out_fname}, with num frame = {bvh.num_frames}")

    return bvh


class MotionSliceSmooth:

    @staticmethod
    def get_weight(k: int) -> np.ndarray:
        x: np.ndarray = (np.arange(0, k, dtype=np.float64) + 1) / k
        w: np.ndarray = (2 * x ** 3) - (3 * x ** 2) + 1
        return 1 - w

    @staticmethod
    def _get_off0(piece_: MotionData):
        sub_pos_0: np.ndarray = piece_.joint_position[0, 0, :]  # (3,)
        # sub_rot_0: Rotation = Rotation(MathHelper.y_decompose(piece_.joint_orientation[0, 0, :])[0])
        sub_rot_0 = Rotation(MotionUtils.decompose_rotation_single_fast(
            piece_.joint_orientation[0, 0, :], np.array([0.0, 1.0, 0.0])))
        # sub_rot_0 = Rotation(piece_.joint_orientation[0, 0, :])
        sub_rot_0_inv: Rotation = sub_rot_0.inv()
        return sub_pos_0, sub_rot_0, sub_rot_0_inv

    # Get root pos and rotation in sub piece
    @staticmethod
    def root_to_local(piece_: MotionData):
        sub_pos_0, sub_rot_0, sub_rot_0_inv = MotionSliceSmooth._get_off0(piece_)
        local_sub_pos: np.ndarray = sub_rot_0_inv.apply(piece_.joint_position[:, 0, :] - sub_pos_0[None, :])  # (k, 3)
        local_sub_rot: Rotation = sub_rot_0_inv * Rotation(piece_.joint_orientation[:, 0, :])  # (k, 4)
        return sub_pos_0, sub_rot_0, local_sub_pos, local_sub_rot

    @staticmethod
    def append_motion_smooth(motion_a: MotionData, motion_b: MotionData, smooth_width: int = 10) -> MotionData:
        assert motion_a.num_frames > smooth_width and motion_b.num_frames > smooth_width
        if smooth_width == 0:
            return MotionSliceSmooth.append_motion_simply(motion_a, motion_b)
        motion_a_end = motion_a.sub_sequence(motion_a.num_frames - smooth_width, None)
        motion_b_start = motion_b.sub_sequence(0, smooth_width)
        motion_merge = MotionSliceSmooth.merge_motion(motion_a_end, motion_b_start)
        result = motion_a.sub_sequence(copy=True)
        result.joint_translation[-smooth_width:, 0] = motion_merge.joint_translation[:, 0, :]
        result.joint_rotation[-smooth_width:] = motion_merge.joint_rotation[:]
        result.joint_position[-smooth_width:] = motion_merge.joint_position[:]
        result.joint_orientation[-smooth_width:] = motion_merge.joint_orientation[:]

        # concat the motion
        append_motion_b = motion_b.sub_sequence(smooth_width, None)
        append_motion_b.joint_rotation[:, 0, :] = ((Rotation(motion_merge.joint_rotation[-1, 0]) * Rotation(motion_b_start.joint_rotation[-1, 0]).inv()) * Rotation(append_motion_b.joint_rotation[:, 0, :])).as_quat()
        append_motion_b.joint_translation[:, 0, :] += motion_merge.joint_translation[-1, 0, :] - motion_b_start.joint_translation[-1, 0, :]
        append_motion_b.recompute_joint_global_info()

        name_b_dict = {node: index for (index, node) in enumerate(motion_b.joint_names)}
        map_a_b = [name_b_dict[node] for node in motion_a.joint_names]
        result._joint_rotation = np.concatenate([result._joint_rotation, append_motion_b.joint_rotation[:, map_a_b, :]], axis=0)
        result._joint_translation = np.concatenate([result._joint_translation, append_motion_b.joint_translation], axis=0)
        result._num_frames += append_motion_b.num_frames
        result._joint_position = np.zeros_like(result.joint_translation)
        result._joint_orientation = np.zeros_like(result.joint_rotation)
        result.recompute_joint_global_info()
        return result

    @staticmethod
    def append_motion_simply(motion_a: MotionData, motion_b: MotionData) -> MotionData:
        a_frame: int = motion_a.num_frames
        b_frame: int = motion_b.num_frames
        b_start_ry, b_start_xz = MathHelper.y_decompose(motion_b.joint_rotation[0, 0])
        b_start_ry_inv = Rotation(b_start_ry).inv()
        b_start_vel = motion_b.joint_position[1, 0, :] - motion_b.joint_position[0, 0, :]
        b_start_vel_local = b_start_ry_inv.apply(b_start_vel)

        sub_b = motion_b.sub_sequence(0, 2, copy=False)
        b_start_angvel = sub_b.compute_angular_velocity(True)[0, 0]
        b_start_angvel_local = b_start_ry_inv.apply(b_start_angvel)

        a_end_ry, a_end_xz = MathHelper.y_decompose(motion_a.joint_rotation[-1, 0])
        a_end_ry = Rotation(a_end_ry)
        b_start_vel_cat = a_end_ry.apply(b_start_vel_local)
        b_start_angvel_cat = a_end_ry.apply(b_start_angvel_local)

        # compute the delta rotation..
        b_start_pos_new = motion_a.joint_position[-1, 0] + b_start_vel_cat
        b_start_quat_new = MathHelper.quat_integrate(motion_a.joint_rotation[None, motion_a.num_frames-1, 0, :], b_start_angvel_cat[None, :], 1.0 / motion_b.fps)
        b_start_quat_new_ry = Rotation(MathHelper.y_decompose(b_start_quat_new)[0])

        append_motion_b = motion_b.sub_sequence(copy=True)
        append_motion_b.joint_translation[:, :, [0, 2]] += b_start_pos_new[None, [0, 2]] - append_motion_b.joint_translation[:, 0:1, [0, 2]]
        append_motion_b.joint_rotation[:, 0, :] = ((b_start_quat_new_ry * b_start_ry_inv) * Rotation(append_motion_b.joint_rotation[:, 0, :])).as_quat()

        # resort the joint index..
        name_b_dict = {node: index for (index, node) in enumerate(motion_b.joint_names)}
        map_a_b = [name_b_dict[node] for node in motion_a.joint_names]

        result = motion_a.sub_sequence(copy=True)
        result._joint_rotation = np.concatenate([result._joint_rotation, append_motion_b.joint_rotation[:, map_a_b, :]], axis=0)
        result._joint_translation = np.concatenate([result._joint_translation, append_motion_b.joint_translation], axis=0)
        result._num_frames += append_motion_b.num_frames
        result._joint_position = np.zeros_like(result.joint_translation)
        result._joint_orientation = np.zeros_like(result.joint_rotation)
        result.recompute_joint_global_info()

        return result

    @staticmethod
    def merge_root(motion_a: MotionData, motion_b: MotionData) -> Tuple[np.ndarray, np.ndarray]:
        # make sure length of motion_a == motion_b
        # return: the smoothed root position and orientation
        assert motion_a.num_frames == motion_b.num_frames
        sub_pos_0, sub_rot_0, local_sub_pos, local_sub_rot = MotionSliceSmooth.root_to_local(motion_a)
        modify_pos_0, modify_rot_0, modify_local_pos, modify_local_rot = MotionSliceSmooth.root_to_local(motion_b)
        w = MotionSliceSmooth.get_weight(motion_a.num_frames)[:, None]  # (0 -> 1)
        local_pos_res: np.ndarray = (1 - w) * local_sub_pos + w * modify_local_pos
        height = (1 - w) * sub_pos_0[1] + w * modify_pos_0[1]
        pos_res: np.ndarray = sub_rot_0.apply(local_pos_res) + sub_pos_0[None, :]
        pos_res[:, 1] = height.flatten()
        # pos_res = smooth_operator(pos_res, GaussianBase(3))

        local_rot_res: np.ndarray = MathHelper.slerp(local_sub_rot.as_quat(), modify_local_rot.as_quat(), w)
        quat_res: np.ndarray = (sub_rot_0 * Rotation(local_rot_res)).as_quat()
        return pos_res, quat_res
    
    @staticmethod
    def merge_all_body(motion_a: MotionData, motion_b: MotionData) -> np.ndarray:  # This function works
        # return smoothed joint rotation
        nj = motion_a.num_joints
        w = MotionSliceSmooth.get_weight(motion_a.num_frames)[:, None]
        sub_joint_q: np.ndarray = motion_a.joint_rotation[:, 1:, :].copy().reshape((-1, 4))
        # note: the joint order is not same. we need to resort.
        name_b_dict = {node: index for (index, node) in enumerate(motion_b.joint_names)}
        map_a_b = [name_b_dict[node] for node in motion_a.joint_names[1:]]
        modify_joint_q = motion_b.joint_rotation[:, map_a_b, :].copy().reshape((-1, 4))
        w_dup: np.ndarray = np.tile(w, (1, nj - 1)).reshape((-1, 1))
        joint_res: np.ndarray = MathHelper.slerp(sub_joint_q, modify_joint_q, w_dup)
        joint_res = joint_res.reshape((motion_a.num_frames, nj - 1, 4))
        return joint_res

    @staticmethod
    def merge_motion(motion_a: MotionData, motion_b: MotionData) -> MotionData:
        pos_res, quat_res = MotionSliceSmooth.merge_root(motion_a, motion_b)
        joint_res = MotionSliceSmooth.merge_all_body(motion_a, motion_b)
        result = motion_a.sub_sequence(copy=True)
        result.joint_translation[:, 0, :] = pos_res[:]
        result.joint_rotation[:, 0, :] = quat_res[:]
        result.joint_rotation[:, 1:, :] = joint_res[:]
        result.recompute_joint_global_info()
        return result

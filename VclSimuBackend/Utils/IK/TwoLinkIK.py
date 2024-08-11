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
from scipy.spatial.transform import Rotation
from typing import List, Optional
from ...Common.MathHelper import MathHelper


_debug_mode = True


def two_link_fk(local_quat: np.ndarray, offset: np.ndarray):
    """
    :param local_quat: shape == (num frame, 2, 4)
    :param offset: shape == (*, 2, 3)

    return:
    local_rot_arr: List[Rotation], len == 2
    global_rot_arr: List[Rotation], len == 2
    pos: np.ndarray, shape == (num frame, 3, 3)
    """
    num_frame: int = local_quat.shape[0]

    local_rot_arr: List[Rotation] = [Rotation(local_quat[:, i, :]) for i in range(2)]
    global_rot_arr: List[Rotation] = [local_rot_arr[0], local_rot_arr[0] * local_rot_arr[1]]
    pos: np.ndarray = np.zeros((num_frame, 3, 3), dtype=np.float64)
    for i in range(1, 3):
        pos[:, i, :] = global_rot_arr[i - 1].apply(offset[:, i - 1]) + pos[:, i - 1, :]

    return local_rot_arr, global_rot_arr, pos


def two_link_ik(local_quat: np.ndarray, offset: np.ndarray, target: np.ndarray, backup_axis1: Optional[np.ndarray]) -> np.ndarray:
    """
    two link IK
    joint 0 -- joint 1 -- end pos
    assume position of joint 0 is (0, 0, 0).
    Position of Joint 0 is A, Position of Joint 1 is B, Position of end site is C, Position of target is T
    First, rotate joint 1, to make |AC| == |AT|
    Then, rotate joint 0, rotate AC to AT.

    :param local_quat: input local quaternion. shape == (num_frame, 2, 4)
    :param offset: offset of 2 bones in joint local coordinate. shape == (*, 2, 3)
    :param target: target end position. shape == (num_frame, 3)
    :return: np.ndarray in shape (num_frame, 2, 4)
    """
    # assert input shape
    num_frame: int = local_quat.shape[0]
    assert local_quat.shape[1:] == (2, 4)
    assert offset.shape[-2:] == (2, 3)
    assert target.shape == (num_frame, 3)
    if offset.ndim == 2:
        offset: np.ndarray = offset[None, ...]

    quat_result: np.ndarray = local_quat.copy()  # (num frame, 2, 4)
    local_rot_arr, global_rot_arr, pos = two_link_fk(local_quat, offset)  # do forward kinematics

    # judge no solution case.
    off_len = np.linalg.norm(offset, axis=-1)  # shape == (*, 2)
    end_len = np.linalg.norm(target - pos[:, 0, :], axis=-1)  # (num frame, )
    assert np.all(end_len <= np.sum(off_len, axis=-1))
    assert np.all(end_len >= off_len[:, 0] - off_len[:, 1])

    # Step 1. rotate joint 1, to let |AC| = |AT|

    # calc current cos B
    ba: np.ndarray = pos[:, 0, :] - pos[:, 1, :]  # (num frame, 3)
    unit_ba: np.ndarray = ba / np.linalg.norm(ba, axis=-1, keepdims=True)  # (num frame, 3)
    bc: np.ndarray = pos[:, 2, :] - pos[:, 1, :]  # (num frame, 3)
    unit_bc: np.ndarray = bc / np.linalg.norm(bc, axis=-1, keepdims=True)  # (num frame, 3)
    cos_abc: np.ndarray = np.clip(np.sum(unit_ba * unit_bc, axis=-1), -1, 1)  # (num frame, )
    angle_abc: np.ndarray = np.arccos(cos_abc)  # (num frame, )

    # calc target cos B
    at: np.ndarray = target - pos[:, 0, :]  # (num frame, 3)
    # Law of Cosines: x^2 + y^2 - 2 x y cos \theta = z^2
    at_sqr: np.ndarray = np.sum(at ** 2, axis=-1)  # (num frame, )
    cos_abc_new: np.ndarray = np.clip((np.sum(off_len ** 2, axis=-1) - at_sqr) / (2 * np.prod(off_len, axis=-1)), -1, 1)  # (num frame,)
    angle_abc_new: np.ndarray = np.arccos(cos_abc_new)  # (num frame,)

    delta_angle: np.ndarray = angle_abc_new - angle_abc  # (num frame,)
    axis_1: np.ndarray = np.cross(unit_ba, unit_bc)  # (num frame, 3)
    # when ba and bc are on the same line, the result is not good..
    axis_1: np.ndarray = global_rot_arr[0].inv().apply(axis_1)
    axis_1[np.abs(axis_1) < 1e-12] = 0
    axis_1_norm = np.linalg.norm(axis_1, axis=-1)
    bad_axis1_place = np.array(axis_1_norm < 0.05)
    axis_1[bad_axis1_place, :] = backup_axis1

    delta_q1: np.ndarray = MathHelper.quat_from_axis_angle(axis_1, delta_angle, True)  # (num frame, 4)
    quat_result[:, 1, :] = (Rotation(delta_q1) * local_rot_arr[1]).as_quat()

    # step 2. Rotate the end effector into place
    local_rot_arr, global_rot_arr, pos = two_link_fk(quat_result, offset)  # do forward kinematics

    ac: np.ndarray = pos[:, 2, :] - pos[:, 0, :]  # recompute position of ac
    if _debug_mode:
        print("delta length: ", np.linalg.norm(at, axis=-1) - np.linalg.norm(ac, axis=-1))

    delta_q0: np.ndarray = MathHelper.quat_between(ac, at)
    quat_result[:, 0, :] = (Rotation(delta_q0) * local_rot_arr[0]).as_quat()

    if _debug_mode:
        _, _, pos = two_link_fk(quat_result, offset)  # do forward kinematics
        delta_end = pos[:, -1, :] - target
        print("Delta End", np.max(np.abs(delta_end)))

    return quat_result  # (num frame, 2, 4)

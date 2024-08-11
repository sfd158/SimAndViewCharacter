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
from typing import Tuple, List

from ..Common.MathHelper import MathHelper
from ..pymotionlib.MotionData import MotionData
from ..Utils.MotionUtils import motion_to_facing_coordinate


def calc_noise_to_signal_ratio(sim_motion: MotionData, ref_motion: MotionData, eps: float = 1e-7) -> float:
    """
    [Liu et al. 2015. Improved Samcon] says,

    To quantitatively measure the noise level,
    we define NSR (Noise-to-Signal Ratio)
    for the simulated motion as

    NSR = \frac{\sigma^2 (m - \hat{m}) }{\sigma^2 (\hat{m}) } \times 100

    where m is the simulation trajectory, and \hat{m} is the reference motion.
    We compute the motion variance s using joint rotations.

    However, this evaluation mode is not perfect. we should consider other evaluation modes.
    """
    # assert sim_motion.num_frames == ref_motion.num_frames and sim_motion.num_joints == ref_motion.num_joints
    # if sim_motion.num_frames == ref_motion.num_frames + 1:
    #    sim_motion = sim_motion.sub_sequence(0, -1, copy=False)

    # resort joint order..
    sim_names = sim_motion.joint_names
    ref_names = ref_motion.joint_names
    sim_new_order = [sim_names.index(node) for node in ref_names]
    assert not (-1 in sim_new_order)
    quat_sim: np.ndarray = sim_motion.joint_rotation.copy()
    # decompose to facing coordinate for computing root quaternion...
    _, quat_sim[:, 0, :] = MathHelper.y_decompose(quat_sim[:, 0, :])
    quat_sim: np.ndarray = np.ascontiguousarray(quat_sim[:, sim_new_order, :])

    # print(f"sim num frame = {sim_motion.num_frames}, ref num frame = {ref_motion.num_frames}")
    quat_ref: np.ndarray = ref_motion.joint_rotation.copy()
    _, quat_ref[:, 0, :] = MathHelper.y_decompose(quat_ref[:, 0, :])

    # remove end sites.
    if sim_motion.end_sites:
        joint_idx = np.arange(0, sim_motion.num_joints, dtype=np.uint64)
        joint_idx = np.delete(joint_idx, sim_motion.end_sites)
        quat_sim = np.ascontiguousarray(quat_sim[:, joint_idx, :])
    if ref_motion.end_sites:
        joint_idx = np.arange(0, ref_motion.num_joints, dtype=np.uint64)
        joint_idx = np.delete(joint_idx, ref_motion.end_sites)
        quat_ref = np.ascontiguousarray(quat_ref[:, joint_idx, :])

    rot_sim: Rotation = Rotation(quat_sim.reshape((-1, 4)))
    rot_ref: Rotation = Rotation(quat_ref.reshape((-1, 4)))

    new_shape: Tuple[int, int] = (np.prod(quat_sim.shape[:2]), 3)
    delta_rotvec: np.ndarray = (rot_sim.inv() * rot_ref).as_rotvec().reshape(new_shape)
    ref_rotvec: np.ndarray = rot_ref.as_rotvec().reshape(new_shape)
    delta_var = np.var(delta_rotvec, axis=0)
    ref_var = np.var(ref_rotvec, axis=0)
    ratio: float = np.sum(delta_var / (ref_var + eps)).item()
    return 100 * ratio


calc_nsr = calc_noise_to_signal_ratio


def eval_motion_base(gt_motion: MotionData, pred_motion: MotionData, in_facing_coordinate: bool = True):
    """
    for evaluation, we can eval in facing coordinate
    (as global position maybe wrong..)
    """
    assert gt_motion.fps == pred_motion.fps
    gt_motion = gt_motion.remove_end_sites(copy=True)
    pred_motion = pred_motion.remove_end_sites(copy=True)
    if in_facing_coordinate:
        gt_motion: MotionData = motion_to_facing_coordinate(gt_motion, True)
        pred_motion: MotionData = motion_to_facing_coordinate(pred_motion, True)

    pred_names: List[str] = pred_motion.joint_names
    gt_names = gt_motion.joint_names
    pred_new_order = [pred_names.index(node) for node in gt_names]
    pred_pos: np.ndarray = pred_motion.joint_position[:, pred_new_order, :]
    gt_pos: np.ndarray = gt_motion.joint_position

    return pred_pos, gt_pos


def mpjpe(pred_pos: np.ndarray, gt_pos: np.ndarray):
    """
    evaluate mpjpe loss
    """
    mpjpe: np.ndarray = np.mean(np.linalg.norm(pred_pos - gt_pos, ord=2, axis=-1))
    return mpjpe


def weighted_mpjpe(predicted: np.ndarray, target: np.ndarray, w: np.ndarray):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    that is, different joint should maintain different weight?
    """
    # assert predicted.shape == target.shape
    # assert w.shape[0] == predicted.shape[0]
    # return np.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))


def calc_motion_mpjpe(gt_motion: MotionData, pred_motion: MotionData, in_facing_coordinate: bool = True):
    """
    compute mpjpe loss between ground truth motion and prediction motion
    """
    pred_pos, gt_pos = eval_motion_base(gt_motion, pred_motion, in_facing_coordinate)
    return mpjpe(pred_pos, gt_pos)


def calc_pcp():
    pass

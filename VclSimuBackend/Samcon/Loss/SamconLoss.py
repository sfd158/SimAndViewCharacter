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

import logging
import ModifyODE as ode
import numpy as np
from typing import Dict, Any, Optional

from .SamconLossBase import SamconLossBase
from ..SamconTargetPose import SamconTargetPose
from ...Common.MathHelper import MathHelper
from ...ODESim.ODECharacter import ODECharacter
from ...Utils.Camera.CameraNumpy import CameraParamNumpy
from ...Utils.ComputeCom2d import Joint2dComIgnoreHandToe
from ...Utils.Dataset.StdHuman import stdhuman_with_root_to_unified_index

from MotionUtils import (
    quat_multiply_forward_fast, quat_to_rotvec_single_fast, quat_to_rotvec_fast, quat_from_rotvec_fast, quat_inv_fast,
    decompose_rotation_single_fast, decompose_rotation_single_pair_fast,
    quat_to_vec6d_single_fast, quat_inv_single_fast, quat_apply_single_fast, quat_multiply_forward_single,
    quat_apply_forward_fast
)

class SamconLoss(SamconLossBase):
    """
    Compute samcon loss in open dynamics engine
    """
    def __init__(self, conf: Dict[str, Any]):
        super(SamconLoss, self).__init__(conf)
        self.camera_param: Optional[CameraParamNumpy] = None
        self.camera_2d_calc: Optional[Joint2dComIgnoreHandToe] = None

        self.t: Optional[int] = 0
        self.mod_t: Optional[int] = 0
        self.facing_rot: Optional[np.ndarray] = None  # facing rotation
        self.facing_rot_inv: Optional[np.ndarray] = None  # inverse of facing rotation
        self.com: Optional[np.ndarray] = None  # global position of center of mass
        self.com_velo: Optional[np.ndarray] = None  # global velocity of center of mass
        self.facing_com: Optional[np.ndarray] = None  #
        self.facing_com_velo: Optional[np.ndarray] = None

        self.target: Optional[SamconTargetPose] = None
        self.gt_position_2d: Optional[np.ndarray] = None
        self.confidence_2d: Optional[np.ndarray] = None

        self.character: Optional[ODECharacter] = None
        self.n_iter: Optional[int] = None
        self.root_body: Optional[ode.Body] = None

        self.facing_ang_momentum: Optional[np.ndarray] = None  # angular momentum in facing coordinate

        self.global_joint_pos: Optional[np.ndarray] = None  # global joint position
        self.facing_joint_pos: Optional[np.ndarray] = None  # facing joint position

        self.local_qs: Optional[np.ndarray] = None  # joint local quaternion
        self.parent_qs_inv: Optional[np.ndarray] = None
        self.local_angvel: Optional[np.ndarray] = None  # joint local angular velocity

        self.facing_coor_end_pos: Optional[np.ndarray] = None

        self.camera_joint_pos_2d: Optional[np.ndarray] = None  # project character joint on 2d camera space.
        self.camera_com_2d: Optional[np.ndarray] = None  # center of mass compute in 2d coordinate

        self.joint_subset: Optional[np.ndarray] = None  # joint subset for compute loss..
        self.joint_subset_with_root: Optional[np.ndarray] = None
        self.joint_weights: Optional[np.ndarray] = None  # joint weights for compute loss..

    def set_loss_attrs(self, samcon_target: SamconTargetPose, character: ODECharacter, n_iter: int):
        self.target = samcon_target
        self.camera_param = self.target.camera_param
        if self.target.pose2d is not None:
            self.gt_position_2d = self.target.pose2d.pos2d
            self.confidence_2d = self.target.pose2d.confidence

        self.character = character
        self.n_iter = n_iter
        self.root_body: Optional[ode.Body] = self.character.root_body

        self.camera_2d_calc = Joint2dComIgnoreHandToe().build(character)

        logging.info(f"set n_iter = {self.n_iter}")

    def __call__(self, frame: int) -> float:
        return self.loss(frame)

    def pre_calc_loss(self, frame: int):
        self.t, self.mod_t = frame, frame % self.target.num_frames
        self.facing_rot = self.character.body_info.calc_facing_quat()
        self.facing_rot_inv: Optional[np.ndarray] = quat_inv_single_fast(self.facing_rot)

        self.com: Optional[np.ndarray] = self.character.body_info.calc_center_of_mass()  # compute by all bodies
        self.com_velo: Optional[np.ndarray] = self.character.body_info.calc_velo_com()
        root_pos = self.root_body.PositionNumpy
        root_pos[1] = 0
        self.facing_com: Optional[np.ndarray] = quat_apply_single_fast(self.facing_rot_inv, self.com - root_pos)
        root_vel = self.root_body.LinearVelNumpy
        root_vel[1] = 0
        self.facing_com_velo: Optional[np.ndarray] = quat_apply_single_fast(self.facing_rot_inv, self.com_velo - root_vel)

        self.facing_coor_end_pos: Optional[np.ndarray] = self.character.character_facing_coor_end_pos(self.facing_rot_inv)
        if self.angular_momentum_coef > 0:  # compute by all bodies
            angular_momentum = self.character.body_info.calc_angular_momentum()
            self.facing_ang_momentum: Optional[np.ndarray] = quat_apply_single_fast(self.facing_rot_inv, angular_momentum)

        if self.w_pose > 0:
            _, _, self.local_qs, self.parent_qs_inv = self.character.joint_info.get_parent_child_qs()
            self.local_angvel = self.character.joint_info.get_local_angvels(self.parent_qs_inv)
            if self.joint_subset is not None:
                self.local_qs: Optional[np.ndarray] = np.ascontiguousarray(self.local_qs[self.joint_subset])
                self.parent_qs_inv: Optional[np.ndarray] = np.ascontiguousarray(self.parent_qs_inv[self.joint_subset])
                self.local_angvel: Optional[np.ndarray] = np.ascontiguousarray(self.local_angvel[self.joint_subset])

        if self.w_global_joint_pos_coef > 0 \
            or self.w_facing_joint_pos_coef > 0 \
            or self.w_camera_joint_pos_2d_coef > 0:
            # export joint anchor..
            joint_pos: np.ndarray = self.character.joint_info.get_global_anchor1()  # should compute by subset of joints..
            if self.joint_subset is not None:
                joint_pos: np.ndarray = joint_pos[self.joint_subset]
            self.global_joint_pos = np.concatenate([self.root_body.PositionNumpy[None, :], joint_pos], axis=0)
        if self.w_facing_joint_pos_coef > 0:  # convert joint to facing coordinate
            facing_joint_pos = self.global_joint_pos.copy()
            facing_joint_pos[:, [0, 2]] -= facing_joint_pos[0:1, [0, 2]]
            self.facing_joint_pos = self.facing_rot_inv.apply(facing_joint_pos)

        if self.w_camera_joint_pos_2d_coef > 0 or self.w_camera_com_2d_coef > 0:  # TODO: Add camera param
            # here we should compute in unified 13 joints
            unified_13_joints: np.ndarray = self.global_joint_pos[stdhuman_with_root_to_unified_index, :]
            camera_joint_pos: np.ndarray = self.camera_param.world_to_camera(unified_13_joints)
            self.camera_joint_pos_2d: np.ndarray = self.camera_param.project_to_2d_linear(camera_joint_pos)

        if self.w_camera_com_2d_coef > 0:  # compute com by 2d coordinate
            # we do not know body position of input data.
            # so, we need to infer com of 2d by 2d joint position
            #
            # Note: This method doesn't work!:
            # assume there is no root joint
            # for each joint, the mass can be viewed as 0.5 * (parent body mass + child body mass)
            # in this way, the total mass will not strictly match the initial character
            # but the error is not large
            self.camera_com_2d: Optional[np.ndarray] = self.camera_2d_calc.calc(self.camera_joint_pos_2d)

    def post_calc_loss(self):
        self.facing_rot: Optional[np.ndarray] = None
        self.facing_rot_inv: Optional[np.ndarray] = None
        self.com: Optional[np.ndarray] = None
        self.com_velo: Optional[np.ndarray] = None
        self.facing_com: Optional[np.ndarray] = None
        self.facing_com_velo: Optional[np.ndarray] = None
        self.facing_ang_momentum: Optional[np.ndarray] = None
        self.t: Optional[int] = None
        self.mod_t: Optional[int] = None

        self.global_joint_pos: Optional[np.ndarray] = None
        self.facing_joint_pos: Optional[np.ndarray] = None

        self.local_qs: Optional[np.ndarray] = None
        self.parent_qs_inv: Optional[np.ndarray] = None
        self.local_angvel: Optional[np.ndarray] = None

        self.facing_coor_end_pos: Optional[np.ndarray] = None

        self.camera_com_2d: Optional[np.ndarray] = None

    def loss(self, frame: int) -> float:
        self.pre_calc_loss(frame)
        res = super(SamconLoss, self).loss()
        # self.post_calc_loss()
        return res

    def loss_debug(self, frame: int):
        self.pre_calc_loss(frame)
        res = super(SamconLoss, self).loss_debug()
        self.post_calc_loss()
        return res

    # loss in Samcon 2010 paper
    def pose_loss(self) -> float:
        # d_q + d_v
        # d_q = abs(theta)
        # d_v = Euclidean distance
        # Should use local coordinate
        frame = self.mod_t
        if self.joint_subset is not None:
            b_local_quat: np.ndarray = self.target.pose.locally.quat[frame, self.joint_subset, :]
            b_local_angvel: np.ndarray = self.target.pose.locally.angvel[frame, self.joint_subset, :]
        else:
            b_local_quat: np.ndarray = self.target.pose.locally.quat[frame]
            b_local_angvel: np.ndarray = self.target.pose.locally.angvel[frame]

        dq: np.ndarray = quat_to_rotvec_fast(quat_multiply_forward_fast(self.local_qs, quat_inv_fast(b_local_quat)))[0]
        dv: np.ndarray = np.linalg.norm(self.local_angvel - b_local_angvel, axis=1)

        # self.joint_weights = np.ones_like(self.joint_weights)

        return np.dot(dq + self.dv_coef * dv, self.joint_weights) / self.joint_weights.size

    def global_root_pos_loss(self):
        return np.linalg.norm(self.root_body.PositionNumpy - self.target.pose.dup_root_pos[self.t])

    # def global_root_quat_loss(self):
    #     # dq_ab: torch.Tensor = flip_quat_by_w(quat_multiply(self.root_quat, quat_inv(self.target.pose.root.quat[frame])))
    #     ret = MathHelper.flip_quat_by_w(
    #         (Rotation(self.root_body.getQuaternionScipy()) * Rotation(self.target.pose.dup_root_quat[self.t]).inv()).as_quat())
    #     return np.linalg.norm(Rotation(ret, False, False).as_rotvec())

    # loss in Samcon 2010 paper
    def root_loss(self) -> float:
        frame = self.mod_t
        a_quat: np.ndarray = quat_multiply_forward_single(self.facing_rot_inv, self.root_body.getQuaternionScipy())
        b_quat: np.ndarray = quat_inv_single_fast(self.target.pose.facing_root.quat[frame])
        dq = quat_to_rotvec_single_fast(quat_multiply_forward_single(a_quat, b_quat))[0]

        a_omega: np.ndarray = quat_apply_single_fast(self.facing_rot_inv, self.root_body.getAngularVelNumpy())
        b_omega: np.ndarray = self.target.pose.facing_root.angvel[frame]
        dv = np.linalg.norm(a_omega - b_omega)
        return dq + self.dv_coef * dv

    # loss in Samcon 2010 paper
    def end_effector_loss(self) -> float:
        frame = self.mod_t
        diff: np.ndarray = np.sum(np.abs(self.facing_coor_end_pos - self.target.pose.facing_coor_end.pos[frame]), axis=1)  # (num end joint, )
        return np.dot(diff, self.character.end_joint_weights) / self.character.end_joint_weights.size

    # loss in Samcon 2010 paper
    def balance_loss(self) -> float:
        frame = self.mod_t
        # # d_v = Euclidean distance
        diff_com_v: np.ndarray = np.linalg.norm(self.facing_com_velo - self.target.balance.facing_com_vel[frame])

        a_ry_rc: np.ndarray = self.facing_com - self.facing_coor_end_pos
        b_ry_rc: np.ndarray = self.target.balance.facing_rc[frame]
        diff_rc: np.ndarray = np.linalg.norm(a_ry_rc - b_ry_rc, axis=1)

        # y = 0?
        loss_rc: np.ndarray = np.dot(diff_rc, self.character.end_joint_weights) / (self.character.end_joint_weights.size * self.character.height)
        result = loss_rc + self.dv_coef * diff_com_v

        return result

    def joint_energy_loss(self) -> float:
        return self.character.accum_energy

    def com_loss(self) -> float:
        frame = self.mod_t
        pos_loss = np.linalg.norm(self.facing_com - self.target.balance.facing_com[frame])
        velo_loss = np.linalg.norm(self.facing_com_velo - self.target.balance.facing_com_vel[frame])

        return pos_loss + self.dv_coef * velo_loss

    def angular_momentum_loss(self) -> float:
        return np.linalg.norm(self.facing_ang_momentum - self.target.balance.facing_angular_momentum[self.mod_t])

    def global_joint_pos_loss(self) -> float:
        gt_pos = self.target.pose.all_joint_global.pos
        if self.joint_subset_with_root is not None:
            gt_frame = gt_pos[self.mod_t, self.joint_subset_with_root, :]
        else:
            gt_frame = gt_pos[self.mod_t]
        return np.linalg.norm(self.global_joint_pos - gt_frame)

    def facing_joint_pos_loss(self) -> float:
        gt_pos = self.target.pose.all_joint_facing.pos
        if self.joint_subset_with_root is not None:
            gt_frame = gt_pos[self.mod_t, self.joint_subset_with_root, :]
        else:
            gt_frame = gt_pos[self.mod_t]
        return np.linalg.norm(self.facing_joint_pos - gt_frame)

    def global_com_loss(self) -> float:
        return np.linalg.norm(self.com - self.target.balance.com[self.mod_t])

    # 1. convert global position into camera space
    # 2. project 3d camera position to 2d position
    # we need not to compute 2d pos in real time
    # we can save them at the start.
    def camera_joint_loss_2d(self) -> float:
        return np.linalg.norm(self.gt_position_2d[self.mod_t] - self.camera_joint_pos_2d)

    # note: as mass of toe and hand is smaller,
    # we can ignore them in compute 2d loss
    def camera_com_2d_loss(self) -> float:
        ret = np.linalg.norm(self.target.pose2d.com_2d[self.mod_t] - self.camera_com_2d)
        return ret

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
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, Union

from .TargetPose import TargetPose
from .Utils import BVHJointMap
from .CharacterWrapper import CharacterWrapper, ODECharacter
from ..pymotionlib import BVHLoader
from ..pymotionlib.MotionData import MotionData
from ..Common.MathHelper import MathHelper
from ..Common.SmoothOperator import GaussianBase, ButterWorthBase, smooth_operator
from ..Utils.MothonSliceSmooth import smooth_motion_data


class BVHToTargetBase(CharacterWrapper):
    """
    Convert bvh motion data to target pose
    """
    def __init__(
        self,
        bvh_data: Union[str, MotionData],
        bvh_fps: int,
        character: ODECharacter,
        ignore_root_xz_pos: bool = False,
        bvh_start: Optional[int] = None,
        bvh_end: Optional[int] = None,
        set_init_state_as_offset: bool = False,
        smooth_type: Union[GaussianBase, ButterWorthBase, None] = None,
        flip = None
    ):
        super(BVHToTargetBase, self).__init__(character)
        # load the character as initial state
        self.character.load_init_state()

        # assume there are end joints
        if not self.character.has_end_joint:
            raise ValueError("End Joint required.")

        # Load BVH File
        if isinstance(bvh_data, str):
            self.bvh: MotionData = BVHLoader.load(bvh_data, ignore_root_xz_pos=ignore_root_xz_pos)
        elif isinstance(bvh_data, MotionData):
            self.bvh = bvh_data
        else:
            raise ValueError
        if flip:
            self.bvh.flip(flip)
        self.bvh: MotionData = self.bvh.sub_sequence(bvh_start, bvh_end, copy=False)

        # hack: handle toe for std-human
        if self.character.is_stdhuman_:
            joint_names = self.character.get_joint_names()
            toe_axis = np.tile(np.array([[1.0, 0.0, 0.0]]), (self.bvh.num_frames, 1))
            for lr in ["l", "r"]:
                toe_name = f"{lr}ToeJoint"
                toe_index: int = joint_names.index(toe_name)
                toe_joint = self.joints[toe_index]
                toe_limit = np.array([toe_joint.AngleLoStop, toe_joint.AngleHiStop])
                # compute hinge angle of toe joint from quaternion
                bvh_index = self.bvh.joint_names.index(toe_name)
                toe_quat: np.ndarray = self.bvh.joint_rotation[:, bvh_index, :]
                toe_quat = MathHelper.flip_quat_by_w(toe_quat)
                angle: np.ndarray = np.linalg.norm(Rotation(toe_quat).as_rotvec(), axis=-1) * np.sign(np.sum(toe_quat[:, 0:3], axis=-1))
                angle = np.clip(angle, *toe_limit)
                self.bvh.joint_rotation[:, bvh_index, :] = MathHelper.quat_from_axis_angle(toe_axis, angle)

            self.bvh.recompute_joint_global_info()

        if set_init_state_as_offset:
            self.set_init_state_as_bvh_offset()

        self.smooth_bvh: Optional[MotionData] = None
        if smooth_type is not None:
            self.do_smooth(smooth_type)  # if we don't use smooth, inverse dynamics result will be very noisy..
            self.smooth_bvh.resample(int(bvh_fps))

        if int(bvh_fps) != self.bvh.fps:
            self.bvh.resample(int(bvh_fps))  # resample

        self.mapper = BVHJointMap(self.bvh, character)

        # Get Raw anchor
        anchor_res = self.character.get_raw_anchor()
        self.raw_anchor1: np.ndarray = anchor_res[0].reshape((-1, 3))  # (joint, 3)

        if self.character.root_init_info is not None:
            self.root_body_offset = self.character.init_root_body_pos() - self.character.root_init_info.pos
        else:
            self.root_body_offset = None

    @property
    def bvh_children(self):
        return self.mapper.bvh_children

    @property
    def character_to_bvh(self):
        return self.mapper.character_to_bvh

    @property
    def end_to_bvh(self):
        return self.mapper.end_to_bvh

    @property
    def bvh_joint_cnt(self):
        """
        bvh joint count
        """
        return len(self.bvh.joint_names)

    @property
    def frame_cnt(self):
        """
        frame count
        """
        return self.bvh.num_frames

    @property
    def character_jcnt(self):
        return len(self.character.joint_info)

    def set_bvh_offset(self, pos_offset: Optional[np.ndarray] = None, quat_offset: Optional[np.ndarray] = None):
        if pos_offset is not None:
            assert pos_offset.shape == (3,)
            delta_pos: np.ndarray = pos_offset - self.bvh.joint_position[0, 0, :]
            self.bvh._joint_translation[:, 0, :] += delta_pos

        if quat_offset is not None:
            assert quat_offset.shape == (4,)
            dq = (Rotation(quat_offset, normalize=False, copy=False)
                  * Rotation(self.bvh.joint_rotation[0, 0, :], copy=False).inv()).as_quat()
            dq, _ = MathHelper.facing_decompose(dq)
            dr = Rotation(dq, copy=False)
            self.bvh._joint_rotation[:, 0, :] = (dr * Rotation(self.bvh.joint_rotation[:, 0, :], copy=False)).as_quat()
            trans_0 = self.bvh.joint_translation[0, 0, :]
            d_trans = self.bvh.joint_translation[:, 0, :] - trans_0
            self.bvh.joint_translation[:, 0, :] = trans_0 + dr.apply(d_trans)

        if pos_offset is not None or quat_offset is not None:
            self.bvh.recompute_joint_global_info()

    def set_init_state_as_bvh_offset(self):
        self.set_bvh_offset(self.character.init_root_body_pos(), self.character.init_root_quat())

    # def refine_hinge_rotation(self):
    #     """
    #     Sometimes, elbow and knee joint may have rotation along other axies..
    #     We should remove these rotations...
    #     """
    #     hinge_id = self.joint_info.hinge_id()
    #     for hid in hinge_id:
    #         axis = self.joints[hid].Axis1RawNumpy
    #         bvh_idx = self.character_to_bvh[hid]
    #         qa, q_noise = MathHelper.axis_decompose(self.bvh.joint_rotation[:, bvh_idx, :], np.array(axis))
    #         self.bvh._joint_rotation[:, bvh_idx, :] = qa

    #     self.bvh.recompute_joint_global_info()

    def do_smooth(self, smooth_type: Union[GaussianBase, ButterWorthBase], test_out_fname: Optional[str] = None):
        self.smooth_bvh: Optional[MotionData] = smooth_motion_data(copy.deepcopy(self.bvh), smooth_type, test_out_fname)

    def init_target_global(self, target: TargetPose, vel_forward: bool = False):
        """
        initialize target in global coordinate
        """
        #  (frame, joint, 3 or 4)
        target.globally.quat = self.bvh.joint_orientation[:, self.character_to_bvh, :]
        target.globally.pos = self.bvh.joint_position[:, self.character_to_bvh, :]

        global_lin_vel = self.bvh.compute_linear_velocity(vel_forward)
        target.globally.linvel = global_lin_vel[:, self.character_to_bvh, :]
        global_ang_vel = self.bvh.compute_angular_velocity(vel_forward)
        target.globally.angvel = global_ang_vel[:, self.character_to_bvh, :]

        return global_lin_vel, global_ang_vel

    def init_target_root(
        self,
        target: TargetPose,
        global_lin_vel: np.ndarray,
        global_ang_vel: np.ndarray
    ):
        """
        Convert bvh to root info
        This method is OK with root joint
        """
        target.root.pos = self.bvh.joint_position[:, 0, :]
        target.root.quat = self.bvh.joint_orientation[:, 0, :]
        target.root.linvel = global_lin_vel[:, 0, :]
        target.root.angvel = global_ang_vel[:, 0, :]

        # compute root body info..
        # there is offset between root joint and root body
        if self.root_body_offset is not None:
            target.root_body.pos = Rotation(target.root.quat).apply(self.root_body_offset) + target.root.pos
            target.root_body.linvel = MathHelper.vec_diff(target.root_body.pos, False, self.bvh.fps)
            target.root_body.quat = target.root.quat
            target.root_body.angvel = target.root.angvel
        else:
            target.root_body = target.root

    def init_facing_root(
        self,
        target: TargetPose,
        global_lin_vel: np.ndarray,
        global_ang_vel: np.ndarray
    ):
        target.facing_root.pos = MathHelper.vec_axis_to_zero(self.bvh.joint_position[:, 0, :], [0, 2])
        ry, facing = MathHelper.facing_decompose(self.bvh.joint_orientation[:, 0, :])
        target.facing_root.quat = facing
        target.facing_root.linvel = MathHelper.vec_axis_to_zero(global_lin_vel[:, :], [0, 2])
        target.facing_root.angvel = Rotation(ry, copy=False).apply(global_ang_vel[:, :])

    def init_locally_coor(self, target: TargetPose, vel_forward: bool = False):
        """
        convert bvh local rotation to target
        """
        target.locally.quat = self.bvh.joint_rotation[:, self.character_to_bvh, :]
        local_ang_vel: np.ndarray = self.bvh.compute_rotational_speed(vel_forward)
        target.locally.angvel = local_ang_vel[:, self.character_to_bvh, :]

    def init_end(
        self,
        target: TargetPose
    ):
        """
        initialize end joints' target info
        """
        target.end.pos = self.bvh.joint_position[:, self.end_to_bvh, :]

    @staticmethod
    def calc_facing_quat(target: TargetPose):
        target.facing_quat, _ = MathHelper.facing_decompose(target.root.quat)

    def init_facing_coor_end(self, target: TargetPose):
        """
        convert bvh end sites to target in facing coordinate
        """
        root_pos = MathHelper.vec_axis_to_zero(target.root.pos, 1)
        ry_rot_inv = Rotation(target.facing_quat).inv()
        target.facing_coor_end.pos = np.copy(target.end.pos)
        for end_idx in range(len(self.end_to_bvh)):
            target.facing_coor_end.pos[:, end_idx, :] = \
                ry_rot_inv.apply(target.end.pos[:, end_idx, :] - root_pos)
        target.facing_coor_end.pos = target.facing_coor_end.pos

    def init_global_child_body(self, target: TargetPose, vel_forward: bool = False):
        """
        convert bvh global info to target body in global coordinate
        """
        #

        target.child_body.pos = np.zeros((self.frame_cnt, self.character_jcnt, 3))
        target.child_body.quat = np.copy(target.globally.quat)
        for jidx in range(self.character_jcnt):
            rot = Rotation(target.globally.quat[:, jidx, :], copy=False)
            target.child_body.pos[:, jidx, :] = \
                target.globally.pos[:, jidx, :] - rot.apply(self.raw_anchor1[jidx])
        target.child_body.linvel = MathHelper.vec_diff(target.child_body.pos, vel_forward, self.bvh.fps)

        # Calc Global Angular Velocity
        target.child_body.angvel = target.globally.angvel.copy()

    def init_all_joint_and_body(self, target: TargetPose):
        """
        joint with root global and local info, all child body info
        """
        facing_rot_inv = Rotation(target.facing_quat)
        if target.globally is not None:
            # build all joint global
            if self.joint_info.has_root:
                target.all_joint_global = target.globally.deepcopy()
            else:
                target.all_joint_global.pos = np.concatenate([target.root.pos[:, None, :], target.globally.pos], axis=1)
                target.all_joint_global.quat = np.concatenate([target.root.quat[:, None, :], target.globally.quat], axis=1)

            # build facing joint pos and quat..
            target.all_joint_facing = copy.deepcopy(target.all_joint_global)
            target.all_joint_facing.pos[:, :, [0, 2]] -= target.all_joint_facing.pos[:, 0:1, [0, 2]]
            for index in range(target.all_joint_global.pos.shape[1]):
                target.all_joint_facing.pos[:, index, :] = facing_rot_inv.apply(target.all_joint_facing.pos[:, index, :])
                target.all_joint_facing.quat[:, index, :] = (facing_rot_inv * Rotation(target.all_joint_global.quat[:, index, :])).as_quat()

        # build all joint local
        if target.locally is not None:
            if self.joint_info.has_root:
                target.all_joint_local = target.locally.deepcopy()
            else:
                res = target.all_joint_local
                # res.locally.pos = np.concatenate([pose.root.pos[:, None, :], pose.locally.pos], axis=1)
                res.quat = np.concatenate([target.root.quat[:, None, :], target.locally.quat], axis=1)
                res.angvel = np.concatenate([target.root.angvel[:, None, :], target.locally.angvel], axis=1)
                # res.locally.linvel = np.concatenate([pose.root.linvel[:, None, :], pose.locally.linvel], axis=1)

        if not self.joint_info.has_root:
            if target.child_body is not None:  # get body info correspond to character
                res = target.character_body
                cat_func = self.character.cat_root_child_body_value
                res.pos = cat_func(target.root_body.pos, target.child_body.pos)
                res.quat = cat_func(target.root_body.quat, target.child_body.quat)
                res.rot_mat = Rotation(res.quat.reshape((-1, 4)), copy=False).as_matrix().reshape(
                    res.quat.shape[:-1] + (3, 3))
                res.linvel = cat_func(target.root_body.linvel, target.child_body.linvel)
                res.angvel = cat_func(target.root_body.angvel, target.child_body.angvel)

            if target.child_body is not None:  # child body info (with root body..)
                res = target.all_child_body
                res.pos = np.concatenate([target.root_body.pos[:, None, :], target.child_body.pos], axis=1)
                res.quat = np.concatenate([target.root_body.quat[:, None, :], target.child_body.quat], axis=1)
                res.linvel = np.concatenate([target.root_body.linvel[:, None, :], target.child_body.linvel], axis=1)
                res.angvel = np.concatenate([target.root_body.angvel[:, None, :], target.child_body.angvel], axis=1)
        else:
            if target.child_body is not None:
                res = target.character_body
                res.pos = target.child_body.pos[:, self.character.joint_to_child_body, :]
                res.quat = target.child_body.quat[:, self.character.joint_to_child_body, :]
                res.rot_mat = Rotation(res.quat.reshape((-1, 4)), copy=False).as_matrix().reshape(
                    res.quat.shape[:-1] + (3, 3))
                res.linvel = target.child_body.linvel[:, self.character.joint_to_child_body, :]
                res.angvel = target.child_body.angvel[:, self.character.joint_to_child_body, :]

    def init_smooth_target(self, target: Optional[TargetPose] = None, vel_forward: bool = False) -> TargetPose:
        res = self.init_target(target, self.smooth_bvh, vel_forward)
        res.smoothed = True
        return res

    def only_init_global_target(self, vel_forward: bool = False) -> TargetPose:
        target = TargetPose()
        global_vel = self.init_target_global(target, vel_forward)
        self.init_target_root(target, *global_vel)
        self.init_global_child_body(target, vel_forward)
        return target

    def init_target(
        self,
        target: Optional[TargetPose] = None,
        bvh: Optional[MotionData] = None,
        vel_forward: bool = False,
        ) -> TargetPose:
        """
        Note:
        in ODE engine,
        a_t = F(x_t, v_t),
        v_{t + 1} = v_{t} + h * a_{t}
        x_{t + 1} = x_{t} + h * v_{t + 1}
        """
        if target is None:
            target = TargetPose()

        bvh_backup = self.bvh
        if bvh is not None:
            self.bvh = bvh

        # Calc Target Pose. The index is character's joint index
        global_vel = self.init_target_global(target, vel_forward)
        self.init_target_root(target, *global_vel)
        self.init_global_child_body(target, vel_forward)

        self.init_locally_coor(target, vel_forward)

        self.calc_facing_quat(target)

        self.init_facing_root(target, target.root.linvel, target.root.angvel)
        self.init_end(target)
        self.init_facing_coor_end(target)
        self.init_all_joint_and_body(target)

        target.num_frames = self.bvh.num_frames
        target.fps = self.bvh.fps
        target.to_continuous()

        self.bvh = bvh_backup
        return target


    def calc_posi_by_rot(self, quat, root_posi):
        """
        calculate joints' global position from their global rotation
        """
        parent_idx_ = self.bvh.joint_parents_idx # 23
        joint_offset_ = self.bvh.joint_offsets # 23
        joint_num = len(parent_idx_)

        rot = np.zeros([joint_num, 4]) # 23
        for i in range(len(quat)):
            rot[self.character_to_bvh[i]] = quat[i]

        joint_posi = np.zeros([joint_num, 3])

        for i in range(joint_num):
            if parent_idx_[i] == -1:
                joint_posi[i, :] = root_posi
            else:
                joint_posi[i, :] = joint_posi[parent_idx_[i], :] + Rotation.from_quat(rot[parent_idx_[i]]).apply(joint_offset_[i])

        return joint_posi # 23

    def calc_body_posi_by_rot(self, quat, joint_posi):
        joint_posi = joint_posi[self.character_to_bvh] # 18

        body_posi = np.zeros((self.character_jcnt, 3))
        body_quat = quat
        for jidx in range(self.character_jcnt):
            rot = Rotation(quat[jidx, :], copy=False)
            body_posi[jidx, :] = \
                joint_posi[jidx, :] - rot.apply(self.raw_anchor1[jidx])

        return body_posi, body_quat
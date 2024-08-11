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

from VclSimuBackend.Utils.Camera.CameraNumpy import CameraParamNumpy
from ..Common.MathHelper import MathHelper
from ..ODESim.TargetPose import TargetPose, SetTargetToCharacter
from ..ODESim.CharacterWrapper import ODECharacter
from ..ODESim.BVHToTarget import BVHToTargetBase
from ..pymotionlib.MotionData import MotionData


class SamconBalanceTarget:

    __slots__ = (
        "com", "com_vel", "facing_com", "facing_com_vel",
        "facing_rc", "facing_angular_momentum", "kinematic_energy")

    def __init__(self, num_frame: Optional[int] = None, num_end_joint: Optional[int] = None):
        # position of Center of Mass globally
        self.com: Optional[np.ndarray] = None  # shape = (nframe, 3)

        # velocity of Center of Mass globally
        self.com_vel: Optional[np.ndarray] = None  # shape = (nframe, 3)

        # position of Center of Mass in facing coordinate
        self.facing_com: Optional[np.ndarray] = None  # shape = (nframe, 3)

        # Velocity of Center of Mass in facing coordinate
        self.facing_com_vel: Optional[np.ndarray] = None  # shape = (nframe, 3)

        # in Samcon 2010 paper.
        self.facing_rc: Optional[np.ndarray] = None  # shape = (nframe, end joint, 3)

        # self.facing_root: Optional[np.ndarray]
        self.facing_angular_momentum: Optional[np.ndarray] = None

        self.kinematic_energy: Optional[np.ndarray] = None
        # self.smooth_energy: Optional[np.ndarray] = None

        if num_frame is not None and num_end_joint is not None:
            self.resize(num_frame, num_end_joint)

    def duplicate(self, times: int = 1):
        res = SamconBalanceTarget()
        if self.com is not None:
            res.com = np.concatenate([self.com] * times, axis=0)
        if self.com_vel is not None:
            res.com_vel = np.concatenate([self.com_vel] * times, axis=0)
        if self.facing_com is not None:
            res.facing_com = np.concatenate([self.facing_com] * times, axis=0)
        if self.facing_com_vel is not None:
            res.facing_com_vel = np.concatenate([self.facing_com_vel] * times, axis=0)
        if self.facing_rc is not None:
            res.facing_rc = np.concatenate([self.facing_rc] * times, axis=0)
        if self.facing_angular_momentum is not None:
            res.facing_angular_momentum = np.concatenate([self.facing_angular_momentum] * times, axis=0)
        if self.kinematic_energy is not None:
            res.kinematic_energy = np.concatenate([self.kinematic_energy] * times, axis=0)

        return res

    def append(self, other):
        if self.com is None:
            self.com: Optional[np.ndarray] = copy.deepcopy(other.com)
        else:
            self.com = np.concatenate([self.com, other.com], axis=0)

        if self.com_vel is None:
            self.com_vel: Optional[np.ndarray] = copy.deepcopy(other.com_vel)
        else:
            self.com_vel = np.concatenate([self.com_vel, other.com_vel], axis=0)

        if self.facing_com is None:
            self.facing_com: Optional[np.ndarray] = copy.deepcopy(other.facing_com)
        else:
            self.facing_com = np.concatenate([self.facing_com, other.facing_com], axis=0)

        if self.facing_com_vel is None:
            self.facing_com_vel = copy.deepcopy(other.facing_com_vel)
        else:
            self.facing_com_vel = np.concatenate([self.facing_com_vel, other.facing_com_vel], axis=0)

        if self.facing_rc is None:
            self.facing_rc = copy.deepcopy(other.facing_rc)
        else:
            self.facing_rc = np.concatenate([self.facing_rc, other.facing_rc], axis=0)

        if self.facing_angular_momentum is None:
            self.facing_angular_momentum = copy.deepcopy(other.facing_angular_momentum)
        else:
            self.facing_angular_momentum = np.concatenate([self.facing_angular_momentum, other.facing_angular_momentum], axis=0)

        if self.kinematic_energy is None:
            self.kinematic_energy = copy.deepcopy(other.kinematic_energy)
        else:
            self.kinematic_energy = np.concatenate([self.kinematic_energy, other.kinematic_energy], axis=0)

    def resize(self, num_frame: int, num_end_joint: int):
        self.com = np.zeros((num_frame, 3))
        self.com_vel = np.zeros((num_frame, 3))
        self.facing_com = np.zeros((num_frame, 3))
        self.facing_com_vel: Optional[np.ndarray] = None
        self.facing_rc = np.zeros((num_frame, num_end_joint, 3))  # P_{com} - Pi in ry coordinate
        self.facing_angular_momentum = np.zeros((num_frame, 3))

    def check_com(self, character: ODECharacter, target_pose: TargetPose):
        tar_set = SetTargetToCharacter(character, target_pose)
        for i in range(target_pose.num_frames):
            tar_set.set_character_byframe(i)
            dfacing_compos = np.linalg.norm(character.character_facing_coor_com() - self.facing_com[i])
            assert dfacing_compos < 1e-10
            dfacing_comvel = np.linalg.norm(character.character_facing_coor_com_velo() - self.facing_com_vel[i])
            assert dfacing_comvel < 1e-10
        print("Check com ok.")
        exit(0)

    def check_facing_rc(self, character: ODECharacter, target_pose: TargetPose):
        tar_set = SetTargetToCharacter(character, target_pose)
        for i in range(target_pose.num_frames):
            tar_set.set_character_byframe(i)
            end_a = character.end_joint.get_global_pos()
            end_b = target_pose.end.pos[i]
            delta_end = np.linalg.norm(end_b - end_a)
            print(i, delta_end)
        print("Check facing rc OK")
        exit(0)

    def init_target_com(self, character: ODECharacter, target_pose: TargetPose):
        mass = character.body_info.mass_val.reshape((1, -1, 1))  # (1, num body, 1)
        body_pos: np.ndarray = target_pose.character_body.pos  # (frame, nb, 3)
        body_velo: np.ndarray = target_pose.character_body.linvel  # (frame, nb, 3)
        self.com = np.sum(mass * body_pos, axis=1) / character.body_info.sum_mass  # CoM Position of each frame
        self.com_vel = np.sum(mass * body_velo, axis=1) / character.body_info.sum_mass

        rot_y_inv = Rotation(target_pose.facing_quat, copy=False).inv()
        root_pos = MathHelper.vec_axis_to_zero(target_pose.root.pos, 1)  # (frame, 3)
        self.facing_com = rot_y_inv.apply(self.com - root_pos)

        # Calc Velocity of Center of Mass
        self.facing_com_vel = rot_y_inv.apply(self.com_vel - MathHelper.vec_axis_to_zero(target_pose.root.linvel, 1))

        end_pos = target_pose.facing_coor_end.pos
        self.facing_rc = self.facing_com[:, None, :] - end_pos
        tar_set = SetTargetToCharacter(character, target_pose)

        # Should Reset
        tar_set.set_character_byframe(0)

    def __len__(self):
        return self.com.shape[0]

    def num_end_joint(self):
        return self.facing_rc.shape[1]

    def sub_seq(self, start: int = 0, end: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True):
        assert skip is None or isinstance(skip, int)
        res = SamconBalanceTarget()
        if end is None:
            end = len(self)
        if end == 0:
            return res

        piece = slice(start, end, skip)
        if self.com is not None:
            res.com = self.com[piece].copy() if is_copy else self.com[piece]
        if self.com_vel is not None:
            res.com_vel = self.com_vel[piece].copy() if is_copy else self.com_vel[piece]
        if self.facing_com is not None:
            res.facing_com = self.facing_com[piece].copy() if is_copy else self.facing_com[piece]
        if self.facing_com_vel is not None:
            res.facing_com_vel = self.facing_com_vel[piece].copy() if is_copy else self.facing_com_vel[piece]
        if self.facing_rc is not None:
            res.facing_rc = self.facing_rc[piece].copy() if is_copy else self.facing_rc[piece]
        if self.facing_angular_momentum is not None:
            res.facing_angular_momentum = self.facing_angular_momentum[piece].copy() \
                if is_copy else self.facing_angular_momentum[piece]
        if self.kinematic_energy is not None:
            res.kinematic_energy = self.kinematic_energy[piece].copy() \
                if is_copy else self.kinematic_energy[piece]
        return res

    def to_continuous(self):
        if self.com is not None:
            self.com = np.ascontiguousarray(self.com)
        if self.com_vel is not None:
            self.com_vel = np.ascontiguousarray(self.com_vel)
        if self.facing_com is not None:
            self.facing_com = np.ascontiguousarray(self.facing_com)
        if self.facing_com_vel is not None:
            self.facing_com_vel = np.ascontiguousarray(self.facing_com_vel)
        if self.facing_rc is not None:
            self.facing_rc = np.ascontiguousarray(self.facing_rc)
        if self.facing_angular_momentum is not None:
            self.facing_angular_momentum = np.ascontiguousarray(self.facing_angular_momentum)
        if self.kinematic_energy is not None:
            self.kinematic_energy = np.ascontiguousarray(self.kinematic_energy)
        return self

class TargetPose2d:
    __slots__ = ("pos2d", "confidence", "com_2d")
    def __init__(self) -> None:
        self.pos2d: Optional[np.ndarray] = None
        self.confidence: Optional[np.ndarray] = None
        self.com_2d: Optional[np.ndarray] = None  # compute center of mass in 2d camera coordinate

    def sub_seq(self, start: int = 0, end: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True):
        assert skip is None or isinstance(skip, int)
        res = TargetPose2d()
        if end is None:
            end = len(self)
        if end == 0:
            return res
        piece = slice(start, end, skip)
        if self.pos2d is not None:
            res.pos2d = self.pos2d[piece].copy() if is_copy else self.pos2d[piece]
        if self.confidence is not None:
            res.confidence = self.confidence[piece].copy() if is_copy else self.confidence[piece]
        if self.com_2d is not None:
            res.com_2d = self.com_2d[piece].copy() if is_copy else self.com_2d[piece]
        return res

    def to_continuous(self):
        if self.pos2d is not None:
            self.pos2d: Optional[np.ndarray] = np.ascontiguousarray(self.pos2d)

        if self.confidence is not None:
            self.confidence: Optional[np.ndarray] = np.ascontiguousarray(self.confidence)

        if self.com_2d is not None:
            self.com_2d: Optional[np.ndarray] = np.ascontiguousarray(self.com_2d)

        return self


class SamconTargetPose:

    __slots__ = ("pose", "balance", "pose2d", "pose2d_unified", "camera_param")

    def __init__(self):
        self.pose: Optional[TargetPose] = None
        self.balance: Optional[SamconBalanceTarget] = None
        self.pose2d: Optional[TargetPose2d] = None  # in std-human format
        self.pose2d_unified: Optional[TargetPose2d] = None  # in unified 13 joint
        self.camera_param: Optional[CameraParamNumpy] = None

    def __len__(self) -> int:
        return self.num_frames

    @property
    def num_frames(self) -> int:
        return self.pose.num_frames

    @property
    def fps(self):
        return self.pose.fps

    @property
    def smoothed(self) -> bool:
        return self.pose.smoothed

    def to_continuous(self):
        if self.pose is not None:
            self.pose.to_continuous()

        if self.balance is not None:
            self.balance.to_continuous()

        if self.pose2d is not None:
            self.pose2d.to_continuous()

        if self.pose2d_unified is not None:
            self.pose2d_unified.to_continuous()

        return self

    # def clear_for_samcon_loss(self):
    #    self.pose.globally = None

    def compute_angular_momentum(self, character: ODECharacter) -> np.ndarray:
        num_frames = self.num_frames
        num_body = len(character.bodies)
        dcm: np.ndarray = Rotation(self.pose.character_body.quat.reshape((-1, 4))).as_matrix().reshape((num_frames, num_body, 3, 3))
        inertia: np.ndarray = character.body_info.calc_body_init_inertia().reshape((1, num_body, 3, 3))
        inertia: np.ndarray = dcm @ inertia @ dcm.transpose((0, 1, 3, 2))
        angular_momentum: np.ndarray = (inertia @ self.pose.character_body.angvel[..., None]).reshape((num_frames, num_body, 3))

        com_to_body: np.ndarray = self.pose.character_body.pos - self.balance.com.reshape((num_frames, 1, 3))  # (frame, body, 3)
        body_mass: np.ndarray = character.body_info.mass_val.reshape((1, -1, 1))
        body_linear_momentum: np.ndarray = self.pose.character_body.linvel * body_mass

        angular_momentum: np.ndarray = angular_momentum + np.cross(com_to_body, body_linear_momentum, axis=-1)  # (frame, body, 3)
        angular_momentum: np.ndarray = np.sum(angular_momentum, axis=1)  # (frame, 3)

        # check compute angular momentum..

        return angular_momentum

    def compute_facing_angular_momentum(self, character: ODECharacter):
        angular_momentum: np.ndarray = self.compute_angular_momentum(character)
        facing_rot_inv: Rotation = Rotation(self.pose.facing_quat).inv()
        self.balance.facing_angular_momentum = facing_rot_inv.apply(angular_momentum)

    def calc_kinematic_engery(self, character: ODECharacter):
        character_body = self.pose.character_body
        velo = character_body.linvel
        mass = character.body_info.mass_val[None, ...]  # (1, num body)
        eng_1 = 0.5 * np.sum(mass * np.sum(velo ** 2, axis=-1), axis=-1)  # (num frame,)

        init_inertia: np.ndarray = character.body_info.calc_body_init_inertia()[None, ...]  # (1, body, 3, 3)

        q: np.ndarray = character_body.quat.reshape((-1, 4))  # (num frame * num body, 4)
        mat: np.ndarray = Rotation(q, copy=False).as_matrix().reshape((self.num_frames, -1, 3, 3))
        inertia: np.ndarray = mat @ init_inertia @ mat.transpose((0, 1, 3, 2))  # (num frame, num body, 3, 3)
        omega: np.ndarray = character_body.angvel[..., None]  # (num frame, num body, 3, 1)
        eng_2: np.ndarray = 0.5 * omega.transpose((0, 1, 3, 2)) @ inertia @ omega  # (frame, body, 1, 1)
        eng_2: np.ndarray = np.sum(eng_2.reshape(self.num_frames, len(character.bodies)), axis=-1)  # (frame, )
        eng: np.ndarray = eng_1 + eng_2  # (num frame, )
        self.balance.kinematic_energy = eng

        return self.balance.kinematic_energy

    def test_calc_angular_momentum(self, character: ODECharacter):
        tar_set = SetTargetToCharacter(character, self.pose)
        for i in range(self.num_frames):
            tar_set.set_character_byframe(i)
            eng = character.character_facing_coord_angular_momentum()
            assert np.max(np.abs(eng - self.balance.facing_angular_momentum[i])) < 1e-10
        print("Check angular momentum OK")
        exit(0)

    def test_calc_kinematic_engery(self, character: ODECharacter):  # Test OK
        tar_set = SetTargetToCharacter(character, self.pose)
        for i in range(self.num_frames):
            tar_set.set_character_byframe(i)
            eng = character.calc_kinetic_energy()
            assert abs(eng - self.balance.kinematic_energy[i]) < 1e-10
        print("Check kinematic engery OK")
        exit(0)

    def sub_seq(self, start: int = 0, end: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True):
        assert skip is None or isinstance(skip, int)
        res = SamconTargetPose()
        if end is None:
            end = self.num_frames
        if end == 0:
            return res
        if self.pose is not None:
            res.pose = self.pose.sub_seq(start, end, skip, is_copy)
        if self.balance is not None:
            res.balance = self.balance.sub_seq(start, end, skip, is_copy)
        if self.pose2d is not None:
            res.pose2d = self.pose2d.sub_seq(start, end, skip, is_copy)
        if self.pose2d_unified is not None:
            res.pose2d_unified = self.pose2d_unified.sub_seq(start, end, skip, is_copy)

        res.camera_param = copy.deepcopy(self.camera_param) if is_copy else self.camera_param
        return res

    def duplicate(self, times: int = 1):
        res = SamconTargetPose()
        if self.pose is not None:
            res.pose = self.pose.duplicate(times)
        if self.balance is not None:
            res.balance = self.balance.duplicate(times)
        if self.pose2d is not None:
            res.pose2d = self.pose2d.duplicate(times)
        if self.pose2d_unified is not None:
            res.pose2d_unified = self.pose2d_unified.duplicate(times)

        return res

    def append(self, other):
        if self.pose is None:
            self.pose: Optional[TargetPose] = copy.deepcopy(other.pose)
        else:
            self.pose.append(other.pose)

        if self.balance is None:
            self.balance: Optional[SamconBalanceTarget] = copy.deepcopy(other.balance)
        else:
            self.balance.append(other.balance)

        if self.pose2d is None:
            self.pose2d: Optional[TargetPose2d] = copy.deepcopy(other.pose2d)
        else:
            self.pose2d.append(other.pose2d)

        if self.pose2d_unified is not None:
            self.pose2d_unified: Optional[TargetPose2d] = copy.deepcopy(other.pose2d_unified)
        else:
            self.pose2d_unified.append(other.pose2d_unified)

        return self

    @staticmethod
    def load2(
        bvh_data: Union[str, MotionData],
        character: ODECharacter, sim_fps: int,
        bvh_start: Optional[int] = None,
        bvh_end: Optional[int] = None,
        load_balance: bool = True,
        return_loader: bool = False
    ):
        target = SamconTargetPose()
        if isinstance(bvh_data, TargetPose):
            target.pose = bvh_data
        else:
            loader = BVHToTargetBase(bvh_data, sim_fps, character, False, bvh_start, bvh_end)
            target.pose = loader.init_target()
        if load_balance:
            target.balance = SamconBalanceTarget(target.num_frames, len(character.end_joint))
            target.balance.init_target_com(character, target.pose)
            target.compute_facing_angular_momentum(character)
            target.calc_kinematic_engery(character)

        return (target, loader) if return_loader else target

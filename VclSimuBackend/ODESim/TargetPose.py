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
from typing import Optional, Union, Iterable, Tuple

from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from .CharacterWrapper import CharacterWrapper, ODECharacter
from ..Common.MathHelper import MathHelper


class TargetBaseType:

    __slots__ = ("pos", "quat", "rot_mat", "linvel", "angvel", "linacc", "angacc")

    def __init__(self):
        self.pos: Optional[np.ndarray] = None  # Position
        self.quat: Optional[np.ndarray] = None  # Quaternion
        self.rot_mat: Optional[np.ndarray] = None

        self.linvel: Optional[np.ndarray] = None  # linear velocity
        self.angvel: Optional[np.ndarray] = None  # angular velocity

        self.linacc: Optional[np.ndarray] = None  # linear acceleration
        self.angacc: Optional[np.ndarray] = None  # angular acceleration

    def append(self, other):
        if self.pos is None:
            self.pos = copy.deepcopy(other.pos)
        else:
            self.pos = np.concatenate([self.pos, other.pos], axis=0)

        if self.quat is None:
            self.quat = copy.deepcopy(other.quat)
        else:
            self.quat = np.concatenate([self.quat, other.quat], axis=0)

        if self.rot_mat is None:
            self.rot_mat = copy.deepcopy(other.rot_mat)
        else:
            self.rot_mat = np.concatenate([self.rot_mat, other.rot_mat], axis=0)

        if self.linvel is None:
            self.linvel = copy.deepcopy(other.linvel)
        else:
            self.linvel = np.concatenate([self.linvel, other.linvel], axis=0)

        if self.angvel is None:
            self.angvel = copy.deepcopy(other.angvel)
        else:
            self.angvel = np.concatenate([self.angvel, other.angvel], axis=0)

        if self.linacc is None:
            self.linacc = copy.deepcopy(other.linacc)
        else:
            self.linacc = np.concatenate([self.linacc, other.linacc], axis=0)

        if self.angacc is None:
            self.angacc = copy.deepcopy(other.angacc)
        else:
            self.angacc = np.concatenate([self.angacc, other.angacc], axis=0)

        return self

    def duplicate(self, times: int = 1):
        res = TargetBaseType()
        if self.pos is not None:
            res.pos = np.concatenate([self.pos] * times, axis=0)
        if self.quat is not None:
            res.quat = np.concatenate([self.quat] * times, axis=0)
        if self.rot_mat is not None:
            res.rot_mat = np.concatenate([self.rot_mat] * times, axis=0)
        if self.linvel is not None:
            res.linvel = np.concatenate([self.linvel] * times, axis=0)
        if self.angvel is not None:
            res.angvel = np.concatenate([self.angvel] * times, axis=0)
        if self.linacc is not None:
            res.linacc = np.concatenate([self.linacc] * times, axis=0)
        if self.angacc is not None:
            res.angacc = np.concatenate([self.angacc] * times, axis=0)

        return res

    def deepcopy(self):
        return copy.deepcopy(self)

    def to_continuous(self):
        if self.pos is not None:
            self.pos = np.ascontiguousarray(self.pos)

        if self.quat is not None:
            self.quat = np.ascontiguousarray(self.quat)

        if self.rot_mat is not None:
            self.rot_mat = np.ascontiguousarray(self.rot_mat)

        if self.linvel is not None:
            self.linvel = np.ascontiguousarray(self.linvel)

        if self.angvel is not None:
            self.angvel = np.ascontiguousarray(self.angvel)

        if self.linacc is not None:
            self.linacc = np.ascontiguousarray(self.linacc)

        if self.angacc is not None:
            self.angacc = np.ascontiguousarray(self.angacc)

    def resize(self, shape: Union[int, Iterable, Tuple[int]], dtype=np.float64):
        self.pos = np.zeros(shape + (3,), dtype=dtype)
        self.quat = np.zeros(shape + (4,), dtype=dtype)
        self.linvel = np.zeros(shape + (3,), dtype=dtype)
        self.angvel = np.zeros(shape + (3,), dtype=dtype)
        self.linacc = np.zeros(shape + (3,), dtype=dtype)
        self.angacc = np.zeros(shape + (3,), dtype=dtype)
        return self

    def __len__(self) -> int:
        if self.pos is not None:
            return self.pos.shape[0]
        elif self.quat is not None:
            return self.quat.shape[0]
        elif self.rot_mat is not None:
            return self.rot_mat.shape[0]
        elif self.linvel is not None:
            return self.linvel.shape[0]
        elif self.angvel is not None:
            return self.angvel.shape[0]
        elif self.linacc is not None:
            return self.linacc.shape[0]
        elif self.angacc is not None:
            return self.angacc.shape[0]
        else:
            return 0

    def sub_seq(self, start: int = 0, end: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True):
        """
        Get sub sequence of TargetBaseType
        """
        assert skip is None or isinstance(skip, int)
        res = TargetBaseType()
        if end is None:
            end = len(self)
        if end == 0:
            return res

        piece = slice(start, end, skip)
        if self.pos is not None:
            res.pos = self.pos[piece].copy() if is_copy else self.pos[piece]
        if self.quat is not None:
            res.quat = self.quat[piece].copy() if is_copy else self.quat[piece]
        if self.rot_mat is not None:
            res.rot_mat = self.rot_mat[piece].copy() if is_copy else self.rot_mat[piece]
        if self.linvel is not None:
            res.linvel = self.linvel[piece].copy() if is_copy else self.linvel[piece]
        if self.angvel is not None:
            res.angvel = self.angvel[piece].copy() if is_copy else self.angvel[piece]
        if self.linacc is not None:
            res.linacc = self.linacc[piece].copy() if is_copy else self.linacc[piece]
        if self.angacc is not None:
            res.angacc = self.angacc[piece].copy() if is_copy else self.angacc[piece]

        return res

    def __str__(self):
        res = "pos" + (" is None" if self.pos is None else ".shape = " + str(self.pos.shape)) + \
              ". quat" + (" is None" if self.quat is None else ".shape = " + str(self.quat.shape)) + \
              ". linvel" + (" is None" if self.linvel is None else ".shape = " + str(self.linvel.shape)) + \
              ". angvel" + (" is None" if self.angvel is None else ".shape = " + str(self.angvel.shape))
        return res

    def set_value(self, pos: Optional[np.ndarray] = None, quat: Optional[np.ndarray] = None,
                  rot_mat: Optional[np.ndarray] = None,
                  linvel: Optional[np.ndarray] = None, angvel: Optional[np.ndarray] = None,
                  linacc: Optional[np.ndarray] = None, angacc: Optional[np.ndarray] = None):
        self.pos = pos
        self.quat = quat
        self.rot_mat = rot_mat
        self.linvel = linvel
        self.angvel = angvel
        self.linacc = linacc
        self.angacc = angacc


class TargetPose:

    __slots__ = ("globally", "locally", "child_body", "root", "root_body", "facing_root", "end", "facing_coor_end",
                 "all_joint_global", "all_joint_local", "all_joint_facing", "all_child_body", "character_body", "facing_quat",
                 "num_frames", "fps", "smoothed", "dup_pos_off_mix", "dup_rot_off_mix",
                 "dup_root_pos", "dup_root_quat")

    def __init__(self):
        # joint info in global coordinate
        # component shape is (num frame, num joint, 3)
        self.globally: TargetBaseType = TargetBaseType()

        # joint info in parent local coordinate
        # component shape is (num frame, character num joint, 3)
        self.locally: TargetBaseType = TargetBaseType()

        # child body's position in global coordinate
        # component shape is (num frame, num child body, 3)
        self.child_body: TargetBaseType = TargetBaseType()

        # root info in global coordinate
        # component shape is (num frame, 3)
        self.root: TargetBaseType = TargetBaseType()

        self.root_body: TargetBaseType = TargetBaseType()

        # root info in facing coordinate
        # component shape is (num frame, 3)
        self.facing_root: TargetBaseType = TargetBaseType()

        # end info in global coordinate
        # component shape is (num frame, num joint, 3)
        self.end: TargetBaseType = TargetBaseType()

        # end site in y rotation (heading) coordinate
        # component shape is (num frame, num joint, 3)
        self.facing_coor_end: TargetBaseType = TargetBaseType()

        # joint global info with root joint
        # component shape is (num frame, num body, 3)
        self.all_joint_global: TargetBaseType = TargetBaseType()

        # joint local info with root joint
        self.all_joint_local: TargetBaseType = TargetBaseType()

        # joint facing info with root joint
        self.all_joint_facing: TargetBaseType = TargetBaseType()

        # all body global info
        # component shape is (num frame, num body, 3)
        # note: body order may be different from ode bodies...
        self.all_child_body: TargetBaseType = TargetBaseType()

        # all body global info, body order matches ode order..
        self.character_body: TargetBaseType = TargetBaseType()

        # shape = (num frame, 4)
        self.facing_quat: Optional[np.ndarray] = None

        self.num_frames: int = 0
        self.fps: int = 0

        self.smoothed: bool = False
        self.dup_pos_off_mix: Optional[np.ndarray] = None  # delta position from (frame - 1) to (frame) for motion duplicate
        self.dup_rot_off_mix: Union[np.ndarray, Rotation, None] = None  # delta quaternion from (frame - 1) to (frame) for motion duplicate
        self.dup_root_pos: Optional[np.ndarray] = None
        self.dup_root_quat: Optional[np.ndarray] = None

    def set_dup_off_mix(self, pos_off_mix: np.ndarray, rot_off_mix: Union[np.ndarray, Rotation]):
        self.dup_pos_off_mix = pos_off_mix
        self.dup_rot_off_mix = rot_off_mix
        self.dup_root_pos = self.root.pos.copy()
        self.dup_root_quat = self.root.quat.copy()

    def compute_global_root_dup(self, dup_count: int):
        if dup_count > 1:
            assert self.dup_pos_off_mix is not None and self.dup_rot_off_mix is not None
            self.compute_global_root_dup_impl(dup_count, self.dup_pos_off_mix, self.dup_rot_off_mix)
        elif dup_count == 1:
            self.dup_root_pos = self.root.pos.copy()
            self.dup_root_quat = self.root.quat.copy()

    def compute_global_root_dup_impl(self, dup_count: int, pos_off_mix: Optional[np.ndarray], rot_off_mix: Union[np.ndarray, Rotation, None]):
        if dup_count <= 1:
            return

        dt = 1.0 / self.fps
        if pos_off_mix is None:
            pos_off_mix = 0.5 * dt * (self.root.linvel[0] + self.root.linvel[-1])
        if rot_off_mix is None: # calc by omega
            omega_ = self.root.angvel[-1].copy()
            last_q_ = self.root.quat[-1].copy()
            end_q_ = MathHelper.quat_integrate(last_q_[None, :], omega_[None, :], dt)
            rot_off_mix: Rotation = (Rotation(last_q_).inv() * Rotation(end_q_))
        if isinstance(rot_off_mix, np.ndarray):
            rot_off_mix: Rotation = Rotation(rot_off_mix)

        self.dup_root_pos = np.zeros((dup_count, self.num_frames, 3), dtype=np.float64)
        self.dup_root_quat = MathHelper.unit_quat_arr((dup_count, self.num_frames, 4))

        self.dup_root_pos[0, :, :] = self.root.pos.copy()
        self.dup_root_quat[0, :, :] = self.root.quat.copy()
        frame_0_rot = Rotation(self.root.quat[0])
        frame_0_rot_inv = frame_0_rot.inv()
        frame_0_coor_rot: Rotation = frame_0_rot_inv * Rotation(self.root.quat)
        frame_0_coor_pos: np.ndarray = frame_0_rot_inv.apply(self.root.pos - self.root.pos[None, 0])

        pos_off_mix = Rotation(self.root.quat[-1]).inv().apply(pos_off_mix)
        for i in range(1, dup_count):
            end_rot = Rotation(self.dup_root_quat[i - 1, -1])
            next_rot_0 = rot_off_mix * end_rot
            self.dup_root_quat[i, :, :] = (next_rot_0 * frame_0_coor_rot).as_quat()
            next_pos_0 = self.dup_root_pos[i - 1, -1] + end_rot.apply(pos_off_mix)
            self.dup_root_pos[i, :, :] = next_pos_0.reshape((1, 3)) + next_rot_0.apply(frame_0_coor_pos)
            self.dup_root_pos[i, :, 1] = self.root.pos[:, 1].copy()

        def debug_func():
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            pos_ = self.dup_root_pos
            for i in range(dup_count):
                ax.plot(pos_[i, :, 0], pos_[i, :, 1], pos_[i, :, 2])
            # pos_ = self.dup_root_pos.reshape((-1, 3))
            # ax.plot(pos_[:, 0], pos_[:, 1], pos_[:, 2])
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 2)
            ax.set_zlim(-3, 3)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()
            exit(0)

        # debug_func()
        self.dup_root_pos = np.concatenate([self.dup_root_pos.reshape((-1, 3)), self.root.pos[None, 0]], axis=0)
        self.dup_root_quat = np.concatenate([self.dup_root_quat.reshape((-1, 4)), self.root.quat[None, 0]], axis=0)

    def append(self, other):
        if self.globally is None:
            self.globally: Optional[TargetBaseType] = copy.deepcopy(other.globally)
        else:
            self.globally.append(other.globally)

        if self.locally is None:
            self.locally: Optional[TargetBaseType] = copy.deepcopy(other.locally)
        else:
            self.locally.append(other.locally)

        if self.child_body is None:
            self.child_body: Optional[TargetBaseType] = copy.deepcopy(other.child_body)
        else:
            self.child_body.append(other.child_body)

        if self.root is None:
            self.root: Optional[TargetBaseType] = copy.deepcopy(other.root)
        else:
            self.root.append(other.root)

        if self.root_body is None:
            self.root_body: Optional[TargetBaseType] = copy.deepcopy(other.root_body)
        else:
            self.root_body.append(other.root_body)

        if self.facing_root is None:
            self.facing_root: Optional[TargetBaseType] = copy.deepcopy(other.facing_root)
        else:
            self.facing_root.append(other.facing_root)

        if self.end is None:
            self.end: Optional[TargetBaseType] = copy.deepcopy(other.end)
        else:
            self.end.append(other.end)

        if self.facing_coor_end is None:
            self.facing_coor_end: Optional[TargetBaseType] = copy.deepcopy(other.facing_coor_end)
        else:
            self.facing_coor_end.append(other.facing_coor_end)

        if self.all_joint_global is None:
            self.all_joint_global: Optional[TargetBaseType] = copy.deepcopy(other.all_joint_global)
        else:
            self.all_joint_global.append(other.all_joint_global)

        if self.all_joint_local is None:
            self.all_joint_local: Optional[TargetBaseType] = copy.deepcopy(other.all_joint_local)
        else:
            self.all_joint_local.append(other.all_joint_local)

        if self.all_joint_facing is not None:
            self.all_joint_facing: Optional[TargetBaseType] = copy.deepcopy(other.all_joint_facing)
        else:
            self.all_joint_facing.append(other.all_joint_facing)

        if self.all_child_body is None:
            self.all_child_body: Optional[TargetBaseType] = copy.deepcopy(other.all_child_body)
        else:
            self.all_child_body.append(other.all_child_body)

        if self.character_body is None:
            self.character_body: Optional[TargetBaseType] = copy.deepcopy(other.character_body)
        else:
            self.character_body.append(other.character_body)

        if self.facing_quat is None:
            self.facing_quat: Optional[np.ndarray] = copy.deepcopy(other.facing_quat)
        else:
            self.facing_quat = np.concatenate([self.facing_quat, other.facing_quat], axis=0)

        self.num_frames += other.num_frames
        return self

    def duplicate(self, times: int = 0):
        res = TargetPose()
        if self.globally is not None:
            res.globally = self.globally.duplicate(times)
        if self.locally is not None:
            res.locally = self.locally.duplicate(times)
        if self.child_body is not None:
            res.child_body = self.child_body.duplicate(times)
        if self.root is not None:
            res.root = self.root.duplicate(times)
        if self.root_body is not None:
            res.root_body = self.root_body.duplicate(times)
        if self.facing_root is not None:
            res.facing_root = self.facing_root.duplicate(times)
        if self.end is not None:
            res.end = self.end.duplicate(times)
        if self.facing_coor_end is not None:
            res.facing_coor_end = self.facing_coor_end.duplicate(times)
        if self.all_joint_global is not None:
            res.all_joint_global = self.all_joint_global.duplicate(times)
        if self.all_joint_local is not None:
            res.all_joint_local = self.all_joint_local.duplicate(times)
        if self.all_joint_facing is not None:
            res.all_joint_facing = self.all_joint_facing.duplicate(times)
        if self.all_child_body is not None:
            res.all_child_body = self.all_child_body.duplicate(times)
        if self.character_body is not None:
            res.character_body = self.character_body.duplicate(times)
        if self.facing_quat is not None:
            res.facing_quat = np.concatenate([self.facing_quat] * times, axis=0)
        res.num_frames = self.num_frames * times
        res.fps = self.fps
        res.smoothed = self.smoothed

        return res

    def __len__(self) -> int:
        return max([len(i) for i in [self.globally, self.locally, self.root, []] if i is not None])

    def deepcopy(self):
        return copy.deepcopy(self)

    def to_continuous(self):
        if self.globally is not None:
            self.globally.to_continuous()

        if self.locally is not None:
            self.locally.to_continuous()

        if self.child_body is not None:
            self.child_body.to_continuous()

        if self.root is not None:
            self.root.to_continuous()

        if self.root_body is not None:
            self.root_body.to_continuous()

        if self.facing_root is not None:
            self.facing_root.to_continuous()

        if self.end is not None:
            self.end.to_continuous()

        if self.facing_coor_end is not None:
            self.facing_coor_end.to_continuous()

        if self.all_joint_global is not None:
            self.all_joint_global.to_continuous()

        if self.all_joint_local is not None:
            self.all_joint_local.to_continuous()

        if self.all_joint_facing is not None:
            self.all_joint_facing.to_continuous()

        if self.all_child_body is not None:
            self.all_child_body.to_continuous()

        if self.character_body is not None:
            self.character_body.to_continuous()

        self.facing_quat = np.ascontiguousarray(self.facing_quat)

        return self

    def sub_seq(self, start: Optional[int] = None, end_: Optional[int] = None, skip: Optional[int] = None, is_copy: bool = True):
        """
        Get sub sequence of TargetPose
        """
        assert skip is None or isinstance(skip, int)
        res = TargetPose()
        if start is None:
            start = 0
        if end_ is None:
            end_ = self.num_frames
        if end_ == 0:
            return res

        piece = slice(start, end_, skip)
        if self.globally is not None:
            res.globally = self.globally.sub_seq(start, end_, skip, is_copy)

        if self.locally is not None:
            res.locally = self.locally.sub_seq(start, end_, skip, is_copy)

        if self.child_body is not None:
            res.child_body = self.child_body.sub_seq(start, end_, skip, is_copy)

        if self.root is not None:
            res.root = self.root.sub_seq(start, end_, skip, is_copy)

        if self.root_body is not None:
            res.root_body = self.root_body.sub_seq(start, end_, skip, is_copy)

        if self.end is not None:
            res.end = self.end.sub_seq(start, end_, skip, is_copy)

        if self.facing_coor_end is not None:
            res.facing_coor_end = self.facing_coor_end.sub_seq(start, end_, skip, is_copy)

        if self.facing_root is not None:
            res.facing_root = self.facing_root.sub_seq(start, end_, skip, is_copy)

        if self.all_joint_global is not None:
            res.all_joint_global = self.all_joint_global.sub_seq(start, end_, skip, is_copy)

        if self.all_joint_local is not None:
            res.all_joint_local = self.all_joint_local.sub_seq(start, end_, skip, is_copy)

        if self.all_joint_facing is not None:
            res.all_joint_facing = self.all_joint_facing.sub_seq(start, end_, skip, is_copy)

        if self.all_child_body is not None:
            res.all_child_body = self.all_child_body.sub_seq(start, end_, skip, is_copy)

        if self.character_body is not None:
            res.character_body = self.character_body.sub_seq(start, end_, skip, is_copy)

        if self.facing_quat is not None:
            res.facing_quat = self.facing_quat[piece].copy() if is_copy else self.facing_quat[piece]

        res.num_frames = len(res)
        res.fps = self.fps
        res.smoothed = self.smoothed

        return res

    def get_facing_body_info(self):
        result = TargetBaseType()
        root_pos: np.ndarray = MathHelper.vec_axis_to_zero(self.root.pos, 1)
        ry, _ = MathHelper.y_decompose(self.root.quat)
        ry_inv = Rotation(ry).inv()
        result.pos = self.character_body.pos - root_pos[:, None, :]
        for body_index in range(1, result.pos.shape[1]):
            result.pos[:, body_index] = ry_inv.apply(result.pos[:, body_index])

        result.quat = self.character_body.quat.copy()
        for body_index in range(1, result.quat.shape[1]):
            result.quat[:, body_index] = None

        result.linvel = None
        result.angvel = None
        return result

class SetTargetToCharacter(CharacterWrapper):
    """
    use for load {frame} to ODE Character
    """
    def __init__(self, character: ODECharacter, target: TargetPose):
        super(SetTargetToCharacter, self).__init__(character)
        self.target: TargetPose = target

    @property
    def num_frames(self):
        return self.target.num_frames

    def set_character_byframe(self, frame: int = 0, other_character: Optional[ODECharacter] = None):
        if other_character is None:
            other_character = self.character
        c_id = other_character.body_info.body_c_id
        ch_body = self.target.character_body
        other_character.world.loadBodyPos(c_id, ch_body.pos[frame].flatten())
        other_character.world.loadBodyQuat(c_id, ch_body.quat[frame].flatten())
        other_character.world.loadBodyLinVel(c_id, ch_body.linvel[frame].flatten())
        other_character.world.loadBodyAngVel(c_id, ch_body.angvel[frame].flatten())

        # state = other_character.save()
        # other_character.load(state)
        # return state

    def set_character_byframe_old(self, frame: int = 0, other_character: Optional[ODECharacter] = None):
        """
        load {frame} to ODE Character
        we don't need to resort joint, because we have joint c id..
        """
        if other_character is None:
            other_character = self.character

        # Set Root Body's Position, Rotation, Linear Velocity, Angular Velocity
        other_character.root_body.PositionNumpy = self.target.root.pos[frame]
        other_character.root_body.setQuaternionScipy(self.target.root.quat[frame])  # rot
        other_character.root_body.LinearVelNumpy = self.target.root.linvel[frame]
        other_character.root_body.setAngularVelNumpy(self.target.root.angvel[frame])

        # Set global position and quaternion via child_body's c id
        other_character.world.loadBodyPos(other_character.joint_info.child_body_c_id,
                                          self.target.child_body.pos[frame].flatten())
        other_character.world.loadBodyQuat(other_character.joint_info.child_body_c_id,
                                           self.target.child_body.quat[frame].flatten())

        # Set child_body's linear velocity
        other_character.world.loadBodyLinVel(other_character.joint_info.child_body_c_id,
                                             self.target.child_body.linvel[frame].flatten())

        # Set child_body's angular velocity
        other_character.world.loadBodyAngVel(other_character.joint_info.child_body_c_id,
                                             self.target.child_body.angvel[frame].flatten())

        state = other_character.save()
        other_character.load(state)
        return state
        # self.check(frame)

    def check(self, frame: int):
        # check root body
        assert np.all(self.character.root_body.PositionNumpy - self.target.root.pos[frame] == 0)
        assert np.all(np.abs(self.character.root_body.getQuaternionScipy() - self.target.root.quat[frame]) < 1e-10)
        assert np.all(self.character.root_body.LinearVelNumpy - self.target.root.linvel[frame] == 0)
        assert np.all(self.character.root_body.getAngularVelNumpy() - self.target.root.angvel[frame] == 0)

        # check body pos
        assert np.all(self.world.getBodyPos(self.body_info.body_c_id[1:]).reshape((-1, 3)) -
                      self.target.child_body.pos[frame] == 0)
        # check body linear velocity
        assert np.all(self.world.getBodyLinVel(self.body_info.body_c_id[1:]).reshape((-1, 3)) -
                      self.target.child_body.linvel[frame] == 0)
        # check body quat
        assert np.all(np.abs(self.world.getBodyQuatScipy(self.body_info.body_c_id[1:]).reshape((-1, 4)) -
                             self.target.child_body.quat[frame]) < 1e-10)

        # check body angvel
        assert np.all(np.abs(self.world.getBodyAngVel(self.body_info.body_c_id[1:]).reshape((-1, 3)) -
                             self.target.child_body.angvel[frame]) < 1e-10)
        # check global angular velocity..

        angvel = np.linalg.norm(self.character.joint_info.get_local_angvels(), axis=1)
        angvel_real = np.linalg.norm(self.target.locally.angvel[frame], axis=1)

        print(np.max(self.character.joint_info.get_local_angvels() - self.target.locally.angvel[frame]))

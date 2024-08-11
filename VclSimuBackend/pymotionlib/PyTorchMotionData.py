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
from optparse import Option
import numpy as np
import os
from typing import List, Optional, Tuple
import torch
from .MotionData import MotionData
# from ..DiffODE import DiffQuat
import RotationLibTorch as DiffQuat
from ..DiffODE.PyTorchMathHelper import PyTorchMathHelper


cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda")
fdir = os.path.dirname(__file__)


class PyTorchMotionData:
    """
    Modified from pymotionlib.MotionData

    For forward kinematics:
    First, copy joint local rotation from MotionData (without gradient)
    Then, copy root and leg rotation from nn.Parameter (with gradient)
    """

    __slots__ = (
        "_skeleton_joints",
        "_skeleton_joint_parents",
        "_skeleton_joint_offsets",
        "_end_sites",
        "_num_joints",
        "_num_frames",
        "_fps",
        "_joint_rotation",
        "_root_translation",
        "_joint_position",
        "_joint_orientation"
    )

    def __init__(self) -> None:
        self._skeleton_joints: Optional[List[str]] = None  # name of each joint
        self._skeleton_joint_parents: Optional[List[int]] = None
        self._skeleton_joint_offsets: Optional[torch.Tensor] = None  # Joint OFFSET in BVH file

        self._end_sites: Optional[List[int]] = None
        self._num_joints = 0

        # animation
        self._num_frames = 0
        self._fps = 0

        self._joint_rotation: Optional[torch.Tensor] = None  # joint local rotation
        self._root_translation: Optional[torch.Tensor] = None

        # pre-computed global information
        self._joint_position: Optional[torch.Tensor] = None
        self._joint_orientation: Optional[torch.Tensor] = None

    @property
    def joint_rotation(self) -> Optional[torch.Tensor]:
        return self._joint_rotation

    @property
    def root_translation(self) -> Optional[torch.Tensor]:
        return self._root_translation

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def joint_position(self) -> Optional[torch.Tensor]:
        return self._joint_position

    @property
    def joint_orientation(self) -> Optional[torch.Tensor]:
        return self._joint_orientation

    @property
    def joint_parents_idx(self) -> Optional[List[int]]:
        return self._skeleton_joint_parents

    @property
    def joint_names(self) -> Optional[List[str]]:
        return self._skeleton_joints

    @property
    def end_sites(self) -> Optional[List[int]]:
        return self._end_sites

    def to_device(self, device):
        result = PyTorchMotionData()
        result._skeleton_joint_parents = self._skeleton_joint_parents
        if self._skeleton_joint_offsets is not None:
            result._skeleton_joint_offsets = self._skeleton_joint_offsets.to(device)
        if self._joint_rotation is not None:
            result._joint_rotation = self._joint_rotation.to(device)
        if self._joint_orientation is not None:
            result._joint_orientation = self._joint_orientation.to(device)
        if self._root_translation is not None:
            result._root_translation = self._root_translation.to(device)
        if self._joint_position is not None:
            result._joint_position = self._joint_position.to(device)
        return result

    def to_device_(self, device):
        """
        convert to pytorch device
        """
        if self._skeleton_joint_offsets is not None:
            self._skeleton_joint_offsets = self._skeleton_joint_offsets.to(device)
        if self._joint_rotation is not None:
            self._joint_rotation = self._joint_rotation.to(device)
        if self._joint_orientation is not None:
            self._joint_orientation = self._joint_orientation.to(device)
        if self._root_translation is not None:
            self._root_translation = self._root_translation.to(device)
        if self._joint_position is not None:
            self._joint_position = self._joint_position.to(device)

        return self

    def sub_sequnece(self, start: Optional[int], end: Optional[int], is_copy: bool = False, detach: bool = False):
        """
        """
        result = PyTorchMotionData()
        result._skeleton_joints = copy.deepcopy(self._skeleton_joints)
        result._skeleton_joint_parents = copy.deepcopy(self._skeleton_joint_parents)
        result._skeleton_joint_offsets = self._skeleton_joint_offsets.detach().clone()
        result._end_sites = copy.deepcopy(self._end_sites)
        result._num_joints = self._num_joints
        result._fps = self._fps

        index = slice(start, end)
        if detach and is_copy:
            func = lambda x: x[index].detach().clone() if x is not None else None
        elif detach and not is_copy:
            func = lambda x: x[index].detach() if x is not None else None
        elif not detach and is_copy:
            func = lambda x: x[index].clone() if x is not None else None
        elif not detach and not is_copy:
            func = lambda x: x[index] if x is not None else None
        else:
            raise NotImplementedError

        if self._joint_rotation is not None:
            result._joint_rotation = func(self._joint_rotation)
        if self._root_translation is not None:
            result._root_translation = func(self._root_translation)
        if self._joint_position is not None:
            result._joint_position = func(self._joint_position)
        if self._joint_orientation is not None:
            result._joint_orientation = func(self._joint_orientation)

        if result._joint_rotation is not None:
            result._num_frames = result._joint_rotation.shape[0]
        else:
            result._num_frames = 0

        return result

    def build_from_motion_data(self, motion: MotionData, dtype=torch.float32, device=cpu_device):
        # 1. copy motion parent, child attributes to here.
        self._skeleton_joints = copy.deepcopy(motion._skeleton_joints)
        self._skeleton_joint_parents = copy.deepcopy(motion._skeleton_joint_parents)
        self._skeleton_joint_offsets: Optional[torch.Tensor] = torch.as_tensor(
            motion._skeleton_joint_offsets, dtype=dtype, device=device
        )
        self._end_sites = copy.deepcopy(motion._end_sites)
        self._num_frames = motion.num_frames
        self._num_joints = motion.num_joints
        self._fps = motion.fps

        # 2. copy joint rotation and joint translation here.
        self.load_rot_trans(motion, dtype, device)

        return self

    def export_to_motion_data(self, is_copy: bool = True) -> MotionData:
        result = MotionData()
        result._skeleton_joints = copy.deepcopy(self._skeleton_joints)
        result._skeleton_joint_parents = copy.deepcopy(self._skeleton_joint_parents)
        result._skeleton_joint_offsets = self._skeleton_joint_offsets.detach().cpu().clone().numpy()
        result._end_sites = copy.deepcopy(self._end_sites)
        result._num_frames = self._num_frames
        result._num_joints = self._num_joints
        result._fps = self._fps

        if self._joint_rotation is not None:
            result._joint_rotation = self._joint_rotation.detach().cpu().clone().numpy() if is_copy else self._joint_rotation.detach().cpu().numpy()
        if self._root_translation is not None:
            result._joint_translation = np.zeros((self._num_frames, self._num_joints, 3))
            result._joint_translation[:, 0, :] = self._root_translation.detach().cpu().clone().numpy()
        if self._joint_position is not None:
            result._joint_position = self._joint_position.detach().cpu().clone().numpy() if is_copy else self._joint_position.detach().cpu().numpy()
        if self._joint_orientation is not None:
            result._joint_orientation = self._joint_orientation.detach().cpu().clone().numpy() if is_copy else self._joint_orientation.detach().cpu().numpy()

        return result

    def load_rot_trans(self, motion: MotionData, dtype=torch.float32, device=cpu_device):
        if motion._joint_rotation is not None:
            self._joint_rotation = torch.as_tensor(motion._joint_rotation, dtype=dtype, device=device)

        if motion._joint_translation is not None:
            self._root_translation = torch.as_tensor(motion._joint_translation[:, 0, :], dtype=dtype, device=device)

        return self

    def clear(self):
        self._skeleton_joint_offsets = self._skeleton_joint_offsets.detach().clone()

        self._joint_position: Optional[torch.Tensor] = None
        self._joint_orientation: Optional[torch.Tensor] = None

        # clear joint local rotation and translation here..
        self._joint_rotation: Optional[torch.Tensor] = None
        self._root_translation: Optional[torch.Tensor] = None

        self._num_frames = 0

        return self

    def set_parameter(self, root_pos_param: torch.Tensor, rotate_param: torch.Tensor, rotate_modify_dim: Optional[List[int]] = None):
        if rotate_modify_dim is None:
            rotate_modify_dim = slice(None, None)
        self._root_translation = root_pos_param
        self._joint_rotation[:, rotate_modify_dim, :] = rotate_param
        self._joint_rotation = self._joint_rotation.contiguous()

        return self

    def to_contiguous(self):
        """
        if x is a contiguous Tensor, data pointer of x.contiguous() and x will be same.
        That is, contiguous operation will not cost much time on contiguous Tensor.
        """
        if self._joint_rotation is not None:
            self._joint_rotation: Optional[torch.Tensor] = self._joint_rotation.contiguous()
        if self._root_translation is not None:
            self._root_translation: Optional[torch.Tensor] = self._root_translation.contiguous()
        if self._joint_orientation is not None:
            self._joint_orientation: Optional[torch.Tensor] = self._joint_orientation.contiguous()
        if self._joint_position is not None:
            self._joint_position: Optional[torch.Tensor] = self._joint_position.contiguous()

        return self

    def compute_joint_global_info(
        self,
        root_translation: torch.Tensor,
        joint_rotation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified from pymotionlib.MotionData
        compute global information based on given local information
        """

        root_translation: torch.Tensor = root_translation.view((-1, 3))
        joint_rotation: torch.Tensor = joint_rotation.view((-1, self._num_joints, 4))

        num_frames, num_joints = joint_rotation.shape[:2]
        joint_position: List[Optional[torch.Tensor]] = [None for _ in range(num_joints)]
        joint_orientation: List[Optional[torch.Tensor]] = [None for _ in range(num_joints)]
        for i, pi in enumerate(self._skeleton_joint_parents):
            if pi < 0:
                assert i == 0 and joint_position[i] is None and joint_orientation[i] is None
                joint_position[i] = root_translation
                joint_orientation[i] = joint_rotation[:, i, :].contiguous()
                continue

            assert joint_position[i] is None and joint_orientation[i] is None
            parent_orient: torch.Tensor = joint_orientation[pi]
            parent_pos: torch.Tensor = joint_position[pi]
            offset_ext: torch.Tensor = self._skeleton_joint_offsets[None, i, :].repeat(num_frames, 1)
            joint_position[i] = DiffQuat.quat_apply(parent_orient, offset_ext) + parent_pos
            child_orient: torch.Tensor = joint_rotation[:, i, :].contiguous()
            child_orient_result = DiffQuat.quat_multiply(parent_orient, child_orient)
            child_orient_result = DiffQuat.quat_normalize(child_orient_result)
            # child_orient_result = DiffQuat.flip_vec_by_dot(child_orient_result)

            joint_orientation[i] = child_orient_result

        joint_position: torch.Tensor = torch.cat([node[:, None] for node in joint_position], dim=1)
        joint_orientation: torch.Tensor = torch.cat([node[:, None] for node in joint_orientation], dim=1)

        return joint_position, joint_orientation

    def recompute_joint_global_info(self):
        # now pre-compute joint global positions and orientations
        assert self._root_translation.shape == (self._num_frames, 3)
        assert self._joint_rotation.shape == (self._num_frames, self._num_joints, 4), f"(self._num_frames, self._num_joints, 4) = {(self._num_frames, self._num_joints, 4)}, self._joint_rotation.shape = {self._joint_rotation.shape}, {self._joint_rotation.device}"
        self._joint_position, self._joint_orientation = self.compute_joint_global_info(
            self._root_translation, self._joint_rotation)

        return self

    def compute_linear_velocity(self, forward: bool = False) -> torch.Tensor:
        return PyTorchMathHelper.vec_diff(self._joint_position, forward, self._fps)

    @staticmethod
    def compute_angvel_frag(rotation: torch.Tensor, fps: float) -> torch.Tensor:
        # here we should use version implemented by Yuan Shen
        num_frames, num_joints = rotation.shape[:2]
        qd: torch.Tensor = torch.diff(rotation, dim=0) * fps
        # qd: torch.Tensor = (rotation[1:] - rotation[:-1]) * fps
        # modified by heyuan Yao
        q: torch.Tensor = rotation[:-1]
        conj_mask = torch.ones_like(q).view(-1, 4)
        conj_mask[:, :3] *= -1
        q_conj = conj_mask * q.view(-1, 4)
        qw: torch.Tensor = DiffQuat.quat_multiply(qd.view(-1, 4), q_conj)

        frag: torch.Tensor = 2 * qw[:, :3].view(num_frames - 1, num_joints, 3)

        return frag

    @staticmethod
    def compute_angvel_base(rotation: torch.Tensor, fps: float, forward: bool = False) -> torch.Tensor:
        frag: torch.Tensor = PyTorchMotionData.compute_angvel_frag(rotation, fps)
        if forward:
            return torch.cat([frag, frag[None, -1]], dim=0)
        else:
            return torch.cat([frag[None, 0], frag], dim=0)


    def compute_angular_velocity(self, forward: bool = False) -> torch.Tensor:
        return self.compute_angvel_base(self._joint_orientation, self._fps, forward)

    def compute_rotational_speed(self, forward: bool = False) -> torch.Tensor:
        return self.compute_angvel_base(self._joint_rotation, self._fps, forward)

    def get_device(self) -> torch.device:
        nodes = [self._root_translation, self._joint_rotation, self._joint_position, self._joint_orientation]
        for node in nodes:
            if node is not None:
                return node.device
        return cpu_device

    def remove_end_sites(self, is_copy: bool = True, detach: bool = True):
        """
        Modified from numpy version in MotionData.py
        """
        # 1. create sub set
        ret = self.sub_sequnece(None, None, is_copy=is_copy, detach=detach)
        if not ret._end_sites:
            return ret

        # modify attr index. joint_idx doesn't requires gradient.
        # As delete operation is not convenient, we use numpy first, and convert to pytorch..
        joint_idx: np.ndarray = np.arange(0, ret._num_joints, dtype=np.int64)
        joint_idx: np.ndarray = np.delete(joint_idx, np.array(ret._end_sites, dtype=np.int64))
        device = self.get_device()
        joint_idx: torch.Tensor = torch.as_tensor(joint_idx, dtype=torch.long, device=device)
        # We need not to modify the root translation here.
        if ret._joint_rotation is not None:
            ret._joint_rotation = ret._joint_rotation[:, joint_idx, :]
        if ret._joint_position is not None:
            ret._joint_position = ret._joint_position[:, joint_idx, :]
        if ret._joint_orientation is not None:
            ret._joint_orientation = ret._joint_orientation[:, joint_idx, :]

        # modify parent index
        for i in range(len(ret._end_sites)):  # here doesn't requires gradient
            end_idx = ret._end_sites[i] - i
            before = ret._skeleton_joint_parents[:end_idx]
            after = ret._skeleton_joint_parents[end_idx + 1:]
            for j in range(len(after)):
                if after[j] > end_idx:
                    after[j] -= 1
            ret._skeleton_joint_parents = before + after

        # modify other attributes. Here doesn't requires gradient, we can simply use numpy version
        ret._skeleton_joints = np.array(ret._skeleton_joints)[joint_idx].tolist()
        ret._skeleton_joint_offsets = np.array(ret._skeleton_joint_offsets)[joint_idx]
        ret._num_joints -= len(ret._end_sites)
        ret._end_sites.clear()

        return ret.to_contiguous()

    def get_adj_matrix(self) -> np.ndarray:
        num_joints = len(self.joint_parents_idx)
        result: np.ndarray = np.zeros(num_joints, dtype=np.int32)
        for idx, parent_idx in enumerate(self.joint_parents_idx):
            if parent_idx == -1:
                result[idx, parent_idx] = 1
                result[parent_idx, idx] = 1
        return result

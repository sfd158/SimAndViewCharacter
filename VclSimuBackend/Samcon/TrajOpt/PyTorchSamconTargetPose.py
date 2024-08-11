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
import torch
from typing import Optional, Union, Iterable, Tuple
from ..SamconTargetPose import SamconTargetPose, TargetPose, SamconBalanceTarget
from ...ODESim.TargetPose import TargetBaseType


class PyTorchTargetBaseType:
    """
    Same as TargetBase of numpy version
    """
    def __init__(self):
        self.pos: Optional[torch.Tensor] = None  # Position
        self.quat: Optional[torch.Tensor] = None  # Quaternion
        self.linvel: Optional[torch.Tensor] = None  # linear velocity
        self.angvel: Optional[torch.Tensor] = None  # angular velocity

        self.linacc: Optional[torch.Tensor] = None  # linear acceleration
        self.angacc: Optional[torch.Tensor] = None  # angular acceleration

    def to_continuous(self):
        """
        Convert to continuous array
        """
        if self.pos is not None:
            self.pos = self.pos.contiguous()

        if self.quat is not None:
            self.quat = self.quat.contiguous()

        if self.linvel is not None:
            self.linvel = self.linvel.contiguous()

        if self.angvel is not None:
            self.angvel = self.angvel.contiguous()

        if self.linacc is not None:
            self.linacc = self.linacc.contiguous()

        if self.angacc is not None:
            self.angacc = self.angacc.contiguous()

        return self

    def resize(self, shape: Union[int, Iterable, Tuple[int]], dtype=torch.float64):
        """
        resize
        """
        self.pos = torch.zeros(shape + (3,), dtype=dtype)
        self.quat = torch.zeros(shape + (4,), dtype=dtype)
        self.linvel = torch.zeros(shape + (3,), dtype=dtype)
        self.angvel = torch.zeros(shape + (3,), dtype=dtype)
        self.linacc = torch.zeros(shape + (3,), dtype=dtype)
        self.angacc = torch.zeros(shape + (3,), dtype=dtype)

        return self

    def __len__(self) -> int:
        if self.pos is not None:
            return self.pos.shape[0]
        if self.quat is not None:
            return self.quat.shape[0]
        if self.linvel is not None:
            return self.linvel.shape[0]
        if self.angvel is not None:
            return self.angvel.shape[0]
        if self.linacc is not None:
            return self.linacc.shape[0]
        if self.angacc is not None:
            return self.angacc.shape[0]

        return 0

    def sub_seq(self, start: int = 0, end: Optional[int] = None, is_copy: bool = True):
        """
        Get sub sequence of
        """
        res = PyTorchTargetBaseType()
        if end is None:
            end = len(self)
        if end == 0:
            return res

        if self.pos is not None:
            res.pos = self.pos[start:end].clone() if is_copy else self.pos[start:end]
        if self.quat is not None:
            res.quat = self.quat[start:end].clone() if is_copy else self.quat[start:end]
        if self.linvel is not None:
            res.linvel = self.linvel[start:end].clone() if is_copy else self.linvel[start:end]
        if self.angvel is not None:
            res.angvel = self.angvel[start:end].clone() if is_copy else self.angvel[start:end]
        if self.linacc is not None:
            res.linacc = self.linacc[start:end].clone() if is_copy else self.linacc[start:end]
        if self.angacc is not None:
            res.angacc = self.angacc[start:end].clone() if is_copy else self.angacc[start:end]

        return res

    def set_value(self, pos: Optional[torch.Tensor] = None, quat: Optional[torch.Tensor] = None,
                  linvel: Optional[torch.Tensor] = None, angvel: Optional[torch.Tensor] = None,
                  linacc: Optional[torch.Tensor] = None, angacc: Optional[torch.Tensor] = None):
        self.pos = pos
        self.quat = quat
        self.linvel = linvel
        self.angvel = angvel
        self.linacc = linacc
        self.angacc = angacc

        return self

    @staticmethod
    def build_from_numpy(other: TargetBaseType, dtype=torch.float64, device=torch.device("cpu")):
        res = PyTorchTargetBaseType()

        if other.pos is not None:
            res.pos = torch.as_tensor(other.pos, dtype=dtype, device=device)
        if other.quat is not None:
            res.quat = torch.as_tensor(other.quat, dtype=dtype, device=device)
        if other.linvel is not None:
            res.linvel = torch.as_tensor(other.linvel, dtype=dtype, device=device)
        if other.angvel is not None:
            res.angvel = torch.as_tensor(other.angvel, dtype=dtype, device=device)
        if other.linacc is not None:
            res.linacc = torch.as_tensor(other.linacc, dtype=dtype, device=device)
        if other.angacc is not None:
            res.angacc = torch.as_tensor(other.angacc, dtype=dtype, device=device)

        return res


class PyTorchBalanceTarget:
    def __init__(self):
        # position of Center of Mass globally
        self.com: Optional[torch.Tensor] = None  # shape = (nframe, 3)

        # velocity of Center of Mass globally
        self.com_vel: Optional[torch.Tensor] = None  # shape = (nframe, 3)

        # position of Center of Mass in facing coordinate
        self.facing_com: Optional[torch.Tensor] = None  # shape = (nframe, 3)

        # Velocity of Center of Mass in facing coordinate
        self.facing_com_vel: Optional[torch.Tensor] = None  # shape = (nframe, 3)

        # in Samcon 2010 paper.
        self.facing_rc: Optional[torch.Tensor] = None  # shape = (nframe, end joint, 3)

        # in Samcon 2016 paper.
        self.facing_angular_momentum: Optional[torch.Tensor] = None

    def to_continuous(self):
        if self.com is not None:
            self.com: Optional[torch.Tensor] = self.com.contiguous()
        if self.com_vel is not None:
            self.com_vel = self.com_vel.contiguous()
        if self.facing_com is not None:
            self.facing_com = self.facing_com.contiguous()
        if self.facing_com_vel is not None:
            self.facing_com_vel = self.facing_com_vel.contiguous()
        if self.facing_rc is not None:
            self.facing_rc = self.facing_rc.contiguous()
        if self.facing_angular_momentum is not None:
            self.facing_angular_momentum = self.facing_angular_momentum.contiguous()

        return self

    def sub_seq(self, start: int = 0, end: Optional[int] = None, is_copy: bool = True):
        res = PyTorchBalanceTarget()
        if end is None:
            end = len(self)
        if end == 0:
            return res

        if self.com is not None:
            res.com = self.com[start:end].clone() if is_copy else self.com[start:end]
        if self.com_vel is not None:
            res.com_vel = self.com_vel[start:end].clone() if is_copy else self.com_vel[start:end]
        if self.facing_com is not None:
            res.facing_com = self.facing_com[start:end].clone() if is_copy else self.facing_com[start:end]
        if self.facing_com_vel is not None:
            res.facing_com_vel = self.facing_com_vel[start:end].clone() if is_copy else self.facing_com_vel[start:end]
        if self.facing_rc is not None:
            res.facing_rc = self.facing_rc[start:end].clone() if is_copy else self.facing_rc[start:end]
        if self.facing_angular_momentum is not None:
            res.facing_angular_momentum = self.facing_angular_momentum[start:end].clone() if is_copy else self.facing_angular_momentum[start:end]

        return res

    @staticmethod
    def build_from_numpy(other: SamconBalanceTarget, dtype=torch.float64, device=torch.device("cpu")):
        res = PyTorchBalanceTarget()
        if other.com is not None:
            res.com = torch.as_tensor(other.com, dtype=dtype, device=device)
        if other.com_vel is not None:
            res.com_vel = torch.as_tensor(other.com_vel, dtype=dtype, device=device)
        if other.facing_com is not None:
            res.facing_com = torch.as_tensor(other.facing_com, dtype=dtype, device=device)
        if other.facing_com_vel is not None:
            res.facing_com_vel = torch.as_tensor(other.facing_com_vel, dtype=dtype, device=device)
        if other.facing_rc is not None:
            res.facing_rc = torch.as_tensor(other.facing_rc, dtype=dtype, device=device)
        if other.facing_angular_momentum is not None:
            res.facing_angular_momentum = torch.as_tensor(other.facing_angular_momentum, dtype=dtype, device=device)

        return res

    def __len__(self):
        pass


class PyTorchTargetPose:
    """
    Tensors in this class don't need gradient.
    """
    def __init__(self):
        # joint info in global coordinate
        # component shape is (num frame, num joint, 3)
        self.globally: PyTorchTargetBaseType = PyTorchTargetBaseType()

        # joint info in parent local coordinate
        # component shape is (num frame, character num joint, 3)
        self.locally: PyTorchTargetBaseType = PyTorchTargetBaseType()

        # child body's position in global coordinate
        # component shape is (num frame, num child body, 3)
        self.child_body: PyTorchTargetBaseType = PyTorchTargetBaseType()

        # root info in global coordinate
        # component shape is (num frame, 3)
        self.root: PyTorchTargetBaseType = PyTorchTargetBaseType()

        # root info in facing coordinate
        # component shape is (num frame, 3)
        self.facing_root: PyTorchTargetBaseType = PyTorchTargetBaseType()

        # end info in global coordinate
        # component shape is (num frame, num joint, 3)
        self.end: PyTorchTargetBaseType = PyTorchTargetBaseType()

        # end site in y rotation (heading) coordinate
        # component shape is (num frame, num joint, 3)
        self.facing_coor_end: PyTorchTargetBaseType = PyTorchTargetBaseType()

        self.all_joint_global: PyTorchTargetBaseType = PyTorchTargetBaseType()
        self.all_joint_facing: PyTorchTargetBaseType = PyTorchTargetBaseType()

        # shape = (num frame, 4)
        self.facing_quat: Optional[torch.Tensor] = None

        self.num_frames = 0
        self.fps = 0

        self.dup_root_pos: Optional[torch.Tensor] = None
        self.dup_root_quat: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return max([len(i) for i in [self.globally, self.locally, self.root, []] if i is not None])

    def to_continuous(self):
        if self.globally is not None:
            self.globally.to_continuous()

        if self.locally is not None:
            self.locally.to_continuous()

        if self.child_body is not None:
            self.child_body.to_continuous()

        if self.root is not None:
            self.root.to_continuous()

        if self.facing_quat is not None:
            self.facing_root.to_continuous()

        if self.end is not None:
            self.end.to_continuous()

        if self.facing_coor_end is not None:
            self.facing_coor_end.to_continuous()

        if self.all_joint_global is not None:
            self.all_joint_global.to_continuous()

        if self.all_joint_facing is not None:
            self.all_joint_facing.to_continuous()

        if self.facing_quat is not None:
            self.facing_quat = self.facing_quat.contiguous()

        return self

    def sub_seq(self, start: Optional[int] = None, end_: Optional[int] = None, is_copy: bool = True):
        """
        Get sub sequence
        """
        res = PyTorchTargetPose()
        if start is None:
            start = 0
        if end_ is None:
            end_ = self.num_frames
        if end_ == 0:
            return res

        if self.globally is not None:
            res.globally = self.globally.sub_seq(start, end_, is_copy)
        if self.locally is not None:
            res.locally = self.locally.sub_seq(start, end_, is_copy)
        if self.child_body is not None:
            res.child_body = self.child_body.sub_seq(start, end_, is_copy)
        if self.root is not None:
            res.root = self.root.sub_seq(start, end_, is_copy)
        if self.facing_root is not None:
            res.facing_root = self.facing_root.sub_seq(start, end_, is_copy)
        if self.end is not None:
            res.end = self.end.sub_seq(start, end_, is_copy)
        if self.facing_coor_end is not None:
            res.facing_coor_end = self.facing_coor_end.sub_seq(start, end_, is_copy)
        if self.all_joint_global is not None:
            res.all_joint_global = self.all_joint_global.sub_seq(start, end_, is_copy)
        if self.all_joint_facing is not None:
            res.all_joint_facing = self.all_joint_facing.sub_seq(start, end_, is_copy)
        if self.facing_quat is not None:
            res.facing_quat = self.facing_quat[start:end_].clone() if is_copy else self.facing_quat[start:end_]

        res.num_frames = len(res)
        res.fps = self.fps

        return res

    @staticmethod
    def build_from_numpy(other: TargetPose, dtype=torch.float64):
        res = PyTorchTargetPose()
        res.fps = other.fps
        res.num_frames = other.num_frames

        build_func = PyTorchTargetBaseType.build_from_numpy
        if other.globally is not None:
            res.globally = build_func(other.globally, dtype=dtype)
        if other.locally is not None:
            res.locally = build_func(other.locally, dtype=dtype)
        if other.child_body is not None:
            res.child_body = build_func(other.child_body, dtype=dtype)
        if other.root is not None:
            res.root = build_func(other.root, dtype=dtype)
        if other.facing_root is not None:
            res.facing_root = build_func(other.facing_root, dtype=dtype)
        if other.end is not None:
            res.end = build_func(other.end, dtype=dtype)
        if other.facing_coor_end is not None:
            res.facing_coor_end = build_func(other.facing_coor_end, dtype=dtype)
        if other.all_joint_global is not None:
            res.all_joint_global = build_func(other.all_joint_global, dtype=dtype)
        if other.all_joint_facing is not None:
            res.all_joint_facing = build_func(other.all_joint_facing, dtype=dtype)
        if other.facing_quat is not None:
            res.facing_quat = torch.as_tensor(other.facing_quat, dtype=dtype)

        return res


class PyTorchSamconTargetPose:
    """
    pytorch version of Samcon Target Pose.
    Tensors in this class don't need gradient.
    """
    def __init__(self):
        self.pose: Optional[PyTorchTargetPose] = None
        self.balance: Optional[PyTorchBalanceTarget] = None

    @property
    def num_frames(self) -> int:
        return self.pose.num_frames

    @property
    def fps(self):
        return self.pose.fps

    def to_continuous(self):
        if self.pose is not None:
            self.pose = self.pose.to_continuous()
        if self.balance is not None:
            self.balance = self.balance.to_continuous()

        return self

    @staticmethod
    def build_from_numpy(other: SamconTargetPose, dtype=torch.float64):
        res = PyTorchSamconTargetPose()
        if other.pose is not None:
            res.pose = PyTorchTargetPose.build_from_numpy(other.pose, dtype=dtype)
        if other.balance is not None:
            res.balance = PyTorchBalanceTarget.build_from_numpy(other.balance, dtype=dtype)

        return res

    def sub_seq(self, start: int = 0, end: Optional[int] = None, is_copy: bool = True):
        res = PyTorchSamconTargetPose()
        if end is None:
            end = self.num_frames
        if end == 0:
            return res
        if self.pose is not None:
            res.pose = self.pose.sub_seq(start, end, is_copy)
        if self.balance is not None:
            res.balance = self.balance.sub_seq(start, end, is_copy)

        return res

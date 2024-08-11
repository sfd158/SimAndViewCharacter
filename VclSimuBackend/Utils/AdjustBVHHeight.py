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

"""
for same motion, height of some joints may < 0
we can adjust by pytorch optimize based method..
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from typing import List, Union

from VclSimuBackend.Common.SmoothOperator import GaussianBase

from ..Common.MathHelper import RotateType
from ..pymotionlib import BVHLoader
from ..pymotionlib.MotionData import MotionData
from ..pymotionlib.PyTorchMotionData import PyTorchMotionData
from ..Utils.MothonSliceSmooth import smooth_motion_data

from ..DiffODE import DiffQuat


class WeightSummary:
    """
    Loss weights for optimize motion
    """
    def __init__(self):
        self.w_height_avoid: float = 100.0
        self.w_smooth: float = 2000.0
        self.w_close_initial: float = 200.0
        self.max_epoch: int = 300


class AdjustBVHHeight:
    def __init__(
        self,
        motion: Union[str, MotionData],
        rotate_type=RotateType.AxisAngle,
        ):
        if isinstance(motion, str):
            motion: MotionData = BVHLoader.load(motion)
        self.motion: MotionData = motion
        self.diff_motion = PyTorchMotionData()
        self.diff_motion.build_from_motion_data(self.motion)
        self.num_frames = motion.num_frames

        self.rotate_type: RotateType = rotate_type
        self.loss_weight = WeightSummary()
        optimize_names: List[str] = [
            "lHip", "lKnee", "lAnkle",
            "rHip", "rKnee", "rAnkle",
            "lTorso_Clavicle", "lShoulder", "lElbow",
            "rTorso_Clavicle", "rShoulder", "rElbow"
            ]
        self.joint_optim_index: List[int] = [motion.joint_names.index(name) for name in optimize_names]

        # not optimize root now..
        rotate_shape = (self.num_frames, len(self.joint_optim_index))
        self.rotate_quat_shape = rotate_shape + (4,)
        rotate_inputs = self.diff_motion.joint_rotation[:, self.joint_optim_index, :].clone()

        if self.rotate_type == RotateType.AxisAngle:
            self.rotate_shape = rotate_shape + (3,)
            rotate_inputs = DiffQuat.quat_to_rotvec(rotate_inputs.view(-1, 4)).view(self.rotate_shape)
            self.rotate_gt: torch.Tensor = rotate_inputs.clone()
        elif self.rotate_type == RotateType.Vec6d:
            self.rotate_shape = rotate_shape + (3, 2)
            rotate_inputs = DiffQuat.quat_to_vec6d(rotate_inputs.view(-1, 4)).view(self.rotate_shape)
            self.rotate_gt: torch.Tensor = rotate_inputs.clone()
        else:
            raise NotImplementedError

        self.rotate_inputs_parameter = nn.Parameter(rotate_inputs, requires_grad=True)
        self.optimizer = Adam([self.rotate_inputs_parameter], lr=0.05)

        self.diff_motion.recompute_joint_global_info()
        pos = self.diff_motion.joint_position[..., 1]
        self.eps = 0.1
        self.not_opt_index = torch.min(pos, dim=-1)[0] > self.eps

    def run(self):
        for epoch in range(self.loss_weight.max_epoch):
            self.optimizer.zero_grad()
            if self.rotate_type == RotateType.Vec6d:
                rotate_quat: torch.Tensor = DiffQuat.vec6d_to_quat(self.rotate_inputs_parameter.view(-1, 3, 2)).view(self.rotate_quat_shape)
            elif self.rotate_type == RotateType.AxisAngle:
                rotate_quat: torch.Tensor = DiffQuat.quat_from_rotvec(self.rotate_inputs_parameter.view(-1, 3)).view(self.rotate_quat_shape)
            else:
                raise NotImplementedError
            self.diff_motion.load_rot_trans(self.motion)
            self.diff_motion.joint_rotation[:, self.joint_optim_index, :] = rotate_quat
            self.diff_motion.recompute_joint_global_info()
            close_to_init = self.loss_weight.w_close_initial * F.mse_loss(self.rotate_inputs_parameter, self.rotate_gt)
            smooth = self.loss_weight.w_smooth * torch.mean((self.rotate_inputs_parameter[1:] - self.rotate_inputs_parameter[:-1]) ** 2)

            pos = self.diff_motion.joint_position[..., 1]
            if torch.any(pos < self.eps):
                height_avoid = self.loss_weight.w_height_avoid * torch.mean(pos[pos < self.eps] ** 2)
            else:
                height_avoid = torch.as_tensor(0.0)

            loss = close_to_init + smooth + height_avoid
            loss.backward()
            self.rotate_inputs_parameter.grad[self.not_opt_index] = 0
            self.optimizer.step()
            print(
                f"epoch = {epoch}, "
                f"loss = {loss.item():.7f}, close to init = {close_to_init.item():.7f}, "
                f"height avoid = {height_avoid.item():.7f}, "
                f"smooth = {smooth.item():.7f}"
            )

        with torch.no_grad():
            self.diff_motion.load_rot_trans(self.motion)
            self.diff_motion.joint_rotation[:, self.joint_optim_index, :] = rotate_quat
            ret_motion = self.diff_motion.export_to_motion_data()
            BVHLoader.save(ret_motion, "test.bvh")
            # smooth ret motion
            smooth_ret_motion = smooth_motion_data(ret_motion, GaussianBase(5), "test-smooth.bvh")
            return ret_motion

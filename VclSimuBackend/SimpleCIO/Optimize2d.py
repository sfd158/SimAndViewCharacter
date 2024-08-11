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
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import LBFGS, SGD, AdamW, Adam
from typing import Dict, Optional
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from ..Common.MathHelper import MathHelper, RotateType
from ..DiffODE import DiffQuat
from ..pymotionlib import BVHLoader
from ..pymotionlib.MotionData import MotionData
from ..pymotionlib.PyTorchMotionData import PyTorchMotionData
from ..Utils.Camera.CameraNumpy import CameraParamNumpy
from ..Utils.Camera.CameraPyTorch import CameraParamTorch
from ..Utils.Dataset.StdHuman import stdhuman_to_unified_name


fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")


class Optimize2d:

    def __init__(self, motion: MotionData, pos_2d: np.ndarray, camera_np: CameraParamNumpy, device=cpu_device) -> None:
        self.device = device
        motion = motion.remove_end_sites()
        print(f"input motion fps = {motion.fps}")
        # maybe we should downsample here...
        self.sub_fps = 10
        self.sub_fps_ratio = int(motion.fps // self.sub_fps)
        self.tot_nj = motion.num_joints

        self.diff_motion = PyTorchMotionData()
        self.diff_motion.build_from_motion_data(motion, torch.float32, device)
        self.rotate_type = RotateType.Vec6d
        self.camera_np = camera_np
        self.camera_torch = CameraParamTorch.build_from_numpy(self.camera_np, torch.float32, device)
        self.unified_index = self.mocap_to_unified(motion)
        self.rotate_gt: torch.Tensor = DiffQuat.quat_to_other_rotate(self.diff_motion.joint_rotation.clone(), self.rotate_type)
        self.rotate_shape = MathHelper.get_rotation_last_shape(self.rotate_type)
        self.rotate_dim: int = np.prod(self.rotate_shape).item()
        self.root_pos_gt: torch.Tensor = self.diff_motion.joint_translation[:, 0, :].clone()

        # down sample and convert as spline..
        sub_rotate_gt: torch.Tensor = self.rotate_gt.view(-1, motion.num_joints, self.rotate_dim)[::self.sub_fps_ratio]
        sub_rotate_gt: torch.Tensor = torch.transpose(sub_rotate_gt, 0, 1)
        sub_pos_gt: torch.Tensor = self.root_pos_gt.view(-1, 3)[::self.sub_fps_ratio]
        sub_time: torch.Tensor = torch.linspace(0, 1, sub_pos_gt.shape[0], device=self.device)
        sub_pos_factor = natural_cubic_spline_coeffs(sub_time, sub_pos_gt)  # Tuple of torch.Tensor
        sub_rotate_factor = natural_cubic_spline_coeffs(sub_time, sub_rotate_gt)  # Tuple of torch.Tensor
        self.spline_sample_t = torch.linspace(0, 1, motion.num_frames, device=self.device)
        print(self.spline_sample_t.dtype)

        self.pos_param = nn.ParameterList([nn.Parameter(node) for node in sub_pos_factor])
        self.rotate_param = nn.ParameterList([nn.Parameter(node) for node in sub_rotate_factor])
        self.optimizer = AdamW([
            {"params": self.pos_param.parameters()},
            {"params": self.rotate_param.parameters()}
        ], lr=5e-5)
        self.epoch = 0
        self.forward_count = 0

        self.best_loss = float("inf")
        self.best_rotate_param: Optional[torch.Tensor] = None
        self.best_pos_param: Optional[torch.Tensor] = None

        self.motion: MotionData = motion
        self.pos_2d_torch: torch.Tensor = torch.as_tensor(pos_2d, dtype=torch.float32, device=device)

    @staticmethod
    def mocap_to_unified(motion: MotionData):
        name_map: Dict[str, int] = {name: index for index, name in enumerate(motion.joint_names)}
        return [name_map[name] for name in stdhuman_to_unified_name]

    def closure(self):
        self.optimizer.zero_grad()
        self.diff_motion.clear()
        self.diff_motion.load_rot_trans(self.motion, torch.float32, self.device)
        # get position and quaternion for all frames.
        pos_spline = NaturalCubicSpline(list(self.pos_param.parameters()))
        pos_sample: torch.Tensor = pos_spline.evaluate(self.spline_sample_t)
        rot_spline = NaturalCubicSpline(list(self.rotate_param.parameters()))
        rot_sample: torch.Tensor = rot_spline.evaluate(self.spline_sample_t)
        rot_sample: torch.Tensor = torch.transpose(rot_sample, 0, 1)
        rot_sample: torch.Tensor = rot_sample.view(rot_sample.shape[:2] + self.rotate_shape)
        quat_sample: torch.Tensor = DiffQuat.convert_to_quat(rot_sample, self.rotate_type)

        self.diff_motion.set_parameter(pos_sample, quat_sample)
        self.diff_motion.recompute_joint_global_info()
        unified_global_3d = self.diff_motion.joint_position[:, self.unified_index, :]
        unified_camera_3d = self.camera_torch.world_to_camera(unified_global_3d)
        unified_camera_2d = self.camera_torch.project_to_2d_linear(unified_camera_3d)
        loss_2d = 100 * F.mse_loss(unified_camera_2d, self.pos_2d_torch)
        loss_smooth_pos = 10 * torch.mean(torch.diff(pos_sample, dim=0) ** 2)
        loss_smooth_rot = 10 * torch.mean(torch.diff(rot_sample, dim=0) ** 2)
        # loss_smooth_rot = 0.0001 * torch.mean(torch.diff(self.rotate_param.view(self.num_frames, -1), dim=-1) ** 2)
        # loss_near_pos = 0.5 * torch.mean((self.pos_param - self.root_pos_gt) ** 2)
        # loss_near_rot = 1 * torch.mean((self.rotate_param - self.rotate_gt) ** 2)
        loss = loss_2d + loss_smooth_pos + loss_smooth_rot # + loss_near_rot #  # + loss_near_pos

        self.forward_count += 1
        if self.forward_count % 20 == 0:
            print(
                f"epoch = {self.epoch}, loss 2d = {loss_2d.item():.4}, "
                f"smooth pos = {loss_smooth_pos.item():.4}, "
                f"smooth rot = {loss_smooth_rot.item():.4}, "
                # f"near pos = {loss_near_pos.item():.4}, "
                # f"near rot = {loss_near_rot.item():.4}"
            )
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_rotate_param = self.rotate_param.state_dict()
            self.best_pos_param = self.pos_param.state_dict()
        if loss.item() > 2 * self.best_loss:
            self.rotate_param.load_state_dict(self.best_rotate_param)
            self.pos_param.load_state_dict(self.best_pos_param)
            print(f"this step is bad. reset to best param")
        else:
            loss.backward()
        return loss

    def export_bvh(self):
        # output diff motion
        if self.epoch % 10 == 0:
            output_motion = self.diff_motion.export_to_motion_data()
            BVHLoader.save(output_motion, f"test-{self.epoch}.bvh")

    def train(self):
        output_motion = self.diff_motion.export_to_motion_data()
        BVHLoader.save(output_motion, f"test-raw.bvh")
        while self.epoch < 10000:
            self.closure()
            self.optimizer.step()
            self.epoch += 1
            self.export_bvh()

        return
        self.rotate_param.requires_grad = True
        self.pos_param.requires_grad = False
        self.optimizer = LBFGS([self.rotate_param], lr=1e-2)
        while self.epoch < 200:
            self.optimizer.step(self.closure)
            self.epoch += 1
            self.export_bvh()

        # self.pos_param.requires_grad = True
        # self.optimizer = LBFGS([self.pos_param, self.rotate_param], lr=1e-3)
        # while self.epoch < 300:
        #     self.optimizer.step(self.closure)
        #     self.epoch += 1
        #     self.export_bvh()


if __name__ == "__main__":
    print("main")
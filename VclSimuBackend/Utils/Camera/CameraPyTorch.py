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

import torch
from torch import nn

from typing import Iterable, Optional, Dict, Any, List
from .CameraNumpy import CameraParamNumpy
import RotationLibTorch as DiffQuat

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
cpu_device = torch.device("cpu")

@torch.jit.script
def reshape_vec3_to_dim2(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == 3
    if x.ndim == 1:
        x1: torch.Tensor = x[None, :]
    elif x.ndim == 2:
        x1: torch.Tensor = x
    else:
        x1: torch.Tensor = x.view(-1, 3)
    return x1


@torch.jit.script
def swap_axis_for_view(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[..., 0, None], -x[..., 2, None], x[..., 1, None]], dim=-1)


@torch.jit.script
def swap_axis_for_view_inv(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[..., 0, None], x[..., 2, None], -x[..., 1, None]], dim=-1)


class CameraParamTorch:
    def __init__(
        self,
        cam_id: int,
        center: Iterable[float],
        focal_length: Iterable[float],
        radial_distortion: Iterable[float],
        tangential_distortion: Iterable[float],
        res_w: int, res_h: int, azimuth: int,
        orientation: Optional[Iterable[float]] = None,
        translation: Optional[Iterable[float]] = None,
        dtype=torch.float32,
        device=cpu_device
    ):
        self.cam_id = cam_id
        self.center: torch.Tensor = torch.as_tensor(center, dtype=dtype, device=device)
        self.focal_length: torch.Tensor = torch.as_tensor(focal_length, dtype=dtype, device=device)
        self.radial_distortion: torch.Tensor = torch.as_tensor(radial_distortion, dtype=dtype, device=device)
        self.tangential_distortion: torch.Tensor = torch.as_tensor(tangential_distortion, dtype=dtype, device=device)
        self.res_w = res_w
        self.res_h = res_h
        self.azimuth = azimuth

        # extrinsic
        self.orientation: torch.Tensor = torch.as_tensor(orientation, dtype=dtype, device=device) if orientation is not None else None
        self.translation: torch.Tensor = torch.as_tensor(translation, dtype=dtype, device=device) if translation is not None else None

        self.rotation_to_camera = self._build_rotation_to_camera()
        self.rotation_to_world: torch.Tensor = DiffQuat.quat_inv(self.rotation_to_camera)

    def get_opt_parameter(self):
        """
        Maybe we need not modify the res_w and res_h in optimization.
        """
        return {
            "translation": nn.Parameter(self.translation.detach().clone(), True),
            "orientation": nn.Parameter(DiffQuat.quat_to_vec6d(self.orientation.detach()), True),
            # "focal_length": nn.Parameter(self.focal_length.detach().clone(), True)
        }

    def load_parameter(self, params: Dict[str, nn.Parameter]):
        self.translation = params["translation"]
        self.orientation = DiffQuat.quat_from_vec6d(params["orientation"])
        # self.focal_length = params["focal_length"]
        return self

    def detach_(self):
        self.focal_length = self.focal_length.detach()
        self.radial_distortion = self.radial_distortion.detach()
        self.tangential_distortion = self.tangential_distortion.detach()
        self.orientation = self.orientation.detach()
        self.translation = self.translation.detach()
        return self

    def to(self, dtype=torch.float32, device=cpu_device):
        res = CameraParamTorch(
            self.cam_id,
            self.center.clone(),
            self.focal_length,
            self.radial_distortion,
            self.tangential_distortion,
            self.res_w,
            self.res_h,
            self.azimuth,
            self.orientation,
            self.translation,
            dtype, device)
        return res

    def to_numpy(self, dtype=np.float32) -> CameraParamNumpy:
        result = CameraParamNumpy(
            self.cam_id,
            self.center.detach().cpu().clone().numpy().astype(dtype),
            self.focal_length.detach().cpu().clone().numpy().astype(dtype),
            self.radial_distortion.detach().cpu().clone().numpy().astype(dtype),
            self.tangential_distortion.detach().cpu().clone().numpy().astype(dtype),
            self.res_w,
            self.res_h,
            self.azimuth,
            self.orientation.detach().cpu().clone().numpy().astype(dtype)
        )
        return result

    @staticmethod
    def build_dict_from_numpy(rhs: Dict, dtype=torch.float32, device=cpu_device) -> Dict:
        assert isinstance(rhs, Dict)
        result = {}
        for key, value in rhs.items():
            if isinstance(value, CameraParamNumpy):
                result[key] = CameraParamTorch.build_from_numpy(value, dtype, device)
            elif isinstance(value, CameraParamTorch):
                result[key] = copy.deepcopy(value)
            elif isinstance(value, List):
                result[key] = CameraParamTorch.build_list_from_numpy(value, dtype, device)
            elif isinstance(value, Dict):
                result[key] = CameraParamTorch.build_dict_from_numpy(value, dtype, device)
            else:
                raise ValueError

        return result

    @staticmethod
    def build_list_from_numpy(rhs: List, dtype=torch.float32, device=cpu_device) -> List:
        assert isinstance(rhs, List)
        result = []
        for key in rhs:
            if isinstance(key, Dict):
                result.append(CameraParamTorch.build_dict_from_numpy(key, dtype, device))
            elif isinstance(key, List):
                result.append(CameraParamTorch.build_list_from_numpy(key, dtype, device))
            elif isinstance(key, CameraParamNumpy):
                result.append(CameraParamTorch.build_from_numpy(key, dtype, device))
            elif isinstance(key, CameraParamTorch):
                result.append(copy.deepcopy(key))
            else:
                raise ValueError

        return result

    @staticmethod
    def build_from_numpy(rhs: CameraParamNumpy, dtype=torch.float32, device=torch.device("cpu")):
        assert isinstance(rhs, CameraParamNumpy)
        res = CameraParamTorch(
            rhs.cam_id, rhs.center, rhs.focal_length, rhs.radial_distortion,
            rhs.tangential_distortion, rhs.res_w, rhs.res_h, rhs.azimuth,
            rhs.orientation, rhs.translation, dtype, device
        )
        return res

    @staticmethod
    def project_to_2d_common(f: torch.Tensor, c: torch.Tensor, k: torch.Tensor, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        print("Warning: this is slow. please use the linear projection instead", flush=True)
        xx: torch.Tensor = torch.clamp(x[..., :2] / x[..., 2:], min=-1, max=1)
        r2: torch.Tensor = torch.sum(xx[..., :2] ** 2, dim=xx.ndim - 1, keepdim=True)
        radial: torch.Tensor = torch.as_tensor(1.0) + \
                               torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=r2.ndim - 1), dim=r2.ndim - 1, keepdim=True)
        tan: torch.Tensor = torch.sum(p * xx, dim=xx.ndim - 1, keepdim=True)
        xxx: torch.Tensor = xx * (radial + tan) + p * r2

        result: torch.Tensor = f * xxx + c
        return result

    @staticmethod
    def divide_fckp(fckp: torch.Tensor) -> torch.Tensor:
        assert fckp.shape[-1] == 9
        f: torch.Tensor = fckp[..., :2]  # (batch size, 2)
        c: torch.Tensor = fckp[..., 2:4]  # (batch size, 2)
        k: torch.Tensor = fckp[..., 4:7]  # (batch size, 3)
        p: torch.Tensor = fckp[..., 7:9]  # (batch size, 2)

        return f, c, k, p

    @staticmethod
    def project_to_2d_batch(fckp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_joint, xyz_dim = x.shape
        assert xyz_dim == 3
        f, c, k, p = CameraParamTorch.divide_fckp(fckp)

        return CameraParamTorch.project_to_2d_common(f, c, k, p, x)

    def get_fckp_tuple(self):
        f = self.focal_length
        c = self.center
        k = self.radial_distortion
        p = self.tangential_distortion
        return f, c, k, p

    def get_fckp_tensor(self) -> torch.Tensor:
        return torch.cat(self.get_fckp_tuple(), dim=0)

    def project_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param
        """
        assert x.shape[-1] == 3
        # focal_length, center, radial_distortion, tangential_distortion
        f, c, k, p = self.get_fckp_tuple()
        return self.project_to_2d_common(f, c, k, p, x)

    def rebuild_to_3d(self, pos2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        alpha = (pos2d - self.center) / self.focal_length
        xy = alpha * depth[..., None]
        xyz = torch.cat([xy, depth[..., None]], dim=-1)
        return xyz

    def project_to_2d_linear(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 3
        f: torch.Tensor = self.focal_length
        c: torch.Tensor = self.center

        xx: torch.Tensor = x[..., :2] / torch.clamp(x[..., 2:], 1e-2, 1e2)
        result: torch.Tensor = f * xx + c
        return result

    @staticmethod
    def convert_rotation_to_world(quat: torch.Tensor, camera_orient: torch.Tensor) -> torch.Tensor:
        """
        Test OK on human3.6 camera parameter..
        """
        matrix: torch.Tensor = DiffQuat.quat_to_matrix(quat)
        mat: torch.Tensor = torch.as_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=quat.dtype, device=quat.device)
        camera_mat: torch.Tensor = DiffQuat.quat_to_matrix(camera_orient)
        new_camera_mat: torch.Tensor = mat @ camera_mat
        new_mat: torch.Tensor = new_camera_mat @ matrix
        new_quat: torch.Tensor = DiffQuat.quat_from_matrix(new_mat)
        return new_quat

    @staticmethod
    def camera_to_world_batch(x: torch.Tensor, cam_trans: torch.Tensor, camera_rot: torch.Tensor, swap_axis: bool = True) -> torch.Tensor:
        x1: torch.Tensor = reshape_vec3_to_dim2(x)  # (*, 3)
        x2: torch.Tensor = DiffQuat.quat_apply(camera_rot, x1)  # (*, 3)
        x3: torch.Tensor = x2 + cam_trans
        x4: torch.Tensor = x3.view(x.shape)

        if swap_axis:
            x4: torch.Tensor = swap_axis_for_view_inv(x4)

        return x4

    def camera_to_world(self, x: torch.Tensor, swap_axis: bool = True) -> torch.Tensor:
        """
        convert 3d position from camera coordinate to world coordinate
        :param
        :return: torch.Tensor, R * x + t
        """
        x1: torch.Tensor = reshape_vec3_to_dim2(x)  # (*, 3)
        quat: torch.Tensor = torch.tile(self.orientation, (x1.shape[0], 1))  # (*, 4)
        x2: torch.Tensor = DiffQuat.quat_apply(quat, x1)  # (*, 3)
        x3: torch.Tensor = x2 + self.translation[None, :]
        x4: torch.Tensor = x3.type_as(x).view(x.shape)

        if swap_axis:
            x4: torch.Tensor = swap_axis_for_view_inv(x4)

        return x4

    def world_to_camera(self, x: torch.Tensor, swap_axis: bool = True) -> torch.Tensor:
        if swap_axis:  # this option is add by Zhenhua Song..
            # in ode simulation, y is up vector
            # however, in human 3.6 dataset, z is up vector..
            x = swap_axis_for_view(x)
        x1: torch.Tensor = reshape_vec3_to_dim2(x)  # (*, 3)
        x2: torch.Tensor = x1 - self.translation[None, :]
        quat: torch.Tensor = torch.tile(DiffQuat.quat_inv(self.orientation[None, :]).view(-1), (x2.shape[0], 1))
        x3: torch.Tensor = DiffQuat.quat_apply(quat, x2)
        x4: torch.Tensor = x3.type_as(x).view(x.shape)

        return x4

    def _build_rotation_to_camera(self):
        dtype, device = self.orientation.dtype, self.orientation.device
        rot = torch.from_numpy(Rotation.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])).as_quat()).to(device, dtype)
        result = DiffQuat.quat_multiply(DiffQuat.quat_inv(self.orientation[None]), rot[None])
        return result

    def convert_rotation_to_camera(self, quat: torch.Tensor) -> torch.Tensor:
        return DiffQuat.quat_multiply(self.rotation_to_camera, quat)

    def convert_rotation_to_world(self, quat: torch.Tensor) -> torch.Tensor:
        return DiffQuat.quat_multiply(self.rotation_to_world, quat)

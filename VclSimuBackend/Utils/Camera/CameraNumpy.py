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
import numba
from scipy.spatial.transform import Rotation
from typing import Iterable, Optional, Tuple, Union
from VclSimuBackend.Common.MathHelper import MathHelper


def swap_axis_for_view(x: np.ndarray):
    """
    used in world to camera method
    to convert from y up to z up
    (x, y, z) => (x, -z, y)
    matrix = (1, 0, 0, 0, 0, -1, 0, 1, 0)
    """
    ret: np.ndarray = x.copy()
    ret[..., 1] = -x[..., 2].copy()
    ret[..., 2] = x[..., 1].copy()
    return ret


y_up_to_z_up = swap_axis_for_view

def swap_axis_for_view_inv(x: np.ndarray):
    """
    used in camera to world method
    to convert from z up to y up
    (x, y, z) => (x, z, -y)
    matrix = (1, 0, 0, 0, 0, 1, 0, -1, 0)
    """
    ret: np.ndarray = x.copy()
    ret[..., 1] = x[..., 2].copy()
    ret[..., 2] = -x[..., 1].copy()
    return ret


z_up_to_y_up = swap_axis_for_view_inv


# define camera format
# Tested on Human 3.6 dataset.
class CameraParamNumpy:
    def __init__(
        self,
        cam_id: Union[int, str],
        center: Iterable[float],
        focal_length: Iterable[float],
        radial_distortion: Iterable[float],
        tangential_distortion: Iterable[float],
        res_w: int,
        res_h: int,
        azimuth: int,
        orientation: Optional[Iterable[float]] = None,
        translation: Optional[Iterable[float]] = None,
        dtype=np.float32,
        do_normalize: bool = True
    ):
        # Intrinsic
        self.cam_id: Union[int, str] = cam_id
        self.center: np.ndarray = np.asarray(center, dtype=dtype)
        self.focal_length: np.ndarray = np.asarray(focal_length, dtype=dtype)
        self.radial_distortion: np.ndarray = np.asarray(radial_distortion, dtype=dtype)
        self.tangential_distortion: np.ndarray = np.asarray(tangential_distortion, dtype=dtype)
        self.res_w: int = res_w
        self.res_h: int = res_h
        self.azimuth: int = azimuth

        # extrinsic
        if orientation is not None:
            orient: np.ndarray = MathHelper.wxyz_to_xyzw(np.asarray(orientation, dtype=dtype))
            self.orientation: Optional[np.ndarray] = orient / np.linalg.norm(orient, axis=-1)
        else:
            self.orientation: Optional[np.ndarray] = None

        self.translation = np.asarray(translation, dtype=dtype) if translation is not None else None

        if do_normalize:
            self.normalize()

    def astype(self, dtype):
        result = CameraParamNumpy(
            self.cam_id,
            self.center,
            self.focal_length,
            self.radial_distortion,
            self.tangential_distortion,
            self.res_w,
            self.res_h,
            self.azimuth,
            self.orientation,
            self.translation,
            dtype,
            False
        )
        return result

    def normalize_screen_coordinates(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == 2
        w, h = self.res_w, self.res_h
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return x / w * 2 - np.array([1, h / w], dtype=x.dtype)

    def image_coordinates(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == 2
        w, h = self.res_w, self.res_h

        # Reverse camera frame normalization
        res: np.ndarray = (x + [1, h / w]) * w / 2

        return res

    def normalize(self):
        # Normalize camera frame
        self.center = self.normalize_screen_coordinates(self.center)
        self.focal_length = self.focal_length / self.res_w * 2
        if self.translation is not None:
            self.translation = self.translation / 1000

        return self

    @staticmethod
    def project_to_2d_common(f: np.ndarray, c: np.ndarray, k: np.ndarray, p: np.ndarray, x: np.ndarray):
        xx: np.ndarray = np.clip(x[..., :2] / x[..., 2:], a_min=-1, a_max=1)
        r2: np.ndarray = np.sum(xx[..., :2] ** 2, axis=xx.ndim - 1, keepdims=True)
        radial: np.ndarray = 1 + np.sum(k * np.concatenate((r2, r2 ** 2, r2 ** 3), axis=r2.ndim - 1),
                                        axis=r2.ndim - 1, keepdims=True)
        tan: np.ndarray = np.sum(p * xx, axis=xx.ndim - 1, keepdims=True)
        xxx: np.ndarray = xx * (radial + tan) + p * r2
        result: np.ndarray = f * xxx + c

        return result

    @staticmethod
    def project_to_2d_batch(fckp: np.ndarray,  x: np.ndarray):
        f, c, k, p = CameraParamNumpy.divide_fckp(fckp)
        return CameraParamNumpy.project_to_2d_common(f, c, k, p, x)

    def project_to_2d(self, x: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D using the Human3.6M camera projection function.
        :return:
        """
        assert x.shape[-1] == 3, ""
        # focal_length, center, radial_distortion, tangential_distortion
        # f, c, k, p = self.get_fckp_tuple()
        return self.project_to_2d_linear(x)

    def camera_to_world(self, x: np.ndarray, swap_axis: bool = True) -> np.ndarray:
        """
        convert 3d position from camera coordinate to world coordinate
        :param
        :return: np.ndarray, R * x + t
        """
        assert x.shape[-1] == 3

        x1: np.ndarray = x if x.ndim <= 2 else x.reshape((-1, 3))
        x2: np.ndarray = Rotation(self.orientation, copy=False).apply(x1)
        x3: np.ndarray = x2 + (self.translation if x2.ndim == 1 else self.translation[None, :])
        x4: np.ndarray = x3.astype(x.dtype).reshape(x.shape)

        if swap_axis:
            x4: np.ndarray = swap_axis_for_view_inv(x4)
        return x4

    def world_to_camera(self, x: np.ndarray, swap_axis: bool = True):
        """
        convert 3d position from world coordinate to camera coordinate
        :param
        :return np.ndarray, R^{-1} * (x - t)
        """
        assert x.shape[-1] == 3
        if swap_axis:  # this option is add by Zhenhua Song..
            # in ode simulation, y is up vector
            # however, in human 3.6 dataset, z is up vector..
            x = swap_axis_for_view(x)
        x1: np.ndarray = x if x.ndim <= 2 else x.reshape((-1, 3))
        x2: np.ndarray = x1 - (self.translation if x1.ndim == 1 else self.translation[None, :])
        x3: np.ndarray = Rotation(self.orientation, copy=False).inv().apply(x2)
        x4: np.ndarray = x3.astype(x.dtype).reshape(x.shape)

        return x4

    def rebuild_to_3d(self, pos2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        alpha = (pos2d - self.center) / self.focal_length
        xy = alpha * depth[..., None]
        xyz = np.concatenate([xy, depth[..., None]], axis=-1)
        return xyz

    def project_to_2d_linear(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == 3
        f = self.focal_length
        c = self.center
        # xx = np.clip(x[..., :2] / x[..., 2:], -1, 1)
        # x' = fx * (x / z) + cx
        # y' = fy * (y / z) + cy
        # ((x' - cx) / fx) z = x
        # ((y' - cy) / fy) z = y
        # az = x, bz = y
        xx: np.ndarray = x[..., :2] / np.clip(x[..., 2:], 1e-2, 1e2)
        result = f * xx + c
        return result

    def get_fckp_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f = self.focal_length
        c = self.center
        k = self.radial_distortion
        p = self.tangential_distortion
        return f, c, k, p

    def get_fckp_ndarray(self) -> np.ndarray:
        return np.concatenate(self.get_fckp_tuple(), axis=0)

    @staticmethod
    def divide_fckp(fckp_: np.ndarray):
        assert fckp_.shape[-1] == 9
        f: np.ndarray = fckp_[..., :2]  # (batch size, 2)
        c: np.ndarray = fckp_[..., 2:4]  # (batch size, 2)
        k: np.ndarray = fckp_[..., 4:7]  # (batch size, 3)
        p: np.ndarray = fckp_[..., 7:9]  # (batch size, 2)

        return f, c, k, p

    def to_torch(self, dtype=None, device="cpu"):
        """
        use import inside this method for saving memory
        when this part is not used, pytorch module will not be imported
        """
        import torch
        from .CameraPyTorch import CameraParamTorch
        if dtype is None:
            dtype = torch.float32
        result = CameraParamTorch.build_from_numpy(self, dtype, device)
        return result

    def convert_rotation_to_camera(self, quat: np.ndarray) -> np.ndarray:
        rot = Rotation.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
        camera_rot = Rotation(self.orientation).inv()
        new_camera_rot = camera_rot * rot
        new_quat: np.ndarray = (new_camera_rot * Rotation(quat.reshape(-1, 4), copy=False, normalize=False)).as_quat().reshape(quat.shape)
        return new_quat

    def convert_rotation_to_world(self, quat: np.ndarray, swap_axis: bool = True) -> np.ndarray:
        if swap_axis:
            rot = Rotation.from_matrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
            new_camera_rot = rot * Rotation(self.orientation)
        else:
            new_camera_rot = Rotation(self.orientation)

        new_quat = (new_camera_rot * Rotation(quat.reshape(-1, 4), copy=False, normalize=False)).as_quat().astype(quat.dtype).reshape(quat.shape)
        return new_quat

    def solve_depth_by_height(self, y: np.ndarray, pos2d: np.ndarray) -> np.ndarray:
        """
        assume the y in global coordinate and uv in camera 2d coordinate is known.
        compute the depth in camera 3d coordinate
        """
        R = Rotation(self.orientation).as_matrix()
        R21, R22, R23 = R[1, :]
        alpha = (pos2d - self.center) / self.focal_length
        coef = (R21 * alpha[..., 0] + R22 * alpha[..., 1] + R23)
        return (y - self.translation[1]) / coef


if __name__ == "__main__":
    pass

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
from scipy.spatial.transform import Rotation
from typing import Optional, Union, List, Tuple

from VclSimuBackend.Utils.Camera.CameraNumpy import CameraParamNumpy, swap_axis_for_view
from VclSimuBackend.Utils.Camera.Human36CameraBuild import CameraParamBuilder
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib import BVHLoader


fdir = os.path.dirname(__file__)
camera_params_dict = CameraParamBuilder.build()


def get_human36_camera(data_idx="S1", cam_idx=0) -> CameraParamNumpy:
    param: CameraParamNumpy = camera_params_dict[data_idx][cam_idx]
    return param


def convert_rotation_to_camera(quat: np.ndarray, camera: CameraParamNumpy) -> np.ndarray:
    """
    Test OK on human3.6 camera parameter..
    """
    matrix: np.ndarray = MathHelper.quat_to_matrix(quat)
    mat: np.ndarray = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    camera_mat: np.ndarray = MathHelper.quat_to_matrix(camera.orientation)
    new_camera_mat: np.ndarray = np.linalg.inv(camera_mat) @ mat
    new_mat: np.ndarray = new_camera_mat @ matrix
    print(new_mat, np.linalg.inv(new_mat))
    new_quat: np.ndarray = MathHelper.matrix_to_quat(new_mat)
    return new_quat


def convert_rotation_to_world(quat: np.ndarray, camera: CameraParamNumpy) -> np.ndarray:
    """
    Test OK on human3.6 camera parameter..
    """
    matrix: np.ndarray = MathHelper.quat_to_matrix(quat)
    mat: np.ndarray = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    camera_mat: np.ndarray = MathHelper.quat_to_matrix(camera.orientation)
    new_camera_mat: np.ndarray = mat @ camera_mat
    new_mat: np.ndarray = new_camera_mat @ matrix
    print(new_mat)
    new_quat: np.ndarray = MathHelper.matrix_to_quat(new_mat)
    return new_quat


def convert_rotation_to_camera_quat(quat: np.ndarray, camera: CameraParamNumpy) -> np.ndarray:
    rot = Rotation.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    camera_rot = Rotation(camera.orientation).inv()
    new_camera_rot = camera_rot * rot
    new_quat: np.ndarray = (new_camera_rot * Rotation(quat)).as_quat()
    return new_quat.astype(quat.dtype)


def convert_rotation_to_world_quat(quat: np.ndarray, camera: CameraParamNumpy) -> np.ndarray:
    rot = Rotation.from_matrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    new_camera_rot = rot * Rotation(camera.orientation)
    new_quat = (new_camera_rot * Rotation(quat)).as_quat().astype(quat.dtype)
    return new_quat


def test_convert_rotation_to_camera():
    bvh_fname = os.path.join(fdir, "../../Tests/CharacterData/WalkF-mocap-100.bvh")
    motion: MotionData = BVHLoader.load(bvh_fname)
    camera: CameraParamNumpy = get_human36_camera()
    world_pos: np.ndarray = motion.joint_position
    camera_pos: np.ndarray = camera.world_to_camera(world_pos)
    new_motion: MotionData = motion.sub_sequence(copy=True)
    new_motion.joint_translation[:, 0, :] = camera_pos[:, 0, :].copy()
    new_quat = convert_rotation_to_camera(motion.joint_rotation[:, 0, :].copy(), camera)
    new_motion.joint_rotation[:, 0, :] = new_quat
    new_motion.recompute_joint_global_info()
    print(np.max(np.abs(new_motion.joint_position[:, :, :] - camera_pos[:, :, :])))  # delta value < 1e-9, it is ok..

    # check new quat..OK
    tmp_quat = convert_rotation_to_world(new_quat, camera)
    ori_quat = motion.joint_rotation[:, 0, :].copy()
    print(tmp_quat[0], ori_quat[0])
    tmp_quat, ori_quat = MathHelper.flip_quat_pair_by_dot(tmp_quat, ori_quat)
    print(np.max(np.abs(tmp_quat - ori_quat)))


def ax_human_limit(sub_plot3d, _traj3d: Optional[np.ndarray] = None):
    radius = 1.7
    _pos_min = _pos_max = _traj3d
    sub_plot3d.set_xlim3d(-radius / 2 + _pos_min[0], radius / 2 + _pos_max[0])
    sub_plot3d.set_ylim3d(-radius / 2 + _pos_min[1], radius / 2 + _pos_max[1])
    sub_plot3d.set_zlim3d(-radius / 2 + _pos_min[2], radius / 2 + _pos_max[2])


def init_2d_plot(in_plot, param: CameraParamNumpy):
    in_plot.invert_yaxis()
    in_plot.set_xlim(0, param.res_w)
    in_plot.set_ylim(param.res_h, 0)
    in_plot.set_aspect('equal')


def draw_human_2d(in_plot, camera_2d: np.ndarray, parent_list: List[int]):
    line_2d_list = []
    num_joints = len(parent_list)
    for joint_idx, parent_idx in zip(range(1, num_joints), parent_list[1:]):
        pos2d_: np.ndarray = camera_2d[[parent_idx, joint_idx], :]
        plot_2d_res = in_plot.plot(pos2d_[:, 0], pos2d_[:, 1])
        line_2d_list.append(plot_2d_res[0])
    return line_2d_list


def draw_human_3d(in_plot, pos3d: np.ndarray, parent_list: List):
    line_3d_list = []
    num_joints = len(parent_list)
    for joint_idx, parent_idx in zip(range(1, num_joints), parent_list[1:]):
        pos3d_: np.ndarray = pos3d[[parent_idx, joint_idx], :]
        plot_3d_res = in_plot.plot(pos3d_[:, 0], pos3d_[:, 1], pos3d_[:, 2])[0]
        line_3d_list.append(plot_3d_res)
    ax_human_limit(in_plot, pos3d[0, :])
    return line_3d_list


def update_human_2d(line_2d_list: List, camera_2d: np.ndarray, parent_list: List):
    num_joints = len(parent_list)
    for joint_idx, parent_idx in zip(range(1, num_joints), parent_list[1:]):
        pos2d_: np.ndarray = camera_2d[[parent_idx, joint_idx], :]
        plot_2d_res = line_2d_list[joint_idx - 1]
        plot_2d_res.set_data(pos2d_[:, 0], pos2d_[:, 1])


def update_human_3d(plot_3d, line_3d_list: List, pos3d: np.ndarray, parent_list: List):
    plot_3d.set_xlabel("x")
    plot_3d.set_ylabel("y")
    plot_3d.set_zlabel("z")

    num_joints = len(parent_list)
    for joint_idx, parent_idx in zip(range(1, num_joints), parent_list[1:]):
        pos3d_: np.ndarray = pos3d[[parent_idx, joint_idx], :]
        plot_3d_res = line_3d_list[joint_idx - 1]
        plot_3d_res.set_data(pos3d_[:, 0], pos3d_[:, 1])
        plot_3d_res.set_3d_properties(pos3d_[:, 2])
    ax_human_limit(plot_3d, pos3d[0, :])


def test_func():
    mat: np.ndarray = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    quat = Rotation.from_matrix(mat).as_quat()
    print(quat)


if __name__ == "__main__":
    test_convert_rotation_to_camera()

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
from scipy.spatial.transform import Rotation as R
import torch
from typing import Union, Dict
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.Common.MathHelper import MathHelper

from VclSimuBackend.Common.DiffQuat import (
    quat_apply, vec6d_to_quat, quat_multiply, y_decompose, quat_inv, quat_to_vec6d, 
    quat_from_matrix, vec6d_to_matrix
)

def sample_rollout(feasible_index: np.ndarray, batch_size: int, rollout_length: int):
    """generate index for rollout sampling

    Args:
        feasible_index (np.ndarray): please make sure [i,i+rollout_length) is useful
        batch_size (int): nop
        rollout_length (int): nop
    """
    begin_idx: np.ndarray = np.random.choice(feasible_index.flatten(), [batch_size, 1])
    bias: np.ndarray = np.arange(rollout_length).reshape(1, -1)
    res_idx: np.ndarray = begin_idx + bias
    return res_idx


def calc_feasible_index(done_flag, rollout_length):
    res_flag = np.ones_like(done_flag).astype(int)
    terminate_idx = np.where(done_flag!=0)[0].reshape(-1, 1)
    bias = np.arange(rollout_length).reshape(1, -1)
    terminate_idx = terminate_idx - bias
    res_flag[terminate_idx.flatten()] = 0
    return np.where(res_flag)[0]


def calc_mean_std(data: Dict[str, np.ndarray]):
    ignores = {"done", "root_q"}
    data = {key: torch.from_numpy(value) if value is not None else None for key, value in data.items()}
    mean = {key: torch.mean(value, 0) if value is not None else None for key, value in data.items()}
    std = {key: torch.clip(torch.std(value, 0), 0.1, 10) if value is not None else None for key, value in data.items()}
    n_data = {key: (value - mean[key]) / std[key] if key not in ignores else value
              for key, value in data.items()}
    return n_data, mean, std



def load_from_file(fname: str):
    """
    Network input:
    x_{t}: Joint position in facing coordinate
    v_{t}: Joint velocity in facing coordinate
    q_{t}: Joint rotation in facing coordinate
    w_{t}: Joint angular velocity in facing coordinate
    
    Network Output:
    dq: Delta rotation of each joint
    dx: Delta position of each joint
    
    Result:
    next rotation: dq * q_{t}
    """
    mocap = BVHLoader.load(fname).resample(20)

    nf = mocap.num_frames
    nj = mocap.num_joints
    pos = mocap.joint_position
    quat = mocap.joint_orientation

    y_quat, _ = MathHelper.y_decompose(quat[:, 0, :])
    y_quat_inv = R(y_quat).inv()

    y_inv_front = y_quat_inv[:-1]
    vel = np.zeros((nf - 1, nj, 3))
    avel = np.zeros((nf - 1, nj, 4))
    avel[..., 3] = 1

    for i in range(nj):
        vel[:, i, :] = y_inv_front.apply(pos[1:, i] - pos[:-1, i])
        avel[:, i, :] = (y_inv_front * R(quat[1:, i, :]) * R(quat[:-1, i, :]).inv()).as_quat()
        quat[:, i, :] = (y_quat_inv * R(quat[:, i, :])).as_quat()
    avel = MathHelper.quat_to_vec6d(avel).reshape(nf - 1, nj, 6)

    pos[:, :, [0, 2]] -= pos[:, 0:1, [0, 2]]
    done = np.zeros(nf - 1, np.float32)
    done[-1] = 1
    ret: Dict[str, np.ndarray] = {"pos": pos[1:], "vel": vel,
           "vec6d": MathHelper.quat_to_vec6d(quat[1:]).reshape(nf - 1, nj, 6), "avel": avel,
           "done": done, "root_q": None}
    
    return {key: value.astype(np.float32) if value is not None else None for key, value in ret.items()}, nj


def load_and_cat_from_file(fname: str, device):
    data_buffer, nj = load_from_file(fname)
    n_data, mean, std = calc_mean_std(data_buffer)
    in_key = ["pos", "vel", "vec6d", "avel"]
    # dim = sum([data_buffer[key].shape[-1] for key in in_key]) * nj
    mean: torch.Tensor = torch.cat([mean[key] for key in in_key], -1).to(device)
    std: torch.Tensor = torch.cat([std[key] for key in in_key], -1).to(device)
    n_data: torch.Tensor = torch.cat([n_data[key] for key in in_key], -1)
    return n_data, mean, std, data_buffer["done"]


def get_next_input(curr_state: torch.Tensor, recon: torch.Tensor, curr_quat: torch.Tensor):
    b, nj = curr_state.shape[:2]
    pred_vel = recon[..., 0:3]
    pred_avel = vec6d_to_matrix(recon[:, :, 3:9].reshape(b, nj, 3, 2))
    next_pos = curr_state[..., 0:3] + pred_vel
    next_quat = quat_multiply(quat_from_matrix(pred_avel), curr_quat)
    root_qy, root_qxz = y_decompose(next_quat[:, 0, :])
    root_qy_inv = quat_inv(root_qy)[:, None].tile(1, nj, 1)
    curr_quat = quat_multiply(root_qy_inv, next_quat)
    curr_pos = quat_apply(root_qy_inv, next_pos - next_pos[:, 0:1, :])
    curr_state = torch.cat([curr_pos, pred_vel,
        quat_to_vec6d(curr_quat).reshape(b, nj, 6),
        pred_avel[..., :2].reshape(b, nj, 6)], dim=-1)
    return curr_state, curr_quat

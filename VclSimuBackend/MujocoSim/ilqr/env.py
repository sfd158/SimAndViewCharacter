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

import os
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_world_size: int = mpi_comm.Get_size()
mpi_rank: int = mpi_comm.Get_rank()

# set num threads

from argparse import ArgumentParser, Namespace
import numpy as np
from gym.envs.mujoco import MujocoEnv
import mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import time
from typing import List, Optional
import xml.etree.ElementTree as ET

from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.Common.MathHelper import MathHelper

from MotionUtils import (
    quat_multiply_forward_fast, quat_to_rotvec_single_fast, quat_to_rotvec_fast, quat_from_rotvec_fast, quat_inv_fast,
    decompose_rotation_single_fast, decompose_rotation_single_pair_fast,
    quat_to_vec6d_single_fast, quat_inv_single_fast, quat_apply_single_fast, wxyz_to_xyzw_single,
    quat_apply_forward_fast, quat_apply_forward_one2many_fast, quat_to_vec6d_fast, six_dim_mat_to_quat_fast, decompose_rotation_pair_one2many_fast,
    quat_apply_single_backward_fast, quat_inv_single_backward_fast,
    fast_llt_linear_solve, fast_cg_linear_solve
)

fdir = os.path.dirname(__file__)


class LQRBase(MujocoEnv):
    xml_path = ""
    metadata = {
        'render_modes': [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        'render_fps': 100
    }
    def __init__(self):
        # read the fps in file
        option_node = ET.parse(self.xml_path).getroot().find("option")
        if option_node is not None:
            timestep = option_node.get("timestep")
            if timestep is not None:
                self.metadata["render_fps"] = int(1.0 / float(timestep))

        super(MujocoEnv, self).__init__(self.xml_path, 1, None, "human")
        self.nu = self.data.ctrl.shape[0]

    def change_camera(self):
        self.renderer.render_step()
        self.viewer.cam.fixedcamid += 1
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer._contacts = True
        self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.renderer.render_step()

    def get_state(self):  # here we should also save the warm start
        return np.concatenate([self.data.qpos, self.data.qvel], axis=0), self.data.qacc_warmstart.copy(), # self.data.act.copy()

    def reset(self, info):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = info["qpos"].copy()
        self.data.qvel[:] = info["qvel"].copy()
        self.data.act[:] = info["act"].copy()
        self.data.qacc_warmstart[:] = info["qacc_warmstart"].copy()
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)


class HumanoidStand(LQRBase):
    # xml_path = os.path.join(fdir, "stdhuman.xml")
    xml_path = r"D:\song\documents\GitHub\mujoco-iLQR\stdhuman.xml"
    y_axis = np.array([0.0, 1.0, 0.0])

    def __init__(self, args):
        super().__init__()
        np.random.seed(233)
        self.args = args
        self.consider_root_loss = False
        self.use_discount = True
        self.gamma = 0.9

        self.zero_velo = np.zeros(self.model.nv - 6)  # for stable pd control
        self.joint_names = np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, node) for node in range(self.model.njnt)])
        self.body_names = np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, node) for node in range(1, self.model.nbody)])
        # # get character attribute
        jnt_bodyid = self.model.jnt_bodyid.copy()
        hinge_ch_body = np.where(np.bincount(jnt_bodyid)[1:] == 1)[0][1:]
        jnt_bodyid -= 1
        self.hinge_index = np.where(np.in1d(jnt_bodyid, hinge_ch_body))[0]
        self.is_hinge = np.zeros(len(self.joint_names), dtype=np.bool_)
        self.is_hinge[self.hinge_index] = True
        self.hinge_names = self.joint_names[self.hinge_index]
        self.ball_index = np.where(~self.is_hinge)[0][1:]
        self.body_mass: np.ndarray = self.model.body_mass[1:].copy()  # (num body,)
        self.total_mass: float = np.sum(self.body_mass)

        # in simulation
        self.quat_y: Optional[np.ndarray] = None
        self.quat_y_inv: Optional[np.ndarray] = None
        self.quat_xz: Optional[np.ndarray] = None
        self.root_xz: Optional[np.ndarray] = None
        self.end_site: Optional[np.ndarray] = None
        self.com: Optional[np.ndarray] = None

        self.kps = self.model.actuator_gear[:, 0].copy()
        self.kds: np.ndarray = self.kps * 0.06
        self.model.actuator_gear[:, 0] = 1
        self.fps = self.metadata["render_fps"]
        self.forward_count = int(self.fps // args.control_fps)
        self.motion = BVHLoader.load(args.mocap_fname).resample(self.fps)
        self.T = args.time_count
        self.discount = self.gamma ** np.arange(self.T + 1)

        self.num_frames = self.motion.num_frames

        self.ref_qpos: np.ndarray = np.zeros((self.num_frames, self.model.nq))
        self.ref_qpos[:, :3] = self.motion.joint_position[:, 0].copy()
        self.ref_qpos[:, 1] += 0.01
        root_quat = self.motion.joint_rotation[:, 0].copy()
        self.ref_qpos[:, 3:7] = MathHelper.xyzw_to_wxyz(root_quat)
        self.euler_order: List[str] = []
        for bvh_index in range(1, self.motion.num_joints):
            if bvh_index in self.motion.end_sites:
                continue
            bvh_jname = self.motion.joint_names[bvh_index]
            is_hinge = bvh_jname in self.hinge_names
            rot = Rotation(self.motion.joint_rotation[:, bvh_index, :])
            if is_hinge:  # convert quaternion to angle. here we should consider plus or minus..
                quat = MathHelper.flip_quat_by_w(rot.as_quat())
                angle: np.ndarray = np.linalg.norm(rot.as_rotvec(), axis=-1) * np.sign(np.sum(quat[:, 0:3], axis=-1))
                self.ref_qpos[:, self.joint_name2id(bvh_jname) + 6] = angle[:]
                self.euler_order.append(chr(ord('X') + np.argmax(np.mean(np.abs(quat[:, 0:3]), axis=0))))
            else:  # decompose rotation as euler angle format..get euler order from joint..
                joint_index: int = min([self.joint_name2id(f"{bvh_jname}_{node}") for node in ["x", "y", "z"]] )
                if joint_index == -1:
                    continue
                euler_order: str = ''.join([node[len(bvh_jname) + 1:] for node in self.joint_names[joint_index:joint_index+3]]).upper()
                angle: np.ndarray = rot.as_euler(euler_order, degrees=False)
                self.ref_qpos[:, joint_index+6:joint_index+9] = angle
                self.euler_order.append(euler_order)
        try:
            angvel = self.motion.compute_rotational_speed(forward=False)
        except:
            pass

        # remove outliers.
        mean_val = np.mean(self.ref_qpos, axis=0)
        delta = np.abs(np.abs(self.ref_qpos - mean_val[None, :]))
        mean_delta = np.mean(delta, axis=0)
        for qdim in range(7, self.model.nq):
            index = np.where(delta[:, qdim] >= max(5 * mean_delta[qdim], 0.5))[0]
            q = self.ref_qpos[:, qdim]
            q[index] = mean_val[qdim]

        self.ref_qpos: np.ndarray = gaussian_filter1d(self.ref_qpos, 5, axis=0)

        ref_qvel = MathHelper.vec_diff(self.ref_qpos, False, self.fps)
        self.ref_qvel = np.concatenate([ref_qvel[:, :3], angvel[:, 0], ref_qvel[:, 7:]], axis=-1)
        self.ref_qacc = MathHelper.vec_diff(self.ref_qvel, False, self.fps)

        root_q_inv = Rotation(root_quat).inv()
        self.ref_facing_velo: np.ndarray = root_q_inv.apply(ref_qvel[:, :3])

        self.init_control = np.zeros((self.num_frames, self.kps.shape[0]))
        self.ref_com = np.zeros((self.num_frames, 3))
        self.ref_end = np.zeros((self.num_frames, self.model.nsensor, 3))
        # self.ref_up = np.zeros((self.num_frames, 3)) # reference up vector
        self.ref_up_vector: np.ndarray = root_q_inv.apply(self.y_axis)

        for frame in range(self.num_frames):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.ref_qpos[frame, :].copy()
            self.data.qvel[:] = self.ref_qvel[frame, :].copy()
            self.data.qacc[:] = self.ref_qacc[frame, :].copy()
            mujoco.mj_inverse(self.model, self.data)
            # torque = kp * (target - curr) - kd * velo
            # target = (torque + kd * velo) / kp + curr
            self.init_control[frame, :] = (self.data.qfrc_inverse[6:] + self.kds * self.ref_qvel[frame, 6:]) / self.kps + self.ref_qpos[frame, 7:]
            mujoco.mj_forward(self.model, self.data)
            self.compute_facing_com()
            self.ref_com[frame, :] = self.com[None, :]
            self.ref_end[frame, :, :] = self.end_site[:, :]

        self.init_control: np.ndarray = gaussian_filter1d(self.init_control, 5, axis=0)
        # for i in range(self.init_control.shape[1]):
        #    plt.plot(self.init_control[:, i])
        #    plt.show()

        self.start_t = 0
        # we need to save qpos, qvel, and qacc_warmstart here.
        # actually, maybe we can store the init state with apply control function..
        self.init_state = []
        self.reset_init_state()

        self.w_qpos = 1.0
        self.w_qvel = 0.001
        self.w_up = 10.0
        self.w_height = 0.1
        self.w_facing_velo = 0.001  # facing velocity of root joint
        self.w_global_pos = 0

        self.diag_dldxx = np.concatenate([np.full(self.model.nq, 2 * self.w_qpos), np.full(self.model.nv, 2 * self.w_qvel)])
        if not self.consider_root_loss:
            self.diag_dldxx[:7] = 0
        self.dldxx = np.diag(self.diag_dldxx)

        self.M_diag = np.arange(6, self.model.nv)
        self.model.dof_damping[6:] = self.kds[:]

    def reset_init_state(self):
        self.init_state = [None for _ in range(self.num_frames)]
        self.reset_by_index(0)
        self.init_state[0] = self.get_state()

    def compute_facing_com(self, data=None):
        if data is None:
            data = self.data
        self.quat_y, self.quat_xz = decompose_rotation_single_pair_fast(wxyz_to_xyzw_single(data.qpos[3:7]), self.y_axis)
        self.quat_xz = quat_to_rotvec_single_fast(self.quat_xz)
        self.quat_y_inv = quat_inv_single_fast(self.quat_y)
        self.root_xz: Optional[np.ndarray] = data.qpos[0:3].copy()
        self.root_xz[1] = 0
        self.end_site: Optional[np.ndarray] = quat_apply_forward_one2many_fast(self.quat_y_inv[None, :], data.site_xpos - self.root_xz[None, :])
        self.com: Optional[np.ndarray] = quat_apply_single_fast(self.quat_y_inv, np.sum(self.body_mass[:, None] * data.xpos[1:], axis=0) / self.total_mass - self.root_xz)

    def reset_by_index(self, index: int):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.ref_qpos[self.forward_count * index].copy()
        self.data.qvel[:] = self.ref_qvel[self.forward_count * index].copy()
        mujoco.mj_forward(self.model, self.data)

    def joint_name2id(self, name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def cost(self, x, u, t):
        cur_time = (t + self.start_t) * self.forward_count
        nq = self.model.nq
        if self.consider_root_loss:
            pos_loss = np.sum((x[:nq] - self.ref_qpos[cur_time]) ** 2)
            vel_loss = np.sum((x[nq:] - self.ref_qvel[cur_time]) ** 2)
        else:
            pos_loss = np.sum((x[7:nq] - self.ref_qpos[cur_time, 7:]) ** 2)
            vel_loss = np.sum((x[nq + 6:] - self.ref_qvel[cur_time, 6:]) ** 2)

        quat_inv = quat_inv_single_fast(wxyz_to_xyzw_single(x[3:7]))
        # here we should also introduce root loss, up direction loss
        if self.w_up > 0:
            up_vec = quat_apply_single_fast(quat_inv, self.y_axis)
            up_loss = np.sum((up_vec - self.ref_up_vector[t]) ** 2)
        else:
            up_loss = 0.0

        # consider height loss
        height_loss = (x[1] - self.ref_qpos[cur_time, 1]) ** 2

        # consider facing velocity loss
        if self.w_facing_velo > 0:
            facing_velo = quat_apply_single_fast(quat_inv, x[nq: nq+3])
            facing_loss = np.sum((facing_velo - self.ref_facing_velo[cur_time, :3]) ** 2)
            # compute angular velocity here
            # facing_angvel = quat_apply_forward_fast()
        else:
            facing_loss = 0.0

        # consider global root position
        if self.w_global_pos > 0:
            global_pos_loss = np.sum((x[:3] - self.ref_qpos[cur_time, :3]) ** 2)
        else:
            global_pos_loss = 0.0

        ret_loss = self.w_qpos * pos_loss + self.w_qvel * vel_loss +\
             self.w_up * up_loss + self.w_height * height_loss + self.w_facing_velo * facing_loss +\
             self.w_global_pos * global_pos_loss

        if self.use_discount:
            ret_loss *= self.discount[t]

        return ret_loss

    def up_gradient(self, q: np.ndarray, t: int):
        quat_inv = quat_inv_single_fast(wxyz_to_xyzw_single(q))
        up_vec = quat_apply_single_fast(quat_inv, self.y_axis)
        up_grad = (2 * self.w_up) * (up_vec - self.ref_up_vector[t])
        # inv: (x, y, z, w) -> (x, y, z, -w)
        q_grad = quat_apply_single_backward_fast(quat_inv, self.y_axis, up_grad)[0]
        q_grad = np.array([-q_grad[3], q_grad[0], q_grad[1], q_grad[2]])
        return q_grad

    def facing_velo_gradient(self, q: np.ndarray, velo: np.ndarray, t: int):
        quat_inv = quat_inv_single_fast(wxyz_to_xyzw_single(q))
        facing_velo = quat_apply_single_fast(quat_inv, velo)
        grad_in = (2 * self.w_facing_velo) * (facing_velo - self.ref_facing_velo[t, :3])
        q_grad, v_grad = quat_apply_single_backward_fast(quat_inv, velo, grad_in)
        q_grad = np.array([-q_grad[3], q_grad[0], q_grad[1], q_grad[2]])
        return q_grad, v_grad

    def dl_dx(self, x: np.ndarray, u: np.ndarray, t: int) -> np.ndarray:
        cur_time = (t + self.start_t) * self.forward_count
        nq = self.model.nq
        ret = np.zeros((nq + self.model.nv))
        ratio = self.discount[t] if self.use_discount else 1.0
        # qpos and qvel loss
        # here we should consider remove root node..
        if self.consider_root_loss:
            ret[:self.model.nq] = (2 * ratio * self.w_qpos) * (x[:nq] - self.ref_qpos[cur_time])
            ret[self.model.nq:] = (2 * ratio * self.w_qvel) * (x[nq:] - self.ref_qvel[cur_time])
        else:
            ret[7:self.model.nq] = (2 * ratio * self.w_qpos) * (x[7:nq] - self.ref_qpos[cur_time, 7:])
            ret[self.model.nq+6:] = (2 * ratio * self.w_qvel) * (x[nq+6:] - self.ref_qvel[cur_time, 6:])

        # here we should also introduce root loss
        if self.w_up > 0:
            q_grad = self.up_gradient(x[3:7], t)
            ret[3:7] += ratio * q_grad

        if self.w_height > 0:
            ret[1] += (2 * ratio * self.w_height) * (x[1] - self.ref_qpos[cur_time, 1])

        if self.w_facing_velo > 0:
            q_grad, v_grad = self.facing_velo_gradient(x[3:7], x[nq:nq+3], t)
            ret[3:7] += ratio * q_grad
            ret[nq:nq + 3] += ratio * v_grad

        if self.w_global_pos > 0:
            ret[0:3] += (ratio * self.w_global_pos) * (x[:3] - self.ref_qpos[cur_time, :3])

        return ret

    def dl_dxx(self, x: np.ndarray, u: np.ndarray, t: int):
        ret = np.diag(self.diag_dldxx)
        # Add gradient for local up vector
        # maybe here we can compute by finite_difference..
        delta = 1e-5
        ratio = self.discount[t] if self.use_discount else 1.0
        ratio_delta = ratio / (2 * delta)
        if self.w_up > 0:
            for i in range(4):
                q = x[3:7].copy()
                q[i] += delta
                g1 = self.up_gradient(q, t)
                q[i] -= 2 * delta
                g2 = self.up_gradient(q, t)
                g = ratio_delta * (g1 - g2)
                ret[3 + i, 3:7] += g[:]

        if self.w_height > 0:
            ret[1] += 2 * ratio * self.w_height

        if self.w_facing_velo > 0:
            nq = self.model.nq
            v = x[nq:nq+3].copy()
            for i in range(4):
                q = x[3:7].copy()
                q[i] += delta
                g1_q, g1_v = self.facing_velo_gradient(q, v, t)
                q[i] -= 2 * delta
                g2_q, g2_v = self.facing_velo_gradient(q, v, t)
                g_q = ratio_delta * (g1_q - g2_q)
                g_v = ratio_delta * (g1_v - g2_v)
                ret[3 + i, 3:7] += g_q[:]
                ret[3 + i, nq:nq+3] += g_v[:]

            q = x[3:7].copy()
            for i in range(3):
                v = x[nq:nq+3].copy()
                v[i] += delta
                g1_q, g1_v = self.facing_velo_gradient(q, v, t)
                v[i] -= 2 * delta
                g2_q, g2_v = self.facing_velo_gradient(q, v, t)
                g_q = ratio_delta * (g1_q - g2_q)
                g_v = ratio_delta * (g1_v - g2_v)
                ret[3:7, nq+i] += g_q[:]
                ret[nq:nq+3, nq+i] += g_v[:]

        return ret

    def reset_init(self):
        x = self.init_state[self.start_t]
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = x[0][:self.model.nq].copy()
        self.data.qvel[:] = x[0][self.model.nq:].copy()
        self.data.qacc_warmstart[:] = x[1].copy()
        mujoco.mj_forward(self.model, self.data)

    def plant_dyn(self, x, u, t, warmstart=None, is_render=False):
        if x is not None:
            self.data.qpos[:] = x[:self.model.nq].copy()
            self.data.qvel[:] = x[self.model.nq:].copy()
        if warmstart is not None:
            self.data.qacc_warmstart[:] = warmstart.copy()
        # self.renderer.render_step()
        for i in range(self.forward_count):
            cur_time = (self.start_t + t) * self.forward_count + i + 1
            signal = self.kps * (u + self.ref_qpos[cur_time, 7:] - self.data.qpos[7:])  # only use P control here. ref_qpos[cur_time, 7:]
            self.data.ctrl[:] = signal[:]
            mujoco.mj_step(self.model, self.data)
            if is_render:
                self.renderer.render_step()

        self.data.time = 0
        return self.get_state()

    def play_bvh(self):
        print(f"Begin play bvh")
        while True:
            for t in range(self.ref_qpos.shape[0]):
                self.reset_by_index(t)
                self.renderer.render_step()
        print(f"After play bvh")

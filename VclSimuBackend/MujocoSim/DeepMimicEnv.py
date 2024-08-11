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

from argparse import ArgumentParser, Namespace
import copy
import gym
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
import math
from mpi4py import MPI
import mujoco
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from scipy.linalg import cho_factor, cho_solve
import time
from typing import List, Optional

from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.MujocoSim.ArgsConfig import parse_args
from MotionUtils import (
    quat_multiply_forward_fast, quat_to_rotvec_single_fast, quat_to_rotvec_fast, quat_from_rotvec_fast, quat_inv_fast,
    decompose_rotation_single_fast, decompose_rotation_single_pair_fast,
    quat_to_vec6d_single_fast, quat_inv_single_fast, quat_apply_single_fast, wxyz_to_xyzw_single,
    quat_apply_forward_fast, quat_apply_forward_one2many_fast, quat_to_vec6d_fast, six_dim_mat_to_quat_fast, decompose_rotation_pair_one2many_fast
)


mpi_comm = MPI.COMM_WORLD
mpi_world_size: int = mpi_comm.Get_size()
mpi_rank: int = mpi_comm.Get_rank()
fdir = os.path.dirname(__file__)
class DeepMimicEnv(MujocoEnv):
    """
    using old version of Mujoco.
    """
    y_axis = np.array([0.0, 1.0, 0.0])
    def __init__(self, args: Namespace) -> None:
        self.metadata = {
            'render_modes': [
                "human",
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            ],
            'render_fps': 100
        }
        super(DeepMimicEnv, self).__init__(args.model_path, 1, None, "human")
        self.args = args
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
        self.kds: np.ndarray = self.kps * 0.05
        self.model.actuator_gear[:, 0] = 1
        self.fps = self.metadata["render_fps"]
        self.forward_count = int(self.fps // args.control_freq)
        self.motion = BVHLoader.load(args.mocap_fname).resample(self.fps)

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
        self.ref_up_vector: np.ndarray = root_q_inv.apply(self.y_axis)

        ref_state = []
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
            ref_state.append(self._get_obs()[None, :])
            self.ref_com[frame, :] = self.com[None, :].copy()
            self.ref_end[frame, :, :] = self.end_site[:, :].copy()

        self.init_control: np.ndarray = gaussian_filter1d(self.init_control, 5, axis=0)
        self.ref_state = np.concatenate(ref_state, axis=0)
        self.state_mean = np.mean(self.ref_state, axis=0, dtype=np.float32)
        self.state_scale = np.std(self.ref_state, axis=0, dtype=np.float32)
        self.state_scale[np.abs(self.state_scale) < 1e-1] = 1
        self.ref_state = (self.ref_state - self.state_mean[None, :]) / self.state_scale[None, :]
        self.start_t = 0
        self.model.dof_damping[6:] = self.kds[:]

        self.state_dim = self._get_obs().size
        self.action_dim = self.model.nu

    def joint_name2id(self, name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    @staticmethod
    def main():
        args = parse_args()
        env = DeepMimicEnv(args)
        while True:
            for i in range(env.num_frames):
                # print(i)
                env.reset(i)
                env.renderer.render_step()
        
        # while True:
        #     obs, reward, done, _ = env.step(env.random_action())
        #     env.render()
        #     if done:
        #        env.reset()
            # time.sleep(0.1)

    def _get_obs(self) -> np.ndarray:
        """
        replace ball joint with 3 hinge joint.
        """
        qpos = self.data.qpos
        qvel = self.data.qvel
        self.quat_y, self.quat_xz = decompose_rotation_single_pair_fast(wxyz_to_xyzw_single(self.data.qpos[3:7]), self.y_axis)
        self.quat_xz = quat_to_rotvec_single_fast(self.quat_xz)
        self.quat_y_inv = quat_inv_single_fast(self.quat_y)
        self.root_xz: Optional[np.ndarray] = self.data.qpos[:3].copy()
        self.root_xz[1] = 0
        self.end_site: Optional[np.ndarray] = quat_apply_forward_one2many_fast(self.quat_y_inv[None, :], self.data.site_xpos - self.root_xz[None, :])
        self.com: Optional[np.ndarray] = quat_apply_single_fast(self.quat_y_inv, np.sum(self.body_mass[:, None] * self.data.xpos[1:], axis=0) / self.total_mass - self.root_xz)
        # qy_inv_dup = np.ascontiguousarray(np.tile(self.quat_y, (self.model.nbody - 1, 1)))
        # facing_body = quat_to_rotvec_fast(quat_multiply_forward_fast(qy_inv_dup, self.data.xquat[1:]))[1].reshape(-1)
        # return np.concatenate([qpos, qvel], dtype=np.float)
        root_vel = quat_apply_single_fast(self.quat_y_inv, qvel[:3])
        root_angvel = quat_apply_single_fast(self.quat_y_inv, qvel[3:6])
        # state = np.concatenate([self.root_body.xpos[1:2], facing_body, self.data.cvel[1:, :3].reshape(-1)], dtype=np.float32)
        state = np.concatenate([self.data.qpos[1:2], self.quat_xz, qpos[7:], root_vel, root_angvel, qvel[6:]], dtype=np.float32)
        return state

    def step(self, action: np.ndarray = None):
        # control by offset or target directly. Assume using cycle motion
        args = self.args
        next_frame = (self.curr_time + self.forward_count) % self.num_frames
        if not args.use_target_pose:
            target_pose = action + self.ref_qpos[next_frame, 7:]
        else:
            target_pose = action

        for i in range(self.forward_count):
            torque = self.kps * (target_pose - self.data.qpos[7:])
            self.data.ctrl[:] = torque[:]
            mujoco.mj_step(self.model, self.data)

        self.curr_time = next_frame
        obs: np.ndarray = self._get_obs()
        com_err = np.linalg.norm(self.com - self.ref_com[self.curr_time])
        fail = com_err > args.com_fail_threshold
        if not fail:
            pose_err = np.linalg.norm(self.data.qpos[7:] - self.ref_qpos[self.curr_time, 7:])
            velo_err = np.linalg.norm(self.data.qvel[6:] - self.ref_qvel[self.curr_time, 6:]) * self.dt
            end_err = np.linalg.norm(self.end_site - self.ref_end[self.curr_time])
            pose_r = args.w_pose * math.exp(-args.a_pose * pose_err)
            com_r = args.w_com * math.exp(-args.a_com * com_err)
            vel_r = args.w_vel * math.exp(-args.a_vel * velo_err)
            end_r = args.w_end_eff * math.exp(-args.a_end_eff * end_err)
            reward = pose_r + com_r + end_r + vel_r
        else:
            reward = 0
        self.step_count += 1
        done = fail or self.step_count >= args.max_rollout_length
        return obs, reward, done, None

    def reset(self, new_frame: Optional[int] = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if new_frame is None:
            new_frame = np.random.randint(0, self.num_frames)
        self.curr_time = new_frame
        self.step_count = 0
        self.data.qpos[:] = self.ref_qpos[self.curr_time].copy()
        self.data.qvel[:] = self.ref_qvel[self.curr_time].copy()
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs

    def reset_by_state(self, info, new_frame):
        mujoco.mj_resetData(self.model, self.data)
        self.curr_time = new_frame
        self.step_count = 0
        self.data.qpos[:] = info["qpos"].copy()
        self.data.qvel[:] = info["qvel"].copy()
        self.data.act[:] = info["act"].copy()
        self.data.qacc_warmstart[:] = info["qacc_warmstart"].copy()
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs

    def random_action(self):
        return 0.05 * np.random.randn(self.action_dim)

    def save_mjstate(self):
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "act": self.data.act.copy(), "qacc_warmstart": self.data.qacc_warmstart.copy()}


if __name__ == "__main__":
    DeepMimicEnv.main()

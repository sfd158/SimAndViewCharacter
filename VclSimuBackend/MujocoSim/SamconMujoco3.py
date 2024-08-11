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
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
import jax
from jax import numpy as jnp
import math
import mujoco
from mujoco import mjx
import numpy as np
import os
import random
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation
import time
from typing import List, Optional, Tuple, Set, Union
import xml.etree.ElementTree as ET

from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.CMA.CMAUpdate import CMAUpdate

from MotionUtils import (
    quat_multiply_forward_fast, quat_to_rotvec_single_fast, quat_to_rotvec_fast, quat_from_rotvec_fast, quat_inv_fast,
    decompose_rotation_single_fast, decompose_rotation_single_pair_fast,
    quat_to_vec6d_single_fast, quat_inv_single_fast, quat_apply_single_fast, wxyz_to_xyzw_single,
    quat_apply_forward_fast, quat_apply_forward_one2many_fast, quat_to_vec6d_fast, six_dim_mat_to_quat_fast, decompose_rotation_pair_one2many_fast
)


fdir = os.path.dirname(__file__)
key = jax.random.PRNGKey(0)

def pickle_start_state(cost, qpos, qvel, warm_acc, n_sample):
    if cost.size == 1:
        start_idx = jnp.zeros(n_sample, jnp.int32)
    else:
        cost = cost[:int(cost.size * (1 - 0.4))]
        cost_min, cost_max = cost[0], cost[-1]
        prob = (1.0 - (cost - cost_min) / (cost_max - cost_min + 1e-9))
        prob /= jnp.sum(prob)  # normalize prob
        start_idx = jax.random.choice(key, cost.size, (n_sample,), True, prob)
    
    return qpos[start_idx], qvel[start_idx], warm_acc[start_idx], start_idx

# pickle_start_state = jax.jit(pickle_start_state)

mjx_model = None

@jax.vmap
def sim_one_iter(
    start_qpos,
    start_qvel,
    start_warmacc,
    ctrl
):
    mjx_data = mjx.make_data(mjx_model)
    qpos = mjx_data.qpos.at[:].set(start_qpos)
    qvel = mjx_data.qvel.at[:].set(start_qvel)
    warmacc = mjx_data.qacc_warmstart.at[:].set(start_warmacc)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, qacc_warmstart=warmacc, ctrl=ctrl)
    ret = mjx.step(mjx_model, mjx_data)
    return ret

sim_one_iter = jax.jit(sim_one_iter)

class SearchTreeNode:
    def __init__(self, cost=None, parent=None, start_qpos=None, start_qvel=None, warm_acc=None, ctrl=None, next_qpos=None, next_qvel=None, next_warm_acc=None) -> None:
        self.cost = cost
        self.parent = parent
        self.start_qpos = start_qpos
        self.start_qvel = start_qvel
        self.warm_acc = warm_acc
        self.ctrl = ctrl
        self.next_qpos = next_qpos
        self.next_qvel = next_qvel
        self.next_warm_acc = next_warm_acc
    
    @staticmethod
    def build_start(data):
        result = SearchTreeNode(jnp.zeros((1,)), None, None, None, None, None,
            jnp.asarray(data.qpos[None]), jnp.asarray(data.qvel[None]), jnp.asarray(data.qacc_warmstart[None]))
        return result


class CharacterBaseEnv(MujocoEnv):
    xml_path = ""
    metadata = {
        'render_modes': [
            "human",
            "rgb_array",
            "depth_array",
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
        
        # renderer = mujoco.Renderer(env.model)
        # self.change_camera()

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
        self.data.qacc_warmstart[:] = info["qacc_warmstart"].copy()
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)


class SimpleSamcon(CharacterBaseEnv):
    """
    Simply run samcon algorithm
    Generate some noisy rollouts, for pre-train PPO network
    We need not Generate full trajectory
    When some trajectory fails, just remove the end of rollout
    """
    xml_path = os.path.join(fdir, "stdhuman-zyx-local.xml")
    y_axis = np.array([0.0, 1.0, 0.0])

    def __init__(self, args):
        super().__init__()
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
        self.body_mass: np.ndarray = self.model.body_mass[1:].copy()  # (num body,)
        self.total_mass: float = np.sum(self.body_mass)

        self.kps = self.model.actuator_gear[:, 0].copy()
        self.kds: np.ndarray = self.kps * 0.05
        self.model.actuator_gear[:, 0] = 1
  
        self.motion = BVHLoader.load(args.mocap_fname).resample(args.control_fps)
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
        
        angvel = self.motion.compute_rotational_speed(forward=False)

        # remove outliers.
        mean_val = np.mean(self.ref_qpos, axis=0)
        delta = np.abs(np.abs(self.ref_qpos - mean_val[None, :]))
        mean_delta = np.mean(delta, axis=0)
        for qdim in range(7, self.model.nq):
            index = np.where(delta[:, qdim] >= max(5 * mean_delta[qdim], 0.5))[0]
            q = self.ref_qpos[:, qdim]
            q[index] = mean_val[qdim]

        self.ref_qpos: np.ndarray = gaussian_filter1d(self.ref_qpos, 5, axis=0)

        ref_qvel = MathHelper.vec_diff(self.ref_qpos, False, args.control_fps)
        self.ref_qvel = np.concatenate([ref_qvel[:, :3], angvel[:, 0], ref_qvel[:, 7:]], axis=-1)

        root_q_inv = Rotation(root_quat).inv()
        self.ref_facing_velo: np.ndarray = root_q_inv.apply(ref_qvel[:, :3])

        self.ref_com = np.zeros((self.num_frames, 3))
        self.ref_end = np.zeros((self.num_frames, self.model.nsensor, 3))
        self.ref_up_vector: np.ndarray = root_q_inv.apply(self.y_axis)

        # for frame in range(self.num_frames):
        #     mujoco.mj_resetData(self.model, self.data)
        #     self.data.qpos[:] = self.ref_qpos[frame, :].copy()
        #     self.data.qvel[:] = self.ref_qvel[frame, :].copy()
            
        #     # torque = kp * (target - curr) - kd * velo
        #     # target = (torque + kd * velo) / kp + curr
        #     self.compute_facing_com()
        #     self.ref_com[frame, :] = self.com[None, :].copy()
        #     self.ref_end[frame, :, :] = self.end_site[:, :].copy()

        # self.start_t = 0

        self.model.dof_damping[6:] = self.kds[:]

        self.cmaes: List[CMAUpdate] = [CMAUpdate(self.ref_qpos[i, 7:].copy(), self.args.sigma, np.ones(self.model.nu)) for i in range(self.num_frames)]

        self.kps_jnp = jnp.asarray(self.kps)

        global mjx_model
        mjx_model = mjx.device_put(self.model)
        self.search_tree = []

    def joint_name2id(self, name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

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

    def search_best_path(self, tree):
        one_idx = len(tree.tree) - 1
        while one_idx >= 0:
            if len(self.tree.tree[one_idx]) == 1:
                break
            one_idx -= 1

        prev_path = [self.tree.tree[i][0] for i in range(one_idx + 1)]
        if len(prev_path) == len(self.tree.tree):
            return prev_path

        best_path, best_cost = None, float("inf")
        for sam_idx, sample in enumerate(self.tree.tree[-1]):
            node, total_cost = sample, 0.0
            curr_path = []
            for level in range(len(self.tree.tree) - 1, one_idx, -1):
                total_cost += node.cost
                curr_path.append(node)
                node = node.parent
            if total_cost < best_cost:
                best_path = curr_path
                best_cost = total_cost

        result = prev_path + best_path[::-1]
        # check parent relation shape
        for sample_idx in range(0, len(result) - 1):
            assert result[sample_idx + 1].parent == result[sample_idx]

        return result

    def play_best_path(self, path: List):
        while True:
            self.load_mjstate(path[0].s0)

            if False:
                self.renderer.render_step()
            for t in range(1, len(path)):
                node = path[t]  # take action
                for i in range(self.forward_count):
                    self.data.ctrl[:] = self.kps * (node.a0[:] - self.data.qpos[7:])
                    mujoco.mj_step(self.model, self.data, 1)
                    self.renderer.render_step()

    def load_mjstate(self, info):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.data.qpos[:] = info["qpos"].copy()
        self.data.qvel[:] = info["qvel"].copy()
        self.data.qacc_warmstart[:] = info["qacc_warmstart"].copy()
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def save_mjstate(self):
        com = np.sum(self.body_mass[:, None] * self.data.xpos[1:], axis=0) / self.total_mass
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "qacc_warmstart": self.data.qacc_warmstart.copy(), "com": com}

    def track_single(self):
        n_sample = 200
        self.reset(0)
        self.search_tree.append(SearchTreeNode.build_start(self.data))
        for index in range(10): # self.num_frames // self.args.control_fps):
            tree_node = self.search_tree[-1]
            next_index = (index + 1)
            qpos, qvel, qwarm_acc, start_index = pickle_start_state(tree_node.cost, tree_node.next_qpos, tree_node.next_qvel, tree_node.next_warm_acc, n_sample)
            pd_offset = jax.random.normal(key, (n_sample, 37))
            ctrl = self.kps_jnp * (pd_offset + jnp.asarray(self.ref_qpos[None, next_index, 7:]) - qpos[:, 7:])
            time_1 = time.time()
            next_data = sim_one_iter(qpos, qvel, qwarm_acc, ctrl)
            time_2 = time.time()
            print(time_2 - time_1)
        
        exit(0)

    def reset(self, new_frame: Optional[int] = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if new_frame is None:
            new_frame = np.random.randint(0, self.num_frames)
        self.curr_time = new_frame
        self.step_count = 0
        self.data.qpos[:] = self.ref_qpos[self.curr_time]
        self.data.qvel[:] = self.ref_qvel[self.curr_time]
        mujoco.mj_forward(self.model, self.data)
        self.compute_facing_com()

    @staticmethod
    def parse_args() -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--model_path", type=str, default=os.path.abspath(os.path.join(fdir, "stdhuman-zyx.xml")))
        parser.add_argument("--mocap_fname", type=str,
            default=os.path.join(fdir, "../../Tests/CharacterData/sfu/0005_Jogging001-mocap-100.bvh"))
        parser.add_argument("--total_sample", type=int, default=10000)
        parser.add_argument("--debug_render", action="store_true", default=True)
        parser.add_argument("--sim_fps", type=int, default=100)
        parser.add_argument("--control_fps", type=int, default=10)
        parser.add_argument("--save_fname", type=str, default="")
        parser.add_argument("--n_sample", type=int, default=6000)
        parser.add_argument("--n_save", type=int, default=800)
        parser.add_argument("--sigma", type=float, default=0.2)
        parser.add_argument("--com_err", type=float, default=0.15)
        # parser.add_argument("--bvh_start", type=int, default=0)
        # parser.add_argument("--bvh_end", type=int, default=120)
        args = parser.parse_args()
        return args

    @staticmethod
    def main(args: Optional[Namespace] = None):
        if args is None:
            args = SimpleSamcon.parse_args()
        np.random.seed(0)
        random.seed(0)
        samcon = SimpleSamcon(args)
        samcon.track_single()


if __name__ == "__main__":
    SimpleSamcon.main()


"""
jit_step = jax.jit(env.step)

"""
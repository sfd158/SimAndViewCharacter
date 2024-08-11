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
from mpi4py import MPI
import math
import mujoco
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
from VclSimuBackend.MujocoSim.ArgsConfig import parse_args
from VclSimuBackend.CMA.CMAUpdate import CMAUpdate
from VclSimuBackend.Samcon.StateTree import Sample, StateTree
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
        self.data.act[:] = info["act"].copy()
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
        np.random.seed(233)
        self.args = args
        self.consider_root_loss = False
        self.use_discount = True
        self.gamma = 0.9

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
        self.forward_count = int(self.fps // args.control_fps)
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
            self.ref_com[frame, :] = self.com[None, :].copy()
            self.ref_end[frame, :, :] = self.end_site[:, :].copy()

        self.init_control: np.ndarray = gaussian_filter1d(self.init_control, 5, axis=0)

        self.start_t = 0
        self.model.dof_damping[6:] = self.kds[:]

        if mpi_rank == 0:
            self.cmaes: List[CMAUpdate] = [CMAUpdate(self.ref_qpos[i, 7:].copy(), self.args.sigma, np.ones(self.model.nu)) for i in range(self.num_frames)]

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

    def update_cma(self, level: List[Sample], curr_frame: int):
        a0: np.ndarray = np.concatenate([sample.a0[None, :] for sample in level], axis=0)  # (?, joint, 3)
        cost: np.ndarray = np.array([sample.cost for sample in level])  # it's sorted..

        # self.cma[k].reset_lambda_mu(cost.size)  # I forget the reason...maybe it's not required...
        self.cmaes[curr_frame].update(a0, cost)  # update cma parameter
        com = self.tree.tree[-1][0].s1["com"][1]
        print(
            f"curr_frame = {curr_frame}, "
            f"sigma = {self.cmaes[curr_frame].sigma:.3f}, "
            f"count = {len(level)},"
            f"com = {com:.3f}"
            f"max cov = {self.cmaes[curr_frame].max_cov:.3f}, "
            f"min cost = {cost[0]:.3f}, max cost = {cost[-1]:.3f}",
            flush=True
        )

    def list_to_2d(self, info: Union[List, np.ndarray]):
        """
        reshape 1d list to shape (worker_size, len / worker_size)
        """
        w, n = mpi_world_size, len(info)
        if isinstance(info, list):
            return [info[w_idx * n // w: (w_idx + 1) * n // w] for w_idx in range(w)]
        elif isinstance(info, np.ndarray):
            return [info[w_idx * n // w: (w_idx + 1) * n // w]
                    if (w_idx+1)*n//w > w_idx*n//w else [] for w_idx in range(w)]
        else:
            raise NotImplementedError

    def pickle_start_state(self, last_level: List[Sample], cost: np.ndarray):
        """
        Selected sample index.
        probility of sample selected with small cost is large

        return:
        start_idx: selected sample indices
        sel_sample: selected samples
        sel_s1: List of s1 in selected samples
        result: shallow copy of selected samples, used as samcon result
        """
        if cost.size > 1:
            upper: float = np.sort(cost)[int(cost.size * (1 - 0.4))]  # large part will be discarded
            cost_new = cost[cost < upper]
        else:
            cost_new = cost

        cost_min: float = np.min(cost_new)
        cost_max: float = np.max(cost_new)
        prob: np.ndarray = (1.0 - (cost_new - cost_min) / (cost_max - cost_min + 1e-9)) ** 1
        prob /= np.sum(prob)  # normalize prob
        start_idx: np.ndarray = np.random.choice(cost_new.size, self.args.n_sample, True, prob)

        # logging.info(f"cost min {cost_min:.3f}, cost max {cost_max:.3f}, cost average {np.mean(cost_new):.3f}")
        sel_sample: List[Sample] = [last_level[idx] for idx in start_idx]
        sel_s1 = [sample.s1 for sample in sel_sample]
        result: List[List[Sample]] = self.list_to_2d([sample.create_child() for sample in sel_sample])
        return start_idx, sel_sample, sel_s1, result

    def sample_level(self, level: List[Sample], start_index: int) -> List[Sample]:
        cost: np.ndarray = np.array([x.cost for x in level])

        # start_idx: selected sample indices, (n_sample,)
        # sel_sample: selected samples, len == n_sample
        # sel_s1: List of s1 in selected samples, len == n_sample
        # result: shallow copy of selected samples used for samcon result, len == num_worker
        _, _, sel_s1, result = self.pickle_start_state(level, cost)

        raw_sample_lists: np.ndarray = self.cmaes[start_index].sample(None, self.args.n_sample)
        info_2d = [self.list_to_2d(info) for info in [sel_s1, raw_sample_lists]]
        send = [tuple([w_idx, start_index] + [info[w_idx] for info in info_2d]) for w_idx in range(mpi_world_size)]
        sub_task = mpi_comm.scatter(send, 0)
        sub_result = self.sim_one_iter(*sub_task)
        recv_list = mpi_comm.gather(sub_result, 0)

        raw_sample_lists = self.list_to_2d(raw_sample_lists)
        samples: List[Sample] = []
        end_index = (start_index + self.forward_count) % self.num_frames
        for recv_idx, recv in enumerate(recv_list):
            w_idx, state1_list, cost_list = recv
            sample_list: np.ndarray = raw_sample_lists[recv_idx]
            for res_idx, res in enumerate(result[w_idx]):
                res.s0 = None
                res.set_val(sample_list[res_idx], cost_list[res_idx], state1_list[res_idx], end_index)
            samples.extend(result[w_idx])
        samples.sort(key=lambda x: x.cost)
        return samples

    def tree_sample(self, tree: StateTree, start_index: int):
        best_cma = copy.deepcopy(self.cmaes[start_index])
        curr_best_cost = self.cmaes[start_index].history_best_cost
        stacked_samples: List[Sample] = []
        best_cost = float("inf")
        for i in range(2):
            samples = self.sample_level(tree.tree[-1], start_index)
            stacked_samples.extend(samples)
            self.update_cma(samples, start_index)
            if samples[0].cost < best_cost:
                best_cma = copy.deepcopy(self.cmaes[start_index])
                best_cost = samples[0].cost
        if curr_best_cost > best_cma.history_best_cost:
            self.cmaes[start_index] = best_cma
        stacked_samples.sort(key=lambda x: x.cost)
        tree.tree.append(stacked_samples[:self.args.n_save])

    def search_best_path(self, tree: StateTree):
        one_idx = len(tree.tree) - 1
        while one_idx >= 0:
            if len(self.tree.tree[one_idx]) == 1:
                break
            one_idx -= 1

        prev_path: List[Sample] = [self.tree.tree[i][0] for i in range(one_idx + 1)]
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

        result: List[Sample] = prev_path + best_path[::-1]
        # check parent relation shape
        for sample_idx in range(0, len(result) - 1):
            assert result[sample_idx + 1].parent == result[sample_idx]

        return result

    def play_best_path(self, path: List[Sample]):
        while True:
            self.load_mjstate(path[0].s0)
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
        self.data.act[:] = info["act"].copy()
        self.data.qacc_warmstart[:] = info["qacc_warmstart"].copy()
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def save_mjstate(self):
        com = np.sum(self.body_mass[:, None] * self.data.xpos[1:], axis=0) / self.total_mass
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "act": self.data.act.copy(), "qacc_warmstart": self.data.qacc_warmstart.copy(), "com": com}

    def generate_rollouts(self):
        """
        Output: List[BodyInfoState], List[Action]
        """
        tot_frame = 0
        sample_buffer = []
        # self.need_render = False # mpi_rank == 0
        while tot_frame < self.args.total_sample:
            # random select start sample
            untrack_set = set(range(self.num_frames))
            while len(untrack_set) > 0 and tot_frame < self.args.total_sample:
                start_index: int = (untrack_set.pop() - 10) % self.num_frames
                self.reset(start_index)
                start_state = self.save_mjstate()
                tree = StateTree()
                self.tree = tree
                tree.tree.append([Sample(None, start_state, start_state)])
                for _ in range(min(self.num_frames, self.args.total_sample - tot_frame)):
                    self.tree_sample(tree, start_index)
                    next_index: int = (start_index + self.forward_count) % self.num_frames
                    com = self.tree[-1][0].s1["com"]
                    if abs(com[1] - self.ref_com[next_index, 1]) > self.args.com_err:
                        tree.tree = tree.tree[:-10]
                        break
                    start_index = next_index

                # gather samples..
                time_set = []
                for level in tree.tree[1:]:
                    sample_buffer.extend(level[:self.args.n_save // 2])
                    time_set.append(level[0].t)
                untrack_set = untrack_set - set(time_set)
                print(f"len(untrack_set) = {len(untrack_set)}", flush=True)

    def track_single(self):
        self.reset(0)
        start_state = self.save_mjstate()
        tree = StateTree()
        self.tree = tree
        tree.tree.append([Sample(None, start_state, start_state)])
        start_index = 0
        for _ in range(10): # self.num_frames // self.args.control_fps):
            self.tree_sample(tree, start_index)
            next_index: int = (start_index + self.forward_count) % self.num_frames
            com: np.ndarray = self.tree[-1][0].s1["com"]
            if abs(com[1] - self.ref_com[next_index, 1]) > self.args.com_err:
                # tree.tree = tree.tree[:-10]
                break
            start_index = next_index
        best_path = self.search_best_path(tree)
        self.play_best_path(best_path)

    def reset(self, new_frame: Optional[int] = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if new_frame is None:
            new_frame = np.random.randint(0, self.num_frames)
        self.curr_time = new_frame
        self.step_count = 0
        self.data.qpos[:] = self.ref_qpos[self.curr_time].copy()
        self.data.qvel[:] = self.ref_qvel[self.curr_time].copy()
        mujoco.mj_forward(self.model, self.data)
        self.compute_facing_com()

    def sim_one_iter(self, work_index: int, start_index: int, start_state: List, offset: np.ndarray):
        end_frame: int = (start_index + self.forward_count) % self.num_frames
        loss_list: np.ndarray = np.zeros(len(start_state))
        end_state_list = []
        for idx, state in enumerate(start_state):
            self.load_mjstate(state)
            target_pose = offset[idx] # + ref_motion[7:]
            for i in range(self.forward_count):
                self.data.ctrl[:] = self.kps * (target_pose[:] - self.data.qpos[7:])
                mujoco.mj_step(self.model, self.data, 1)

            end_state_list.append(self.save_mjstate())

            self.compute_facing_com()
            pose_err = np.linalg.norm(self.data.qpos[7:] - self.ref_qpos[end_frame, 7:])
            velo_err = np.linalg.norm(self.data.qvel[6:] - self.ref_qvel[end_frame, 6:]) # * self.dt
            end_err = np.linalg.norm(self.end_site - self.ref_end[end_frame])
            com_err = np.linalg.norm(self.com - self.ref_com[end_frame])
            loss_list[idx] = 2 * pose_err + 0.001 * velo_err + 30 * end_err + 10 * com_err

        return work_index, end_state_list, loss_list

    def run_sub_worker(self):
        print(f"start child worker", flush=True)
        while True:
            input_data = mpi_comm.scatter(None, 0)
            result = self.sim_one_iter(*input_data)
            mpi_comm.gather(result)

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
        np.random.seed(mpi_rank)
        random.seed(mpi_rank)
        samcon = SimpleSamcon(args)
        if mpi_rank == 0:
            samcon.track_single()
        else:
            samcon.run_sub_worker()


if __name__ == "__main__":
    SimpleSamcon.main()

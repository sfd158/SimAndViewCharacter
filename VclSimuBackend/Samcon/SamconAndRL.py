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
1. Initialize samcon cma center using policy network
2. Sampling and build search tree
3. train the policy network using search tree.
"""

from argparse import ArgumentParser, Namespace
import copy
from mpi4py import MPI
import numpy as np
import os
import random
from threadpoolctl import threadpool_limits
import ModifyODE as ode

from typing import Optional, List, Tuple, Union
from VclSimuBackend.CMA.CMAUpdate import CMAUpdate
from VclSimuBackend.Common.MathHelper import MathHelper, RotateType
from VclSimuBackend.ODESim.PDControler import DampedPDControler
from VclSimuBackend.Render.Renderer import RenderWorld

from VclSimuBackend.pymotionlib import BVHLoader

from VclSimuBackend.Samcon.Loss.SamconLoss import SamconLoss
from VclSimuBackend.Samcon.SamconTargetPose import SamconTargetPose
from VclSimuBackend.Samcon.StateTree import Sample, StateTree

from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState

from MotionUtils import (
    quat_to_rotvec_fast,
    quat_from_rotvec_fast,
    six_dim_mat_to_quat_fast,
    quat_multiply_forward_fast
)

mpi_comm = MPI.COMM_WORLD
mpi_world_size: int = mpi_comm.Get_size()
mpi_rank: int = mpi_comm.Get_rank()
fdir = os.path.dirname(__file__)


class SimpleSamcon:
    """
    Simply run samcon algorithm
    """
    def __init__(self, args: Namespace) -> None:
        # 1. load character and trajectory
        # 2. random generate some rollouts.
        self.args = args
        self.scene = JsonSceneLoader().load_from_file(args.scene_fname)
        self.character = self.scene.character0
        self.num_joints = len(self.character.joints)
        self.scene.set_sim_fps(args.sim_freq)
        self.sample_mask = self.character.joint_info.gen_sample_mask()[None, ...]
        self.damped_pd: DampedPDControler = DampedPDControler(self.character)
        self.aux_control = self.scene.build_aux_controller()
        self.forward_count: int = self.args.sim_freq // self.args.control_fps
        self.target = BVHToTargetBase(args.bvh_fname, args.sim_freq, self.character).init_target()
        self.samcon_target = SamconTargetPose.load2(self.target, self.character, args.sim_freq)
        target = self.samcon_target.pose
        self.set_tar = SetTargetToCharacter(self.character, target)
        self.local_quat: np.ndarray = np.concatenate([target.locally.quat, target.locally.quat[:20]], axis=0)
        self.com_y = np.ascontiguousarray(self.samcon_target.balance.com[:, 1])
        self.num_frames = target.num_frames
        self.loss = SamconLoss(args.conf)
        self.loss.target = self.samcon_target
        self.loss.character = self.character
        self.loss.root_body = self.character.root_body
        self.loss.joint_weights = self.character.joint_info.weights
        if mpi_rank == 0:
            print(f"Load motion from {args.bvh_fname}, num_frame = {self.num_frames}", flush=True)

        self.cmaes: List[CMAUpdate] = [CMAUpdate(
            np.zeros(self.num_joints * 3), self.args.sigma, np.ones(self.num_joints * 3),
            self.args.n_sample, self.args.n_save) for _ in range(self.num_frames)]


    def update_cma(self, level: List[Sample], curr_frame: int):
        a0: np.ndarray = np.concatenate([sample.a0[np.newaxis, :, :] for sample in level], axis=0)  # (?, joint, 3)
        cost: np.ndarray = np.array([sample.cost for sample in level])  # it's sorted..

        # self.cma[curr_frame].reset_lambda_mu(cost.size)  # I forget the reason...maybe it's not required...
        # here, we should use the un-masked action
        self.cmaes[curr_frame].update(a0.reshape(-1, self.num_joints * 3), cost)  # update cma parameter

        print(
            f"curr_frame = {curr_frame}, "
            f"sigma = {self.cmaes[curr_frame].sigma:.3f}, "
            f"count = {len(level)},"
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

    def pickle_start_state(
        self, last_level: List[Sample], cost: np.ndarray
    ) -> Tuple[np.ndarray, List[Sample], List[BodyInfoState], List[List[Sample]]]:
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
        sel_s1: List[BodyInfoState] = [sample.s1 for sample in sel_sample]
        result: List[List[Sample]] = self.list_to_2d([sample.create_child() for sample in sel_sample])
        return start_idx, sel_sample, sel_s1, result

    def sample_level(self, level: List[Sample], start_index: int) -> List[Sample]:
        cost: np.ndarray = np.array([x.cost for x in level])

        # start_idx: selected sample indices, (n_sample,)
        # sel_sample: selected samples, len == n_sample
        # sel_s1: List of s1 in selected samples, len == n_sample
        # result: shallow copy of selected samples used for samcon result, len == num_worker
        _, _, sel_s1, result = self.pickle_start_state(level, cost)

        raw_sample_lists: np.ndarray = self.cmaes[start_index].sample(None, self.args.n_sample).reshape((-1, self.num_joints, 3))
        info_2d = [self.list_to_2d(info) for info in [sel_s1, self.sample_mask * raw_sample_lists]]
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
            if best_cost <= 3:
                break
        if curr_best_cost > best_cma.history_best_cost:
            self.cmaes[start_index] = best_cma
        stacked_samples.sort(key=lambda x: x.cost)
        tree.tree.append(stacked_samples[:self.args.n_save])

    def search_best_path(self, tree: StateTree):
        one_idx = len(tree.tree) - 1
        while one_idx >= 0:
            if len(tree.tree[one_idx]) == 1:
                break
            one_idx -= 1

        prev_path: List[Sample] = [tree.tree[i][0] for i in range(one_idx + 1)]
        if len(prev_path) == len(tree.tree):
            return prev_path

        best_path, best_cost = None, float("inf")
        for sam_idx, sample in enumerate(tree.tree[-1]):
            node, total_cost = sample, 0.0
            curr_path = []
            for level in range(len(tree.tree) - 1, one_idx, -1):
                total_cost += node.cost
                curr_path.append(node)
                node = node.parent
            if total_cost < best_cost:
                best_path = curr_path
                best_cost = total_cost

        print(f"best cost of path is {best_cost:.3f}")
        result: List[Sample] = prev_path + best_path[::-1]
        # check parent relation shape
        for sample_idx in range(0, len(result) - 1):
            assert result[sample_idx + 1].parent == result[sample_idx]

        return result

    def track_single(self):
        self.set_tar.set_character_byframe(0)
        start_state = self.character.save()
        tree = StateTree()
        self.tree = tree
        tree.tree.append([Sample(None, start_state, start_state)])
        start_index = 0
        for _ in range(50): # self.num_frames // self.args.control_fps):
            self.tree_sample(tree, start_index)
            next_index: int = (start_index + self.forward_count) % self.num_frames
            # com: np.ndarray = self.tree[-1][0].s1["com"]
            # if abs(com[1] - self.ref_com[next_index, 1]) > self.args.com_err:
            #    break
            start_index = next_index
        best_path = self.search_best_path(tree)
        # self.play_best_path(best_path)

    def play_best_path(self, path: List[Sample]):
        while True:
            self.character.load(path[0].s0)
            for t in range(1, len(path)):
                node = path[t]  # take action
                for i in range(self.forward_count):
                    # self.data.ctrl[:] = self.kps * (node.a0[:] - self.data.qpos[7:])
                    # self.damped_pd.add_torques_by_quat(action)
                    # here we can add aux control signal
                    self.scene.damped_step_fast_collision()

    def sim_one_iter(self, work_index: int, start_index: int, start_state: List[BodyInfoState], offset: np.ndarray):
        """
        offset.shape = (batch, num joint, 3) in axis angle format.
        """
        ref_motion: np.ndarray = self.local_quat[start_index + 1: start_index + self.forward_count + 1]
        # convert offset to quaternion.
        offset_quat: np.ndarray = quat_from_rotvec_fast(
            offset.astype(np.float64).reshape((-1, 3))).reshape((len(start_state), self.num_joints, 4))
        end_frame: int = (start_index + self.forward_count) % self.num_frames
        loss_list: np.ndarray = np.empty(len(start_state))
        end_state_list = []
        for idx, state in enumerate(start_state):
            self.character.load(state)
            sub_offset = offset_quat[idx]
            for i in range(self.forward_count):
                action: np.ndarray = quat_multiply_forward_fast(sub_offset, ref_motion[i])
                self.damped_pd.add_torques_by_quat(action)
                body_pos, body_rot = self.character.body_info.get_body_pos(), self.character.body_info.get_body_rot()
                aux_torque = self.aux_control.compute_body_torque_gravity(body_pos, body_rot)
                self.character.body_info.add_body_torque(aux_torque)
                # here we can add aux control signal
                self.scene.damped_step_fast_collision()
            end_state_list.append(self.character.save())
            loss_list[idx] = self.loss.loss(end_frame)
        return work_index, end_state_list, loss_list

    def run_sub_worker(self):
        print(f"start child worker", flush=True)
        with threadpool_limits(limits=1, user_api='blas'):
            while True:
                input_data = mpi_comm.scatter(None, 0)
                result = self.sim_one_iter(*input_data)
                mpi_comm.gather(result)

    @staticmethod
    def parse_args() -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--bvh_fname", type=str, default=os.path.join(fdir, "../../Tests/CharacterData/sfu/0008_Walking001-mocap-100.bvh"))
        parser.add_argument("--scene_fname", type=str, default=os.path.join(fdir, "../../Tests/CharacterData/StdHuman/world.json"))
        parser.add_argument("--total_sample", type=int, default=100)
        parser.add_argument("--debug_render", action="store_true", default=True)
        parser.add_argument("--sim_freq", type=int, default=100)
        parser.add_argument("--control_fps", type=int, default=10)
        parser.add_argument("--save_fname", type=str, default=os.path.join(fdir, "SamconAndRL.ckpt"))
        parser.add_argument("--n_sample", type=int, default=2000)
        parser.add_argument("--n_save", type=int, default=200)
        parser.add_argument("--sigma", type=float, default=0.2)
        parser.add_argument("--com_fail_threshold", type=float, default=0.2)

        # for policy network
        parser.add_argument("--mode", type=str, default="train")
        parser.add_argument("--output_target", action="store_true", default=False)
        parser.add_argument("--only_one_frame_target", action="store_true", default=False)
        
        parser.add_argument(
            "--rotate_type_out", type=str, default="AxisAngle",
            help="rotate representation in action. default = AxisAngle."
        )
        parser.add_argument("--network_type", default="mlp")
        parser.add_argument("--use_cycle", action="store_true", default=True)

        # parser.add_argument("--")
        args = parser.parse_args()
        args.forward_count = args.sim_freq // args.control_fps
        args.conf = {"loss_list": [{
            "w_pose": 5,
            "w_root": 3,
            "w_end_effector": 30,
            "w_balance": 10,
            "w_com": 1,
            "w_ang_momentum": 0.05,
            "dv_coef": 0.1,
        }]}
        
        np.random.seed(0)
        random.seed(0)
        ode.SetInitSeed(0)
        return args

    @staticmethod
    def main(args: Optional[Namespace] = None):
        if args is None:
            args = SimpleSamcon.parse_args()
        samcon = SimpleSamcon(args)
        if mpi_rank == 0:
            samcon.track_single()
        else:
            samcon.run_sub_worker()


if __name__ == "__main__":
    SimpleSamcon.main()

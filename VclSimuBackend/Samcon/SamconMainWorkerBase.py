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
import json
import numpy as np
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from .SamconWorkerBase import SamHlp, WorkerInfo, LoadTargetPoseMode
from .SamconWorkerBase import SamconWorkerBase, WorkerInfo, WorkerMode, CMAInfo
from .StateTree import StateTree, Sample
from .SamconWorkerFull import SamconWorkerFull

from ..ODESim.TargetPose import SetTargetToCharacter
from ..ODESim.ODECharacter import ODECharacter
from ..ODESim.ODEScene import ODEScene
from ..ODESim.BodyInfoState import BodyInfoState
from ..ODESim.BVHToTarget import BVHToTargetBase
from ..pymotionlib.MotionData import MotionData
from ..pymotionlib import BVHLoader


class SamconMainWorkerBase(SamconWorkerBase):

    def __init__(
        self,
        samhlp: SamHlp,
        worker_info: Optional[WorkerInfo],
        worker: Optional[SamconWorkerFull] = None,
        scene: Optional[ODEScene] = None,
        sim_character: Optional[ODECharacter] = None,
        load_target_mode: LoadTargetPoseMode = LoadTargetPoseMode.BVH_MOCAP
        ):
        """
        param:
        samhlp,
        worker_info,
        worker: Optional[SamconWorkerFull].
        """
        super(SamconMainWorkerBase, self).__init__(samhlp, worker_info, scene, sim_character)
        self.load_target_mode: LoadTargetPoseMode = load_target_mode
        self.cma_info: CMAInfo = CMAInfo(self.conf, self.character)

        self.worker: Optional[SamconWorkerFull] = worker

        self.tree: StateTree = StateTree()

        self.init_com_h: float = 0.0
        self.save_path: str = samhlp.best_path_fname()
        self.dump_path: str = samhlp.main_dump_fname()

        if load_target_mode == LoadTargetPoseMode.BVH_MOCAP:
            self.motion_input, self.target, self.inv_dyn_target = samhlp.load_target(self.scene, self.character)
        elif load_target_mode == LoadTargetPoseMode.PICKLE_TARGET:  # TODO: Test
            self.motion_input, self.target, self.inv_dyn_target, confidence, self.contact_label, camera_param = \
                samhlp.load_inv_dyn_from_pickle(
                    self.scene, self.character, self.conf["filename"]["invdyn_target"])
            # self.loss.camera_param = camera_param  # we should also scatter camera param to child workers.
        else:
            raise NotImplementedError

        BVHLoader.save(self.motion_input, os.path.join(samhlp.save_folder_i_dname(), "motion-input.bvh"))
        self.tar_set = SetTargetToCharacter(self.character, self.target.pose)

        self.target.pose.compute_global_root_dup(1)

        self.calc_n_iter()

        self.init_search_tree()

        if self.worker is not None and self.worker_info.comm_size > 1:
            self.worker.recieve_target_data(self._get_send_target_piece())

        self.com_list: Optional[np.ndarray] = None
        self.facing_com_list: Optional[np.ndarray] = None
        self.reset_com_list()

        if isinstance(self.conf["filename"]["bvh"], MotionData):
            self.conf["filename"]["bvh"] = ""

        logging.info(json.dumps(self.conf, indent=4, separators=(',', ':')))

        # dump modified json file into the folder..
        dump_conf = copy.deepcopy(self.conf)
        dump_fname_conf = {key: value for key, value in dump_conf["filename"].items() if not key.startswith("_")}
        dump_fname_conf["bvh"] = "motion-input.bvh"
        with open(os.path.join(samhlp.save_folder_i_dname(), "SamconConfig.json"), "w") as fout:
            json.dump(dump_conf, fout)

    @property
    def sim_fps(self):
        return int(self.scene.sim_fps)

    def _get_send_target_piece(self):
        basic_piece = (self.target, self.inv_dyn_target)
        return basic_piece

    def reset_com_list(self, width: Optional[int] = None):
        width: int = (self.n_iter + 1) if width is None else width
        self.com_list: np.ndarray = np.full((self.n_iter + 1, 3), np.inf)
        self.facing_com_list: np.ndarray = self.com_list.copy()  # TODO: fit duplicate window..
        _com, _facing_com = self.body_info.calc_com_and_facing_com_by_body_state(self.tree.tree[0][0].s0)
        self.com_list[0] = _com
        self.facing_com_list[0] = _facing_com

    def com_is_too_far(self, t: int, com: Optional[np.ndarray] = None, max_err_ratio: float = 0.2) -> bool:
        """
        return True if Center of Mass of Character is too far away from reference motion.
        """
        if com is None:
            com = self.body_info.calc_com_by_body_state(self.tree.level(t)[0].s0)

        ref_com: np.ndarray = self.target.balance.com[t * self.sim_cnt]
        return np.linalg.norm(ref_com - com) >= max_err_ratio * self.character.height

    def _calc_facing_com_is_too_far(self, com: np.ndarray, facing_com: np.ndarray,
                                    t: int, com_max_err_ratio: float = 0.2, com_y_max_err_ratio: float = 0.07) -> bool:
        # calc difference between facing_com and reference motion
        idx: int = (t * self.sim_cnt) % self.target.num_frames
        ref_com_y: np.ndarray = self.target.balance.com[idx, 1]
        ref_facing_com: np.ndarray = self.target.balance.facing_com[idx]

        com_ratio = com_max_err_ratio * self.character.height
        com_y_ratio = com_y_max_err_ratio * self.character.height
        return np.abs(com[1] - ref_com_y) >= com_y_ratio or np.linalg.norm(facing_com - ref_facing_com) >= com_ratio

    def facing_com_is_too_far(self, t: int, com_max_err_ratio: float = 0.2, com_y_max_err_ratio: float = 0.07) -> bool:
        """
        return True if Center of Mass of Character is too far away from reference motion in facing coordinate.
        """
        com, facing_com = self.com_list[t], self.facing_com_list[t]
        if com[0] == np.inf or facing_com[0] == np.inf:
            com, facing_com = self.character.body_info.calc_com_and_facing_com_by_body_state(self.tree.tree[t][0].s1)
        return self._calc_facing_com_is_too_far(com, facing_com, t, com_max_err_ratio, com_y_max_err_ratio)

    def init_search_tree(self):
        """
        initialize the search tree
        """
        self.tree.reset()
        self.tar_set.set_character_byframe(0)
        state0 = self.character.save()
        self.init_com_h = self.character.body_info.calc_center_of_mass()[1]
        self.tree.insert(0, [Sample(state0=state0, state1=state0)])

    def select_and_save_samples2(self, t: int, samples: List[Sample], save_cnt: Optional[int] = None) -> List[Sample]:
        """
        only retain n_save samples in the t-th layer.
        return: List[Sample], the t-th layer, which is already sorted.
        """
        self.tree.insert(t, samples)
        self.tree.level(t).sort(key=lambda sample: sample.cost)
        # only take n_save best results
        self.tree.tree[t][:] = self.tree.tree[t][0:self.n_save if save_cnt is None else save_cnt]

        # calc com pos at level t
        com, facing_com = self.body_info.calc_com_and_facing_com_by_body_state(self.tree.tree[t][0].s1)
        self.com_list[t] = com
        self.facing_com_list[t] = facing_com

        # calc index at level t
        for idx, sample in enumerate(self.tree.level(t)):
            sample.index = idx

        return self.tree.level(t)

    def search_best_path(self) -> Tuple[Dict[str, Any], float]:
        # find last level with depth == 1
        ret_dict = {"conf": self.conf}
        one_idx = len(self.tree.tree) - 1
        while one_idx >= 0:
            if len(self.tree.tree[one_idx]) == 1:
                break
            one_idx -= 1

        prev_path: List[Sample] = [self.tree.tree[i][0] for i in range(one_idx + 1)]
        prev_loss: float = sum([node.cost for node in prev_path])
        if len(prev_path) == len(self.tree.tree):
            ret_dict["best_path"] = prev_path
            return ret_dict, prev_loss

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

        loss: float = prev_loss + best_cost
        ret_dict["best_path"] = result
        return ret_dict, loss

    def list_to_2d(self, info: Union[List, np.ndarray]):
        """
        reshape 1d list to shape (worker_size, len / worker_size)
        """
        w, n = self.worker_size, len(info)
        if isinstance(info, list):
            return [info[w_idx * n // w: (w_idx + 1) * n // w] for w_idx in range(w)]
        elif isinstance(info, np.ndarray):
            return [info[w_idx * n // w: (w_idx + 1) * n // w]
                    if (w_idx+1)*n//w > w_idx*n//w else [] for w_idx in range(w)]
        else:
            raise NotImplementedError

    def scatter(self, mode: int, t: int, *args):
        """
        Scatter information to workers

        param:
        mode: WorkerMode. Possible Type:
        t: time step
        args: list of 1d information
        """
        if self.comm_size > 1:
            info_2d = [self.list_to_2d(info) for info in args]
            send = [(0,)] + [tuple([w_idx, mode, t] + [info[w_idx] for info in info_2d])
                             for w_idx in range(self.worker_size)]
            self.comm.scatter(send, 0)
            del info_2d
            del send
        else:
            self.worker.call_single_thread(mode, t, args)

    def gather(self):
        """
        Gather information from workers
        """
        if self.comm_size > 1:
            gather_info = self.comm.gather(None, 0)
            res = list(filter(lambda x: x is not None, gather_info))
            del gather_info
            return res
        else:
            res = [(0,) + self.worker.last_buffer]
            self.worker.last_buffer = None
            return res

    def stop_other_workers(self):
        """
        Send stop command to other workers
        """
        if self.comm_size > 1:
            self.comm.scatter([(None, WorkerMode.STOP.value, None, None)] * self.comm_size, 0)
        else:  # if self.comm_size == 1, do nothing.
            pass

    def send_target_pose_no_scatter(self):
        """
        Compute target pose in main worker, and then send them to sub workers.
        """
        basic_piece = self._get_send_target_piece()
        if self.comm_size > 1:
            self.comm.scatter([(None, WorkerMode.SEND_TARGET_NO_SCATTER, None, [None]) for _ in range(self.worker_size + 1)])
            data = (None, WorkerMode.RECIEVE_TARGET.value, None) + basic_piece
            for i in tqdm(range(1, self.comm_size), f"Send target to workers, frame = {self.target.num_frames}"):
                self.comm.send(data, i)
            self.comm.gather(None)
        else:
            if self.worker is not None:
                self.worker.recieve_target_data(basic_piece)

    @staticmethod
    def get_cost(saved_samples: List[Sample]) -> np.ndarray:
        """
        return: np.ndarray
        """
        return np.array([x.cost for x in saved_samples])

    def pickle_start_state(self, last_level: List[Sample], cost: np.ndarray, sample_cnt: Optional[int] = None):
        """
        Selected sample index.
        probility of sample selected with small cost is large

        return:
        start_idx: selected sample indices
        sel_sample: selected samples
        sel_s1: List of s1 in selected samples
        result: shallow copy of selected samples, used as samcon result
        """
        if sample_cnt is None:
            sample_cnt = self.n_sample

        if cost.size > 1:
            upper: float = np.sort(cost)[int(cost.size * (1 - self.cost_bound))]  # large part will be discarded
            cost_new = cost[cost < upper]
        else:
            cost_new = cost
        cost_min: float = np.min(cost_new)
        cost_max: float = np.max(cost_new)
        prob: np.ndarray = (1.0 - (cost_new - cost_min) / (cost_max - cost_min + 1e-9)) ** self.cma_info.cost_exponent
        prob /= np.sum(prob)  # normalize prob
        start_idx: np.ndarray = np.random.choice(cost_new.size, sample_cnt, True, prob)

        # logging.info(f"cost min {cost_min:.3f}, cost max {cost_max:.3f}, cost average {np.mean(cost_new):.3f}")
        sel_sample: List[Sample] = [last_level[idx] for idx in start_idx]
        sel_s1: List[BodyInfoState] = [sample.s1 for sample in sel_sample]
        result: List[List[Sample]] = self.list_to_2d([sample.create_child() for sample in sel_sample])
        return start_idx, sel_sample, sel_s1, result

    def bvh_loss_eval(self, sim_motion: Union[str, MotionData]):
        """
        Evaluate Samcon Loss via simulated motion and reference motion in bvh format
        """
        if isinstance(sim_motion, str):
            sim_motion = BVHLoader.load(sim_motion)
        print(f"sim motion num frame = {sim_motion.num_frames}, target num frame = {self.target.num_frames}")
        sim_motion._fps = int(self.scene.sim_fps)
        tar = BVHToTargetBase(sim_motion, sim_motion.fps, self.character).init_target()
        tar_set = SetTargetToCharacter(self.character, tar)
        tar_set.set_character_byframe(0)
        loss_list = []
        for k in range(1, self.n_iter + 1):
            t = (k % self.n_iter) * self.sim_cnt
            tar_set.set_character_byframe(t)
            loss_val = self.loss.loss(self.character, self.target, t)
            print(f"k = {k}, loss_val = {loss_val}")
            loss_list.append(loss_val)

        # print("loss", loss_list)
        print(f"avg loss = {sum(loss_list) / len(loss_list)}")
        return loss_list

    def enable_worker_duplicate(self):
        if self.worker is not None:
            self.worker.duplicate_mode = True
        if self.comm_size > 1:
            self.comm.scatter([(None, WorkerMode.ENABLE_DUPLICATE, None, [None]) for _ in range(self.worker_size + 1)])
        self.comm.gather(None)

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
import gc
import logging
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from ..SamconWorkerBase import LoadTargetPoseMode, WorkerMode
from ..SamconMainWorkerBase import WorkerInfo, SamconMainWorkerBase, SamHlp
from ..SamconUpdateScene import SamconUpdateScene
from ..SamconWorkerFull import SamconWorkerFull
from ..StateTree import Sample, StateTree

from ...Common.Helper import Helper
from ...Common.MathHelper import MathHelper
from ...CMA.CMAUpdate import CMAUpdate

from ...ODESim.ODEScene import ODEScene
from ...ODESim.ODECharacter import ODECharacter
from ...ODESim.BodyInfoState import BodyInfoState
from ...ODESim.TargetPose import SetTargetToCharacter, TargetPose
from ...pymotionlib.MotionData import MotionData
from ...pymotionlib import BVHLoader
from ...Utils.MeanQuaternion import calc_mean_quaternion
from ...Utils.Evaluation import calc_nsr


class SamconMainWorkerCMA(SamconMainWorkerBase):
    """
    Improved Samcon Algorithm
    """

    def __init__(
        self,
        samhlp: SamHlp,
        worker_info: Optional[WorkerInfo],
        worker: Optional[SamconWorkerFull] = None,
        scene: Optional[ODEScene] = None,
        sim_character: Optional[ODECharacter] = None,
        load_target_mode: LoadTargetPoseMode = LoadTargetPoseMode.BVH_MOCAP
    ):
        super(SamconMainWorkerCMA, self).__init__(
            samhlp, worker_info, worker, scene, sim_character, load_target_mode)
        self.n_sample: int = self.conf["worker_cma"]["n_sample"]
        self.n_save: int = self.conf["worker_cma"]["n_save"]

        self.init_n_sample()

        self.init_n_iter: int = self.n_iter

        self.cma: Optional[List[CMAUpdate]] = None
        self.best_cma: Optional[List[CMAUpdate]] = []
        # self.best_cost_history: Optional[np.ndarray] = None

        self.loss_window: Optional[np.ndarray] = None
        self.k_start: Optional[int] = None
        self.last_k_start: Optional[int] = None
        self.dup_idx: Optional[int] = 0

        self.not_good_count: int = 0
        self.rollback_count: int = 0

    def rebuild_func(
        self,
        info_str: Optional[str] = "",
        recalc_cma: bool = True
    ):
        print(f"begin rebuild func {info_str}", flush=True)

        if recalc_cma:
            _func = self.joint_info.sample_win.flatten()
            cmaes: List[CMAUpdate] = [CMAUpdate(np.zeros(self.num_joints * 3),
                                                self.cma_info.init_sigma,
                                                _func.copy(),
                                                self.n_sample,
                                                self.n_save) for _ in range(self.n_iter + 1)]
            self.cma = cmaes

        if len(self.best_cma) < len(self.cma):
            self.best_cma.extend(copy.deepcopy(self.cma[-len(self.cma) + len(self.best_cma):]))

        # if self.best_cost_history is None:
        #    self.best_cost_history = np.full(len(self.cma), 23333.0, dtype=np.float64)

        self.loss_window: np.ndarray = np.full((self.cma_info.loss_window, len(self.cma)), 23333.0, dtype=np.float64)
        self.k_start: int = 1
        self.last_k_start: int = 0
        self.reset_com_list(self.n_iter + 1)

    def set_cma_sigma(self, value: float):
        if self.cma is not None:
            for index in range(len(self.cma)):
                self.cma[index].sigma = value
                self.cma[index].init_sigma = value

    def loss_win_no_change(self, idx: int) -> bool:
        win: np.ndarray = self.loss_window[:, idx]
        res = np.abs(np.mean(win) - win[-1]) < 1e-2
        logging.info(f"loss win at {idx} = {win}, res = {res}")
        if np.any(win == np.inf):
            return False
        return res

    def print_mem(self):
        Helper.show_curr_mem()
        logging.info(f"Total samples = {self.tree.total_sample()}")

    def _get_dump_dict_base(self):
        result = {
            "k_start": self.k_start,
            "last_kstart": self.last_k_start,
            "loss_window": self.loss_window,
            "com_list": self.com_list,
            "facing_com_list": self.facing_com_list,
            "dup_idx": self.dup_idx
        }
        return result

    def dumps_path_mode(
        self,
        ret_best_path: Dict[str, Any],
        idx=None,
        save_tree: bool = False
    ):
        ext = "" if idx is None else str(idx)
        with open(self.dump_path + ext, "wb") as fout:
            dump_dict = self._get_dump_dict_base()
            if save_tree:
                dump_dict["tree"] = self.tree
                dump_dict["k_start"] = self.k_start

            dump_dict["ret_best_path"] = ret_best_path
            pickle.dump(dump_dict, fout)

    def loads(self, load_tree: bool = False):
        """
        load from file
        """
        fname: str = self.dump_path
        if os.path.exists(self.dump_path):
            with open(fname, "rb") as f:
                state_dict: Dict[str, Any] = pickle.load(f)
            # self.cma: Optional[List[CMAUpdate]] = state_dict.get("cma", self.cma)
            if load_tree:
                self.tree: Optional[StateTree] = state_dict.get("tree", self.tree)
                self.k_start: Optional[int] = state_dict["k_start"]
            # self.last_k_start: Optional[int] = state_dict["last_kstart"]
            # self.loss_window: Optional[np.ndarray] = state_dict.get("loss_window", None)

            # self.com_list: Optional[np.ndarray] = state_dict["com_list"]
            # self.facing_com_list: Optional[np.ndarray] = state_dict["facing_com_list"]

            self.dup_idx: Optional[int] = state_dict["dup_idx"]

            ret_best_path: Optional[Dict[str, Any]] = state_dict.get("ret_best_path")
            if ret_best_path is not None:
                self.rebuild_func()
                best_path: List[Sample] = ret_best_path["best_path"]
                logging.info(f"Load best path, with len = {len(best_path)}")
                print(f"Load best path, with len = {len(best_path)}", flush=True)

                tree: List[List[Sample]] = [[best_path[0]]]
                for idx in range(1, len(best_path)):
                    best_path[idx].parent = best_path[idx - 1]
                    tree.append([best_path[idx]])
                    self.cma[idx].x_mean = best_path[idx].a0.reshape(-1).copy()
                self.tree.tree = tree

            logging.info(f"Load From {fname}")
            print(f"After loading from file, dup_idx = {self.dup_idx}", flush=True)
        else:
            logging.info(f"{fname} not exist. ignore.")
            # logging.info(f"initial cma: sigma = {self.cma[0].sigma}, diag of cov matrix: {np.diagonal(self.cma[0].c)}")
            print(f"{fname} not exist. ignore.", flush=True)

    def _recv_ode_sim(
        self, k: int,
        result: List[List[Sample]],
        save_cnt: Optional[int],
        sample_lists: List[np.ndarray],
        help_mess: str = ""
        ):
        recv_list = self.gather()
        samples: List[Sample] = []
        for recv_idx, recv in enumerate(recv_list):
            w_idx, falldown, cost_list, state1_list = recv
            sample_list: np.ndarray = sample_lists[recv_idx]
            for res_idx, res in enumerate(result[w_idx]):
                res.s0 = None
                res.set_val(sample_list[res_idx], cost_list[res_idx], state1_list[res_idx], k, falldown[res_idx])
            samples.extend(result[w_idx])
        del recv_list
        if save_cnt is not None:
            self.select_and_save_samples2(k, samples, save_cnt)
        else:
            self.tree.insert(k, samples)
        # self.print_level_k(k, help_mess)

    def sample_level_k_base(
        self, k: int, dup_idx: int, sample_cnt: Optional[int] = None,
        save_cnt: Optional[int] = None, pbar: Optional[tqdm] = None
        ):
        level = self.tree.level(k - 1)
        cost = self.get_cost(level)

        # start_idx: selected sample indices, (n_sample,)
        # sel_sample: selected samples, len == n_sample
        # sel_s1: List of s1 in selected samples, len == n_sample
        # result: shallow copy of selected samples used for samcon result, len == num_worker
        sigma_list = self.cma_info.sigma_ratio
        _, _, sel_s1, result = self.pickle_start_state(level, cost, sample_cnt * len(sigma_list))

        # send info:
        # * mode: WorkerMode.CMA.value
        # * time: k
        # * last time state: sel_s1
        # * cma mean at time k:
        # * cma sigma at time k
        # return: falldown_list, c_list, se_list

        # do simple line search here..
        old_sigma = self.cma[k].sigma
        raw_sample_lists = []
        for sigma_ratio in sigma_list:
            self.cma[k].sigma = old_sigma * sigma_ratio
            raw_sample_list_: np.ndarray = self.cma[k].sample(None, sample_cnt).reshape((-1, self.num_joints, 3))
            raw_sample_lists.append(raw_sample_list_)

            # for debug..
            # if np.isnan(raw_sample_list_).any():
            #     logging.info(self.cma[k].sigma, sample_cnt)
            #     raise ValueError

        self.cma[k].sigma = old_sigma
        raw_sample_lists = np.concatenate(raw_sample_lists, axis=0)
        sample_lists: np.ndarray = self.joint_info.sample_mask[None, ...] * raw_sample_lists
        sample_qs = Rotation.from_rotvec(sample_lists.reshape(-1, 3)).as_quat().reshape((-1, self.num_joints, 4))

        # for debug..
        # if np.isnan(sample_lists).any():
        #     logging.info(sample_lists.tolist())
        #     raise ValueError

        self.scatter(WorkerMode.ODE_SIMU, k, sel_s1, sample_qs)
        self._recv_ode_sim(k, result, save_cnt, self.list_to_2d(raw_sample_lists))
        k_best_cost = self.tree.tree[k][0].cost
        descript = f"{k}, {dup_idx}, {k_best_cost:.3f}, {self.com_list[k, 1]:.3f}"
        if pbar is not None:
            pbar.set_description(descript)
        logging.info(f"{descript}, hist cost = {self.best_cma[k].history_best_cost}")

        self.loss_window[:-1, k] = self.loss_window[1:, k]
        self.loss_window[-1, k] = k_best_cost

        self.cma[k].forward_g += 1
        self.cma[k].history_best_cost = min(self.cma[k].history_best_cost, k_best_cost)

    def sample_level_k(
        self,
        k: int,
        sample_cnt: Optional[int] = None,
        save_cnt: Optional[int] = None,
        pbar: Optional[tqdm] = None
        ):
        hist_best_cost = self.best_cma[k].history_best_cost
        # For the first trial, maybe we can optimize at the process of sample..
        self.sample_level_k_base(k, 0, sample_cnt, save_cnt, pbar=pbar)
        def func():
            cma_backup = copy.deepcopy(self.cma[k])
            loss_backup = self.tree.level(k)[0].cost
            level_backup = self.tree.tree[k]
            return cma_backup, loss_backup, level_backup

        best_param, best_cost, best_samples = func()
        stacked_samples: List[Sample] = self.tree.tree[k].copy()
        def repeat_func(best_param_, best_cost_, best_samples_, repeat_count: int):
            for opt_idx in range(1, repeat_count + 1):
                self.update_cma_k(k)
                self.tree.tree[k] = []
                self.sample_level_k_base(k, opt_idx, sample_cnt, save_cnt, pbar=pbar)
                stacked_samples.extend(self.tree.tree[k])
                if self.tree.tree[k][0].cost < best_cost:
                    best_param_, best_cost_, best_samples_ = func()
                if abs(self.loss_window[-1, k] - self.loss_window[-2, k]) < 1e-6:
                    break
            return best_param_, best_cost_, best_samples_

        best_param, best_cost, best_samples = repeat_func(best_param, best_cost, best_samples, self.cma_info.first_trial_iteration)
        stacked_samples.sort(key=lambda x: x.cost)
        self.tree.tree[k] = stacked_samples[:sample_cnt]

        # for debug visualize
        if self.worker_info.worker_size > 1:
            self.character.load(stacked_samples[0].s1)

        # update cma param by new tree..
        # if self.cma_info.first_trial_iteration > 1:
        #    self.update_cma_k(k)
        if self.tree.tree[k][0].cost < hist_best_cost:
            self.cma[k] = best_param
            self.best_cma[k] = copy.deepcopy(self.cma[k])
            self.best_cma[k].history_best_cost = self.tree.tree[k][0].cost
        cost_k = self.tree.tree[k][0].cost
        if cost_k > 1.6 * hist_best_cost and cost_k > 5 + hist_best_cost and False:
            logging.info(f"resample using best history at {k}")
            stacked_samples = self.tree.tree[k]
            self.tree.tree[k] = []
            self.sample_level_k_base(k, "Re", sample_cnt, save_cnt, pbar=pbar)
            self.tree.tree[k].extend(stacked_samples)
            self.tree.tree[k].sort(key=lambda x: x.cost)
            self.not_good_count += 1

    def start_is_good(self) -> bool:
        """
        judge whether start is good enough
        """
        return not self.facing_com_is_too_far(self.k_start, self.cma_info.com_good_ratio, self.cma_info.com_good_ratio)

    def facing_com_is_bad(self, k):
        return self.facing_com_is_too_far(k, self.cma_info.com_err_ratio, self.cma_info.com_y_err_ratio)

    def step_is_bad(self, k: int, com_is_bad_flag: Optional[bool] = None) -> bool:
        if not self.cma_info.no_falldown_forward:
            if com_is_bad_flag is None:
                com_is_bad_flag = self.facing_com_is_bad(k)
            cost_k = self.tree.tree[k][0].cost
            best_k = self.best_cma[k].history_best_cost
            res = (cost_k >= 2 * best_k and cost_k >= 8 + best_k) or com_is_bad_flag
        else:
            res = self.tree.level(k)[0].falldown

        return res

    def start_in_tree(self) -> bool:
        return self.k_start < self.tree.num_level()

    def start_far_stop(self, k_stop: int) -> bool:
        return self.k_start <= k_stop - self.cma_info.start_decay

    def is_start_cma_min_iter(self):
        return self.cma[self.k_start].forward_g >= self.cma_info.min_iteration

    def is_start_forward(self):
        return self.start_is_good() or self.cma[self.k_start].forward_g >= self.cma_info.iteration \
                or self.loss_win_no_change(self.k_start)

    def direct_forward(self, pbar: tqdm, flag: bool = True):
        for idx in range(self.tree.num_level() - 1, self.k_start + 1, -1):
            if not self.facing_com_is_too_far(idx, self.cma_info.com_good_ratio, self.cma_info.com_good_ratio):
                if flag:
                    pbar.update(idx - self.k_start)
                    logging.info(f"direct forward {self.k_start} -> {idx}")
                    self.k_start = idx
                    if self.k_start < self.tree.num_level() - 1:
                        self.k_start += 1
                    break
                else:
                    flag = True

    def k_start_move_forward(self, k_stop: int, pbar: tqdm):
        logging.info(
            f"before move forward: start = {self.k_start}, stop = {k_stop},"
            f" n_iter = {self.n_iter}, num_level = {self.tree.num_level()}"
        )
        if self.k_start >= self.n_iter and self.start_is_good():
            return True

        if self.cma[self.k_start].forward_g < self.cma_info.min_iteration:
            logging.info(f"break at {self.k_start}, forward_g = {self.cma[self.k_start].forward_g}")
            return False

        while self.k_start < min(self.tree.num_level(), len(self.cma)) and \
                self.cma[self.k_start].forward_g >= self.cma_info.iteration:
            self.k_start += 1
            pbar.update(1)

        if self.k_start >= len(self.cma):
            return True

        while self.start_in_tree() and self.is_start_forward():
            self.k_start += 1
            pbar.update(1)

        logging.info(f"After move forward: start = {self.k_start}")
        return self.k_start >= self.n_iter

    # def judge_early_converge(self, judge_idx: int):
    #     if self.tree.num_level() > judge_idx and \
    #             (self.cma[judge_idx].sigma < self.cma_info.cma_small_eps or
    #              self.cma[judge_idx].sigma >= self.cma_info.cma_large_eps or
    #              self.cma[judge_idx].max_cov >= self.cma_info.cma_large_eps):
    #         logging.info(f"Early converge. reset cma param at {judge_idx}.")
    #         self.cma[judge_idx].reset_to_init(reset_generation=False)

    def eval_best_path(self, fout: str = "test.bvh") -> Tuple[Dict[str, Any], List[BodyInfoState], MotionData]:
        ret_best_path, _ = self.search_best_path()
        fout = os.path.join(os.path.dirname(self.dump_path), fout) if fout is not None else ""

        eval_func = SamconUpdateScene.eval_best_path  # compute best path
        sim_full_result, motion = eval_func(self.worker, ret_best_path["best_path"], fout)

        with open(self.save_path, "wb") as fout_handle:
            pickle.dump(ret_best_path, fout_handle)

        if fout:
            logging.info(f"After eval best path. save to {fout}")
        return ret_best_path, sim_full_result, motion

    def cma_k_gen_info(self, k: int):
        return f"k = {k}, cma generation, {self.cma[k].g}, forward g = {self.cma[k].forward_g}, "

    def update_cma_k(self, k: int):
        level = self.tree.level(k)
        a0 = np.concatenate([sample.a0[np.newaxis, :, :] for sample in level], axis=0)  # (?, joint, 3)
        cost = np.array([sample.cost for sample in level])  # it's sorted..

        # self.cma[k].reset_lambda_mu(cost.size)  # I forget the reason...maybe it's not required...
        # here, we should use the un-masked action
        self.cma[k].update(a0.reshape(-1, self.num_joints * 3), cost)  # update cma parameter

        logging.info(
            self.cma_k_gen_info(k) +
            f"sigma = {self.cma[k].sigma:.3f}, "
            f"count = {len(level)},"
            f"max cov = {self.cma[k].max_cov:.3f}, "
            f"min cost = {cost[0]:.3f}, max cost = {cost[-1]:.3f}"
        )

    def check_need_rollback(self, k: int, pbar: Optional[tqdm] = None, is_bad_flag: Optional[bool] = None):
        if is_bad_flag is None:
            is_bad_flag = self.step_is_bad(k)
        if is_bad_flag or self.not_good_count >= 3:
            # We can rollback here..
            self.rollback_count += 1
            self.not_good_count = 0
            start_old = self.k_start
            self.k_start = max(1, self.k_start - self.cma_info.start_decay)
            if pbar is not None:
                pbar.update(self.k_start - start_old)
            logging.info(f"rollback from {start_old} to {self.k_start}")
            return True
        else:
            return False

    def run_single(self, fout_name: str = "test.bvh"):
        pbar: tqdm = tqdm(total=self.n_iter)
        pbar.update(self.k_start - 1)
        self.rollback_count = 0
        self.not_good_count = 0

        while self.k_start <= self.n_iter:
            self.k_start, start_tmp = min(self.k_start, self.tree.num_level()), self.k_start
            pbar.update(self.k_start - start_tmp)
            logging.info(f"while self.k_start({self.k_start}) <= self.n_iter({self.n_iter})")
            if self.check_need_rollback(self.k_start - 1, pbar):
                if self.rollback_count <= 10 and self.cma[self.k_start - 1].forward_g < self.cma_info.iteration:
                    continue
                else:
                    fail_info = f"{self.k_start} is bad. The samcon fails."
                    logging.info(fail_info)
                    print(fail_info, flush=True)
                    break

            self.tree.tree = self.tree.tree[:self.k_start]

            # using sliding window: avoid memory explode..
            k_stop: int = min(self.k_start + self.cma_info.sliding_window, self.n_iter)

            for k in range(self.k_start, k_stop + 1):
                self.sample_level_k(k, self.n_sample, self.n_save, pbar)  # sample at time k
                com_is_bad_flag = self.facing_com_is_bad(k)
                step_is_bad_flag = self.step_is_bad(k, com_is_bad_flag)
                need_rollback_flag = com_is_bad_flag or (step_is_bad_flag and self.cma[k].forward_g <= self.cma_info.iteration)
                if self.check_need_rollback(k, pbar, need_rollback_flag):
                    k_stop = k
                    while k_stop > 1 and self.facing_com_is_bad(k_stop):
                        self.cma[k_stop].sigma = max(1.0, 1.2 * self.cma[k_stop].sigma)
                        k_stop -= 1
                    k_stop = max(1, k_stop - self.cma_info.stop_decay)
                    self.k_start = min(self.k_start, k_stop)
                    self.tree.tree = self.tree.tree[:k_stop + 1]
                    self.eval_best_path()
                    break

            logging.info(f"k_stop = {k_stop}, tree level count = {self.tree.num_level()}")

            if self.cma_info.direct_forward:
                self.direct_forward(pbar, False)

            if self.k_start_move_forward(k_stop, pbar):
                break

            self.print_mem()
            if self.k_start - self.last_k_start >= 50:
                logging.info("clear dead nodes")
                self.tree.clear_dead_nodes(self.k_start - 2)

            self.last_k_start = self.k_start

        pbar.close()
        print(f"Calc best path", flush=True)

        func_ret = self.eval_best_path(fout_name)
        nsr_sim_motion = func_ret[-1].remove_end_sites(True)
        if nsr_sim_motion.num_frames > self.motion_input.num_frames:
            nsr_sim_motion: MotionData = nsr_sim_motion.sub_sequence(0, self.motion_input.num_frames)

        motion_start: int = 0
        while self.tree.tree[motion_start][0].a0 is None:
            motion_start += 1
        motion_start: int = (motion_start - 1) * self.sim_cnt
        nsr_ref_motion: MotionData = self.motion_input.sub_sequence(
            motion_start, motion_start + nsr_sim_motion.num_frames).remove_end_sites(copy=False)
        nsr_value: float = calc_nsr(nsr_sim_motion, nsr_ref_motion)
        print(f"nsr value = {nsr_value:.3f}", flush=True)
        logging.info(f"nsr value = {nsr_value}")
        self.tree.tree.clear()
        gc.collect()
        self.init_search_tree()
        return func_ret

    def export_pd_target_as_bvh(self, sim_full_result: List[BodyInfoState], motion: MotionData):
        # here we should dump the inverse dynamics into a single bvh file
        cat_pd_target = [node.pd_target[None, ...] for node in sim_full_result[:-1]]
        cat_pd_target.append(cat_pd_target[-1])
        cat_pd_target = np.concatenate(cat_pd_target, axis=0)
        pd_target_motion: MotionData = self.to_bvh.forward_kinematics(
            motion.joint_translation[:, 0, :], motion.joint_rotation[:, 0, :], cat_pd_target
        )
        # insert the end site
        pd_target_motion = self.to_bvh.insert_end_site(pd_target_motion)
        return pd_target_motion

    # Not work
    def run_single_hlp(self):
        self.dup_idx = 0
        self.send_target_pose_no_scatter()
        while self.dup_idx < 1:
            self.rebuild_func()
            self.loads()

            ret_best_path, simu_list, motion = self.run_single()
            self.dup_idx += 1
            self.dumps_path_mode(ret_best_path)
            self.dumps_path_mode(ret_best_path, str(self.dup_idx))
            pdtarget_motion: MotionData = self.export_pd_target_as_bvh(simu_list, motion)
            pdtarget_fname: str = f"{self.save_path}-pd-target-{self.dup_idx}.bvh"
            BVHLoader.save(pdtarget_motion, pdtarget_fname)
            print(f"output the pd target at {pdtarget_fname}")

        self.stop_other_workers()

    def set_loss_list_index(self, index: int):
        if self.worker_info.comm_size > 1:
            info = [(None, int(WorkerMode.SET_LOSS_LIST_INDEX), None, index)] * self.worker_info.comm_size
            self.comm.scatter(info)
            self.comm.gather(None)
        else:
            if self.worker is not None:
                self.worker.loss.set_loss_list_index(index)

    def dump_duplicate_by_same_start(self, tmp_fname: str, actions, samcon_buffer, duplicate_index):
        result_dict = {
            "dup_idx": self.dup_idx,
            "duplicate_index": duplicate_index,
            "actions": actions,
            "samcon_buffer": samcon_buffer,
            "best_cma": self.best_cma
        }
        result_dict.update(Helper.save_numpy_random_state())
        with open(tmp_fname, "wb") as fout:
            pickle.dump(result_dict, fout)
        with open(tmp_fname + str(self.dup_idx), "wb") as fout:
            pickle.dump(result_dict, fout)

    def no_enough_memory_callback(self, free_mem: float):
        """
        if memory is not enough, exit the program to avoid system collapse
        """
        no_enough_mem = f"available mem = {free_mem:.3f}. exit.."
        logging.info(no_enough_mem)
        print(no_enough_mem, flush=True)
        self.stop_other_workers()
        exit(0)

    def duplicate_by_same_start(self):
        self.cma_info.dup_count = 1
        self.send_target_pose_no_scatter()
        self.dup_idx = 0
        sigmas = self.cma_info.init_sigma_list
        tmp_fname = os.path.join(self.samhlp.save_folder_i_dname(), "duplicate_same_start.bin")

        if os.path.exists(tmp_fname):
            with open(tmp_fname, "rb") as fin:
                result_dict = pickle.load(fin)
            Helper.load_numpy_random_state(result_dict)
            self.dup_idx = result_dict["dup_idx"]
            actions: np.ndarray = result_dict["actions"]
            duplicate_index: int = result_dict["duplicate_index"]
            samcon_buffer: List = result_dict["samcon_buffer"]
            self.best_cma: List[CMAUpdate] = result_dict.get("best_cma", self.best_cma)
            print(f"load from {tmp_fname}", flush=True)
        else:
            actions: np.ndarray = np.zeros((self.n_iter, self.num_joints, 3), dtype=np.float64)
            print(f"{tmp_fname} not exist. ignore..", flush=True)
            duplicate_index: int = 0
            samcon_buffer = []

        while self.dup_idx < len(sigmas):
            Helper.check_enough_memory(self.no_enough_memory_callback)

            self.cma_info.init_sigma = sigmas[self.dup_idx]
            self.rebuild_func(f"dup_idx = {self.dup_idx}")
            for k_start in range(1, self.n_iter + 1):
                self.cma[k_start].x_mean = actions[k_start - 1].flatten()
            last_cma = copy.deepcopy(self.cma)
            duplicate_count = 5 if (self.dup_idx < len(sigmas) - 1) else 1
            while duplicate_index < duplicate_count:
                dump_postfix = f"{self.dup_idx}-{duplicate_index}"
                self.rebuild_func(f" duplicate_index = {duplicate_index}, sigma = {self.cma_info.init_sigma}", False)
                self.cma = copy.deepcopy(last_cma)
                self.init_search_tree()

                ret_best_path, pd_target, motion = self.run_single(f"{dump_postfix}.bvh")
                # compute mean action
                best_path = ret_best_path["best_path"]
                if len(best_path) > 0:
                    avg_loss = sum([node.cost for node in best_path]) / len(best_path)
                else:
                    avg_loss = 23333.0
                if len(best_path) == self.n_iter + 1:
                    samcon_buffer.append(best_path)
                else:
                    print(f"Failed, length = {len(best_path)}, avg loss = {avg_loss}", flush=True)
                    # if there are good samples in the buffer, we can use them as elite samples..

                self.dumps_path_mode(ret_best_path, dump_postfix)
                self.dumps_path_mode(ret_best_path)
                BVHLoader.save(motion, os.path.join(self.samhlp.save_folder_i_dname(), f"test-{dump_postfix}.bvh"))

                duplicate_index += 1
                self.dump_duplicate_by_same_start(tmp_fname, actions, samcon_buffer, duplicate_index)

            duplicate_index = 0
            assert len(samcon_buffer) <= duplicate_count
            actions = np.mean(np.concatenate(
                [np.concatenate([sample.a0[None, ...] for sample in best_path[1:]], axis=0)[None, ...] for best_path in samcon_buffer], axis=0), axis=0)
            samcon_buffer.clear()
            self.dup_idx += 1
            self.dump_duplicate_by_same_start(tmp_fname, actions, samcon_buffer, duplicate_index)

        self.stop_other_workers()

    def fine_tine_forward(
        self,
        width: int,
        pbar: Optional[tqdm] = None,
        sigma: float = 0.01,
        max_epoch: int = 10,
        sample_cnt: int = 200,
        discount_term: float = 0.99
    ):
        """
        Note: as original samcon algorithm takes action locally,
        here we use sliding window to optimize.
        It's hard to find a good solution by gradient based trajectory optimization with a bad initial solution
        So, sampling based method may works.

        Parameters:
        sigma: as the sample sequence is longer, here we should use a smaller sigma
        max_epoch: as the sample loss is more global, larger CMA update epoch will lead to a global minima.
        however, in order to avoid local minima, we should not run original samcon CMA update algorithm for too many times.
        """
        self.cma_info.dup_count = 1
        def gen_scatter_result(sample_list_: List[Sample]):
            ret_: List[List[Sample]] = self.list_to_2d([sample_.create_child() for sample_ in sample_list_])
            return ret_

        sample_win: np.ndarray = np.concatenate([self.joint_info.sample_win.reshape(-1) for _ in range(width)])
        self.cma_info.init_sigma = sigma
        mean_action: np.ndarray = np.concatenate([node.x_mean for node in self.cma[self.k_start: self.k_start + width]])
        cma_update: CMAUpdate = CMAUpdate(mean_action, self.cma_info.init_sigma, sample_win)

        # select start samples.
        prev_level = self.tree.tree[self.k_start - 1]
        prev_cost: np.ndarray = self.get_cost(prev_level)
        _, sel_sample, _, _ = self.pickle_start_state(prev_level, prev_cost, sample_cnt)
        min_cost, best_action, best_subtree = float("inf"), None, None

        for iteration in range(max_epoch):
            # sample actions.
            sample_lists_raw: np.ndarray = cma_update.sample(None, sample_cnt).reshape((sample_cnt, width, self.num_joints, 3))
            sample_lists: np.ndarray = sample_lists_raw * self.joint_info.sample_mask[None, None, ...]
            sample_lists: np.ndarray = np.transpose(sample_lists, (1, 0, 2, 3))  # (width, sample_cnt, nj, 3)
            sample_qs: np.ndarray = Rotation.from_rotvec(sample_lists.reshape(-1, 3)).as_quat().reshape(sample_lists.shape[:-1] + (4,))  # (width, sample_cnt, nj, 4)
            self.tree.tree = self.tree.tree[:self.k_start]

            # TODO:
            # 1. use elite sample to average here (instead of cma update)
            # 2. use trajectory optimization here (first use CMA, then use optimization based method)
            #    - sub task: take trajectory optimization as a sub module
            #    - sub task: scatter time may be reduced..

            # forward simulation in child workers.
            scatter_last_s1 = [sample.s1 for sample in sel_sample]
            scatter_result = gen_scatter_result(sel_sample)
            for k in range(width):
                self.scatter(WorkerMode.ODE_SIMU, self.k_start + k, scatter_last_s1, sample_qs[k])
                # here, we should not sort these actions...
                self._recv_ode_sim(self.k_start + k, scatter_result, None, self.list_to_2d(sample_lists[k]))
                curr_level = self.tree.tree[self.k_start + k]
                for sam_idx in range(sample_cnt):
                    curr_level[sam_idx].index = sam_idx
                scatter_last_s1 = [sample.s1 for sample in curr_level]
                scatter_result = gen_scatter_result(curr_level)

            # compute loss in sample window
            cost_width = np.array([[sample.cost for sample in level] for level in self.tree[self.k_start: self.k_start + width]])  # （width, nsample)
            # should we use discounted return here...?
            discount: np.ndarray = (discount_term ** np.arange(0, width))[:, None]  # (width, 1), 0.99 works here
            update_cost: np.ndarray = np.sum(cost_width * discount, axis=0)
            min_index: int = np.argmin(update_cost)
            if cost_width[0, min_index] < min_cost:
                min_cost = update_cost[min_index]
                best_action = sample_lists_raw[min_index].copy()
                best_subtree = self.tree.tree[self.k_start: self.k_start + width]

            cma_update.update(sample_lists_raw.reshape((sample_cnt, -1)), update_cost)

            if pbar is not None:
                best_loss_win = [self.tree.tree[self.k_start + k][min_index].cost for k in range(width)]
                loss_info = " ".join([f"{node:.3f}" for node in best_loss_win])
                pbar.set_description(f"{self.k_start}, {loss_info}, sum = {sum(best_loss_win):.3f}, sig*cov={cma_update.sigma * cma_update.max_cov:.4f}")

        # save best actions.
        for k in range(width):
            self.cma[self.k_start + k].x_mean = best_action[k].reshape(-1)
            best_subtree[k].sort(key=lambda x: x.cost)
        self.tree.tree[self.k_start: self.k_start + width] = best_subtree

    def fine_tune_with_window(
        self,
        piece_num: int = 2,
        max_epoch: int = 100,
        sigma: float = 0.05,
        sample_cnt: int = 800,
        forward_n_iter: Optional[int] = None,
        do_clear: bool = True
    ):
        if forward_n_iter is None:
            forward_n_iter: int = self.init_n_iter
        self.k_start: int = 1
        pbar: tqdm = tqdm(total=max_epoch)
        while self.k_start + 2 * piece_num <= forward_n_iter:
            # forward
            self.fine_tine_forward(2 * piece_num, pbar, sigma, max_epoch, sample_cnt)
            # if self.step_is_bad(self.k_start + piece_num):
            #     self.k_start = max(1, self.k_start - 10)
            # else:
            self.k_start += 1
            self.tree.clear_dead_nodes(self.k_start - 1)

            Helper.check_enough_memory(self.no_enough_memory_callback)

        self.k_start: int = forward_n_iter + 1 - 2 * piece_num

        # forward
        if self.k_start > 1:
            self.fine_tine_forward(2 * piece_num, pbar, sigma, max_epoch)
        self.k_start = forward_n_iter

        pbar.close()
        func_ret = self.eval_best_path()
        if do_clear:
            self.tree.tree.clear()
            self.init_search_tree()

        nsr_sim_motion = func_ret[-1].remove_end_sites(True)
        nsr_ref_motion: MotionData = self.motion_input.sub_sequence(0, forward_n_iter * self.sim_cnt + 1, True).remove_end_sites(False)
        nsr_value: MotionData = calc_nsr(nsr_sim_motion, nsr_ref_motion)
        print(f"Fine-tune with window: nsr value = {nsr_value:.3f}. motion len = {func_ret[-1].num_frames}", flush=True)

        return func_ret

    def fine_tune_with_window_hlp(self):
        self.rebuild_func()
        self.loads()

        self.send_target_pose_no_scatter()
        self.dup_idx = 0
        self.fine_tune_with_window(max_epoch=50)
        self.stop_other_workers()

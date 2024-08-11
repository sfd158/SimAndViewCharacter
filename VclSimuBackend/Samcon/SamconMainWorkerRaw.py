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

import logging
import numpy as np
from tqdm import tqdm
import time
from typing import Tuple, Set

from ..ODESim.BodyInfoState import BodyInfoState
from .SamconMainWorkerBase import *
from .SamconWorkerFull import SamconWorkerFull


class SamconMainWorkerRaw(SamconMainWorkerBase):
    def __init__(
        self,
        samhlp: SamHlp,
        worker_info: WorkerInfo,
        use_raw_offset: bool = False,
        worker: Optional[SamconWorkerFull] = None
    ):
        """
        param:
        conf_name: configuration file name
        save_dir: str, save directory
        main_mode: MainWorkerMode.
        """
        super(SamconMainWorkerRaw, self).__init__(samhlp, worker_info, worker)
        self.use_raw_offset = use_raw_offset
        if use_raw_offset:
            self.tree.level(0)[0].offset = self.worker.offset_raw(0, ([self.tree.level(0)[0].s1],))[0].reshape((-1, 4))
            self.character.load(self.tree.level(0)[0].s1)

        self.rollback_time: int = self.conf["worker_raw"]["rollback_time"]
        self.rollback_retain: float = self.conf["worker_raw"]["rollback_retain"]

    def run(self):
        """
        Run Samcon 2010 Algorithm.
        Some information is computed in other workers.
        """
        start_t = time.time()

        t = 1
        pbar = tqdm(total=self.n_iter)
        while t <= self.n_iter:
            self.samples.clear()
            cost = self.get_cost(self.tree.level(t-1))
            t_new, cost, last_level = self.check_failed_rollback(t, cost)
            if t_new != t:
                logging.info(f"In Run: Rollback from {t} to {t_new}")
                pbar.update(t_new - t)
                t = t_new

            # start_idx: selected sample indices,
            # sel_sample: selected samples,
            # sel_s1: List of s1 in selected samples,
            # result: shallow copy of selected samples used for samcon result
            start_idx, sel_sample, sel_s1, result = self.pickle_start_state(last_level, cost)
            if self.use_raw_offset:  # offset method in Samcon 2010 paper
                self.calc_offset_raw(last_level, start_idx, t)
                offs: List[np.ndarray] = [sample.offset[None, ...] for sample in sel_sample]
                sel_offset: np.ndarray = np.concatenate(offs, axis=0) if len(offs) > 1 else offs[0]
                logging.info(f"Begin Sample at {t} with offset. len(sel_s1) = {len(sel_s1)}, sel_offset.shape = {sel_offset.shape}")
                self.scatter(WorkerMode.SAMCON.value, t, sel_s1, sel_offset)
            else:
                logging.info(f"Begin Sample at {t}")
                self.scatter(WorkerMode.SAMCON.value, t, sel_s1)

            recv_type = List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[BodyInfoState], np.ndarray]]
            recv_list: recv_type = self.gather()
            for recv in recv_list:
                # w_idx: index of each divided task
                # sample_list: sampled rotation vector in shpae (*, len(joints), 3)
                # falldown: fall down flag of each sample result.
                # cost_list: np.ndarray cost of each sample
                # state1_list: simulation result of each sample
                # com_h: CoM's height of each sample
                w_idx, sample_list, falldown, cost_list, state1_list, com_h = recv
                for res_idx, res in enumerate(result[w_idx]):
                    res.set_val(sample_list[res_idx], cost_list[res_idx], state1_list[res_idx], t, com_h[res_idx])
                self.samples.extend(result[w_idx])

            self.select_and_save_samples2(t)
            self.tree.clear_dead_nodes()
            t += 1
            pbar.update(1)

        pbar.close()
        # stop other workers
        self.stop_other_workers()
        best_path, cost = self.search_best_path()
        end_t = time.time()
        logging.info(f"Time Total: {end_t - start_t}, path length = {len(best_path)}, cost = {cost[0]}")
        return best_path

    def check_failed_rollback(self, t: int, cost: np.ndarray):
        """
        param:
        t: time step
        cost: samcon cost
        """
        level = self.tree.level(t-1)
        logging.info(f"In Check Failed Rollback, cost {cost[0]}")
        min_idx: int = np.argmin(cost).item()
        logging.info(f"CoM of sample is {level[min_idx].com_h1}")
        if self.reconstruction_idx_fails(level, min_idx):
            logging.info(f"RollBack from time {t}")
            t = self.simple_rollback(t)
            level = self.tree.level(t - 1)
            cost = self.get_cost(level)
            logging.info(f"new cost {cost[0]}, {level}")
        return t, cost, level

    def calc_offset_raw(self, level: List[Sample], start_idx: np.ndarray, t: int):
        """
        offset method in Samcon 2010.

        """
        sample_list = self.get_nooffset_sample(level, start_idx)
        if len(sample_list) == 0:
            logging.info("Not need to calc offset")
        elif len(sample_list) < self.worker_size:
            for sample in sample_list:
                sample.offset = self.worker.offset_raw(0, ([sample.s1],))[0].reshape((-1, 4))
        else:
            self.scatter(WorkerMode.OFFSET_RAW.value, t, [sample.s1 for sample in sample_list])

            # gather offset from workers
            sample_list_new: List[List[Sample]] = self.list_to_2d(sample_list)
            recv_list: List = self.gather()
            for recv in recv_list:
                w_idx: int = recv[0]
                offset: np.ndarray = recv[1]  # (n, joint, 4)
                logging.info(f"offset shape = {offset.shape}")
                for i in range(offset.shape[0]):
                    sample_list_new[w_idx][i].offset = offset[i]

    def simple_rollback(self, t: int, lower_bound: int = 1):
        """
        param:
        t: time step
        return new time step
        """
        # Retain Some Good Samples..
        t_new = max(t - self.rollback_time, lower_bound)
        logging.info(f"t = {t}, t_new = {t_new}")
        remain: Set[Sample] = set()

        for i in range(self.tree.num_level() - 1, t_new-1, -1):
            assert i != 0
            samples = self.tree.level(i)
            sample_0 = set(samples)
            sample_1 = list(sample_0 - remain)
            sample_1.sort(key=lambda sample: sample.cost)
            sample_2 = sample_1[0:max(int(len(samples) * self.rollback_retain) - len(remain), 0)]
            samples.clear()
            samples.extend(sample_2 + list(remain))
            remain = {sample.parent for sample in samples}
            logging.info(f"RollBack Layer {i}. {len(self.tree.level(i))} samples remained.")

        # test
        for i in range(self.tree.num_level()):
            logging.info(f"level {i}, length = {len(self.tree.level(i))}")
        while t_new > 1 and len(self.tree.level(t_new - 1)) == 0:
            t_new -= 1
        while len(self.tree.level(-1)) == 0 and self.tree.num_level() > t_new:
            logging.info(f"Level {self.tree.num_level() - 1} is empty. pop.")
            self.tree.tree.pop(-1)

        return t_new

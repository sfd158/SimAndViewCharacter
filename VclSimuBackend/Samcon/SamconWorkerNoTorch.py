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
from scipy.spatial.transform import Rotation
from typing import Any, Tuple, List, Optional, Union

from .SamconWorkerBase import SamHlp, SamconWorkerBase, WorkerInfo, CMAInfo, WorkerMode
from .SamconTargetPose import SamconTargetPose
from ..CMA.CMAUpdate import CMAUpdate
from ..Common.MathHelper import MathHelper
from ..ODESim.BodyInfoState import BodyInfoState
from ..ODESim.ODEScene import ODEScene
from ..ODESim.ODECharacter import ODECharacter
from ..ODESim.TargetPose import SetTargetToCharacter, TargetPose
from ..Server.v2.ServerForUnityV2 import ServerThreadHandle


class SamconWorkerNoTorch(SamconWorkerBase):
    def __init__(
        self,
        samhlp: SamHlp,
        worker_info: WorkerInfo,
        scene: Optional[ODEScene] = None,
        sim_character: Optional[ODECharacter] = None
        ):
        """
        conf_name: samcon configuration filename
        worker_info:
        """
        super(SamconWorkerNoTorch, self).__init__(samhlp, worker_info, scene, sim_character)
        self.cma_info = CMAInfo(self.conf, self.character)
        self.cma = CMAUpdate(np.zeros(self.num_joints * 3, dtype=np.float64), self.cma_info.init_sigma)

        self.target_local_quat: Optional[np.ndarray] = None
        self.last_buffer = None

        # use for Unity Server
        # run simulation on sub thread
        # hang up this thread after a forward simultion step
        self.as_sub_thread: bool = False

        self.duplicate_mode: bool = False
        self.sim_one_iter_callback_before = None

    def sub_thread_callback_before(self, t: int, i: int, iteration: int):
        if self.as_sub_thread:
            frame_index: int = (t - 1) * self.sim_cnt + i
            ServerThreadHandle.sub_thread_wait_for_run()
            self.scene.str_info = f"e{iteration}, a{t}, f{frame_index}"
            self.character.curr_frame_index = frame_index

    def sub_thread_callback_after(self):
        if self.as_sub_thread:
            ServerThreadHandle.sub_thread_run_end()

    def _eval_sample(
        self,
        t: int,
        ss_list: List[BodyInfoState],
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_list: np.ndarray = actions
        sample_q: np.ndarray = Rotation.from_rotvec(sample_list.reshape(-1, 3))._quat.reshape(
            sample_list.shape[:-1] + (4,))
        falldown, c_list, se_list, _, com_h = self.simulate_with_offset(t, (ss_list, sample_q), True, True, False, True)

        return sample_list, sample_q, falldown, c_list, se_list, com_h

    def get_pd_target_array(self, t: int) -> np.ndarray:
        start: int = ((t - 1) % self.n_iter) * self.sim_cnt + 1
        if start > 0:
            end: int = min(start + self.sim_cnt, self.target_local_quat.shape[0])
            pd_target: np.ndarray = self.target_local_quat[start:end]
        else:
            pd_target: np.ndarray = self.target_local_quat[start:]

        if self.duplicate_mode and pd_target.shape[0] < self.sim_cnt:
            pd_target: np.ndarray = np.concatenate([pd_target, self.target_local_quat[0:1]], axis=0)
        return pd_target, start

    def sim_one_iter(
        self, t: int, ss: Optional[BodyInfoState],
        off_rot: Optional[Rotation],
        iteration=None,
        forward_count: Optional[int] = None,
        loss_index: Optional[int] = None
        ) -> bool:
        if ss is not None:
            self.character.load(ss)  # at the first sample, the geometry order will not change in ode space...

        self.character.accum_energy = 0.0
        falldown = False
        if forward_count is None:
            forward_count = self.sim_cnt
        if loss_index is None:
            loss_index = t * self.sim_cnt

        pd_target, start_index = self.get_pd_target_array(t)
        self.character.accum_loss = 0.0
        for i in range(forward_count):
            try:
                local_q: np.ndarray = (off_rot * Rotation(pd_target[i], False, False)).as_quat()
            except ValueError as value_err:
                print(off_rot.as_quat().tolist())
                print(pd_target[i].tolist())
                logging.info(off_rot.as_quat().tolist())
                logging.info(pd_target[i].tolist())
                raise value_err

            self.damped_pd.add_torques_by_quat(local_q)
            # support multi simulation type
            self.scene.simu_func()
            if self.use_dense_loss:
                self.character.accum_loss += self.loss.loss(start_index + i)

            falldown = falldown or self.character.fall_down  # Check Fall Down..

        self.character.accum_energy /= self.fps
        return falldown

    def get_forward_count(self, t: int) -> Tuple[int, int]:
        if not self.duplicate_mode and t * self.sim_cnt >= self.target.num_frames:
            forward_count: int = self.sim_cnt - 1
            loss_index: int = t * self.sim_cnt - 1
        else:
            forward_count: int = self.sim_cnt
            loss_index: int = t * self.sim_cnt

        return forward_count, loss_index

    def simulate_with_offset(
        self,
        t: int,
        data: Tuple[List[BodyInfoState], Optional[np.ndarray]]
        ):
        ss_list, offset, off_rot = data[0], data[1], None

        # for debug mode..
        if np.isnan(offset).any():
            print(offset)
            logging.info(offset)
            raise ValueError(f"np.isnan(offset).any()")

        if offset is not None and offset.shape[1] < self.num_joints:
            new_offset: np.ndarray = MathHelper.unit_quat_arr((offset.shape[0], self.num_joints, 4))
            new_offset[:, self.track_joint_index, :] = offset  # TODO
            offset = np.ascontiguousarray(new_offset)

        falldown_list: np.ndarray = np.zeros(len(ss_list), dtype=np.bool_)
        c_list: np.ndarray = np.zeros(len(ss_list), dtype=np.float64)
        se_list: List[Optional[BodyInfoState]] = [None for _ in range(len(ss_list))]

        forward_count, loss_index = self.get_forward_count(t)
        for idx, ss in enumerate(ss_list):
            if offset is not None:
                off_rot = Rotation(offset[idx] if offset.ndim == 3 else offset, False, False)
            falldown_list[idx] = self.sim_one_iter(t, ss, off_rot, idx, forward_count)
            if not self.use_dense_loss:
                c_list[idx] = self.loss.loss(loss_index)
            else:
                c_list[idx] = self.character.accum_loss / self.sim_cnt

            se_list[idx] = self.character.save()

        return falldown_list, c_list, se_list

    def recieve_target_data(self, data: Tuple) -> None:
        """
        Recieve target pose and inverse dynamics pose from Main Worker.
        """
        self.target: SamconTargetPose = data[0]
        self.tar_set = SetTargetToCharacter(self.character, self.target.pose)
        self.inv_dyn_target: Optional[TargetPose] = data[1]
        if self.inv_dyn_target is not None:
            logging.info("Use inverse dynamics")
            self.target_local_quat = self.inv_dyn_target.locally.quat
        else:
            logging.info("Not use inverse dynamics")
            self.target_local_quat = self.target.pose.locally.quat

        self.n_iter = self.calc_n_iter()
        self.loss.set_loss_attrs(self.target, self.character, self.n_iter)

    def call_single_thread(self, mode: int, t: int, data):
        self.last_buffer = tuple()

        if mode == WorkerMode.ODE_SIMU:
            # data: (ss_list, sample_q)
            falldown_list, c_list, se_list = self.simulate_with_offset(t, data)
            self.last_buffer = (falldown_list, c_list, se_list)
        elif mode == WorkerMode.SEND_TARGET_NO_SCATTER:
            self.get_target_no_scatter()
        elif mode == WorkerMode.SET_LOSS_LIST_INDEX:
            self.loss.set_loss_list_index(data[0])
        elif mode == WorkerMode.ENABLE_DUPLICATE:
            self.duplicate_mode = True

        if data is not None:
            del data
        return self.last_buffer

    def get_target_no_scatter(self):
        buf: Tuple[Optional[int], int, Optional[int], Any] = self.comm.recv()
        w_idx, mode, t, data = buf[0], buf[1], buf[2], buf[3:]
        self.recieve_target_data(data)

    def run(self) -> None:
        """
        wait for signal from Main Worker
        TODO: we should modify to async mode.
        Now, all the works will wait for main worker..
        """

        while True:
            buf: Tuple[Optional[int], int, Optional[int], Any] = self.comm.scatter(None, 0)
            # logging.info(f"recieve {buf}")
            # print("in run, buf = ", buf)
            # w_idx: worker index. unique index of each part of task.
            # mode: mode of worker. WorkerMode.SAMCON, WorkerMode.OFFSET_RAW, WorkerMode.CMA is supported.
            # t: now time.
            # data: data recieved from main worker
            w_idx, mode, t, data = buf[0], buf[1], buf[2], buf[3:]
            if mode == WorkerMode.STOP.value:
                logging.info(f"Worker {self.comm_rank} on Node {self.node_name} Exit.")
                break

            # if len(data[0]) == 0:
            #    self.comm.gather(None, 0)

            send_data = (w_idx,) + self.call_single_thread(mode, t, data)
            self.comm.gather(send_data, 0)  # send result to main worker
            # logging.info(f"recieve = {buf}, scatter to main worker: {send_data}")
            del send_data
            self.last_buffer = None

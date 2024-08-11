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
How about use CMA + Trajectory Optimization..?
That is, for a simple sliding window, we optimize with a LBFGS step, and then optimize by CMA
As L-BFGS optimizer will converge to local minima after several steps,
it will be hard to achieve global minima only using L-BFGS optimizer.


"""

import copy
import logging
import numpy as np
import torch
from torch.nn import Parameter
from typing import Any, Callable, Tuple, List, Optional, Dict

from .SamconWorkerBase import SamHlp, WorkerInfo, WorkerMode
from .SamconTargetPose import SamconTargetPose
from .SamconWorkerNoTorch import SamconWorkerNoTorch
from ..ODESim.BodyInfoState import BodyInfoState
from ..ODESim.TargetPose import TargetPose
from ..ODESim.ODECharacter import ODECharacter
from ..ODESim.ODEScene import ODEScene

from .Loss.SingleDiffFrameSamconLoss import SamconLossExtractor, PyTorchSamconTargetPose
from ..DiffODE.DiffODEWorld import DiffODEWorld
from ..DiffODE.Build import BuildFromODEScene
from ..DiffODE.DiffPDControl import DiffDampedPDControler
from ..DiffODE.DiffQuat import quat_from_rotvec, quat_multiply
from ..DiffODE.DiffFrameInfo import DiffFrameInfo
from ..DiffODE.DiffContact import DiffContactInfo
from ..Server.v2.ServerForUnityV2 import ServerThreadHandle

from ..Utils.Camera.Human36CameraBuild import CameraParamBuilder
from ..Utils.Camera.CameraPyTorch import CameraParamTorch
from ..Utils.Camera.CameraNumpy import CameraParamNumpy


class RootPDControlParam:
    def __init__(self, conf: Optional[Dict[str, Any]]) -> None:
        self.in_use: bool = False
        self.initial_kp: float = 1000
        self.initial_kd: float = 100
        self.kp: float = 1000
        self.kd: float = 100
        self.decay_ratio: float = 0.5
        self.decay_count: int = 5

        if conf is not None:
            self.update_param(conf)

    def reset_init_kp_kd(self):
        self.kp = self.initial_kp
        self.kd = self.initial_kd

    def update_param(self, conf_dict: Dict[str, Any]):
        self.in_use: bool = conf_dict.get("in_use", self.in_use)
        self.kp: float = conf_dict.get("kp", self.kp)
        self.kd: float = conf_dict.get("kd", self.kd)
        self.initial_kp = self.kp
        self.initial_kd = self.kd
        self.decay_ratio = conf_dict.get("decay_ratio", self.decay_ratio)
        self.decay_count = conf_dict.get("decay_count", self.decay_count)

    def export_dict(self):
        return dict(
            in_use=self.in_use,
            kp=self.kp,
            kd=self.kd,
            decay_ratio=self.decay_ratio,
            decay_count=self.decay_count
        )

    def zero_kp_kd(self, debug_print: bool = False):
        self.kp = 0.0
        self.kd = 0.0
        if debug_print:
            print("set kp = 0, and kd = 0")

    def decay_kp_kd(self, debug_print: bool = False):
        self.kp *= self.decay_ratio
        self.kd *= self.decay_ratio
        if debug_print:
            print(f"kp = {self.kp:.3f}, kd = {self.kd:.3f}")


class SamconWorkerFull(SamconWorkerNoTorch):
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
        super(SamconWorkerFull, self).__init__(samhlp, worker_info, scene, sim_character)
        self.root_pd_param = RootPDControlParam(self.conf["traj_optim"]["root_virtual_force"])

        self.builder: Optional[BuildFromODEScene] = None
        self.diff_world: Optional[DiffODEWorld] = None
        self.torch_target: Optional[PyTorchSamconTargetPose] = None
        self.torch_stable_pd: Optional[DiffDampedPDControler] = None
        self.torch_target_local_quat: Optional[torch.Tensor] = None
        self.torch_loss: Optional[SamconLossExtractor] = None
        self.camera_torch: Optional[CameraParamTorch] = None

        self.contact_info_list: Optional[List[DiffContactInfo]] = None

        self.in_sim_hack_func: Optional[Callable] = None

    @property
    def curr_frame(self) -> Optional[DiffFrameInfo]:
        return self.diff_world.curr_frame if self.diff_world is not None else None

    def build_diff_target(self) -> None:
        self.torch_target = PyTorchSamconTargetPose.build_from_numpy(self.target)
        self.torch_target_local_quat: torch.Tensor = torch.from_numpy(self.target_local_quat)
        # here we should consider camera param
        if self.loss.camera_param is not None:
            self.torch_loss.camera_param = CameraParamTorch.build_from_numpy(self.loss.camera_param, torch.float32)
        self.torch_loss.set_loss_attrs(self.diff_world, self.torch_target, self.n_iter)

    def build_diff_world(self) -> DiffODEWorld:
        self.builder = BuildFromODEScene(self.scene)
        self.diff_world = self.builder.build()
        self.stable_pd = DiffDampedPDControler.build_from_joint_info(self.character.joint_info)
        self.torch_loss = SamconLossExtractor(self.conf, self.character)

        self.build_diff_target()
        return self.diff_world

    def get_torch_pd_target_array(self, t: int) -> torch.Tensor:
        """
        get pd target for tracking
        """
        start: int = (t - 1) * self.sim_cnt + 1
        end: int = min(start + self.sim_cnt, self.torch_target_local_quat.shape[0])
        pd_target: torch.Tensor = self.torch_target_local_quat[start:end]
        if self.duplicate_mode and pd_target.shape[0] < self.sim_cnt:
            pd_target: torch.Tensor = torch.cat([pd_target, self.torch_target_local_quat[0:1]], dim=0)
        return pd_target

    def diffode_forward(
        self,
        t: int,
        state: Optional[BodyInfoState],
        action: torch.Tensor,
        forward_count: Optional[int] = None,
        ) -> torch.Tensor:
        """
        TODO: Here we should support add hack for contact
        """
        if state is not None:
            self.diff_world.import_from_state(state)
        self.character.accum_energy = 0.0
        q1: torch.Tensor = quat_from_rotvec(action)

        # if we add hack for contact, consider contact hack loss
        # that is, if contact height doesn't match the hacked contact, we should add penality.
        tot_contact_loss = torch.as_tensor(0.0, dtype=torch.float64)

        pd_target: torch.Tensor = self.get_torch_pd_target_array(t)  # Target Pose for PD Controller

        if forward_count is None:
            forward_count = self.sim_cnt

        for i in range(forward_count):
            frame_index = (t - 1) * self.sim_cnt + i
            q0: torch.Tensor = pd_target[i]
            q_tot: torch.Tensor = quat_multiply(q1, q0)
            self.curr_frame.stable_pd_control_fast(q_tot)

            if self.contact_info_list is not None:
                self.add_hack_local_contact(self.contact_info_list[frame_index])
                self.torch_loss.frame_info = self.curr_frame
                contact_loss = self.torch_loss.hack_contact_loss(frame_index)
                tot_contact_loss += contact_loss

            self.diff_world.step()

        return tot_contact_loss

    def add_hack_local_contact(self, diff_contact: DiffContactInfo) -> DiffContactInfo:
        """
        convert contact position into current frame
        this operation doesn't require gradient.
        """
        return self.diff_world.add_hack_local_contact(diff_contact)

    def diffode_forward_with_save(
        self,
        t: int,
        state: Optional[BodyInfoState],
        action: torch.Tensor,
        root_pd_action: Optional[torch.Tensor] = None,
        iteration: Optional[int] = None,
        forward_count: Optional[int] = None
        ) -> Tuple[List[BodyInfoState], torch.Tensor]:
        if state is not None:
            self.diff_world.import_from_state(state)

        act_shape: int = action.shape[-1]
        if act_shape == 3:  # Input is in axis angle format
            q1: torch.Tensor = quat_from_rotvec(action)
        elif act_shape == 4:  # Input is in quaternion format
            q1: torch.Tensor = action
        else:
            raise ValueError

        tot_contact_loss = torch.as_tensor(0.0, dtype=torch.float64)
        saved_state: List[BodyInfoState] = []
        pd_target: torch.Tensor = self.get_torch_pd_target_array(t)

        if forward_count is None:
            forward_count = self.sim_cnt

        if self.contact_info_list is not None:
            assert len(self.contact_info_list) >= forward_count

        for i in range(forward_count):
            frame_index = (t - 1) * self.sim_cnt + i
            if self.as_sub_thread:
                ServerThreadHandle.sub_thread_wait_for_run()
                self.scene.str_info = f"e{iteration}, a{t}, f{frame_index}"
                self.character.curr_frame_index = frame_index

            q0: torch.Tensor = pd_target[i]
            q_tot: torch.Tensor = quat_multiply(q1, q0)
            self.curr_frame.stable_pd_control_fast(q_tot)

            # add virtual force pd control on root..
            if self.root_pd_param.in_use and root_pd_action is not None:
                self.add_virtual_root_force(frame_index + 1, root_pd_action)

            if self.contact_info_list is not None:
                self.add_hack_local_contact(self.contact_info_list[frame_index])
                self.torch_loss.frame_info = self.curr_frame
                # contact_loss = self.torch_loss.hack_contact_loss(frame_index)
                # tot_contact_loss += contact_loss

            # compute contact loss here..
            self.diff_world.step(do_contact=False)  # forward simulation
            if self.in_sim_hack_func:
                self.in_sim_hack_func(frame_index + 1)
            # import time
            # time.sleep(1)  # for visualize using Long Ge's framework..

            state = self.character.save()
            state.pd_target = q_tot.detach().numpy().copy()  # save pd control target pose
            saved_state.append(state)

            if self.as_sub_thread:
                ServerThreadHandle.sub_thread_run_end()  # wait for Unity server..

        # self.curr_frame.contact_hack_loss = tot_contact_loss
        contact_log_info = f"t = {t}, contact loss = {tot_contact_loss.item():.4f}"
        print(contact_log_info)
        logging.info(contact_log_info)

        return saved_state, tot_contact_loss

    def simulate_diffode(self, t: int, data: Tuple[List[BodyInfoState], np.ndarray]):
        """
        Input: the best k samples at time t.
        Output is the simulation result (with ODE)
        Variable: control signal for each input sample
        return: character state at time t + 1 after trajectory optimization by one step? (or more step?)
        """
        ss_list: List[BodyInfoState] = data[0]
        actions: torch.Tensor = torch.from_numpy(data[1])
        last_actions: Optional[torch.Tensor] = actions.clone() if self.cma_info.traj_opt_epoch > 1 else None
        use_lbfgs: bool = self.cma_info.optim == torch.optim.LBFGS
        for idx, ss in enumerate(ss_list):
            action = Parameter(actions[idx])
            best_action: Optional[torch.Tensor] = None
            best_loss = float('inf')
            opt: torch.optim.Optimizer = self.cma_info.optim([action], lr=self.cma_info.traj_opt_lr)
            loss_hist = []

            def closure() -> torch.Tensor:  # for L-BFGS optimizer
                opt.zero_grad(set_to_none=True)
                tot_contact_loss = self.diffode_forward(t, copy.deepcopy(ss), action)
                loss_val_: torch.Tensor = self.torch_loss.loss(t * self.sim_cnt)
                loss_val_ += tot_contact_loss
                loss_val_.backward()
                loss_hist.append(loss_val_.item())
                return loss_val_

            for epoch in range(self.cma_info.traj_opt_epoch):
                prev_action: torch.Tensor = action.detach().clone()

                if not use_lbfgs:
                    closure()
                    opt.step()
                else:
                    opt.step(closure)

                if loss_hist[-1] < best_loss:
                    best_loss = loss_hist[-1]
                    best_action = prev_action.clone()

                # print(f"t = {t}, idx = {idx}, epoch = {epoch}, loss = {loss_hist[-1]:.4f}, len(loss_hist) = {len(loss_hist)}")
                logging.info(f"t = {t}, idx = {idx}, epoch = {epoch}, loss = {loss_hist[-1]:.4f}, len(loss_hist) = {len(loss_hist)}")
                loss_hist.clear()

            if self.cma_info.traj_opt_epoch == 1:
                best_action = action.detach().clone()

            if last_actions is not None:
                last_actions[idx] = action.detach().clone()
            actions[idx] = best_action

        res0 = self._eval_sample(t, ss_list, actions)

        if last_actions is not None:
            res1 = self._eval_sample(t, ss_list, last_actions)
            for idx, loss_val in enumerate(res1[3]):
                logging.info(f"idx = {idx}, ode eval = {loss_val:.4f}")
            res0 = self.merge_sim_result([res0[3], res1[3]], list(zip(res0, res1)))
        return res0

    def merge_sim_result(self, costs: List[np.ndarray], bufs: List):
        sample_cnt: int = len(bufs[0][0])
        num_input: int = len(bufs[0])

        min_place = np.concatenate([cost[None, ...] for cost in costs], axis=0)
        min_place = np.argmin(min_place, axis=0)
        assert min_place.shape == (sample_cnt,)
        logging.info(f"min_place = {min_place}")

        bufs_all = [np.concatenate([np.asarray(buf[n_in])[None, ...] for n_in in range(num_input)], axis=0) for buf in bufs]
        idx = np.arange(0, sample_cnt)
        bufs_all = [buf[min_place, idx] for buf in bufs_all]
        bufs_all[4] = bufs_all[4].tolist()
        return tuple(bufs_all)

    def get_local_quat_torch(self, i: int, t: int) -> torch.Tensor:
        return self.torch_target_local_quat[(t - 1) * self.sim_cnt + i, :, :]

    def eval_loss_torch(self, t: int) -> torch.Tensor:
        t %= self.n_iter
        return self.torch_loss.loss(self.curr_frame, self.torch_target, t * self.sim_cnt)

    def recieve_target_data(self, data: Tuple[SamconTargetPose, TargetPose]):
        """
        Recieve target pose and inverse dynamics pose from Main Worker.
        """
        super().recieve_target_data(data)

        traj_opt_dict = self.conf["worker_cma"].get("traj_opt")
        if traj_opt_dict and traj_opt_dict["in_use"]:
            self.build_diff_world()

    def call_single_thread(self, mode: int, t: int, data):
        logging.info(f"t = {t}, mode = {mode}")

        self.last_buffer = tuple()

        if mode == WorkerMode.ODE_SIMU:
            # data: (ss_list, sample_q)
            falldown_list, c_list, se_list = self.simulate_with_offset(t, data)
            self.last_buffer = (falldown_list, c_list, se_list)
        elif mode == WorkerMode.TRAJECTORY_OPTIM:
            self.last_buffer = self.simulate_diffode(t, data)
        elif mode == WorkerMode.IDLE:  # do nothing here
            pass
        elif mode == WorkerMode.SEND_TARGET_NO_SCATTER:
            self.get_target_no_scatter()
        elif mode == WorkerMode.RECIEVE_TARGET.value:
            self.recieve_target_data(data)

        del data
        return self.last_buffer

    def get_target_no_scatter(self):
        buf: Tuple[Optional[int], int, Optional[int], Any] = self.comm.recv()
        w_idx, mode, t, data = buf[0], buf[1], buf[2], buf[3:]
        self.call_single_thread(mode, t, data)

    def run(self):
        """
        wait for signal from Main Worker
        """

        while True:
            buf: Tuple[Optional[int], int, Optional[int], Any] = self.comm.scatter(None, 0)
            # w_idx: worker index. unique index of each part of task.
            # mode: mode of worker. WorkerMode.SAMCON, WorkerMode.OFFSET_RAW, WorkerMode.CMA is supported.
            # t: now time.
            # data: data recieved from main worker
            w_idx, mode, t, data = buf[0], buf[1], buf[2], buf[3:]
            if mode == WorkerMode.STOP.value:
                logging.info("Worker %d on Node %s Exit." % (self.comm_rank, self.node_name))
                break

            if len(data[0]) == 0:
                self.comm.gather(None, 0)

            send_data = (w_idx,) + self.call_single_thread(mode, t, data)
            self.comm.gather(send_data, 0)  # send result to main worker
            del send_data

    def add_virtual_root_force(
        self,
        frame: int,
        signal: torch.Tensor
    ) -> torch.Tensor:
        if self.root_pd_param.kp == 0.0 and self.root_pd_param.kd == 0.0:
            pass

        assert isinstance(signal, torch.Tensor)
        curr_frame = self.diff_world.curr_frame
        curr_root_pos: torch.Tensor = curr_frame.root_pos.view(3)
        target_root_pos: torch.Tensor = self.torch_target.pose.root.pos[frame].view(3)
        curr_root_vel: torch.Tensor = curr_frame.root_velo.view(3)
        force: torch.Tensor = self.root_pd_param.kp * (signal.view(3) + target_root_pos - curr_root_pos) - self.root_pd_param.kd * curr_root_vel
        force: torch.Tensor = force.view(3, 1)
        root_index: int = curr_frame.root_body_index
        if curr_frame.body_force is None:
            curr_frame.body_frame.body_force = torch.zeros_like(curr_frame.body_pos)
        if curr_frame.body_force.requires_grad is False:
            curr_frame.body_force[root_index] = force
        else:
            # I know this implement is ugly.. but it is ok..
            root_force_mask: torch.Tensor = torch.zeros_like(curr_frame.body_force)
            root_force_mask[root_index] = force
            curr_frame.body_frame.body_force = curr_frame.body_force + root_force_mask

        return force

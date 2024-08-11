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

from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
import copy
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
import torch
from torch import nn
from typing import Optional, Union, List, Dict, Any

from .MainWorkerCMANew import SamconMainWorkerCMA
from ..StateTree import Sample
from ..SamconWorkerFull import SamconWorkerFull
from ..SamconMainWorkerBase import WorkerInfo, SamHlp
from ...Common.MathHelper import MathHelper
from ...DiffODE import DiffQuat


class TrajStateCls:
    def __init__(self, state, loss, action) -> None:
        self.state = state
        self.loss = loss
        self.action = action


class TrajResState:
    def __init__(self, state_list, tot_saved_path, action_hist, forward_count):
        self.state_list = state_list
        self.tot_saved_path = tot_saved_path
        self.action_hist = action_hist
        self.forward_count = forward_count


class TrajOptBVHBatch(SamconMainWorkerCMA):
    def __init__(self, samhlp: SamHlp, worker_info: WorkerInfo, worker: Optional[SamconWorkerFull], load_samcon_res: bool = True):
        super().__init__(samhlp, worker_info, worker=worker)
        self.k_start = 1
        traj_optim_conf: Dict[str, Any] = self.conf["traj_optim"]
        self.piece_num: int = traj_optim_conf["piece_num"]
        self.l2_reg_coef: float = traj_optim_conf["l2_reg_coef"]
        self.l1_reg_coef: float = traj_optim_conf["l1_reg_coef"]
        self.initial_lr: float = traj_optim_conf["initial_lr"]
        self.total_epoch: int = traj_optim_conf["total_epoch"]

        self.batch_size: int = traj_optim_conf["batch_size"]
        self.initial_noise_sigma: float = traj_optim_conf["initial_noise_sigma"]

        self.traj_result = TrajResState(
            [TrajStateCls(copy.deepcopy(self.tree.level(0)[0].s1), 0.0, None)] +
            [TrajStateCls(None, None, None) for _ in range(self.n_iter)],
            [copy.deepcopy(self.tree.level(0)[0].s1)] + [None for _ in range(self.n_iter * self.sim_cnt)],
            self.load_samcon_res_func(self.samhlp.best_path_fname() if load_samcon_res else None),
            np.zeros((self.n_iter + 1,), dtype=np.int32)
        )
        self.epoch_writers = [SummaryWriter(os.path.join(self._writer_logdir, "epoch" + str(epoch_idx))) for epoch_idx in range(self.total_epoch)]
        self.send_target_pose_no_scatter()
        self.worker.build_diff_world()

    def load_samcon_res_func(self, best_path: Union[str, List[Sample], None]):
        # Load samcon result
        nj: int = len(self.joints)
        unit_func = lambda: torch.from_numpy(MathHelper.quat_to_vec6d(MathHelper.unit_quat_arr((self.n_iter + 1, nj, 4))))
        if best_path is None:
            return unit_func()
        if isinstance(best_path, str):
            if not os.path.isfile(best_path):
                print(f"best path filename {best_path} not exist. ignore..")
                return unit_func()
            with open(best_path, "rb") as fin:
                # raise ValueError("Test mode, not use initial solution")
                best_path: List[Sample] = pickle.load(fin)
        if len(best_path) != self.n_iter + 1:
            print(f"length of best path = {len(best_path)} doesn't match.")
            return unit_func()
        _cat_action = np.concatenate([np.zeros((1, nj, 3))] + [node.a0.reshape((1, nj, 3)) for node in best_path[1:]], axis=0)
        _cat_action = self.joint_info.sample_mask[None, ...] * _cat_action
        _cat_action: np.ndarray = _cat_action.reshape(((self.n_iter + 1) * nj, 3))
        _result: np.ndarray = np.ascontiguousarray(
            Rotation.from_rotvec(_cat_action).as_matrix().reshape((self.n_iter + 1, nj, 3, 3))[..., :2])
        return torch.from_numpy(_result)

    def pre_print(self, epoch, batch_idx, norm_grad, axis_angle, width, mid_state):
        with torch.no_grad():
            grad_norm_info = norm_grad.flatten().tolist() if norm_grad is not None else None
            max_action_info = list(np.max(axis_angle.view(width, -1).numpy(), axis=-1))
            mean_action_info = list(np.mean(np.abs(axis_angle.view(width, -1).numpy()), axis=-1))
            norm_action_info = torch.sqrt(torch.sum(axis_angle.view(width, -1) ** 2, dim=-1)).tolist()

            print(f"\n=====k_start = {self.k_start}, epoch = {epoch}, batch = {batch_idx}, ", end="")
            loss_info_str = " ".join([f"{node.loss:.3f}" for node in mid_state])
            print(f"loss = {loss_info_str}, total loss = {sum([node.loss for node in mid_state]):.3f}, =====")
            print(f"max action  = ", " ".join([f"{node:.4f}" for node in max_action_info]))
            print(f"mean action = ", " ".join([f"{node:.4f}" for node in mean_action_info]))
            if grad_norm_info is not None:
                print(f"grad_norm   = ", " ".join([f"{node:.4f}" for node in grad_norm_info]))
            print(f"action norm = ", " ".join([f"{node:.4f}" for node in norm_action_info]))

    def closure(self, epoch: int, batch_idx: int,
                noise: Optional[torch.Tensor],
                action_piece: torch.Tensor,
                width: int,
                lr_win: torch.Tensor,
                best_info: TrajStateCls):
        # noise is in the format of axis-angle
        character = self.character
        mid_path, mid_state = [], []
        self.worker.diff_world.import_from_state(self.traj_result.state_list[self.k_start - 1].state)
        tot_loss = torch.as_tensor(0.0)
        action_quat: torch.Tensor = DiffQuat.vec6d_to_quat(action_piece)
        if noise is not None:  # Add noise to action_quat
            noise = DiffQuat.quat_from_rotvec(noise.view(-1, 3))
            action_quat = DiffQuat.quat_multiply(noise, action_quat.view(-1, 4)).view_as(action_quat)

        axis_angle = DiffQuat.quat_to_rotvec(action_quat.view(-1, 4)).view(action_quat.shape[:-1] + (3,))

        # if epoch < width - 1:
        #     forward_count = epoch + 1
        # else:
        forward_count = width
        for piece in range(forward_count):
            diff_fwd_save: List[BodyInfoState] = self.worker.diffode_forward_with_save(self.k_start + piece, None, action_quat[piece])
            loss = self.worker.eval_loss_torch(self.k_start + piece)
            mid_state.append(TrajStateCls(character.save(), loss.item(), None))
            com, facing_com = character.body_info.calc_center_of_mass(), character.character_facing_coor_com()
            if self._calc_facing_com_is_too_far(com, facing_com, self.k_start + piece,
                                                self.cma_info.com_err_ratio, self.cma_info.com_y_err_ratio):
                print(f"Failed at {self.k_start + piece}")
                break
            tot_loss += loss
            mid_path.extend(diff_fwd_save)

        # export the body state to file, for debug..
        fdir = os.path.dirname(__file__)
        out_fname = f"k_start-{self.k_start}-epoch-{epoch}-batch-{batch_idx}.bin"
        with open(os.path.join(fdir, out_fname), "wb") as dump_fout:
            pickle.dump(mid_path, dump_fout)
        print(f"output path to {out_fname}")
        exit(0)

        self.traj_result.forward_count[self.k_start:self.k_start + len(mid_state)] += 1
        if tot_loss.item() < best_info.loss and len(mid_state) == width:
            best_info.action, best_info.loss = action_piece.detach().clone(), tot_loss.item()
            self.traj_result.state_list[self.k_start:self.k_start + len(mid_state)] = mid_state
            tot_path_idx: int = (self.k_start - 1) * self.sim_cnt + 1
            self.traj_result.tot_saved_path[tot_path_idx: tot_path_idx + len(mid_path)] = mid_path

        l1_reg_loss = self.l1_reg_coef * torch.mean(torch.abs(axis_angle))
        l2_reg_loss = self.l2_reg_coef * torch.mean(axis_angle ** 2)
        tot_loss += l2_reg_loss + l1_reg_loss
        if len(mid_state) > 0:
            tot_loss.backward()

        if action_piece.grad is not None:
            with torch.no_grad():
                shape0 = action_piece.shape[0]
                shape_norm = (shape0,) + tuple(1 for _ in range(action_piece.ndim - 1))
                norm_grad = torch.linalg.norm(action_piece.grad.view(shape0, -1), dim=-1).view(shape_norm)
                action_piece.grad *= lr_win
        else:
            norm_grad = None

        self.pre_print(epoch, batch_idx, norm_grad, axis_angle, width, mid_state)

        with torch.no_grad():
            clip_grad_info = torch.sqrt(torch.sum(action_piece.grad.view(width, -1) ** 2, dim=-1))
            print(f"cliped_grad = ", " ".join([f"{node:.4f}" for node in clip_grad_info]))

        return tot_loss

    def test_direct_trajopt(self):
        """
        Direct trajectory optimization by Diff ODE.
        for each sliding window:
            for each epoch:
                for each sample:
                    action_i = action center + sample
                    forward simulation
                    compute loss
                save action_i with best cost
                back-prop (the optimize parameter is action center)
        """
        self.batch_size = 1
        print(f"use inv dyn: {self.inv_dyn_target is not None}")
        print(f"n_iter = {self.n_iter}, len(state_list) = {len(self.traj_result.state_list)}")
        character = self.worker.character
        nj: int = len(self.joints)

        def optimize_func(width: int, max_epoch: int = 4, lr: float = 1e-2):
            action_piece = nn.Parameter(self.traj_result.action_hist[self.k_start: self.k_start + width].clone())
            lr_win: torch.Tensor = torch.from_numpy(0.7 ** np.arange(width - 1, -1, -1)).view(width, 1, 1, 1)
            print("lr_win = ", lr_win.flatten())
            opt = torch.optim.SGD([action_piece], lr=lr)
            best_info = TrajStateCls(None, float("inf"), None)

            for epoch in range(max_epoch):
                # generate random noise..
                if self.batch_size > 1:
                    noise: Optional[torch.Tensor] = self.initial_noise_sigma * torch.randn((self.batch_size - 1, width, nj, 3), dtype=torch.float64)
                else:
                    noise: Optional[torch.Tensor] = None
                # self.pre_print(self, epoch, "total", norm_grad, axis_angle, width, mid_state)
                # print(f"grad_norm   = ", " ".join([f"{node:.4f}" for node in grad_norm_info]))
                opt.zero_grad()
                self.closure(epoch, 0, None, action_piece, width, lr_win, best_info)
                for batch in range(self.batch_size - 1):
                    self.closure(epoch, batch + 1, noise[batch], action_piece, width, lr_win, best_info)
                with torch.no_grad():
                    action_piece.grad /= self.batch_size
                opt.step()
                if epoch == 5:
                   lr_win: torch.Tensor = torch.from_numpy(0.85 ** np.arange(width - 1, -1, -1)).view(width, 1, 1, 1)
                if epoch == 10:
                    lr_win: torch.Tensor = torch.from_numpy(0.9 ** np.arange(width - 1, -1, -1)).view(width, 1, 1, 1)
                if (epoch + 1) % 2 == 0:
                    opt.param_groups[0]["lr"] *= 0.85

            self.traj_result.action_hist[self.k_start: self.k_start + width] = best_info.action.detach().clone()

        def forward_optim():
            def forward_hlp():
                init_width = min(self.k_start + 2 * self.piece_num, self.n_iter + 1) - self.k_start
                optimize_func(init_width, self.total_epoch, self.initial_lr)
                torch.save({"k_start": self.k_start, "action": self.traj_result.action_hist, "conf": self.conf},
                            os.path.join(self.samhlp.save_folder_i_dname(), "test-traj-opt.ckpt"))

            while self.k_start + 2 * self.piece_num <= self.n_iter:
                forward_hlp()
                self.k_start += self.piece_num
            self.k_start = self.n_iter + 1 - 2 * self.piece_num
            forward_hlp()

            # Export to bvh file
            for path_idx, path_node in enumerate(self.traj_result.tot_saved_path):
                if path_node is None:
                    break
                character.load(path_node)
                self.worker.to_bvh.append_no_root_to_buffer()

            save_fname = os.path.join(self.samhlp.save_folder_i_dname(), "test-traj-opt.bvh")
            self.worker.to_bvh.to_file(save_fname)
            print(f"save bvh to {save_fname}")

        forward_optim()

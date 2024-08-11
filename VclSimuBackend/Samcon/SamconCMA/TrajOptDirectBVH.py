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
import logging
import numpy as np
import operator
import os
import pickle
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
import torch
from torch import nn

from typing import Callable, Optional, Union, List, Iterable, Dict, Any

from VclSimuBackend.ODESim.TargetPose import TargetPose

from .MainWorkerCMANew import SamconMainWorkerCMA
from ..StateTree import Sample
from ..SamconWorkerFull import SamconWorkerFull, RootPDControlParam
from ..SamconMainWorkerBase import WorkerInfo, SamHlp, LoadTargetPoseMode
from ..SamconUpdateScene import SamconUpdateScene
from ...Common.MathHelper import MathHelper, RotateType
from ...Common.Helper import Helper
from ...DiffODE import DiffQuat
from ...ODESim.ODEScene import ODEScene
from ...ODESim.ODECharacter import ODECharacter
from ...Utils.Evaluation import calc_nsr


class TrajStateCls:
    def __init__(self, state, loss, action, root_ctrl = None) -> None:
        self.state = state
        self.loss = loss
        self.action = action
        self.root_ctrl = root_ctrl


class TrajResState:
    def __init__(self, state_list, tot_saved_path, action_hist, forward_count, root_control_hist = None):
        self.state_list = state_list
        self.tot_saved_path = tot_saved_path
        self.action_hist = action_hist
        self.forward_count = forward_count
        self.root_control_hist = root_control_hist


class DirectTrajOptBVH(SamconMainWorkerCMA):
    def __init__(
        self,
        samhlp: SamHlp,
        worker_info: Optional[WorkerInfo],
        worker: Optional[SamconWorkerFull],
        scene: Optional[ODEScene] = None,
        sim_character: Optional[ODECharacter] = None,
        load_target_mode: LoadTargetPoseMode = LoadTargetPoseMode.BVH_MOCAP,
        inv_dyn_target_quat: Optional[np.ndarray] = None
    ):
        super().__init__(samhlp, worker_info, worker, scene, sim_character, load_target_mode)
        traj_optim_conf: Dict[str, Any] = self.conf["traj_optim"]

        self.root_pd_param: RootPDControlParam = self.worker.root_pd_param
        if inv_dyn_target_quat is not None:
            self.inv_dyn_target = TargetPose()
            self.inv_dyn_target.num_frames = inv_dyn_target_quat.shape[0]
            self.inv_dyn_target.locally.quat = inv_dyn_target_quat

        self.send_target_pose_no_scatter()
        self.ckpt_fname = os.path.join(self.samhlp.save_folder_i_dname(), "test-traj-opt.ckpt")
        self.initial_bvh_export = self.to_bvh.deepcopy()

        self.k_start: int = 1

        self.piece_num: int = traj_optim_conf["piece_num"]
        self.l2_reg_coef: float = traj_optim_conf["l2_reg_coef"]
        self.l1_reg_coef: float = traj_optim_conf["l1_reg_coef"]
        self.initial_lr: float = traj_optim_conf["initial_lr"]
        self.lr_decay: float = traj_optim_conf["lr_decay"]
        self.total_epoch: int = traj_optim_conf["total_epoch"]
        self.reject_ratio: float = traj_optim_conf["reject_ratio"]
        self.reject_lr_decay: float = traj_optim_conf["reject_lr_decay"]

        self.load_samcon_result: bool = traj_optim_conf.get("load_samcon_result", True)
        self.use_grad_clip: bool = traj_optim_conf.get("use_grad_clip", True)
        self.optim_var_type: RotateType = RotateType[traj_optim_conf.get("optim_var_type", "Vec6d")]
        self.debug_mode: bool = traj_optim_conf.get("debug_mode", False)  # flag for debug

        self.reject_count: int = 0
        self.traj_result = None
        self.initialize_traj_result()

        use_epoch_writer = False
        self._writer_logdir: str = os.path.join(samhlp.save_folder_i_dname(), samhlp.log_dir_name) + Helper.get_curr_time()
        logging.info(f"Create logging dir at {self._writer_logdir}")
        self.writer = SummaryWriter(os.path.join(self._writer_logdir, "global"))

        if use_epoch_writer:
            self.epoch_writers = [
                SummaryWriter(os.path.join(self._writer_logdir, "epoch" + str(epoch_idx)))
                for epoch_idx in range(self.total_epoch)
            ]
        else:
            self.epoch_writers = None

        self.in_sim_hack_func: Optional[Callable] = None

        self.worker.build_diff_world()  # also set camera param in this method

    def build_empty_state_list(self, samcon_actions: Optional[torch.Tensor] = None):
        result_state_list = [TrajStateCls(copy.deepcopy(self.tree.level(0)[0].s1), 0.0, samcon_actions[0].detach().clone() if samcon_actions is not None else None)] + \
                [TrajStateCls(None, 23333.3, samcon_actions[i].detach().clone() if samcon_actions is not None else None) for i in range(1, self.n_iter + 1)]
        return result_state_list

    def initialize_traj_result(self):
        if not os.path.exists(self.ckpt_fname):
            best_path: Optional[str] = self.samhlp.best_path_fname()
            if isinstance(best_path, str) and self.load_samcon_result:
                if not os.path.isfile(best_path):
                    print(f"best path filename {best_path} not exist. ignore..")
                    best_path = None
                else:
                    with open(best_path, "rb") as fin:
                        print(f"load best path from {best_path}")
                        best_path_dict = pickle.load(fin)
                        assert os.path.split(best_path_dict["conf"]["filename"]["bvh"])[1] == os.path.split(self.conf["filename"]["bvh"])[1]
                        best_path: List[Sample] = best_path_dict["best_path"]
                    best_path: List[Sample] = best_path[:self.n_iter + 1]
            else:
                best_path = None

            samcon_actions: torch.Tensor = self.load_samcon_res_func(best_path)
            result_state_list = self.build_empty_state_list(samcon_actions)
            if best_path is not None:
                for i in range(1, self.n_iter + 1):
                    result_state_list[i].state = best_path[i].s1
                    result_state_list[i].loss = 2 * best_path[i].cost
                update_scene = SamconUpdateScene(self.worker, best_path, None, True, False)
                _ = list(iter(update_scene.uhelp))
                tot_saved_path, samcon_ret_motion = update_scene.ret_sim_hist, update_scene.ret_motion
            else:
                tot_saved_path = [copy.deepcopy(self.tree.level(0)[0].s1)] + [None for _ in range(self.n_iter * self.sim_cnt)]

            self.traj_result = TrajResState(
                result_state_list,
                tot_saved_path,
                samcon_actions,
                np.zeros((self.n_iter + 1,), dtype=np.int32),
                torch.zeros((self.n_iter + 1, 3), dtype=torch.float64)  # virtual root force..
            )
        else:
            ckpt_result = torch.load(self.ckpt_fname)
            self.k_start = ckpt_result["k_start"]
            self.traj_result = ckpt_result["traj_result"]
            print(f"load middle state from {self.ckpt_fname}")

    def divide_action_to_smaller(self, new_sample_fps: int):
        # it does not work..
        assert new_sample_fps % self.sample_fps == 0
        assert self.traj_result.action_hist is not None
        action_hist = self.traj_result.action_hist[1:]
        repeat_count: int = new_sample_fps // self.sample_fps
        piece_shape = action_hist.shape[1:]
        action_hist = action_hist.repeat((repeat_count,) + (1,) * (action_hist.ndim))
        action_hist = torch.transpose(action_hist, 0, 1).reshape(-1, *piece_shape)
        self.traj_result.action_hist = torch.cat([self.traj_result.action_hist[0:1], action_hist], dim=0)
        self.traj_result.forward_count = np.zeros(self.n_iter * repeat_count + 1, dtype=np.int32)

        self.set_sample_fps(new_sample_fps)
        self.worker.set_sample_fps(new_sample_fps)
        self.piece_num *= repeat_count
        self.n_iter *= repeat_count
        self.worker.n_iter *= repeat_count
        self.traj_result.state_list = self.build_empty_state_list(self.traj_result.action_hist)
        print(f"Divide new sample fps to {new_sample_fps}")

    def load_samcon_res_func(self, best_path: Union[str, List[Sample], None]):
        nj: int = self.num_joints
        if self.optim_var_type == RotateType.Vec6d:
            unit_func = lambda: torch.from_numpy(
                MathHelper.quat_to_vec6d(
                    MathHelper.unit_quat_arr((self.n_iter + 1, nj, 4))
                )
            )
        elif self.optim_var_type == RotateType.AxisAngle:
            unit_func = lambda: torch.zeros((self.n_iter + 1, nj, 3), dtype=torch.float64)
        else:
            raise NotImplementedError

        if best_path is None:
            return unit_func()

        if len(best_path) != self.n_iter + 1:
            print(f"length of best path = {len(best_path)} doesn't match.")
            return unit_func()
        _cat_action: np.ndarray = np.concatenate(
            operator.add(
                [np.zeros((1, nj, 3))],
                [node.a0.reshape((1, nj, 3)) for node in best_path[1:]]
            ),
            axis=0
        )
        _cat_action: np.ndarray = self.joint_info.sample_mask[None, ...] * _cat_action
        _cat_action: np.ndarray = _cat_action.reshape(((self.n_iter + 1) * nj, 3))
        if self.optim_var_type == RotateType.Vec6d:
            _result: np.ndarray = np.ascontiguousarray(
                Rotation.from_rotvec(_cat_action).as_matrix().reshape((self.n_iter + 1, nj, 3, 3))[..., :2]
            )
        elif self.optim_var_type == RotateType.AxisAngle:
            _result = _cat_action
        else:
            raise ValueError

        return torch.from_numpy(_result)

    def writer_append(self, info: str, nodes: Iterable[float]):
        for node_idx, node in enumerate(nodes):
            self.writer.add_scalar(f"TimePieces{self.k_start + node_idx}/{info}", node,
                                    global_step=self.traj_result.forward_count[self.k_start + node_idx])

    # def epoch_writer_append(self, epoch: int, info: str, nodes: Iterable[float]):
    #     if self.epoch_writers is None:
    #         return
    #     writer = self.epoch_writers[epoch]
    #     for node_idx, node in enumerate(nodes):
    #         writer.add_scalar(f"SlidingWindows_{self.k_start}/{info}", node, self.k_start + node_idx)

    def pre_print_log(self, epoch: int, opt: torch.optim.Optimizer, norm_grad: torch.Tensor,
                      width: int, mid_state, axis_angle: torch.Tensor,
                      l1_reg_loss: Union[float, torch.Tensor],
                      l2_reg_loss: Union[float, torch.Tensor],
                      tot_loss: torch.Tensor,
                      root_ctrl: Optional[torch.Tensor] = None,
                      do_print: bool = True):
        mid_loss = [node.loss for node in mid_state]
        with torch.no_grad():
            grad_norm_info = norm_grad.flatten().tolist() if norm_grad is not None else None
            max_action_info = list(np.max(axis_angle.view(width, -1).numpy(), axis=-1))
            mean_action_info = list(np.mean(np.abs(axis_angle.view(width, -1).numpy()), axis=-1))
            norm_action_info = torch.sqrt(torch.sum(axis_angle.view(width, -1) ** 2, dim=-1)).tolist()
            if root_ctrl is not None:
                root_norm = torch.linalg.norm(root_ctrl, dim=1)
                root_grad_norm = torch.linalg.norm(root_ctrl.grad, dim=1) if root_ctrl.grad is not None else None
            else:
                root_norm = root_grad_norm = None

            if do_print:
                print(f"\n=====k_start = {self.k_start}, epoch = {epoch}, ", end="")
                loss_info_str = " ".join([f"{tmp_loss:.3f}" for tmp_loss in mid_loss])
                print(f"loss = {loss_info_str}, total loss = {sum(mid_loss):.3f}, "
                        f"lr = {opt.param_groups[0]['lr']:.6f} =====")
                print(f"max action  = ", " ".join([f"{node:.4f}" for node in max_action_info]))
                print(f"mean action = ", " ".join([f"{node:.4f}" for node in mean_action_info]))
                if grad_norm_info is not None:
                    print(f"grad_norm   = ", " ".join([f"{node:.4f}" for node in grad_norm_info]))
                print(f"action norm = ", " ".join([f"{node:.4f}" for node in norm_action_info]))
                if root_norm is not None:
                    print(f"root norm = ", " ".join([f"{node:4f}" for node in root_norm]))
                if root_grad_norm is not None:
                    print(f"root grad = ", " ".join([f"{node:4f}" for node in root_grad_norm]))

            # Write to Summary Writer
            self.writer_append("loss", mid_loss)
            # if grad_norm_info is not None:
            #     self.writer_append("grad norm", grad_norm_info)
            # self.writer_append("max action", max_action_info)
            # self.writer_append("mean action", mean_action_info)
            # self.writer_append("norm action", norm_action_info)
            # self.writer.add_scalar(f"SlidingWindows_{self.k_start}/l1", l1_reg_loss.item(), epoch)
            # self.writer.add_scalar(f"SlidingWindows_{self.k_start}/l2", l2_reg_loss.item(), epoch)
            # self.writer.add_scalar(f"SlidingWindows_{self.k_start}/Total", tot_loss.item(), epoch)
            # self.writer.add_scalar(f"SlidingWindows_{self.k_start}/learning_rate", opt.param_groups[0]['lr'], epoch)
            # self.epoch_writer_append(epoch, "loss win", mid_loss)
            # if grad_norm_info is not None:
            #     self.epoch_writer_append(epoch, "grad norm win", grad_norm_info)
            # self.epoch_writer_append(epoch, "max action win", max_action_info)
            # self.epoch_writer_append(epoch, "mean action win", mean_action_info)
            # self.epoch_writer_append(epoch, "norm action win", norm_action_info)

    # def post_print_log(self, epoch: int, action_piece: torch.Tensor, width: int):
    #     if action_piece.grad is None:
    #         return
    #     with torch.no_grad():
    #         clip_grad_info = torch.sqrt(torch.sum(action_piece.grad.view(width, -1) ** 2, dim=-1))
    #         print(f"cliped_grad = ", " ".join([f"{node:.4f}" for node in clip_grad_info]))
    #         self.writer_append("cliped grad", clip_grad_info.tolist())
    #         self.epoch_writer_append(epoch, "clip grad norm win", clip_grad_info)

    @staticmethod
    def clip_gradient(vec: torch.Tensor, clip_value = 30): # 20):
        with torch.no_grad():
            shape0: int = vec.shape[0]
            norm_grad = torch.linalg.norm(
                vec.grad.view(shape0, -1), dim=-1
            ).view(shape0, *([1] * (vec.ndim - 1)))
            grad_clip = norm_grad.clone()
            grad_clip[grad_clip < clip_value] = 1
            grad_clip[grad_clip >= clip_value] /= clip_value
            vec.grad /= grad_clip
        return norm_grad

    def closure(self,
                epoch: int,
                opt: torch.optim.Optimizer,
                action_piece: torch.Tensor,
                root_ctrl_piece: Optional[torch.Tensor],
                width: int,
                best_info: TrajStateCls,
                zero_grad: bool = True):
        if zero_grad:
            opt.zero_grad()
        mid_path, mid_state = [], []
        self.worker.diff_world.import_from_state(self.traj_result.state_list[self.k_start - 1].state)
        tot_loss = torch.as_tensor(0.0)

        # different optimize variable type
        if self.optim_var_type == RotateType.Vec6d:
            action_quat: torch.Tensor = DiffQuat.vec6d_to_quat(action_piece)
            axis_angle = DiffQuat.quat_to_rotvec(action_quat.view(-1, 4)).view(action_quat.shape[:-1] + (3,))
        elif self.optim_var_type == RotateType.AxisAngle:
            action_quat: torch.Tensor = DiffQuat.quat_from_rotvec(
                action_piece.view(-1, 3)).view(action_piece.shape[:-1] + (4,)
            )
            axis_angle: torch.Tensor = action_piece
        else:
            raise ValueError

        # forward simulation
        for piece in range(width):
            curr_t: int = self.k_start + piece
            forward_count, loss_index = self.worker.get_forward_count(curr_t)
            if self.root_pd_param.in_use:
                root_pd_action: Optional[torch.Tensor] = root_ctrl_piece[piece]
            else:
                root_pd_action: Optional[torch.Tensor] = None

            diff_fwd_save, contact_hack_loss = self.worker.diffode_forward_with_save(
                curr_t, None, action_quat[piece], root_pd_action, epoch, forward_count)

            loss = self.worker.torch_loss.loss(loss_index)
            loss += contact_hack_loss
            curr_state = self.worker.diff_world.character.save()
            mid_state.append(TrajStateCls(curr_state, loss.item(), None))
            tot_loss += loss
            mid_path.extend(diff_fwd_save)

        self.traj_result.forward_count[self.k_start:self.k_start + len(mid_state)] += 1

        # dump mid path and action piece to file, for visualize..
        pickle_fname = os.path.join(self.samhlp.save_folder_i_dname(), f"kstart-{self.k_start}-epoch-{epoch}.pickle")
        with open(pickle_fname, "wb") as fout:
            pickle.dump(mid_path, fout)

        # if the loss become bad, e.g. current loss > ratio * best loss. we should reject this update..
        norm_grad = None
        l1_reg_loss = self.l1_reg_coef * torch.mean(torch.abs(axis_angle))
        l2_reg_loss = self.l2_reg_coef * torch.mean(axis_angle ** 2)
        if tot_loss.item() > self.reject_ratio * best_info.loss:
            tot_loss.backward()
            opt.zero_grad()
            print(f"reject update. reset to action with best loss..")
            action_piece.data[:] = best_info.action.clone()
            self.reject_count += 1
            if self.reject_count >= 2:
                # maybe we can use a smaller learning rate here...?
                opt.param_groups[0]["lr"] *= self.reject_lr_decay
                self.reject_count = 0
        else:
            # save best state
            if tot_loss.item() < best_info.loss and len(mid_state) == width:
                best_info.action, best_info.loss = action_piece.detach().clone(), tot_loss.item()
                if root_ctrl_piece is not None:
                    best_info.root_ctrl = root_ctrl_piece.detach().clone()
                self.traj_result.state_list[self.k_start:self.k_start + len(mid_state)] = mid_state
                tot_path_idx: int = (self.k_start - 1) * self.sim_cnt + 1
                self.traj_result.tot_saved_path[tot_path_idx: tot_path_idx + len(mid_path)] = mid_path

            tot_loss += l2_reg_loss + l1_reg_loss
            if len(mid_state) > 0:
                tot_loss.backward()
            else:
                print(f"Warning !!! len(mid_state) == 0.")

            # clip action norm
            if action_piece.grad is not None:
                norm_grad = self.clip_gradient(action_piece, 10)  # 20

            if root_ctrl_piece is not None and root_ctrl_piece.grad is not None:
                root_ctrl_piece.grad[-1] *= 20
                root_ctrl_piece.grad[-2] *= 10
                root_ctrl_piece.grad[:-3] *= 0.8
                self.clip_gradient(root_ctrl_piece, 50)


        self.pre_print_log(epoch, opt, norm_grad, width, mid_state, axis_angle, l1_reg_loss, l2_reg_loss, tot_loss, root_ctrl_piece)
        return tot_loss

    def best_loss_calc(self, init_width):
        return sum([self.traj_result.state_list[i].loss for i in range(self.k_start, self.k_start + init_width)])

    def optimize_func_with_root_force(self, width: int, max_epoch: int = 4, lr: float = 1e-2):
        self.root_pd_param.reset_init_kp_kd()
        for decay_index in range(self.root_pd_param.decay_count):
            # for simple, we can only decay the kp and kd param..
            action_piece_in: torch.Tensor = self.traj_result.action_hist[self.k_start: self.k_start + width].clone()
            root_piece_in: torch.Tensor = self.traj_result.root_control_hist[self.k_start: self.k_start + width].clone()
            action_piece: torch.Parameter = nn.Parameter(action_piece_in)
            root_piece: torch.Parameter = nn.Parameter(root_piece_in)
            opt = torch.optim.SGD([action_piece, root_piece], lr=lr)
            best_info = TrajStateCls(action_piece_in.detach().clone(), 2 * self.best_loss_calc(width), None)
            self.reject_count = 0
            for epoch in range(max_epoch):
                self.closure(epoch, opt, action_piece, root_piece, width, best_info, True)
                opt.step()
                if (epoch + 1) % 2 == 0:
                    for groups in opt.param_groups:
                        groups["lr"] *= self.lr_decay
            # clear previous best path..

            print(f"At decay index = {decay_index}, ", end="")
            if decay_index < self.root_pd_param.decay_count - 1:
                self.root_pd_param.decay_kp_kd(True)
                for k in range(self.k_start, self.k_start + width):
                    self.traj_result.state_list[k].loss = 23333.3
            else:
                self.root_pd_param.zero_kp_kd(True)

        self.traj_result.action_hist[self.k_start: self.k_start + width] = best_info.action.detach().clone()

    def optimize_func(self, width: int, max_epoch: int = 4, lr: float = 1e-2):
        action_piece_in = self.traj_result.action_hist[self.k_start: self.k_start + width].clone()
        action_piece = nn.Parameter(action_piece_in)
        use_lbfgs = True
        # if self.k_start > 1:
        #   lr *= 0.1
        if not use_lbfgs:
            opt = torch.optim.SGD([action_piece], lr=lr)
        else:
            opt = torch.optim.LBFGS([action_piece], lr=lr, max_iter=10)  #line_search_fn="strong_wolfe")

        best_info = TrajStateCls(action_piece_in.detach().clone(), 2 * self.best_loss_calc(width), None)

        self.reject_count = 0
        for epoch in range(max_epoch):
            if False:  # use batch to optimize
                opt.zero_grad()
                for batch_size in range(5):
                    noise = opt.param_groups[0]["lr"] * torch.randn(action_piece.shape, dtype=torch.float64)
                    action_closure = action_piece + noise
                    self.closure(epoch, opt, action_closure, None, width, best_info, False)
                action_piece.grad /= 5
            # else:
            #     # noise = 10 * opt.param_groups[0]["lr"] * torch.randn(action_piece.shape, dtype=torch.float64)
            #     # action_closure = action_piece + noise
            #     self.closure(epoch, opt, action_piece, None, width, best_info, True)
            if use_lbfgs:
                opt.step(lambda : self.closure(epoch, opt, action_piece, None, width, best_info, True))
            else:
                self.closure(epoch, opt, action_piece, None, width, best_info, True)
                opt.step()
            if (epoch + 1) % 2 == 0:
                opt.param_groups[0]["lr"] *= self.lr_decay
            if best_info.loss < 2.5 * width:
                break

        self.traj_result.action_hist[self.k_start: self.k_start + width] = best_info.action.detach().clone()

    def save_middle_state(self, ckpt_fname: Optional[str] = None):
        if ckpt_fname is None:
            ckpt_fname = self.ckpt_fname
        torch.save({"k_start": self.k_start, "traj_result": self.traj_result, "conf": self.conf}, ckpt_fname)

    def forward_optim(self, trial_index: int = 0):
        def forward_hlp():
            init_width = min(self.k_start + 2 * self.piece_num, self.n_iter + 1) - self.k_start
            prev_best_loss = self.best_loss_calc(init_width)
            if not self.root_pd_param.in_use:  # no root control
                self.optimize_func(init_width, self.total_epoch, self.initial_lr)
            else:  # with root control
                self.optimize_func_with_root_force(init_width, self.total_epoch, self.initial_lr)

            after_best_loss = self.best_loss_calc(init_width)
            print(f"prev best cost {prev_best_loss:.3f}, after best cost = {after_best_loss:.3f}")
            self.save_middle_state()
            self.save_middle_state(f"{self.ckpt_fname}{self.k_start}")
            need_rollback = after_best_loss >= 1.4 * prev_best_loss

            return need_rollback

        rollback_count = 0
        while self.k_start + 2 * self.piece_num <= self.n_iter:
            need_rollback = forward_hlp()
            # we should add rollback here..
            if need_rollback:
                self.k_start = max(1, self.k_start - 10)
                print(f"rollback to {self.k_start}, rollback count = {rollback_count}")
                rollback_count += 1
            else:
                self.k_start += min(5, self.piece_num)
                # self.k_start += 1
            if rollback_count > 3:
                print(f"failed to reconstruct")
                return

        self.k_start = max(1, self.n_iter + 1 - 2 * self.piece_num)
        forward_hlp()

        # eval loss globally
        print("Global Path Loss:")
        for node_idx, node in enumerate(self.traj_result.state_list):
            if node_idx == 0:
                continue
            self.writer.add_scalar("Global Path Loss", node.loss, node_idx)
            print(f"index = {node_idx}, loss = {node.loss:.3f}")
        print(f"Avg Loss = {sum([node.loss for node in self.traj_result.state_list[1:]]) / self.n_iter:.3f}")

        # dump the total saved path
        with open(os.path.join(self.samhlp.save_folder_i_dname(), f"final-result.pickle"), "wb") as fout:
            pickle.dump(self.traj_result.tot_saved_path, fout)

        # Export to bvh file
        for path_idx, path_node in enumerate(self.traj_result.tot_saved_path):
            if path_node is None:
                break
            self.worker.diff_world.character.load(path_node)
            self.worker.to_bvh.append_no_root_to_buffer()

        save_fname = os.path.join(self.samhlp.save_folder_i_dname(), f"test-traj-opt{trial_index}.bvh")
        nsr_sim_motion = self.worker.to_bvh.to_file(save_fname).remove_end_sites(True)
        print(f"save bvh to {save_fname}")

        # evaluate nsr value
        if nsr_sim_motion.num_frames > self.motion_input.num_frames:
            nsr_sim_motion = nsr_sim_motion.sub_sequence(0, self.motion_input.num_frames)

        nsr_ref_motion = self.motion_input.sub_sequence(0, nsr_sim_motion.num_frames).remove_end_sites(True)

        nsr_value = calc_nsr(nsr_sim_motion, nsr_ref_motion)
        print(f"nsr value = {nsr_value:.4f}")

        self.to_bvh = self.initial_bvh_export.deepcopy()

    def test_direct_trajopt(self):
        """
        Direct trajectory optimization by Diff ODE.

        """
        print(f"use inv dyn: {self.inv_dyn_target is not None}")
        print(f"n_iter = {self.n_iter}, len(state_list) = {len(self.traj_result.state_list)}")

        self.forward_optim()

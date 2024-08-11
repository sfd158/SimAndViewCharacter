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

from argparse import Namespace
from datetime import datetime
from enum import IntEnum
import gc
from mpi4py import MPI
import numpy as np
import random
import os
import torch
from torch import nn
from torch.autograd import Function
from typing import Optional, Union, Any, Tuple, List, Dict

from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo
from VclSimuBackend.DiffODE.DiffFrameInfo import diff_frame_import_from_tensor, DiffFrameInfo
from VclSimuBackend.DiffODE.DiffODEWorld import DiffODEWorld
from VclSimuBackend.DiffODE.Build import BuildFromODEScene

from VclSimuBackend.pymotionlib.PyTorchMotionData import PyTorchMotionData

from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.Common.MathHelper import RotateType
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter

from VclSimuBackend.Samcon.OptimalGait.ContactWithKinematic import ContactLabelExtractor
from VclSimuBackend.Samcon.SamconWorkerBase import SamconWorkerBase

from VclSimuBackend.Samcon.PolicyTrain.SamconDataLoader import SamconDataLoader


comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda")


class TrainingInstruction(IntEnum):
    DiffODELossReduced = 0
    DiffODELossMaximal = 1
    DataPrepare = 2
    Stop = 3


class SimulationLoss:
    """
    evaluate with diff ode for physics loss..
    Note: the contact label doesn't provides the gradient directly.
    """
    def __init__(self, args: Namespace, conf_fname: str) -> None:
        self.args: Namespace = args
        self.rotate_type: Optional[RotateType] = args.rotate_type
        # here we should load the character from the file
        self.scene: ODEScene = SamconWorkerBase.load_scene_with_conf(conf_fname)
        self.character: Optional[ODECharacter] = self.scene.character0
        self.extractor: ContactLabelExtractor = ContactLabelExtractor(self.scene, self.character)
        self.builder: BuildFromODEScene = BuildFromODEScene(self.scene)
        self.diff_world: Optional[DiffODEWorld] = self.builder.build()
        self.data_loader: SamconDataLoader = SamconDataLoader(self.scene, self.character, self.args)

    @property
    def curr_frame(self) -> DiffFrameInfo:
        return self.diff_world.curr_frame

    def prepare_data(self, fname_list: List[str]):
        tot_result: List = []
        # for debug in child worker..

        for fname in fname_list:
            if not os.path.exists(fname):
                continue
            raw_name = os.path.split(fname)[1]
            data_label: str = raw_name[:2]
            if data_label not in self.data_loader.camera_numpy.keys():
                data_label: str = "S1"
            print(f"comm_rank = {comm_rank}, prepare = {raw_name}", flush=True)
            ret_list = self.data_loader.prepare_one_piece(fname, data_label, pbar=None)
            # print(f"after prepare {fname}, comm_rank = {comm_rank}")
            tot_result.extend(ret_list)
        return tot_result

    @torch.enable_grad()
    def compute_loss_maximal_coordinate(
        self,
        index: int,
        start_index: int,
        body_pos_numpy: np.ndarray,
        body_matrix_numpy: np.ndarray,
        body_quat_numpy: np.ndarray,
        pd_target_numpy: np.ndarray,
        contact_label_numpy: np.ndarray,
        divide_index_numpy: np.ndarray
    ):
        fps: int = self.scene.sim_fps
        num_frame: int = body_pos_numpy.shape[0]
        coef: float = float(num_frame - 1) / (self.args.batch_size - 1)
        assert body_matrix_numpy.shape[0] == pd_target_numpy.shape[0] == contact_label_numpy.shape[0] == divide_index_numpy.shape[0]
        body_pos_float64: np.ndarray = body_pos_numpy.astype(np.float64)
        body_matrix_float64: np.ndarray = body_matrix_numpy.astype(np.float64)
        body_quat_float64: np.ndarray = body_quat_numpy.astype(np.float64)
        pd_target_float64: np.ndarray = pd_target_numpy.astype(np.float64)
        body_pos: torch.Tensor = nn.Parameter(torch.from_numpy(body_pos_float64))
        body_matrix: torch.Tensor = nn.Parameter(torch.from_numpy(body_matrix_float64))
        body_quat: torch.Tensor = nn.Parameter(torch.from_numpy(body_quat_float64))
        pd_target: torch.Tensor = nn.Parameter(torch.from_numpy(pd_target_float64))
        inv_divide_index: torch.Tensor = torch.from_numpy(~divide_index_numpy)[:-1]
        num_body: int = len(self.character.bodies)
        body_c_id: np.ndarray = self.character.body_info.body_c_id

        # Note: velocity of some frame is not used, which will make no contribution to gradient
        body_vel: torch.Tensor = (body_pos[1:] - body_pos[:-1]) * fps # (width - 1, num body, 3)   # compute linear velocity by position
        body_omega: torch.Tensor = PyTorchMotionData.compute_angvel_frag(body_quat, fps)  # compute angular velocity by rotation
        sim_body_pos_list, sim_body_mat_list, sim_body_velo_list, sim_body_omega_list = [], [], [], []

        dummy_pos, dummy_mat = torch.zeros((1,) + body_pos.shape[1:]), torch.zeros((1,) + body_matrix.shape[1:])
        dummy_velo, dummy_omega = torch.zeros((1,) + body_vel.shape[1:]), torch.zeros((1,) + body_omega.shape[1:])

        for frame in range(num_frame - 1):
            if frame == 0:
                import_velo = torch.zeros_like(body_pos[0])
                import_omega = torch.zeros_like(import_velo)
            else:
                import_velo = body_vel[frame - 1]
                import_omega = body_omega[frame - 1]

            if divide_index_numpy[frame] == 1:
                sim_body_pos_list.append(dummy_pos)
                sim_body_mat_list.append(dummy_mat)
                sim_body_velo_list.append(dummy_velo)
                sim_body_omega_list.append(dummy_omega)
                continue

            # load body state here..
            diff_frame_import_from_tensor(self.curr_frame, body_pos[frame], import_velo, body_matrix[frame],
                body_quat[frame], import_omega, self.scene.world, body_c_id)
            # create contact using given contact label. Note: create contact doesn't requires gradient
            ret_contact = self.extractor.compute_contact_by_sub_body(body_pos_float64[frame], body_quat_float64[frame], contact_label_numpy[frame])
            diff_contact: DiffContactInfo = self.extractor.convert_to_diff_contact_single(*ret_contact)
            diff_contact.cfm.fill_(1e-5)
            self.diff_world.add_hack_local_contact(diff_contact)
            # add control signal
            self.curr_frame.stable_pd_control_fast(pd_target[frame])
            self.diff_world.step()
            # compute the loss here..we can also compute the contact height loss here..
            sim_body_pos_list.append(self.curr_frame.body_pos.view(1, num_body, 3))
            sim_body_mat_list.append(self.curr_frame.body_rot.view(1, num_body, 3, 3))
            sim_body_velo_list.append(self.curr_frame.body_velo.view(1, num_body, 3))
            sim_body_omega_list.append(self.curr_frame.body_omega.view(1, num_body, 3))

        # Note: we should not compute loss when divide_index_numpy[frame] == 1..we can use a mask instead
        sim_body_pos: torch.Tensor = torch.cat(sim_body_pos_list, dim=0)  # (width - 1, num body, 3)
        sim_body_mat: torch.Tensor = torch.cat(sim_body_mat_list, dim=0)  # (width - 1, num body, 3, 3)
        sim_body_velo: torch.Tensor = torch.cat(sim_body_omega_list, dim=0) # (width - 1, num body, 3)
        sim_body_omega: torch.Tensor = torch.cat(sim_body_omega_list, dim=0)  # (width - 1, num body, 3)

        pos_loss: torch.Tensor = (coef * self.args.w_phys_pos) * torch.mean((inv_divide_index.view(-1, 1, 1) * (sim_body_pos - body_pos[1:])) ** 2)
        rot_loss: torch.Tensor = (coef * self.args.w_phys_rot) * torch.mean((inv_divide_index.view(-1, 1, 1, 1) * (sim_body_mat - body_matrix[1:])) ** 2)
        velo_loss: torch.Tensor = (coef * self.args.w_phys_velo) * torch.mean((inv_divide_index.view(-1, 1, 1) * (sim_body_velo - body_vel)) ** 2)
        omega_loss: torch.Tensor = (coef * self.args.w_phys_omega) * torch.mean((inv_divide_index.view(-1, 1, 1) * (sim_body_omega - body_omega)) ** 2)
        total_loss: torch.Tensor = pos_loss + rot_loss + velo_loss + omega_loss
        total_loss.backward()

        loss_dict = {
            "simu_pos_loss": pos_loss.item(),
            "simu_rot_loss": rot_loss.item(),
            "simu_velo_loss": velo_loss.item(),
            "simu_omega_loss": omega_loss.item(),
            "simu_total_loss": total_loss.item()
        }
        # return the gradient
        body_pos_grad: np.ndarray = body_pos.grad.to(torch.float32).numpy()
        body_mat_grad: np.ndarray = body_matrix.grad.to(torch.float32).numpy()
        body_quat_grad: np.ndarray = body_quat.grad.to(torch.float32).numpy()
        pd_target_grad: np.ndarray = pd_target.grad.to(torch.float32).numpy()

        return index, start_index, loss_dict, body_pos_grad, body_mat_grad, body_quat_grad, pd_target_grad

    def compute_loss(
        self,
        root_pos_numpy: np.ndarray,  # (in shape batch size, 3)
        joint_rotate_numpy: np.ndarray,  # (in shape batch size, num joint + 1, ?)
        potential_rotate: Optional[np.ndarray],
        pd_target_numpy: np.ndarray,  # (in shape batch size, num joint, ?)
        contact_label_numpy: np.ndarray,  # (in shape batch size, num body)
        divide_index_numpy: np.ndarray, # (in shape batch size)
    ) -> Tuple[np.ndarray, np.ndarray, None, np.ndarray]:
        """
        Here we need only evaluate the simulation loss, contact height loss.
        We need not evaluate other parts, for they have been evaluated in main worker on GPU.

        Note: the input data type are torch.float32
        We should convert to torch.float64 for forward simulation in diff-ode

        Note: we can also run this method at maximal coordinate
        The global position and rotation is already computed by previous part..
        """
        raise NotImplementedError
        num_frame: int = root_pos_numpy.shape[0]
        assert num_frame == joint_rotate_numpy.shape[0]
        root_pos: nn.Parameter = nn.Parameter(torch.as_tensor(root_pos_numpy, dtype=torch.float64))
        joint_rotate: nn.Parameter = nn.Parameter(torch.as_tensor(joint_rotate_numpy, dtype=torch.float64))
        pd_target: nn.Parameter = nn.Parameter(torch.as_tensor(pd_target_numpy, dtype=torch.float64))

        for frame in range(num_frame - 1):
            if divide_index_numpy[frame] == 1:  # we should not compute physics loss
                continue
            # generate contact from contact label.
            sub_contact_label: np.ndarray = contact_label_numpy[frame]

        root_pos_grad: np.ndarray = root_pos.grad.numpy()
        joint_rotate_grad: np.ndarray = joint_rotate.grad.numpy()
        pd_target_grad: np.ndarray = pd_target.grad.numpy()

        return root_pos_grad, joint_rotate_grad, None, pd_target_grad


simu_loss_handle: Optional[SimulationLoss] = None

def simu_loss_init(args: Namespace, conf_fname: str) -> SimulationLoss:
    global simu_loss_handle
    if simu_loss_handle is None:
        simu_loss_handle = SimulationLoss(args, conf_fname)
    return simu_loss_handle


def stop_child_workers():
    """
    stop all child workers.
    """
    assert comm_rank == 0
    comm.barrier()
    comm.scatter([(TrainingInstruction.Stop, None) for _ in range(comm_size)])


def prepare_data_parallel(input_dirs: List[str]):
    assert comm_rank == 0, f"In assert, comm_rank == {comm_rank}"
    start_time = datetime.now()
    random.shuffle(input_dirs)
    divide_index: List[int] = np.linspace(0, len(input_dirs), comm_size + 1, dtype=np.int32).tolist()
    divide_list = [(TrainingInstruction.DataPrepare, input_dirs[divide_index[i]: divide_index[i + 1]]) for i in range(comm_size)]
    comm.barrier()
    worker_mode, forward_simu_info = comm.scatter(divide_list)
    data_result = simu_loss_handle.prepare_data(forward_simu_info)
    comm.barrier()
    for rank in range(1, comm_size):
        print(f"recieve from rank {rank}")
        while True:
            rank_result = comm.recv(None, rank)
            if rank_result is None:
                break
            data_result.extend(rank_result)

    print(f"After load data from {len(input_dirs)} files", flush=True)
    Helper.print_total_time(start_time)
    return data_result


def prepare_data_parallel_slow(input_dirs: List[str]):
    """
    Note: we need to compute mean and std after load data..
    When there are too much data, we can not send the data batchly..
    """
    assert comm_rank == 0, f"In assert, comm_rank == {comm_rank}"
    # divide the input dirs into several parts.
    # random.shuffle(input_dirs)
    input_dirs.sort()
    divide_index: List[int] = np.linspace(0, len(input_dirs), comm_size + 1, dtype=np.int32).tolist()
    divide_list = [input_dirs[divide_index[i]: divide_index[i + 1]] for i in range(comm_size)]
    max_count: int = max([len(node) for node in divide_list])
    total_result = []
    for i in range(max_count):
        print(f"before barrier, i = {i}", flush=True)
        comm.barrier()
        print("after barrier", flush=True)
        scatter_list = [(TrainingInstruction.DataPrepare, divide_list[j][i:i+1] if len(divide_list[j]) > i else []) for j in range(comm_size)]
        worker_mode, forward_simu_info = comm.scatter(scatter_list)
        data_result = simu_loss_handle.prepare_data(forward_simu_info)
        # print(f"before gather, comm_rank = {comm_rank}", flush=True)
        comm.barrier()
        data_result_list = comm.gather(data_result)
        # print(f"after gather, comm_rank = {comm_rank}", flush=True)
        data_result_list = sum(data_result_list, [])
        total_result.extend(data_result_list)

    print(f"After load data from {len(input_dirs)} files")
    return total_result


def run_child_worker():
    print(f"start child worker {comm_rank} / {comm_size}", flush=True)
    while True:
        comm.barrier()
        worker_mode, *forward_simu_info = comm.scatter(None)  # here we should also recieve stop info..
        # print(f"get instruction {worker_mode} at worker {comm_rank}")
        if worker_mode == TrainingInstruction.Stop:
            print(f"stop child worker {comm_rank} / {comm_size}", flush=True)
            break
        elif worker_mode == TrainingInstruction.DataPrepare:
            # here we should prepare the input data in child worker
            result = simu_loss_handle.prepare_data(*forward_simu_info)
            print(f"prepare success at rank {comm_rank}", flush=True)
            comm.barrier()
            # here we should send result by batch..
            send_num: int = 200
            while len(result) > 0:
                send_res = result[:send_num]
                result = result[send_num:]
                print(f"at rank {comm_rank}, len(send_res) = {len(send_res)}, len(result) = {len(result)}")
                comm.send(send_res, 0)
            comm.send(None, 0)
            continue
        elif worker_mode == TrainingInstruction.DiffODELossReduced:
            result = simu_loss_handle.compute_loss(*forward_simu_info)  # compute simulation loss on multiply CPU cores.
        elif worker_mode == TrainingInstruction.DiffODELossMaximal:
            result = simu_loss_handle.compute_loss_maximal_coordinate(*forward_simu_info)
        else:
            raise ValueError(f"{worker_mode} not supported.")
        # we need to gather the computed data to the main worker.
        # print(f"before gather, comm_rank = {comm_rank}", flush=True)
        comm.barrier()
        comm.gather(result)
        # print(f"after gather, comm_rank = {comm_rank}", flush=True)
        gc.collect()

class SimuLossParallel(Function):  # TODO: Test..
    """
    Here we can scatter and gather with np.float32, for saving memory and time..
    Input: root pos, joint rotate, pd target, contact label, divide index
    Output: the simulation loss

    This function is called at the main worker
    Note: when calling torch.autograd.Function.apply, there is no gradient..
    """
    loss_mode: TrainingInstruction = TrainingInstruction.DiffODELossMaximal

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert comm_rank == 0
        if simu_loss_handle is None:
            raise ValueError("You should call method simu_loss_init before estimate physics loss when training neural network.")

        pos_torch: torch.Tensor = args[0]  # in reduced coordinate, it is root position; in maximal coordinate, it is body position
        rotate_torch: torch.Tensor = args[1]  # in reduced coordinate, it is joint rotation; in maximal coordinate, it is body rotation
        rotate_potential_torch: Optional[torch.Tensor] = args[2]
        tot_pd_target_torch: torch.Tensor = args[3]
        tot_contact_label_torch: torch.Tensor = args[4]
        tot_separate_index_torch: torch.Tensor = args[5]

        pos: np.ndarray = pos_torch.detach().cpu().numpy()
        rotate: np.ndarray = rotate_torch.detach().cpu().numpy()
        if rotate_potential_torch is not None:
            rotate_potential: Optional[np.ndarray] = rotate_potential_torch.detach().cpu().numpy()
        else:
            rotate_potential: Optional[np.ndarray] = None

        tot_pd_target: np.ndarray = tot_pd_target_torch.detach().cpu().numpy()
        tot_contact_label: np.ndarray = tot_contact_label_torch.detach().cpu().numpy()
        tot_separate_index: np.ndarray = tot_separate_index_torch.detach().cpu().numpy()

        # This is 0-1 label. 0 means we can do forward simulation with next frame,
        # 1 means we should not do forward simulation with next frame, because they belong to different motion sequence.
        device: torch.device = pos_torch.device
        width: int = pos.shape[0]
        assert width == rotate.shape[0] == tot_pd_target.shape[0] == tot_contact_label.shape[0] == tot_separate_index.shape[0]
        # we need to divide the input data into several pieces..

        def main_worker_prepare_base(num_divide: int = comm_size):
            divide_piece_: np.ndarray = np.linspace(0, width - 1, num_divide + 1, dtype=np.int32)
            divide_start_: np.ndarray = divide_piece_[:-1]
            divide_end_: np.ndarray = divide_piece_[1:] + 1

            def divide_func_(x_: Optional[np.ndarray], i_: int) -> Optional[np.ndarray]:
                if x_ is None:
                    return None
                else:
                    return x_[divide_start_[i_]: divide_end_[i_]]

            # here we should also scatter the mode..
            divide_list_ = [(
                SimuLossParallel.loss_mode,
                i,  # index
                divide_start_[i].item(),  # start_index
                divide_func_(pos, i),  # position
                divide_func_(rotate, i),
                divide_func_(rotate_potential, i),
                divide_func_(tot_pd_target, i),
                divide_func_(tot_contact_label, i),
                divide_func_(tot_separate_index, i)
            ) for i in range(num_divide)]

            return divide_start_, divide_end_, divide_list_

        divide_start, divide_end, divide_list = main_worker_prepare_base()

        def compute_loss_func(forward_simu_info_):
            # format:
            # index, start index, loss dict, pos grad, rot mat grad, potential rot (quaternion) grad, pd target grad, contact grad
            if worker_mode == TrainingInstruction.DiffODELossReduced:
                single_simu_res_ = simu_loss_handle.compute_loss(*forward_simu_info_)  # The main worker should also compute in Diff-ODE
            elif worker_mode == TrainingInstruction.DiffODELossMaximal:
                single_simu_res_ = simu_loss_handle.compute_loss_maximal_coordinate(*forward_simu_info_)
            else:
                raise ValueError
            return single_simu_res_

        # divide the loss into several piece.
        if comm_size > 1:
            comm.barrier()
            worker_mode, *forward_simu_info = comm.scatter(divide_list, root=0)
            single_simu_res = compute_loss_func(forward_simu_info)
            comm.barrier()
            total_simu_res = comm.gather(single_simu_res)
        else:
            total_simu_res = []
            for node in divide_list:
                worker_mode, *forward_simu_info = node
                single_simu_res = compute_loss_func(forward_simu_info)
                total_simu_res.append(single_simu_res)

        # set the gradient to original parameter.
        if True:
            grad_pos_param: torch.Tensor = torch.zeros(pos.shape, dtype=torch.float32)
            grad_rotate_param: torch.Tensor = torch.zeros(rotate.shape, dtype=torch.float32)
            if rotate_potential_torch is not None:
                grad_rotate_potential: Optional[torch.Tensor] = torch.zeros(rotate_potential_torch.shape, dtype=torch.float32)
            else:
                grad_rotate_potential: Optional[torch.Tensor] = None

            grad_pd_target_param: torch.Tensor = torch.zeros(tot_pd_target.shape, dtype=torch.float32)
            # maybe the contact label is also differentiable..
            if tot_contact_label_torch.requires_grad:
                grad_contact_label: Optional[torch.Tensor] = torch.zeros(tot_contact_label.shape, dtype=torch.float32)
            else:
                grad_contact_label: Optional[torch.Tensor] = None

            avg_loss: Dict[str, Union[float, torch.Tensor]] = {key: 0.0 for key in total_simu_res[0][2].keys()}
            for node in total_simu_res:
                index: int = node[0]
                for key in avg_loss.keys():  # compute the average loss, for record.
                    avg_loss[key] += node[2][key]
                piece = slice(divide_start[index], divide_end[index])
                grad_pos_param[piece] += torch.from_numpy(node[3])
                grad_rotate_param[piece] += torch.from_numpy(node[4])
                if rotate_potential_torch is not None:
                    grad_rotate_potential[piece] += torch.from_numpy(node[5])
                grad_pd_target_param[piece] += torch.from_numpy(node[6])
                if grad_contact_label is not None:
                    grad_contact_label[piece] += torch.from_numpy(node[7])

        # save the gradient for backward. Here we should convert to the input device (e.g. cpu or cuda)
        ctx.grad_pos_param = grad_pos_param.to(device)
        ctx.grad_rotate_param = grad_rotate_param.to(device)
        ctx.grad_rotate_potential = grad_rotate_potential.to(device) if grad_rotate_potential is not None else None
        ctx.grad_pd_target_param = grad_pd_target_param.to(device)
        ctx.grad_contact_label = grad_contact_label.to(device) if grad_contact_label is not None else None

        return torch.as_tensor(avg_loss["simu_total_loss"], device=device), avg_loss  # The return value is a single tensor..

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], None]:
        # when the forward output is a single value, we need not to consider the input gradient
        grad_pos: torch.Tensor = ctx.grad_pos_param
        grad_rotate_param: torch.Tensor = ctx.grad_rotate_param
        grad_rotate_potential: torch.Tensor = ctx.grad_rotate_potential
        grad_pd_target: torch.Tensor = ctx.grad_pd_target_param
        grad_contact_label: Optional[torch.Tensor] = ctx.grad_contact_label

        # clear the saved result here
        ctx.grad_root_pos_param = None
        ctx.grad_rotate_param = None
        ctx.grad_rotate_potential = None
        ctx.grad_pd_target_param = None
        ctx.grad_contact_label = None

        return grad_pos, grad_rotate_param, grad_rotate_potential, grad_pd_target, grad_contact_label, None

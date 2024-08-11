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

import os
from sympy import arg

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import torch
torch.set_num_threads(1)

from argparse import Namespace
import itertools
import copy
from datetime import datetime
from enum import IntEnum
from mpi4py import MPI
import numpy as np
import platform
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW, LBFGS
from torch.optim.lr_scheduler import StepLR
from typing import Optional, Union, List, Tuple, Dict, Any

from MotionUtils import six_dim_mat_to_quat_fast

from VclSimuBackend.Common.SmoothOperator import smooth_operator, GaussianBase
from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.Samcon.OptimalGait.OptimizeEachFrameBase import ContactPlanMode, draw_foot_contact_plan, only_use_phys_loss
from VclSimuBackend.Samcon.OptimalGait.OptimizeEachFrame import OptimizeEachFrame, parse_args
from VclSimuBackend.Samcon.OptimalGait.ContactPlan import ContactPlan, ContactLabelState
from VclSimuBackend.Samcon.OptimalGait.ContactWithKinematic import ContactLabelExtractor
from VclSimuBackend.Samcon.SamconWorkerBase import WorkerInfo
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo
from VclSimuBackend.DiffODE import DiffQuat
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter, TargetPose
from VclSimuBackend.Render.Renderer import RenderWorld


fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")
comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
is_windows: bool = "Windows" in platform.platform()


class OptimizeWorkerMode(IntEnum):
    Optimize = 0
    ContactPlan = 1
    RecieveContactPlan = 2
    Evaluate = 3
    Stop = 4


class OptimizeParallel(OptimizeEachFrame):
    """
    Here we should run each process at a single CPU core.
    """
    def __init__(self, args: Namespace, scene: Optional[ODEScene] = None) -> None:
        super().__init__(args, scene)
        self.worker_info = WorkerInfo()
        self.ret = None

        if comm_rank == 0:
            self.init_main_parameter(args.index_t, args.index_t + args.width)

        if comm_size == 1 and is_windows:
            self.render: Optional[RenderWorld] = RenderWorld(self.scene)
            self.render.start()

        # self.handle_contact_kinematic(0, self.args.index_t, 0, self.init_root_pos.numpy(), self.init_joint_vec6d.numpy())
        # self.compute_contact_info_by_mess()
        comm.barrier()

    def save_result_parallel(self, save_filename: Optional[str] = None):
        """
        for the final result, we need to compute the contact force and contact torque in each frame...
        Maybe the contact force is noisy....for simple visualization, maybe we can merge the contact force into a same body..
        1. remove redundant force of x, y, z component. (a simple method is compute the total force,
        and divide them by the length of contact point..)
        2.
        """
        # we should also save the new contact here..
        save_result: Dict[str, Any] = {
            "root_pos_param": self.main_root_pos_param.detach(),
            "joint_vec6d_param": self.main_joint_vec6d_param.detach(),
            "pd_target_param": self.main_pd_target_param.detach() if self.main_pd_target_param is not None else None,
            "optim_state": self.optim.state_dict() if self.optim is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "contact_mess": self.contact_mess,
            "epoch": self.epoch
        }
        if self.main_eval_force_list is not None:
            save_result["extract_contact_force"] = self.extract_contact_force

        if self.main_body_velo_param is not None:
            save_result["body_velo_param"] = self.main_body_velo_param.detach()
        if self.main_body_angvel_param is not None:
            save_result["body_angvel_param"] = self.main_body_angvel_param.detach()

        if self.main_root_vel_param is not None:
            save_result["root_vel_param"] = self.main_root_vel_param.detach()
        if self.main_angvel_param is not None:
            save_result["angvel_param"] = self.main_angvel_param.detach()

        # Actually, these 3 parameters are not used.
        if self.main_contact_label_param is not None:
            save_result["contact_label_param"] = self.main_contact_label_param.detach()
        if self.main_contact_force_param is not None:
            save_result["contact_force_param"] = self.main_contact_force_param.detach()
        if self.main_contact_torque_param is not None:
            save_result["contact_torque_param"] = self.contact_torque_param.detach()

        if save_filename is not None and len(save_filename) > 0:
            torch.save(save_result, save_filename)
        # here we can also save the kinematics state as bvh file, for visualize..
        # bvh_fname: str = save_filename + ".bvh"
        # motion: MotionData = self.diff_motion.export_to_motion_data()
        # motion: MotionData = self.to_bvh.insert_end_site(motion)
        # BVHLoader.save(motion, bvh_fname)
        return save_result

    def load_result_parallel(self, save_result: Union[str, Dict[str, Any]]):
        if isinstance(save_result, str):
            if not os.path.exists(save_result):
                print(f"result not exist. ignore.", flush=True)
                return
            else:
                print(f"Load the result from {save_result}", flush=True)
                saved_result_: Dict[str, Any] = torch.load(save_result)
        else:
            saved_result_ = save_result

        if self.main_root_pos_param is not None:
            self.main_root_pos_param.data = saved_result_["root_pos_param"]
        else:
            self.main_root_pos_param = nn.Parameter(self.main_root_pos_param)

        if self.main_joint_vec6d_param is not None:
            self.main_joint_vec6d_param.data = saved_result_["joint_vec6d_param"]
        else:
            self.main_joint_vec6d_param = nn.Parameter(saved_result_["joint_vec6d_param"])

        if "pd_target_param" in saved_result_ and saved_result_["pd_target_param"] is not None:
            if self.main_pd_target_param is not None:
                self.main_pd_target_param.data = saved_result_["pd_target_param"]
            else:
                self.main_pd_target_param = nn.Parameter(saved_result_["pd_target_param"])

        if self.args.use_cio:
            if "contact_label_param" in saved_result_:
                self.main_contact_label_param.data = saved_result_["contact_label_param"]
            if "contact_force_param" in saved_result_:
                self.main_contact_force_param.data = saved_result_["contact_force_param"]
            if "contact_torque_param" in saved_result_:
                self.main_contact_torque_param.data = saved_result_["contact_torque_param"]

        if "body_velo_param" in saved_result_:
            if self.main_body_velo_param is not None:
                self.main_body_velo_param.data = saved_result_["body_velo_param"]
            else:
                pass

        if "body_angvel_param" in saved_result_:
            if self.main_body_angvel_param is not None:
                self.main_body_angvel_param.data = saved_result_["body_angvel_param"]
            else:
                self.main_body_angvel_param = nn.Parameter(saved_result_["body_angvel_param"])

        if "root_vel_param" in saved_result_:
            self.main_root_vel_param.data = saved_result_["root_vel_param"]
        if "angvel_param" in saved_result_:
            self.main_angvel_param.data = saved_result_["angvel_param"]

        if self.optim is not None:
            self.optim.load_state_dict(saved_result_["optim_state"])
        if self.scheduler is not None and saved_result_["scheduler"] is not None:
            self.scheduler.load_state_dict(saved_result_["scheduler"])
        self.epoch: int = saved_result_["epoch"]
        if "contact_mess" in saved_result_:
            self.contact_mess: List[List[int]] = saved_result_["contact_mess"]
            if comm_rank == 0:
                print(f"load contact mess from dump file.", flush=True)

    def init_main_kinematic_optim(self):
        param_list: List[nn.Parameter] = [] # self.main_root_pos_param, self.main_joint_vec6d_param]
        if self.main_root_pos_param is not None:
            param_list.append(self.main_root_pos_param)
        if self.main_joint_vec6d_param is not None:
            param_list.append(self.main_joint_vec6d_param)

        self.main_pd_target_param: Optional[nn.Parameter] = None
        self.main_body_velo_param: Optional[nn.Parameter] = None
        self.main_body_angvel_param = None
        if self.args.use_lbfgs:
            self.optim = LBFGS(param_list, lr=self.args.lr)
        else:
            self.optim = AdamW(param_list, lr=self.args.lr)
        # self.scheduler: Optional[StepLR] = None
        self.scheduler: StepLR = StepLR(self.optim, self.args.lr_decay_epoch, self.args.lr_decay_ratio)

    def init_main_kinematic_optim(self):
        # if self.main_root_pos_param is None:
        #    self.init_main_parameter()
        param_list: List[nn.Parameter] = [self.main_root_pos_param, self.main_joint_vec6d_param]
        self.main_pd_target_param = None
        self.main_body_velo_param = None
        self.main_body_angvel_param = None
        if self.args.use_lbfgs:
            self.optim = LBFGS(param_list, lr=self.args.lr)
        else:
            self.optim = AdamW(param_list, lr=self.args.lr)
        self.scheduler: Optional[StepLR] = None

    def init_main_optim(self):
        # Note: we should recompute the velocity and angular velocity here..
        self.init_main_parameter(self.args.index_t, self.args.index_t + self.args.width)
        param_list: List[nn.Parameter] = [
            self.main_root_pos_param, self.main_joint_vec6d_param, self.main_pd_target_param
        ]
        if self.main_contact_label_param is not None:
            param_list.extend([
                self.main_contact_label_param, self.main_contact_torque_param, self.main_contact_torque_param])
        if self.main_body_velo_param is not None:
            param_list.append(self.main_body_velo_param)
        if self.main_body_angvel_param is not None:
            param_list.append(self.main_body_angvel_param)

        # Note: velocity in reduced coordinate is not used. ignore..
        if self.main_root_vel_param is not None:
            param_list.append(self.main_root_vel_param)
        if self.main_angvel_param is not None:
            param_list.append(self.main_angvel_param)

        # optimize by cio mode..
        if self.main_contact_label_param is not None:
            param_list.append(self.main_contact_label_param)
        if self.main_contact_force_param is not None:
            param_list.append(self.main_contact_force_param)
        if self.main_contact_torque_param is not None:
            param_list.append(self.main_contact_torque_param)

        # build optimizer
        if self.args.use_lbfgs:
            self.optim = LBFGS(param_list, lr=self.args.lr)
        else:
            self.optim = AdamW(param_list, lr=self.args.lr)
        self.scheduler: StepLR = StepLR(self.optim, self.args.lr_decay_epoch, self.args.lr_decay_ratio)

    def init_main_kinematic_parameter(self, start: int, end: int) -> torch.Tensor:
        """
        return joint rotation matrix
        """
        piece = slice(start, end)
        if self.main_root_pos_param is None:
            self.main_root_pos_param: nn.Parameter = nn.Parameter(self.init_root_pos[piece].clone())  # (width, 3)
        else:
            print(f"main root pos param exists. ignore", flush=True)

        if self.main_joint_vec6d_param is None:  # (width, num joint, 3, 2)
            self.main_joint_vec6d_param: nn.Parameter = nn.Parameter(self.init_joint_vec6d[piece].clone())
            joint_rotmat: torch.Tensor = DiffQuat.vec6d_to_matrix(self.main_joint_vec6d_param.detach())
        else:
            joint_rotmat: torch.Tensor = DiffQuat.vec6d_to_matrix(self.main_joint_vec6d_param.detach())
            self.main_joint_vec6d_param: nn.Parameter = nn.Parameter(joint_rotmat[..., :2].contiguous())
            print(f"joint rotmat exists. ignore", flush=True)

        return joint_rotmat

    def init_main_parameter(self, start: int, end: int):
        """
        initialize the optimization parameter for main worker
        """
        piece = slice(start, end)
        joint_rotmat: torch.Tensor = self.init_main_kinematic_parameter(start, end)

        if self.main_pd_target_param is None:  # (width, num joint, 3, 2)
            self.main_pd_target_param: nn.Parameter = nn.Parameter(self.init_pd_target_vec6d[piece].clone())
        else:
            pd_target_mat: torch.Tensor = DiffQuat.vec6d_to_matrix(self.main_pd_target_param.detach())
            self.main_pd_target_param: nn.Parameter = nn.Parameter(pd_target_mat[..., :2].contiguous())
            print(f"init pd target")

        if self.args.use_cio:  # (width, num body)
            self.main_contact_label_param: nn.Parameter = nn.Parameter(torch.from_numpy(self.contact_label).clone())

        if self.args.also_optimize_velo:
            if self.args.optimize_velo_in_maximal:
                with torch.no_grad():
                    # Note: here we should do forward kinematics again,
                    # to gather the newest body velocity and angular velocity
                    joint_quat: torch.Tensor = DiffQuat.quat_from_matrix(joint_rotmat) # (width, num joint, 4)

                    # forward kinematics to compute joint global orientation and position
                    self.diff_motion._root_translation = self.main_root_pos_param.detach()
                    self.diff_motion._joint_rotation = joint_quat  # (width, num joint, 4)
                    self.diff_motion._num_frames = self.main_root_pos_param.shape[0]
                    self.diff_motion.recompute_joint_global_info()

                    # here we should compute body position by joint position..
                    kine_body_pos, kine_body_quat = self.mocap_import.import_mocap_base_batch(self.diff_motion)  # (width, num body, 3), (width, num body, 4)
                    # here we should compute linear and angular velocity
                    kine_body_vel: torch.Tensor = torch.diff(kine_body_pos, dim=0) * self.args.simulation_fps # (width - 1, num body, 3)
                    kine_body_ang: torch.Tensor = self.diff_motion.compute_angvel_frag(kine_body_quat, self.args.simulation_fps)  # (width - 1, num body, 3)
                    kine_body_vel: torch.Tensor = torch.cat([kine_body_vel[0, None], kine_body_vel], dim=0)  # (width, num body, 3)
                    kine_body_ang: torch.Tensor = torch.cat([kine_body_ang[0, None], kine_body_ang], dim=0)  # (width, num body, 3)
                    self.diff_motion.clear()

                self.main_body_velo_param: nn.Parameter = nn.Parameter(kine_body_vel.detach())
                self.main_body_angvel_param: nn.Parameter = nn.Parameter(kine_body_ang.detach())
            else:
                raise NotImplementedError
                # self.main_root_vel_param: nn.Parameter = nn.Parameter(self.init_root_velo.clone())
                # self.main_angvel_param: nn.Parameter = nn.Parameter(self.init_joint_angvel.clone())

    def main_worker_prepare_base(self, mode: OptimizeWorkerMode, num_divide: int = comm_size):
        args: Namespace = self.args
        width: int = args.width
        # note: here we should compute physics loss at multiple frame
        # we should avoid compute another loss at overlapping frames in the middle
        divide_piece: np.ndarray = np.linspace(0, width - 1, num_divide + 1, dtype=np.int32)
        divide_start: np.ndarray = divide_piece[:-1]
        divide_end: np.ndarray = divide_piece[1:] + args.phys_loss_num_forward

        # body kinematic pose and pd control target pose
        root_pos_numpy: np.ndarray = self.main_root_pos_param.detach().numpy()  # (num frame, 3)
        root_joint_numpy: np.ndarray = self.main_joint_vec6d_param.detach().numpy()  # (num frame, num joint, 3, 2)
        pd_target_numpy: np.ndarray = self.main_pd_target_param.detach().numpy() if self.main_pd_target_param is not None else None  # (num frame, num joint, 3, 2)

        # actually, this part is not used.
        root_vel_numpy: Optional[np.ndarray] = self.main_root_vel_param.detach().numpy() if self.main_root_vel_param is not None else None
        angvel_numpy: Optional[np.ndarray] = self.main_angvel_param.detach().numpy() if self.main_angvel_param is not None else None

        body_vel_numpy: Optional[np.ndarray] = self.main_body_velo_param.detach().numpy() if self.main_body_velo_param is not None else None
        body_angvel_numpy: Optional[np.ndarray] = self.main_body_angvel_param.detach().numpy() if self.main_body_angvel_param is not None else None

        if self.main_contact_label_param is not None:
            contact_param_numpy: Optional[np.ndarray] = self.main_contact_label_param.detach().numpy()
            contact_force_numpy: Optional[np.ndarray] = self.main_contact_force_param.detach().numpy()
            contact_torque_numpy: Optional[np.ndarray] = self.main_contact_torque_param.detach().numpy()
        else:
            contact_param_numpy: Optional[np.ndarray] = None
            contact_force_numpy: Optional[np.ndarray] = None
            contact_torque_numpy: Optional[np.ndarray] = None

        def divide_func(x_: Optional[np.ndarray], i: int) -> Optional[np.ndarray]:
            if x_ is None:
                return None
            else:
                return x_[divide_start[i]: divide_end[i]]

        # here we should also scatter the mode..
        divide_list = [(
            mode,
            i,
            self.args.index_t + divide_start[i].item(),
            self.epoch,
            divide_func(root_pos_numpy, i),
            divide_func(root_joint_numpy, i),
            divide_func(pd_target_numpy, i),
            divide_func(body_vel_numpy, i),
            divide_func(body_angvel_numpy, i),
            divide_func(root_vel_numpy, i),  # it may be None
            divide_func(angvel_numpy, i),  # it may be None
            divide_func(contact_param_numpy, i),
            divide_func(contact_force_numpy, i),
            divide_func(contact_torque_numpy, i)
        ) for i in range(num_divide)]

        return divide_start, divide_end, divide_list

    def main_worker_contact_plan(self):
        forward_count, self.args.phys_loss_num_forward = self.args.phys_loss_num_forward, 1
        self.init_main_parameter(self.args.index_t, self.args.index_t + self.args.width)
        # if self.main_pd_target_param is None:
        #    self.main_pd_target_param = nn.Parameter(self.init_pd_target_vec6d.clone())
        old_contact_mess: List[List[int]] = copy.deepcopy(self.contact_mess)
        # num_divide: int = max(1, comm_size - 1)
        divide_start, divide_end, divide_list = self.main_worker_prepare_base(OptimizeWorkerMode.ContactPlan)
        if comm_size > 1:
            comm.barrier()
            worker_mode, *forward_simu_info = comm.scatter(divide_list, root=0)
            # maybe we can modify the contact label kinematicly..
            # here we use different type of contact plan. simple / mcmc
            single_contact_res = self.handle_contact_planninng(*forward_simu_info)
            print(f"After handle contact info in main worker.", flush=True)
            total_contact_res: List[Tuple[int, int, List[List[int]]]] = comm.gather(single_contact_res)
        else:
            total_contact_res = []
            for node_idx, node in enumerate(divide_list):
                worker_mode, *forward_simu_info = node
                single_contact_res = self.handle_contact_planninng(*forward_simu_info)
                total_contact_res.append(single_contact_res)

        # we can log the difference between original contact label and new contact label.
        # here we need to merge the contact label, and scatter to all of child workers.
        # Note: we need also sync the target pose..
        for node in total_contact_res:
            index, start_index, ret_mess = node
            #print(
            #    f"index = {index}, start_index = {start_index}, "
            #    f"divide_start = {divide_start[index]}, divide_end = {divide_end[index]}", flush=True
            #)
            self.contact_mess[divide_start[index]: divide_end[index] - 1] = ret_mess

        # smooth the contact sequence.
        self.contact_mess: List[List[int]] = ContactLabelExtractor.smooth_contact_mess(self.contact_mess)
        comm.bcast((
            self.contact_mess,
            self.main_root_pos_param.detach().numpy(),
            self.main_joint_vec6d_param.detach().numpy()))
        # here we can output the old and new contact..

        print(self.character.get_body_name_list(), flush=True)
        for frame in range(self.args.width):
            print(f"index = {frame}, old = {old_contact_mess[frame]}, new = {self.contact_mess[frame]}")

        # for the main worker, we need to recompute the planning..
        self.handle_contact_optimize_info(0, 0, 0,
            self.main_root_pos_param.detach().numpy(),
            self.main_joint_vec6d_param.detach().numpy(), None
        )

        # we need to recompute the DiffContactInfo here..
        self.compute_contact_info_by_mess()  # we need not to gather any data here..
        comm.barrier()
        self.args.phys_loss_num_forward = forward_count

        print(f"After handle the contact info", flush=True)

    def main_worker_epoch(self, is_evaluate: bool = False):
        """
        Here we should scatter the parameter to child workers,
        then compute loss and gradient.
        """
        divide_start, divide_end, divide_list = self.main_worker_prepare_base(OptimizeWorkerMode.Optimize)
        # here we should write single thread version and multi thread version
        if comm_size > 1:
            comm.barrier()
            worker_mode, *forward_simu_info = comm.scatter(divide_list, root=0)
            single_simu_res = self.handle_forward_simu_info(*forward_simu_info, is_evaluate=is_evaluate)
            # The main worker should also compute in Diff-ODE
            total_simu_res = comm.gather(single_simu_res)
        else:
            total_simu_res = []
            for node in divide_list:
                worker_mode, *forward_simu_info = node
                result = self.handle_forward_simu_info(*forward_simu_info, is_evaluate=is_evaluate)
                total_simu_res.append(result)
            # here we can save the total simu result for debug..
            # self.tmp_total_simu_res = total_simu_res

        # set the gradient to original parameter.
        # with torch.no_grad():
        if True:
            if self.main_root_pos_param is not None and not is_evaluate:
                grad_root_pos_param: torch.Tensor = torch.zeros_like(self.main_root_pos_param)
            else:
                grad_root_pos_param: Optional[torch.Tensor] = None

            if self.main_joint_vec6d_param is not None and not is_evaluate:
                grad_joint_param: torch.Tensor = torch.zeros_like(self.main_joint_vec6d_param)
            else:
                grad_joint_param: Optional[torch.Tensor] = None

            if self.main_pd_target_param is not None and not is_evaluate:
                grad_pd_target_param: torch.Tensor = torch.zeros_like(self.main_pd_target_param)
            else:
                grad_pd_target_param: Optional[torch.Tensor] = None

            if self.main_body_velo_param is not None and not is_evaluate:
                grad_body_velo: torch.Tensor = torch.zeros_like(self.main_body_velo_param)
                grad_body_angvel: torch.Tensor = torch.zeros_like(self.main_body_angvel_param)
            else:
                grad_body_velo: Optional[torch.Tensor] = None
                grad_body_angvel: Optional[torch.Tensor] = None

            if self.main_contact_label_param is not None and not is_evaluate:
                grad_contact_label_param: torch.Tensor = torch.zeros_like(self.main_contact_label_param)
                grad_contact_force_param: torch.Tensor = torch.zeros_like(self.main_contact_force_param)
                grad_contact_torque_param: torch.Tensor = torch.zeros_like(self.main_contact_torque_param)
            else:
                grad_contact_label_param: Optional[torch.Tensor] = None
                grad_contact_force_param: Optional[torch.Tensor] = None
                grad_contact_torque_param: Optional[torch.Tensor] = None

            avg_loss: Dict[str, Union[float, torch.Tensor]] = {
                key: 0.0 for key in total_simu_res[0]["loss_dict"].keys()}
            if is_evaluate:
                self.main_eval_force_list = [None for _ in range(self.main_root_pos_param.shape[0])]
            else:
                self.main_eval_force_list = None

            for node in total_simu_res:
                index: int = node["index"]
                for key in avg_loss.keys():  # compute the average loss, for record.
                    avg_loss[key] += node["loss_dict"][key]
                piece = slice(divide_start[index], divide_end[index])
                if not is_evaluate:
                    if grad_root_pos_param is not None and node["root_pos_param_grad"] is not None:
                        grad_root_pos_param[piece] += torch.from_numpy(node["root_pos_param_grad"])
                    if grad_joint_param is not None and node["joint_vec6d_param_grad"] is not None:
                        grad_joint_param[piece] += torch.from_numpy(node["joint_vec6d_param_grad"])
                    if grad_pd_target_param is not None and node["pd_target_param_grad"] is not None:
                        grad_pd_target_param[piece] += torch.from_numpy(node["pd_target_param_grad"])
                    # ah, we need to compute velocity gradient here..
                    if grad_body_velo is not None and node["body_velo_param_grad"] is not None:
                        grad_body_velo[piece] += torch.from_numpy(node["body_velo_param_grad"])
                    if grad_body_angvel is not None and node["body_angvel_param_grad"] is not None:
                        grad_body_angvel[piece] += torch.from_numpy(node["body_angvel_param_grad"])

                    if self.main_contact_label_param is not None and node["contact_label_param_grad"] is not None:
                        grad_contact_label_param[piece] += torch.from_numpy(node["contact_label_param_grad"])
                    if self.main_contact_force_param is not None and node["contact_force_param_grad"] is not None:
                        grad_contact_force_param[piece] += torch.from_numpy(node["contact_force_param_grad"])
                    if self.main_contact_torque_param is not None and node["contact_torque_param_grad"] is not None:
                        grad_contact_torque_param[piece] += torch.from_numpy(node["contact_torque_param_grad"])

                # handld the evaluated contact force..
                if node["eval_force_list"] is not None:
                    self.main_eval_force_list[piece] = node["eval_force_list"]
                    # print(self.main_root_pos_param.shape, "test shape", len(self.main_eval_force_list), len(node["eval_force_list"]), piece)
                    self.extract_contact_force = self.extract_contact_message_for_save(self.main_eval_force_list)
                else:  # we need to do nothing here.
                    self.main_eval_force_list = None

            # clear the optimization param
            if self.main_root_pos_param is not None and self.main_root_pos_param.requires_grad:
                self.main_root_pos_param.grad = grad_root_pos_param  # The gradient is computed in sub workers
            if self.main_joint_vec6d_param is not None and self.main_joint_vec6d_param.requires_grad:
                self.main_joint_vec6d_param.grad = grad_joint_param
            if self.main_pd_target_param is not None and self.main_pd_target_param.requires_grad:
                self.main_pd_target_param.grad = grad_pd_target_param

            if self.main_body_velo_param is not None and self.main_body_velo_param.requires_grad:
                self.main_body_velo_param.grad = grad_body_velo
            if self.main_body_angvel_param is not None and self.main_body_angvel_param.requires_grad:
                self.main_body_angvel_param.grad = grad_body_angvel

            if self.main_contact_label_param is not None and self.main_contact_label_param.requires_grad:
                self.main_contact_label_param.grad = grad_contact_label_param
            if self.main_contact_force_param is not None and self.main_contact_force_param.requires_grad:
                self.main_contact_force_param.grad = grad_contact_force_param
            if self.main_contact_torque_param is not None and self.main_contact_torque_param.requires_grad:
                self.main_contact_torque_param.grad = grad_contact_torque_param

        # As the gradient is computed, do gradient descent in main worker
        return avg_loss

    def main_worker_optimize(self, epoch: int):
        args: Namespace = self.args
        ret: Dict[str, Union[float, torch.Tensor]] = self.main_worker_epoch(is_evaluate=False)
        tot_loss_item: float = ret["tot_loss"]

        if args.print_log_info:
            self.print_loss(epoch, ret, False, True)  # here we should record info with summary writer, not print directly.

        if tot_loss_item < self.best_loss:
            # here we can save the best param in a dict..
            self.best_param = self.save_result_parallel()
            self.best_loss = tot_loss_item

        if args.pos_grad_clip is not None:  # clip the gradient.
            if self.main_root_pos_param.grad is not None:
                clip_grad_norm_(self.main_root_pos_param, args.pos_grad_clip)

        if args.rot_grad_clip is not None:
            if self.main_joint_vec6d_param is not None and self.main_joint_vec6d_param.grad is not None:
                clip_grad_norm_(self.main_joint_vec6d_param, args.rot_grad_clip)
            if self.main_pd_target_param is not None and self.main_pd_target_param.grad is not None:
                clip_grad_norm_(self.main_pd_target_param, args.rot_grad_clip)

        # here we should also clip the contact force and torque..
        if self.main_contact_label_param is not None and self.main_contact_label_param.grad is not None and args.cio_contact_label_clip is not None:
            clip_grad_norm_(self.main_contact_label_param, args.cio_contact_label_clip)

        if self.main_contact_force_param is not None and self.main_contact_force_param.grad is not None and args.cio_contact_force_clip is not None:
            clip_grad_norm_(self.main_contact_force_param, args.cio_contact_force_clip)
        if self.main_contact_torque_param is not None and self.main_contact_torque_param.grad is not None and args.cio_contact_force_clip is not None:
            clip_grad_norm_(self.main_contact_torque_param, args.cio_contact_force_clip)

        return ret

    def get_simu_target_set(
        self,
        root_pos: Union[torch.Tensor, np.ndarray, None] = None,
        joint_vec6d: Union[torch.Tensor, np.ndarray, None] = None
    ) -> SetTargetToCharacter:
        if root_pos is None:
            root_pos: torch.Tensor = self.main_root_pos_param
        if joint_vec6d is None:
            joint_vec6d: torch.Tensor = self.main_joint_vec6d_param
        if isinstance(root_pos, torch.Tensor):
            root_pos: np.ndarray = root_pos.detach().numpy()
        if isinstance(root_pos, np.ndarray):
            root_pos: np.ndarray = np.ascontiguousarray(root_pos, dtype=np.float64)

        if isinstance(joint_vec6d, torch.Tensor):
            joint_vec6d: np.ndarray = joint_vec6d.detach().numpy()
        if isinstance(joint_vec6d, np.ndarray):
            joint_vec6d: np.ndarray = np.ascontiguousarray(joint_vec6d, dtype=np.float64)
        joint_quat: np.ndarray = MathHelper.vec6d_to_quat(joint_vec6d)
        # motion_hierarchy = self.to_bvh.forward_kinematics(root_pos, joint_quat[:, 0], joint_quat[:, 1:])
        # for pd target, we should use to_bvh. for other motion, we should use motiondata..
        num_frame, num_joint = joint_vec6d.shape[:2]
        motion_hierarchy: MotionData = self.motion.get_hierarchy(True)
        motion_hierarchy._joint_translation = np.zeros((num_frame, num_joint, 3), dtype=np.float64)
        motion_hierarchy._joint_translation[:, 0, :] = root_pos[:]
        motion_hierarchy._joint_rotation = joint_quat
        motion_hierarchy._joint_position = np.zeros_like(motion_hierarchy.joint_translation)
        motion_hierarchy._joint_orientation = np.zeros_like(motion_hierarchy.joint_rotation)
        motion_hierarchy._num_frames = num_frame
        motion_hierarchy.recompute_joint_global_info()
        motion_hierarchy_tar: TargetPose = BVHToTargetBase(
            motion_hierarchy, self.scene.sim_fps, self.character).init_target()
        motion_tar_set: SetTargetToCharacter = SetTargetToCharacter(self.character, motion_hierarchy_tar)
        return motion_tar_set

    def output_result(self, use_inf_iter: bool = True):
        # Here we can start a render in main worker, for visualize the final result..
        num_frame, num_joint = self.main_joint_vec6d_param.shape[:2]
        motion_tar_set = self.get_simu_target_set()
        for frame_index in range(0, num_frame):
            motion_tar_set.set_character_byframe(frame_index)
            self.to_bvh.append_no_root_to_buffer()
        self.to_bvh.to_file(os.path.join(self.args.output_fdir, "OptimizeEachFrame.bvh"), False)
        if self.gt_mocap_cp is not None:
            out_sub_mocap = self.gt_mocap_cp.sub_sequence(
                self.args.index_t, self.args.index_t + self.args.width, copy=False)
            BVHLoader.save(out_sub_mocap, os.path.join(self.args.output_fdir, "GTMotion.bvh"))

        if is_windows:
            # we can render the contact label here..
            # convert the array to continuous ndarray..
            continuous_contact: np.ndarray = self.convert_contact_mess_to_ndarray()
            # draw_foot_contact_plan(continuous_contact, self.contact_label)
            if self.render is None:
                self.render = RenderWorld(self.scene)
                self.render.start()
            if self.ref_character is not None:
                self.ref_character.set_render_color(np.array([1.0, 0.0, 0.0]))
            # for _ in range(1):
            if use_inf_iter:
                for_range = itertools.count()
            else:
                for_range = range(1)

            for _ in for_range:
                for frame_index in range(0, num_frame):
                    motion_tar_set.set_character_byframe(frame_index)
                    if self.gt_tar_set is not None and False:
                        self.gt_tar_set.set_character_byframe(self.args.index_t + frame_index)
                    if self.ref_gt_tarset is not None:
                        self.ref_gt_tarset(self.args.index_t + frame_index)
                    time.sleep(0.05)
        else:
            print(f"Linux doesn't support rendering")

    def render_result(self):
        self.load_result_parallel(self.args.checkpoint_fname)
        self.output_result()

    def zero_dynamic_grad(self):
        if self.main_pd_target_param is not None and self.main_pd_target_param.grad is not None:
            self.main_pd_target_param.grad.zero_()
        if self.main_body_velo_param is not None and self.main_body_velo_param.grad is not None:
            self.main_body_velo_param.grad.zero_()
        if self.main_body_angvel_param is not None and self.main_body_angvel_param.grad is not None:
            self.main_body_angvel_param.grad.zero_()

    def main_opt_wrapper(self):
        args: Namespace = self.args
        pbar: tqdm = tqdm(total=args.max_epoch)
        pbar.update(self.epoch)

        self.ret: Optional[Dict[str, Any]] = None

        def closure():
            self.optim.zero_grad(True)
            self.ret = self.main_worker_optimize(self.epoch)
            if self.epoch < args.optimize_root_epoch:
                self.zero_dynamic_grad()
                if self.main_joint_vec6d_param is not None and self.main_joint_vec6d_param.grad is not None:
                    self.main_joint_vec6d_param.grad.zero_()

            elif self.epoch < args.phys_loss_epoch:
                self.zero_dynamic_grad()

            return torch.as_tensor(self.ret["tot_loss"])

        self.init_main_kinematic_optim()
        while self.epoch < args.max_epoch:
            start_time = time.time()
            # self.output_result(False)
            if self.epoch == args.phys_loss_epoch:  # initialize phys optimization
                # here we can scatter new contact plan by the new kinematic pose..
                # I think use kinematic position and velocity simply is OK...
                if False: # contact plan only use kinematic info.
                    scatter_info = [(
                        OptimizeWorkerMode.RecieveContactPlan,
                        self.main_root_pos_param.detach().numpy(),
                        self.main_joint_vec6d_param.detach().numpy()
                    ) for _ in range(comm_size)]
                    comm.barrier()
                    worker_mode, *simu_info = comm.scatter(scatter_info)
                    self.handle_contact_label_by_body_height(*simu_info)
                    comm.barrier()
                if args.use_contact_plan:
                    pbar.close()
                    self.main_worker_contact_plan()
                    pbar: tqdm = tqdm(total=args.max_epoch)
                    pbar.update(self.epoch)
                # else:  # we should also print contact label to edit here..
                for frame, contact in enumerate(self.contact_mess):
                    print(f"{frame}:{contact}", flush=True)
                # here we can visualize the contact planning result.
                # self.draw_contact_mess()

                self.init_main_optim()

            if args.use_lbfgs:
                self.optim.step(closure)
            else:
                closure()
                self.optim.step()  # As gradient has been saved at the parameter, we can optimize these parameters.

            # if self.epoch >= args.phys_loss_epoch:
            if self.scheduler is not None:
                self.scheduler.step()
            message = f"time: {time.time() - start_time:.2f}, loss: {self.ret['tot_loss']:.6f}"
            pbar.set_description(message)
            pbar.update()
            if self.epoch % args.save_num_epoch == 0:
                save_fname: str = os.path.join(args.output_fdir, f"OptimizeEachFrame.ckpt{self.epoch}")
                self.save_result_parallel(save_fname)

            self.epoch += 1

        # print total time usage, and final loss.
        pbar.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def main_worker(self):
        args: Namespace = self.args
        self.load_result_parallel(args.checkpoint_fname)
        # here we should optimize the contact sequence..for contact optimize:
        # 1. search the best contact label in each worker
        # 2. gather the new contact label in main worker
        # 3. scatter the contact label from main worker to other workers
        # pipeline: contact planning, optimization, contact planning again, optimzation again

        total_start_time = datetime.now()
        # max_epoch: int = args.max_epoch
        # max_proc = 5
        # if args.use_contact_plan:
        #     for proc_index in range(max_proc):
        #         self.main_worker_contact_plan()
        #         # if proc_index == max_proc - 1:
        #         #    self.args.max_epoch += 2 * max_epoch
        #         self.main_opt_wrapper()
        #         self.args.max_epoch += max_epoch
        # else:
        self.main_opt_wrapper()
        # self.main_opt_wrapper()
        # stop child workers

        Helper.print_total_time(total_start_time)
        # here we should also save the result to the log file..
        if self.ret is not None:
            with open(os.path.join(args.output_fdir, "log.txt"), "w") as fout:
                for key, value in sorted(self.ret.items()):
                    out_info = f"{key} = {value:.6f}\n"
                    print(out_info, end="", flush=True)
                    fout.write(out_info)

            print(f"Save the final log to log.txt", flush=True)

        # here we can do final evaluate.
        print(f"Begin evaluate force", flush=True)
        self.main_worker_epoch(is_evaluate=True)
        print(f"After evaluate force", flush=True)
        print(f"Begin save result parallel")
        self.save_result_parallel(os.path.join(args.output_fdir, f"OptimizeEachFrame.ckpt"), )
        print(f"End save result parallel")

        comm.barrier()
        comm.scatter([(OptimizeWorkerMode.Stop,)] * comm_size, root=0)

        self.convert_for_samcon_parallel()
        self.output_result()

    def main_worker_grad_test(self):
        args: Namespace = self.args

        root_pos_noise = 0.01 * torch.rand_like(self.main_root_pos_param)
        rot_noise = 0.01 * torch.rand_like(self.main_joint_vec6d_param)
        pd_noise = 0.01 * torch.rand_like(self.main_pd_target_param)

        def add_noise(root_pos: nn.Parameter, rot: nn.Parameter, pd: nn.Parameter):
            with torch.no_grad():
                root_pos.data += root_pos_noise
                rot.data += rot_noise
                pd.data += pd_noise

        for epoch_ in range(args.max_epoch):
            start_time = time.time()
            add_noise(self.main_root_pos_param, self.main_joint_vec6d_param, self.main_pd_target_param)
            self.main_worker_optimize(epoch_)

            self.init_parameter(args.index_t, args.width)
            self.best_loss: float = float("inf")
            add_noise(self.root_pos_param, self.joint_vec6d_param, self.pd_target_param)
            ret = self.closure(epoch_)
            # print(torch.max(torch.abs(multi_root_grad - self.root_pos_param.grad)))
            print(
                ret["loss_contact_h"] - self.tmp_total_simu_res[0][2]["loss_contact_h"] - self.tmp_total_simu_res[1][2]["loss_contact_h"])
            print(ret["tot_loss"] - self.tmp_total_simu_res[0][2]["tot_loss"] - self.tmp_total_simu_res[1][2]["tot_loss"])
            print("delta grad", torch.max(self.main_root_pos_param.grad - self.root_pos_param.grad))
            exit(0)
            # check root, rotation, pd target smooth loss OK.
            # check root, rotation, pd target close loss for initial solu..we should add some noise to check loss..
            # check pos, rot, linvel, angvel loss OK
            # check contact loss OK
            self.optim.step()  # As gradient has been saved at the parameter, we can optimize these parameters.

        # stop child workers
        comm.barrier()
        comm.scatter([-1] * self.worker_info.comm_size, root=0)

    def callback(self, index_: int, contact_: DiffContactInfo):
        with torch.no_grad():
            self.at_contact_plan = True
            ret = self.forward_process(index_, 2, 0, [contact_], False)["tot_loss"].item()
            self.at_contact_plan = False
            return ret

    def callback_optim(self, index_: int, contact_: DiffContactInfo):
        # build the optimizer, and optimize for some steps.
        self.optim = AdamW([
            self.root_pos_param, self.joint_vec6d_param, self.pd_target_param,
            self.body_velo_param, self.body_angvel_param
            ], lr=self.args.lr
        )
        best_loss: float = float("inf")
        for sub_epoch in range(self.args.optimize_num_for_plan):
            self.optim.zero_grad(set_to_none=True)
            ret: torch.Tensor = self.forward_process(index_, 2, 0, [contact_])["tot_loss"]
            ret.backward()
            self.optim.step()
            best_loss = min(best_loss, ret.item())

        self.optim: Optional[AdamW] = None
        return best_loss

    def handle_mcmc_contact_optimize_info(
        self,
        index: int,
        start_index: int,
        epoch: int,
        root_pos_numpy: np.ndarray,
        joint_numpy: Optional[np.ndarray],
        pd_target_numpy: Optional[np.ndarray],
        body_velo_numpy: Optional[np.ndarray] = None,
        body_angvel_numpy: Optional[np.ndarray] = None,
        root_velo_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        joint_angvel_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        contact_param_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
        contact_force_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
        contact_torque_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
    ) -> Tuple[int, int, List[List[int]]]:
        """
        we can enumerate contact for several frames.
        How to handle gap in different workers..?
        1. compute local contact loss
        2. enumerate contact by loss..

        """
        # Note: the loss is saved at self.planning_list
        args: Namespace = self.args
        index, start_index, ret_mess = self.handle_contact_optimize_info(
            index,
            start_index,
            epoch,
            root_pos_numpy,
            joint_numpy,
            pd_target_numpy,
            body_velo_numpy,
            body_angvel_numpy,
            root_velo_numpy, joint_angvel_numpy, contact_param_numpy, contact_force_numpy, contact_torque_numpy
        )

        # set the parameter here
        self.root_pos_param = nn.Parameter(torch.from_numpy(root_pos_numpy).clone())
        self.joint_vec6d_param = nn.Parameter(torch.from_numpy(joint_numpy).clone())
        self.pd_target_param = nn.Parameter(torch.from_numpy(pd_target_numpy).clone())
        self.body_velo_param = nn.Parameter(torch.from_numpy(body_velo_numpy).clone())
        self.body_angvel_param = nn.Parameter(torch.from_numpy(body_angvel_numpy).clone())

        # stack out the parameters
        saved_params = self.save_result()
        self.clear_param()

        def run_optimization(idx_start_: int, idx_end_: int, contact_list_: List[DiffContactInfo]) -> float:
            # 1. set the parameter
            self.load_result(saved_params, idx_start_ - start_index, idx_end_ - start_index)
            witdh_: int = idx_end_ - idx_start_
            assert self.root_pos_param.shape[0] == witdh_
            # 2. build the optimizer
            optim_ = self.build_optim()
            # 3. run forward simulation to compute loss..
            loss_ = torch.as_tensor(0.0, dtype=torch.float64)
            for search_epoch_ in range(args.contact_plan_mcmc_opt_step):
                optim_.zero_grad(True)
                try:
                    ret_dict_ = self.forward_process(
                        idx_start_, witdh_,
                        search_epoch_ + 23333,
                        contact_list_,
                        is_evaluate=False
                    )
                except Exception as err:
                    print(start_index, idx_start_, witdh_, self.root_pos_param.shape, self.body_velo_param.shape, self.joint_vec6d_param.shape)
                    raise err
                loss_: torch.Tensor = ret_dict_["tot_loss"]
                # print(loss_.item())
                loss_.backward()
                optim_.step()

            self.clear_param()
            # 4. return the optimized loss.
            return loss_.item()

        # find contact between previous contact and curr contact
        def find_potential_contact(
            prev_hash_: int,
            idx_: Union[int, ContactPlan],
            prev_prev_hash: Optional[int] = None
        ) -> List[int]:
            if isinstance(idx_, int):
                contact_plan_: ContactPlan = self.planning_list[idx_]
            else:
                contact_plan_: ContactPlan = idx_

            # here we can search contact comb within 2 different contacts..
            reachable_hash_list_: List[int] = []
            best_loss = contact_plan_.sorted_contact_map[0][1]
            full_mask: int = (1 << self.num_body) - 1
            for contact_hash_, contact_loss_ in contact_plan_.contact_map.items():
                # compute different number between contact_hash_ and previous hash
                if contact_hash_ == 0:  # for hack.. TODO
                    continue
                if contact_loss_ > 5 * best_loss:
                    continue
                if prev_prev_hash is not None:
                    if MathHelper.count_ones((full_mask - prev_prev_hash) & (full_mask - contact_hash_) & prev_hash_) >= 2:
                        continue
                    if MathHelper.count_ones(prev_prev_hash & contact_hash_ & (full_mask - prev_hash_)) >= 2:
                        continue
                diff_hash_: int = contact_hash_ ^ prev_hash_
                diff_count_: int = MathHelper.count_ones(diff_hash_)
                if diff_count_ >= 3:
                    continue
                reachable_hash_list_.append(contact_hash_)

            # use the best contact as default contact
            if len(reachable_hash_list_) == 0:
                reachable_hash_list_.append(prev_hash_)
                # reachable_hash_list_.append(contact_plan_.sorted_contact_map[0][0])
                # print(f"use default contact in find potential contact")
            return reachable_hash_list_

        # serach the contacts in self.planning_list
        width: int = root_pos_numpy.shape[0]
        final_contact_plan: List[Optional[List[int]]] = [None for _ in range(width)]  # for save the contact result
        final_contact_plan[0] = self.contact_mess[start_index]
        # here we can use a simple sliding window...
        if comm_rank == 0:
            pbar = tqdm(total=width)
        else:
            pbar = None

        for frame in range(start_index, start_index + width, args.contact_plan_mcmc_frame - 1):
            # here we should save the contact sequence..Dict[Tuple[int...], float] is just OK
            # because python uses hash value for indexing...
            # There is small prob for hash conflict..ignore the conflict simply
            if pbar is not None:
                pbar.update(args.contact_plan_mcmc_frame - 1)
            saved_combination: Dict[Tuple, float] = {}
            contact_info_list: List[DiffContactInfo] = []
            hash_list: List[int] = []
            forward_count: int = min(args.contact_plan_mcmc_frame, start_index + width - frame - 1)
            # print(f"frame = {frame}, start_index = {start_index}, width = {width}, forward_count = {forward_count}", flush=True)
            if forward_count <= 2:
                break
            for sub_piece_index in range(0, forward_count):  # Test the best predicted sequence here.
                planner: ContactPlan = self.planning_list[frame + sub_piece_index]
                best_hash: int = planner.sorted_contact_map[0][0]
                diff_contact: DiffContactInfo = planner.to_subcontact_by_hash(best_hash)
                contact_info_list.append(diff_contact)
                hash_list.append(best_hash)

            # run optimization by given contact list
            eval_loss = run_optimization(frame, frame + forward_count, contact_info_list)
            saved_combination[tuple(hash_list)] = eval_loss

            # eval the suboptimal solution of each frame
            search_sub_optimal = False
            if search_sub_optimal:
                for sub_optimal in range(0, forward_count):
                    contact_info_list: List[DiffContactInfo] = []
                    hash_list: List[int] = []
                    for sub_piece_index in range(0, forward_count):
                        planner: ContactPlan = self.planning_list[frame + sub_piece_index]
                        if sub_optimal == sub_piece_index:
                            use_index = 1
                        else:
                            use_index = 0
                        best_hash: int = planner.sorted_contact_map[use_index][0]
                        diff_contact: DiffContactInfo = planner.to_subcontact_by_hash(best_hash)
                        contact_info_list.append(diff_contact)
                        hash_list.append(best_hash)
                    # run optimization
                    eval_loss = run_optimization(frame, frame + forward_count, contact_info_list)
                    saved_combination[tuple(hash_list)] = eval_loss

            use_mcmc_random_search = True
            if use_mcmc_random_search:
                for duplicate_piece in range(args.contact_plan_mcmc_number):  # maybe 10 different pieces are enough..
                    hash_list: List[int] = []
                    # 1. TODO: sample the initial contact..
                    hash_list.append(ContactLabelState.list_to_hash(final_contact_plan[frame - start_index]))
                    # 2. search the next contacts
                    for sub_piece_index in range(1, forward_count):
                        planner: ContactPlan = self.planning_list[frame + sub_piece_index]
                        potential_contact = find_potential_contact(hash_list[-1], planner, hash_list[-2] if len(hash_list) > 1 else None)
                        # select a contact with prob..
                        # contact_loss = [(idx, planner.contact_map[node]) for idx, node in enumerate(potential_contact)]
                        # contact_loss.sort(key=lambda x: x[1])
                        # prob = np.array([node[0] for node in contact_loss])
                        # prob = prob / np.sum(prob)
                        contact_loss = []
                        for node_idx, node in enumerate(potential_contact):
                            if not node in planner.contact_map:
                                contact_loss.append(planner.sorted_contact_map[0][1])
                                # planner.compute_loss_by_contact_label(ContactLabelState.list_from_hash(node))
                            else:
                                contact_loss.append(planner.contact_map[node])
                        contact_loss = np.array(contact_loss)
                        # contact_loss = np.array([planner.contact_map[node] for node in potential_contact])
                        if len(contact_loss) > 1:
                            contact_loss = (contact_loss - np.min(contact_loss)) / (np.max(contact_loss) - np.min(contact_loss))
                            prob = (1 - contact_loss) ** 2
                            prob = prob / np.sum(prob)
                        else:
                            prob = np.array([1])
                        select: int = np.random.choice(np.arange(0, len(prob)), 1, p=prob).item()
                        new_hash = potential_contact[select]
                        hash_list.append(new_hash)
                        # we need to find a solution, match these cases:
                        # At most one contact point of two adjacent frames can be changed
                        # A contact may be remained for at least 0.05 seconds.

                    hash_tuple = tuple(hash_list)
                    if hash_tuple in saved_combination:
                        continue
                    # 3. get DiffContactList by current hash list
                    contact_info_list = []
                    for sub_piece_index in range(0, forward_count):
                        planner: ContactPlan = self.planning_list[frame + sub_piece_index]
                        hash_value = hash_list[sub_piece_index]
                        contact_info_list.append(planner.to_subcontact_by_hash(hash_value))

                    # 4. optimize, and save the result
                    eval_loss = run_optimization(frame, frame + forward_count, contact_info_list)
                    saved_combination[hash_tuple] = eval_loss

            # get the best saved result
            saved_combination_list: List[Tuple[Tuple, float]] = list(saved_combination.items())
            saved_combination_list.sort(key=lambda x: x[1])
            best_contact_seq: Tuple = saved_combination_list[0][0]
            # convert to List[int] as contact label
            best_contact_label_seq = []
            for sub_piece_index, best_hash in enumerate(best_contact_seq):
                best_label = ContactLabelState.list_from_hash(best_hash)
                best_contact_label_seq.append(best_label)
            # print(f"len contact label seq = {len(best_contact_label_seq)}")
            # save the best contact sequence.
            final_contact_plan[frame - start_index: frame - start_index + forward_count] = best_contact_label_seq

        # do some clear processing.
        if pbar is not None:
            pbar.close()

        return index, start_index, final_contact_plan[:width]

    def handle_contact_kinematic(
        self,
        index: int,
        start_index: int,
        epoch: int,
        root_pos_numpy: np.ndarray,
        root_joint_numpy: Optional[np.ndarray],
        pd_target_numpy: Optional[np.ndarray] = None,
        body_velo_numpy: Optional[np.ndarray] = None,
        body_angvel_numpy: Optional[np.ndarray] = None,
        root_velo_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        joint_angvel_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        contact_param_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
        contact_force_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
        contact_torque_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
    ):
        """
        handle contact info using kinematic pose
        if the velocity and height is small, we can set a contact label.
        we can add on the current contact..
        """
        # here we can also smmoth the input root pos
        root_pos_numpy: np.ndarray = smooth_operator(root_pos_numpy, GaussianBase(3))
        width: int = root_pos_numpy.shape[0]
        # export as motion data..
        with torch.no_grad():
            self.diff_motion.clear()
            self.diff_motion._num_frames = root_joint_numpy.shape[0]
            self.diff_motion._root_translation = torch.from_numpy(root_pos_numpy)
            self.diff_motion._joint_rotation = torch.from_numpy(MathHelper.vec6d_to_quat(root_joint_numpy))
            self.diff_motion.recompute_joint_global_info()
            kine_body_pos, kine_body_quat = self.mocap_import.import_mocap_base_batch(self.diff_motion)
            kine_body_vel: torch.Tensor = torch.diff(kine_body_pos, dim=0) * self.args.simulation_fps # (width - 1, num body, 3)
            kine_body_vel: torch.Tensor = torch.cat([kine_body_vel[0, None], kine_body_vel], dim=0)  # (width, num body, 3)

            vel_eps = 0.5
            height_eps = 0.5
            # when linear velocity and height is small, we can create a new contact here..
            rel_vel = kine_body_vel - kine_body_vel[:, :1, :]
            rel_vel[:, 0, :] = kine_body_vel[:, 0, :]
            length_rel_vel = torch.linalg.norm(rel_vel, dim=-1).view(-1).numpy()

            body_vel_len: np.ndarray = torch.linalg.norm(kine_body_vel, dim=-1).view(-1).numpy()
            body_height: np.ndarray = (kine_body_pos[..., 1].numpy() - self.body_min_contact_h.view(1, -1).numpy()).reshape(-1)
            body_height -= np.min(body_height)
            vel_flag: np.ndarray = np.where(body_vel_len < vel_eps)[0]
            rel_vel_flag: np.ndarray = np.where(length_rel_vel < vel_eps)
            tot_vel_flag = np.union1d(rel_vel_flag, vel_flag)
            height_flag: np.ndarray = np.where(body_height < height_eps)[0]
            common_flag: np.ndarray = np.intersect1d(tot_vel_flag, height_flag)
            if len(common_flag) > 0:
                frames = common_flag // self.num_body
                print(f"create total contact {len(common_flag)}", flush=True)
                bodies = common_flag % self.num_body
                for i in range(frames.shape[0]):
                    curr_frame = int(frames[i] + start_index)
                    if self.contact_mess[curr_frame] is None:
                        self.contact_mess[curr_frame] = []
                    self.contact_mess[curr_frame].append(int(bodies[i]))
                    # Note: if the delta height is too large, we should ignore this contact..or shift down this contact..
                for i in range(start_index, start_index + width):
                    # maybe we should remove contact with too large height...?
                    # if self.contact_mess[i] is not None:
                    unique_idx = np.unique(self.contact_mess[i])
                    self.contact_mess[i] = unique_idx.tolist()
                    try:
                        if len(unique_idx) > 0:
                            self.contact_label[i, unique_idx] = 1
                        else:
                            pass
                    except Exception as err:
                        print(i, unique_idx, self.contact_label.shape)
                        raise err

            else:
                print("No need to add contact kinematic")

            ret_mess: List[List[int]] = self.contact_mess[start_index: start_index + width - 1]
            # as we should not pickle torch.Tensor directly, we need to recompute DiffContactInfo at each worker
            # rather than send them using MPI directly..
            if comm_rank == 0:
                print(f"after calling handle_contact_kinematic")

            return index, start_index, ret_mess


    def handle_contact_optimize_info(
        self,
        index: int,
        start_index: int,
        epoch: int,
        root_pos_numpy: np.ndarray,
        root_joint_numpy: Optional[np.ndarray],
        pd_target_numpy: Optional[np.ndarray],
        body_velo_numpy: Optional[np.ndarray] = None,
        body_angvel_numpy: Optional[np.ndarray] = None,
        root_velo_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        joint_angvel_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        contact_param_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
        contact_force_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
        contact_torque_numpy: Optional[np.ndarray] = None,  # use for CIO, unused here.
    ) -> Optional[Tuple[int, int, List[List[int]]]]:
        """
        This function is called at each sub-worker.
        Usage: simply get the contact neighbour of each frame..
        """
        args_old = copy.deepcopy(self.args)
        self.args = only_use_phys_loss(self.args)
        self.args.phys_loss_num_forward = 1
        self.args.optimize_root_epoch = self.args.phys_loss_epoch = 0

        # here we can also smmoth the input root pos
        root_pos_numpy: np.ndarray = smooth_operator(root_pos_numpy, GaussianBase(3))
        root_pos_param: torch.Tensor = torch.from_numpy(root_pos_numpy)
        joint_vec6d_param: torch.Tensor = torch.from_numpy(root_joint_numpy)
        body_velo_param: Optional[torch.Tensor] = torch.from_numpy(body_velo_numpy) \
            if body_velo_numpy is not None else None
        body_angvel_param: Optional[torch.Tensor] = torch.from_numpy(body_angvel_numpy) \
            if body_angvel_numpy is not None else None

        width: int = root_pos_numpy.shape[0]
        backup_writer: Optional[SummaryWriter] = self.writer
        self.writer: Optional[SummaryWriter] = None

        # export as motion data..
        mocap: TargetPose = self.get_simu_target_set(root_pos_numpy, root_joint_numpy).target
        if self.args.optimize_num_for_plan > 0:
            callback = self.callback_optim
        else:
            callback = self.callback

        self.planning_list: List[Optional[ContactPlan]] = ContactPlan.build_by_mocap(
            mocap, self.scene, self.character, callback, self.extractor, start_index)

        if pd_target_numpy is None:  # only used for compute self.planning_list
            self.writer = backup_writer
            self.args = args_old
            return None

        # here we can compute velocity of center of mass..
        com_velo: np.ndarray = (1.0 / self.character.body_info.sum_mass) * np.sum(
            self.character.body_info.mass_val.reshape((1, self.num_body)) * mocap.character_body.linvel[..., 1], axis=1)
        com_acc: np.ndarray = MathHelper.vec_diff(com_velo, False, self.scene.sim_fps)
        # we can smooth the com acc..
        com_acc: np.ndarray = smooth_operator(com_acc, GaussianBase(5))
        if com_acc.shape[0] > 2:
            com_acc[:2] = com_acc[2]
        old_contact_w, self.args.w_contact_height = self.args.w_contact_height, 0
        pbar: Optional[tqdm] = tqdm(total=width - 1, desc="Contact planning") if comm_rank == 0 else None
        pd_target_param: torch.Tensor = torch.from_numpy(pd_target_numpy)
        for frame in range(start_index, start_index + width - 1):
            if pbar is not None:
                pbar.update()
            planer: ContactPlan = self.planning_list[frame]
            idx: int = frame - start_index
            self.root_pos_param: Optional[nn.Parameter] = nn.Parameter(root_pos_param[idx: idx + 2])
            self.joint_vec6d_param: Optional[nn.Parameter] = nn.Parameter(joint_vec6d_param[idx: idx + 2])
            self.pd_target_param: Optional[nn.Parameter] = nn.Parameter(pd_target_param[idx: idx + 2])
            self.body_velo_param: Optional[nn.Parameter] = nn.Parameter(body_velo_param[idx: idx + 2])
            self.body_angvel_param: Optional[nn.Parameter] = nn.Parameter(body_angvel_param[idx: idx + 2])
            # here we should also consider contact on previous frame..
            best_value, best_hash, best_label, sub_contact = planer.search_solution(
                float(com_acc[idx]),
                self.contact_mess[frame],
                self.contact_mess[frame - 1] if frame > 0 else None
            )
            # print(f"frame = {frame}, new_label = {best_label}, old_label = {self.contact_mess[frame]}")
            self.contact_mess[frame] = best_label
            self.contact_info_list[frame] = sub_contact

        # here we should clear the input param
        self.clear_param()

        if comm_rank == 0:
            pbar.close()
        self.writer = backup_writer
        self.args.w_contact_height = old_contact_w

        ret_mess: List[List[int]] = self.contact_mess[start_index: start_index + width - 1]
        # as we should not pickle torch.Tensor directly, we need to recompute DiffContactInfo at each worker
        # rather than send them using MPI directly..
        self.args = args_old
        return index, start_index, ret_mess

    def handle_contact_planninng(self, *args):
        """
        wrapper for different type of contact mode
        """
        if self.args.contact_plan_mode == ContactPlanMode.Greedy:
            return self.handle_contact_optimize_info(*args)
        elif self.args.contact_plan_mode == ContactPlanMode.Kinematic:
            return self.handle_contact_kinematic(*args)
        elif self.args.contact_plan_mode == ContactPlanMode.MCMC:
            return self.handle_mcmc_contact_optimize_info(*args)
        elif self.args.contact_plan_mode == ContactPlanMode.SMC:
            raise NotImplementedError
        else:
            raise ValueError

    def handle_forward_simu_info(
        self,
        index: int,
        start_index: int,
        epoch: int,
        root_pos_numpy: np.ndarray,
        root_joint_numpy: Optional[np.ndarray],
        pd_target_numpy: Optional[np.ndarray],
        body_velo_numpy: Optional[np.ndarray] = None,
        body_angvel_numpy: Optional[np.ndarray] = None,
        root_velo_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        joint_angvel_numpy: Optional[np.ndarray] = None,  # velocity in reduced coordinate
        contact_param_numpy: Optional[np.ndarray] = None,  # use for CIO
        contact_force_numpy: Optional[np.ndarray] = None,  # use for CIO
        contact_torque_numpy: Optional[np.ndarray] = None,  # use for CIO
        is_evaluate: bool = False,
    ):
        self.root_pos_param: Optional[nn.Parameter] = nn.Parameter(torch.from_numpy(root_pos_numpy))
        self.joint_vec6d_param: Optional[nn.Parameter] = nn.Parameter(torch.from_numpy(root_joint_numpy))
        if pd_target_numpy is not None:
            self.pd_target_param: Optional[nn.Parameter] = nn.Parameter(torch.from_numpy(pd_target_numpy))
        else:
            self.pd_target_param: Optional[nn.Parameter] = None

        if body_velo_numpy is not None:
            self.body_velo_param: nn.Parameter = nn.Parameter(torch.from_numpy(body_velo_numpy))
        else:
            self.body_velo_param: Optional[nn.Parameter] = None

        if body_angvel_numpy is not None:
            self.body_angvel_param: Optional[nn.Parameter] = nn.Parameter(torch.from_numpy(body_angvel_numpy))
        else:
            self.body_angvel_param: Optional[nn.Parameter] = None

        if root_velo_numpy is not None:
            self.root_velo_param: Optional[nn.Parameter] = nn.Parameter(torch.from_numpy(root_velo_numpy))
        else:
            self.root_velo_param: Optional[nn.Parameter] = None

        if joint_angvel_numpy is not None:
            self.joint_angvel_param: Optional[nn.Parameter] = nn.Parameter(torch.from_numpy(joint_angvel_numpy))
        else:
            self.joint_angvel_param: Optional[nn.Parameter] = None

        if contact_param_numpy is not None:  # use for optimize with CIO
            self.contact_label_param: nn.Parameter = nn.Parameter(torch.from_numpy(contact_param_numpy))
            self.contact_force_param: nn.Parameter = nn.Parameter(torch.from_numpy(contact_force_numpy))
            self.contact_torque_param: nn.Parameter = nn.Parameter(torch.from_numpy(contact_torque_numpy))
        else:
            self.contact_label_param: Optional[nn.Parameter] = None
            self.contact_force_param: Optional[nn.Parameter] = None
            self.contact_torque_param: Optional[nn.Parameter] = None

        width: int = root_pos_numpy.shape[0]
        ret_dict = self.forward_process(start_index, width, epoch, is_evaluate=is_evaluate)
        loss: torch.Tensor = ret_dict["tot_loss"]
        if loss == 0.0 or is_evaluate:
            pass  # we need not compute gradient here.
        else:
            loss.backward()

        # for evaluate the force and torque, the other gathered value is same as optimization..
        if self.eval_force_list is not None:
            eval_force_list: Optional[List[Tuple]] = [
                (node[0].detach().numpy(), node[1].detach().numpy()) if node is not None else None for node in self.eval_force_list
            ]
        else:
            eval_force_list: Optional[List[Tuple]] = None

        gather_value = {
            "index": index,
            "start_index": start_index,
            "loss_dict": {key: value.item() for key, value in ret_dict.items()},
            "root_pos_param_grad": np.asarray(self.root_pos_param.grad.numpy()) if self.root_pos_param is not None and self.root_pos_param.grad is not None else None,
            "joint_vec6d_param_grad": np.asarray(self.joint_vec6d_param.grad.numpy()) if self.joint_vec6d_param is not None and self.joint_vec6d_param.grad is not None else None,
            "pd_target_param_grad": np.asarray(self.pd_target_param.grad.numpy()) if self.pd_target_param is not None and self.pd_target_param.grad is not None else None,
            "body_velo_param_grad": np.asarray(self.body_velo_param.grad.numpy()) if self.body_velo_param is not None and self.body_velo_param.grad is not None else None,
            "body_angvel_param_grad": np.asarray(self.body_angvel_param.grad.numpy()) if self.body_angvel_param is not None and self.body_angvel_param.grad is not None else None,
            "root_velo_param_grad": np.asarray(self.root_velo_param.grad.numpy()) if self.root_velo_param is not None else None,
            "joint_angvel_param_grad": np.asarray(self.joint_angvel_param.grad.numpy()) if self.joint_angvel_param is not None else None,
            "contact_label_param_grad": np.asarray(self.contact_label_param.numpy()) if self.contact_label_param is not None else None,
            "contact_force_param_grad": np.asarray(self.contact_force_param.numpy()) if self.contact_force_param is not None else None,
            "contact_torque_param_grad": np.asarray(self.contact_torque_param.numpy()) if self.contact_torque_param is not None else None,
            "eval_force_list": eval_force_list
        }

        self.root_pos_param.grad = None
        self.joint_vec6d_param.grad = None
        if self.pd_target_param is not None:
            self.pd_target_param.grad = None
        if self.body_velo_param is not None:
            self.body_velo_param.grad = None
        if self.body_angvel_param is not None:
            self.body_angvel_param.grad = None

        if self.root_velo_param is not None:
            self.root_velo_param.grad = None
        if self.joint_angvel_param is not None:
            self.joint_angvel_param.grad = None

        if self.contact_label_param is not None:
            self.contact_label_param.grad = None
            self.contact_force_param.grad = None
            self.contact_torque_param.grad = None

        self.root_pos_param: Optional[nn.Parameter] = None
        self.joint_vec6d_param: Optional[nn.Parameter] = None
        self.pd_target_param: Optional[nn.Parameter] = None

        self.body_velo_param: Optional[nn.Parameter] = None
        self.body_angvel_param: Optional[nn.Parameter] = None

        self.root_velo_param: Optional[nn.Parameter] = None
        self.joint_angvel_param: Optional[nn.Parameter] = None

        self.contact_label_param: Optional[nn.Parameter] = None
        self.contact_force_param: Optional[nn.Parameter] = None
        self.contact_torque_param: Optional[nn.Parameter] = None
        self.diff_motion.clear()

        return gather_value

    def child_worker(self):
        args: Namespace = self.args
        while True:
            comm.barrier()
            worker_mode, *forward_simu_info = comm.scatter(None)  # here we should also recieve stop info..
            if worker_mode == OptimizeWorkerMode.Stop:
                break
            elif worker_mode == OptimizeWorkerMode.Optimize:
                single_simu_res = self.handle_forward_simu_info(*forward_simu_info)  # The main worker should also compute in Diff-ODE
                comm.gather(single_simu_res)
            elif worker_mode == OptimizeWorkerMode.RecieveContactPlan:
                self.handle_contact_label_by_body_height(*forward_simu_info)
                comm.barrier()
            elif worker_mode == OptimizeWorkerMode.ContactPlan:
                single_simu_res = self.handle_contact_planninng(*forward_simu_info)

                comm.gather(single_simu_res)
                # here we need also get new contact label from main worker.
                self.contact_mess, root_pos, joint_vec6d = comm.bcast(None)
                # convert the new contact label to DiffContact sequence.
                self.handle_contact_optimize_info(0, 0, 0, root_pos, joint_vec6d, None)
                self.compute_contact_info_by_mess()  # we need not to gather any data here..
                comm.barrier()  # for sync
            elif worker_mode == OptimizeWorkerMode.Evaluate:
                single_simu_res = self.handle_forward_simu_info(*forward_simu_info, is_evaluate=True)  # The main worker should also compute in Diff-ODE
                comm.gather(single_simu_res)
            else:
                raise ValueError

    @staticmethod
    def main():
        args = parse_args()
        optimize = OptimizeParallel(args)
        if args.only_vis_result:
            optimize.render_result()
        else:
            if optimize.worker_info.comm_rank == 0:
                optimize.main_worker()
            else:
                optimize.child_worker()


if __name__ == "__main__":
    OptimizeParallel.main()

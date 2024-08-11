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
optimize each frame
"""

from argparse import Namespace
import logging
import math
from mpi4py import MPI
import numpy as np
import os

import pickle
import platform
import time
import torch
from torch import nn
from typing import Optional, Dict, Tuple, Union, List, Any
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR
from VclSimuBackend.ODESim.ODEScene import ODEScene, SceneContactInfo

from VclSimuBackend.Samcon.OptimalGait.OptimizeEachFrameBase import OptimizeEachFrameBase, parse_args, disable_phys_loss, save_phys_loss, load_phys_loss
from VclSimuBackend.Samcon.OptimalGait.ContactPlan import ContactPlan
from VclSimuBackend.DiffODE import DiffQuat
from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo
from VclSimuBackend.DiffODE.DiffFrameInfo import diff_frame_import_from_tensor
from VclSimuBackend.Utils.Dataset.StdHuman import stdhuman_with_root_to_unified_index
from VclSimuBackend.Utils.Evaluation import calc_motion_mpjpe

from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib.PyTorchMotionData import PyTorchMotionData
from VclSimuBackend.pymotionlib import BVHLoader


fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")
comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
is_windows: bool = "Windows" in platform.platform()


# actually, this function is unused.
def soft_height_weight_func0(x, eps) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[x < -eps] = 1
    result[x > eps] = 1
    retain_index = torch.where(result == 0)
    result[retain_index] = 0.5 - 0.5 * torch.cos((math.pi / eps) * x[retain_index])
    return result


# actually, this function is unused
def soft_velo_weight_func0(x, eps) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[x < 0] = 1
    result[x > eps] = -1
    retain_index = torch.where(result == 0)
    result[retain_index] = 0.5 + 0.5 * torch.cos((math.pi / eps) * x[retain_index])
    result[x > eps] = 0
    return result


# actually, this function is unused
def soft_height_weight_func(x, eps) -> torch.Tensor:
    result = torch.ones_like(x)
    result[x > eps] = x[x > eps] - eps + 1
    return result


# actually, this function is unused.
def soft_height_weight_func0(x, eps) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[x < -eps] = 1
    result[x > eps] = 1
    retain_index = torch.where(result == 0)
    result[retain_index] = 0.5 - 0.5 * torch.cos((math.pi / eps) * x[retain_index])
    return result


# actually, this function is unused
def soft_velo_weight_func0(x, eps) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[x < 0] = 1
    result[x > eps] = -1
    retain_index = torch.where(result == 0)
    result[retain_index] = 0.5 + 0.5 * torch.cos((math.pi / eps) * x[retain_index])
    result[x > eps] = 0
    return result


# actually, this function is unused
def soft_height_weight_func(x, eps) -> torch.Tensor:
    result = torch.ones_like(x)
    result[x > eps] = x[x > eps] - eps + 1
    return result


def soft_height_weight_func0(x, eps) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[x < -eps] = 1
    result[x > eps] = 1
    retain_index = torch.where(result == 0)
    result[retain_index] = 0.5 - 0.5 * torch.cos((math.pi / eps) * x[retain_index])
    return result


def soft_velo_weight_func0(x, eps) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[x < 0] = 1
    result[x > eps] = -1
    retain_index = torch.where(result == 0)
    result[retain_index] = 0.5 + 0.5 * torch.cos((math.pi / eps) * x[retain_index])
    result[x > eps] = 0
    return result


def soft_height_weight_func(x, eps) -> torch.Tensor:
    result = torch.ones_like(x)
    result[x > eps] = x[x > eps] - eps + 1
    return result

class OptimizeEachFrame(OptimizeEachFrameBase):

    def __init__(self, args: Namespace, scene: Optional[ODEScene] = None) -> None:
        super().__init__(args, scene)
        self.init_root_pos: Optional[torch.Tensor] = None  # initial root position
        self.init_body_pos: Optional[torch.Tensor] = None  # initial body position
        self.init_body_mat: Optional[torch.Tensor] = None  # initial body rotation matrix
        self.init_body_velo: Optional[torch.Tensor] = None  # initial body velocity
        self.init_body_omega: Optional[torch.Tensor] = None  # initial body angular velocity

        self.init_pd_target_vec6d: Optional[torch.Tensor] = None
        self.init_joint_quat: Optional[torch.Tensor] = None
        self.init_joint_vec6d: Optional[torch.Tensor] = None
        self.init_root_velo: Optional[torch.Tensor] = None
        self.init_joint_angvel: Optional[torch.Tensor] = None

        self.root_pos_param: Optional[nn.Parameter] = None
        self.joint_vec6d_param: Optional[nn.Parameter] = None
        self.pd_target_param: Optional[nn.Parameter] = None
        self.root_velo_param: Optional[nn.Parameter] = None
        self.joint_angvel_param: Optional[nn.Parameter] = None
        self.body_velo_param: Optional[nn.Parameter] = None
        self.body_angvel_param: Optional[nn.Parameter] = None

        self.contact_label_param: Optional[nn.Parameter] = None  # use for CIO mode
        self.contact_force_param: Optional[nn.Parameter] = None  # optimize contact force for CIO
        self.contact_torque_param: Optional[nn.Parameter] = None  # optimize contact torque for CIO

        self.main_root_pos_param: Optional[nn.Parameter] = None  # (width, 3), use for parallel optimization
        self.main_root_vel_param: Optional[nn.Parameter] = None  # (width, 3)
        self.main_joint_vec6d_param: Optional[nn.Parameter] = None
        self.main_pd_target_param: Optional[nn.Parameter] = None  # (width, num joint, 3, 2)
        self.main_angvel_param: Optional[nn.Parameter] = None  # (width, num joint, 3)
        self.main_body_velo_param: Optional[nn.Parameter] = None
        self.main_body_angvel_param: Optional[nn.Parameter] = None

        self.main_contact_label_param: Optional[nn.Parameter] = None  # use for CIO mode
        self.main_contact_force_param: Optional[nn.Parameter] = None
        self.main_contact_torque_param: Optional[nn.Parameter] = None

        self.optim: Optional[Optimizer] = None
        self.scheduler: Optional[StepLR] = None
        self.epoch: int = 0
        self.build_initial_solution()

        # for contact planning
        self.planning_list: Optional[List[ContactPlan]] = None

        self.eval_force_list: Optional[List] = None
        self.main_eval_force_list: Optional[List] = None
        self.extract_contact_force = None

        self.at_contact_plan = False

    def clear_param(self):
        self.root_pos_param: Optional[nn.Parameter] = None
        self.joint_vec6d_param: Optional[nn.Parameter] = None
        self.pd_target_param: Optional[nn.Parameter] = None
        self.root_velo_param: Optional[nn.Parameter] = None
        self.joint_angvel_param: Optional[nn.Parameter] = None
        self.body_velo_param: Optional[nn.Parameter] = None
        self.body_angvel_param: Optional[nn.Parameter] = None

        self.contact_label_param: Optional[nn.Parameter] = None  # use for CIO mode
        self.contact_force_param: Optional[nn.Parameter] = None  # optimize contact force for CIO
        self.contact_torque_param: Optional[nn.Parameter] = None  # optimize contact torque for CIO

        self.optim: Optional[Optimizer] = None
        self.scheduler: Optional[StepLR] = None

    def convert_tensor_to_param(self):
        def base_func(x) -> Optional[nn.Parameter]:
            if x is not None:
                if isinstance(x, torch.Tensor):
                    if not isinstance(x, nn.Parameter):
                        return nn.Parameter(self.root_pos_param)
            else:
                return x
        self.root_pos_param = base_func(self.root_pos_param)
        self.joint_vec6d_param = base_func(self.joint_vec6d_param)
        self.pd_target_param = base_func(self.pd_target_param)
        self.body_velo_param = base_func(self.body_velo_param)
        self.body_angvel_param = base_func(self.body_angvel_param)

    def convert_param_to_tensor(self):
        pass

    # def initialize_contact_force(self):
    #    """
    #    Here we can pre compute the contact force.
    #    The simple way is that: we can compute delta linear momentum and delta angular momentum,
    #    and divide the total force on different contacts.
    #    """
    #    pass

    def convert_for_samcon(self, root_pos: torch.Tensor, joint_rot: torch.Tensor, pd_target: torch.Tensor, save_folder: Optional[str] = None):
        """
        Save the optimized result, as samcon format..
        """
        with torch.no_grad():
            args: Namespace = self.args
            if save_folder is None or len(save_folder) == 0:
                save_folder = args.opt_result_sub_dir

            if comm_rank == 0:
                if not os.path.exists(save_folder):  # create sub dir
                    os.makedirs(save_folder)

            # here we need also insert the end sites,,, or we may need to resort the joint order as std human
            ode_joint_rot = np.ascontiguousarray(DiffQuat.vec6d_to_quat(joint_rot).numpy()[:, self.mocap_import.character_to_bvh])
            ret_motion: MotionData = self.to_bvh.forward_kinematics(root_pos.numpy(), ode_joint_rot[:, 0], ode_joint_rot[:, 1:])
            ret_motion: MotionData = self.to_bvh.insert_end_site(ret_motion)
            bvh_fname: str = os.path.join(save_folder, "eval-mocap-predict.bvh")
            BVHLoader.save(ret_motion, bvh_fname)
            print(f"save bvh file to {bvh_fname}")

            # we should also output the inverse dynamics as bvh format
            if pd_target is not None:
                invdyn_quat: np.ndarray = DiffQuat.vec6d_to_quat(pd_target.detach()).numpy()
                invdyn_motion: MotionData = self.to_bvh.forward_kinematics(root_pos.numpy(), ode_joint_rot[:, 0], invdyn_quat)
                invdyn_fname: str = os.path.join(save_folder, "invdyn-output.bvh")
                BVHLoader.save(invdyn_motion, invdyn_fname)
            else:
                invdyn_quat = None

            # we need also need to save the initial solution
            BVHLoader.save(self.raw_motion.sub_sequence(args.index_t, args.index_t + args.width),
                os.path.join(save_folder, "network-init-pred.bvh"))

            # we need also save the initial invdyn solution
            init_invdyn_quat: np.ndarray = DiffQuat.vec6d_to_quat(self.init_pd_target_vec6d).numpy()
            init_invdyn_motion: MotionData = self.to_bvh.forward_kinematics(root_pos.numpy(), ode_joint_rot[:, 0],
                init_invdyn_quat[args.index_t: args.index_t + args.width])
            BVHLoader.save(init_invdyn_motion, os.path.join(save_folder, "network-init-pd-target.bvh"))

            # here we should use the optimized contact label
            optimized_label: np.ndarray = self.convert_contact_mess_to_ndarray(args.index_t, args.index_t + args.width)
            result: Dict[str, Any] = {
                "pred_motion": "eval-mocap-predict.bvh",
                "invdyn_target": invdyn_quat,
                "camera_param": self.camera_param,
                "pos2d": self.torch_joint_pos_2d[args.index_t: args.index_t + args.width].detach().numpy(),
                "confidence": self.confidence[args.index_t: args.index_t + args.width],  # predict
                "pred_contact_label": optimized_label,  # actually, the contact label is unused in Samcon algorithm
                "original_contact_label": self.contact_label[args.index_t: args.index_t + args.width]
            }
            # we should also save the optimized contact label...
            if self.extract_contact_force is not None:
                result["extract_contact_force"] = self.extract_contact_force

            save_bin_fname: str = os.path.join(save_folder, "network-output.bin")
            with open(save_bin_fname, "wb") as fout:
                pickle.dump(result, fout)
            print(f"Dump the result for samcon at {save_bin_fname}", flush=True)

    def convert_for_samcon_main(self, save_folder: Optional[str] = None):
        self.convert_for_samcon(self.root_pos_param, self.joint_vec6d_param, self.pd_target_param, save_folder)

    def convert_for_samcon_parallel(self, save_folder: Optional[str] = None):
        self.convert_for_samcon(self.main_root_pos_param, self.main_joint_vec6d_param, self.main_pd_target_param, save_folder)

    def init_parameter(self, start: int, width: int):
        piece = slice(start, start + width)
        self.root_pos_param: nn.Parameter = nn.Parameter(self.init_root_pos[piece].clone())  # (width, 3)
        self.joint_vec6d_param: nn.Parameter = nn.Parameter(self.init_joint_vec6d[piece].clone())  # (width, num joint, 3, 2)
        self.pd_target_param: nn.Parameter = nn.Parameter(self.init_pd_target_vec6d[piece].clone())  # (width, num joint, 3, 2)
        opt_list: List[nn.Parameter] = [self.root_pos_param, self.joint_vec6d_param, self.pd_target_param]
        if self.args.use_cio:
            # here we should also optimize the contact label.
            self.contact_label_param: nn.Parameter = nn.Parameter(torch.as_tensor(self.contact_label))
            opt_list.append(self.contact_label_param)

        self.optim: AdamW = AdamW(opt_list, lr=self.args.lr)
        self.scheduler: Optional[StepLR] = StepLR(self.optim, self.args.lr_decay_epoch, self.args.lr_decay_ratio)

    def build_optim(self) -> AdamW:
        optim_list: List[nn.Parameter] = []
        if self.root_pos_param is not None:
            optim_list.append(self.root_pos_param)
        if self.joint_vec6d_param is not None:
            optim_list.append(self.joint_vec6d_param)
        if self.pd_target_param is not None:
            optim_list.append(self.pd_target_param)
        if self.body_velo_param is not None:
            optim_list.append(self.body_velo_param)
        if self.body_angvel_param is not None:
            optim_list.append(self.body_angvel_param)
        optim: AdamW = AdamW(optim_list, lr=self.args.lr)
        return optim

    def build_initial_solution(self):
        self.init_root_pos: torch.Tensor = torch.from_numpy(self.motion.joint_position[:, 0]).contiguous()  # (tot frame, 3)
        # convert initial character to pytorch tensor
        self.init_body_pos: torch.Tensor = torch.from_numpy(self.set_tar.target.character_body.pos)  # (tot frame, num body, 3)
        self.init_body_quat: torch.Tensor = torch.from_numpy(self.set_tar.target.character_body.quat)  # (tot frame, num body, 4)
        self.init_body_mat: torch.Tensor = torch.from_numpy(self.set_tar.target.character_body.rot_mat)  # (tot frame, num body, 3, 3)
        self.init_body_velo: torch.Tensor = torch.from_numpy(self.set_tar.target.character_body.linvel)  # (tot frame, num body, 3)
        self.init_body_omega: torch.Tensor = torch.from_numpy(self.set_tar.target.character_body.angvel)  # (tot frame, num body, 3)

        self.init_root_velo: torch.Tensor = self.init_body_velo[:, 0].clone()
        self.init_pd_target_vec6d: torch.Tensor = DiffQuat.quat_to_vec6d(self.torch_target_quat)  # (tot frame, num joint, 3, 2)
        self.init_joint_angvel: torch.Tensor = torch.from_numpy(self.motion.compute_rotational_speed(False))

        # Note: the initial joint index and body index doesn't match.
        joint_init_quat: torch.Tensor = torch.as_tensor(self.motion.joint_rotation)  # (tot frame, num joint, 4)
        self.init_joint_quat: torch.Tensor = joint_init_quat.clone()
        self.init_joint_vec6d: torch.Tensor = DiffQuat.quat_to_vec6d(joint_init_quat)  # (tot frame, num joint, 3, 2)

    def save_result(self, save_filename: Optional[str] = None) -> Dict[str, Any]:
        result_dict: Dict[str, Any] = {
            "root_pos_param": self.root_pos_param.detach() if self.root_pos_param is not None else None,
            "joint_vec6d_param": self.joint_vec6d_param.detach() if self.joint_vec6d_param is not None else None,
            "pd_target_param": self.pd_target_param.detach() if self.pd_target_param is not None else None,
            "optim_state": self.optim.state_dict() if self.optim is not None else None,
            "epoch": self.epoch,
            "global_step": self.global_step
        }

        if self.body_velo_param is not None:
            result_dict["body_velo_param"] = self.body_velo_param.detach()
        if self.body_angvel_param is not None:
            result_dict["body_angvel_param"] = self.body_angvel_param.detach()

        if self.contact_label_param is not None:
            result_dict["contact_label_param"] = self.contact_label_param.detach()
        if self.contact_force_param is not None:
            result_dict["contact_force_param"] = self.contact_force_param.detach()
        if self.contact_torque_param is not None:
            result_dict["contact_torque_param"] = self.contact_torque_param.detach()

        if save_filename is not None:
            torch.save(result_dict, save_filename)

        return result_dict

    def load_result(self, save_result: Union[str, Dict[str, Any]], start: Optional[int] = None, end: Optional[int] = None):
        if isinstance(save_result, str) and os.path.exists(save_result):
            saved_result_: Dict[str, Any] = torch.load(save_result)
            print(f"load result from {save_result}")
        else:
            saved_result_: Dict[str, Any] = save_result

        piece = slice(start, end)

        root_pos_param: Optional[torch.Tensor] = saved_result_.get("root_pos_param")
        if root_pos_param is not None:
            if self.root_pos_param is not None:
                self.root_pos_param.data = root_pos_param[piece].clone()
            else:
                self.root_pos_param = nn.Parameter(root_pos_param[piece].clone())

        joint_vec6d_param: Optional[torch.Tensor] = saved_result_.get("joint_vec6d_param")
        if joint_vec6d_param is not None:
            if self.joint_vec6d_param is not None:
                self.joint_vec6d_param.data = joint_vec6d_param[piece].clone()
            else:
                self.joint_vec6d_param = nn.Parameter(joint_vec6d_param[piece].clone())

        pd_target_param: Optional[torch.Tensor] = saved_result_.get("pd_target_param")
        if pd_target_param is not None:
            if self.pd_target_param is not None:
                self.pd_target_param.data = pd_target_param[piece].clone()
            else:
                self.pd_target_param = nn.Parameter(pd_target_param[piece].clone())

        body_velo_param: Optional[torch.Tensor] = saved_result_["body_velo_param"]
        if body_velo_param is not None:
            if self.body_velo_param is not None:
                self.body_velo_param.data = body_velo_param[piece].clone()
            else:
                self.body_velo_param = nn.Parameter(body_velo_param[piece].clone())

        body_angvel_param: Optional[torch.Tensor] = saved_result_["body_angvel_param"]
        if body_angvel_param is not None:
            if self.body_angvel_param is not None:
                self.body_angvel_param.data = body_angvel_param[piece].clone()
            else:
                self.body_angvel_param = nn.Parameter(body_angvel_param[piece].clone())

        optim_state = saved_result_.get("optim_state")
        if optim_state is not None and self.optim is not None:
            self.optim.load_state_dict(optim_state)

        self.global_step = saved_result_["global_step"]
        self.epoch: int = saved_result_["epoch"]
        if "contact_label_param" in saved_result_:
            if self.contact_label_param is not None:
                self.contact_label_param.data = saved_result_["contact_label_param"][piece].clone()
            else:
                self.contact_label_param - nn.Parameter(saved_result_["contact_label_param"][piece].clone())

        if "contact_force_param" in saved_result_:
            if self.contact_force_param is not None:
                self.contact_force_param.data = saved_result_["contact_force_param"][piece].clone()
            else:
                self.contact_force_param = nn.Parameter(saved_result_["contact_force_param"][piece].clone())

        if "contact_torque_param" in saved_result_:
            if self.contact_torque_param is not None:
                self.contact_torque_param.data = saved_result_["contact_torque_param"][piece].clone()
            else:
                self.contact_torque_param = nn.Parameter(saved_result_["contact_torque_param"][piece].clone())

    def compute_momentum(
        self,
        body_position: torch.Tensor,
        body_rot_matrix: torch.Tensor,
        body_linear_velo: torch.Tensor,
        body_angular_velo: torch.Tensor,
        global_com: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute momentum and angular momentum with CoM
        return shape: (num frame, 3)
        """
        # reference to Karen Liu: A Quick Tutorial on Multibody Dynamics
        tot_mass_inv: torch.Tensor = 1.0 / self.tot_mass
        if global_com is None:
            global_com: torch.Tensor = tot_mass_inv * torch.sum(self.body_mass.view(1, -1, 1) * body_position, dim=1)

        num_frames: int = body_position.shape[0]
        num_body: int = self.num_body

        dcm: torch.Tensor = body_rot_matrix
        inertia: torch.Tensor = self.body_inertia.view(1, num_body, 3, 3)
        inertia: torch.Tensor = dcm @ inertia @ torch.transpose(dcm, -1, -2).contiguous()  # (num_frames, num_body, 3, 3)
        angular_momentum: torch.Tensor = (inertia @ body_angular_velo[..., None]).view(num_frames, num_body, 3)

        com_to_body: torch.Tensor = body_position - global_com.view(num_frames, 1, 3)  # (frame, num body, 3)
        body_linear_momentum: torch.Tensor = body_linear_velo * self.body_mass.view(1, -1, 1)  # (frame, num body, 3)
        angular_momentum: torch.Tensor = angular_momentum + torch.cross(com_to_body, body_linear_momentum, dim=-1)  # (frame, num body, 3)
        angular_momentum: torch.Tensor = torch.sum(angular_momentum, dim=1)  # (frame, 3)

        # compute linear momentum of center of mass
        linear_momentum: torch.Tensor = torch.sum(body_linear_momentum, dim=1)
        angular_momentum: torch.Tensor = angular_momentum  # (frame, 3)

        return linear_momentum, angular_momentum

    def fk_forward_process(self, start: Optional[int] = None, width: Optional[int] = None):
        if start is None:
            start: int = 0
        if width is None:
            width: int = self.joint_vec6d_param.shape[0]
        fps: int = self.scene.sim_fps
        joint_vec6d: torch.Tensor = self.joint_vec6d_param[start: start + width]
        joint_rot_mat: torch.Tensor = DiffQuat.vec6d_to_matrix(joint_vec6d)  # (width, num joint, 3, 3) here the vector 6d has been normalized.
        joint_6d: torch.Tensor = joint_rot_mat[..., :2].contiguous()  # (width, num joint, 3, 2)
        joint_quat: torch.Tensor = DiffQuat.quat_from_matrix(joint_rot_mat)  # (width, num joint, 4)

        # forward kinematics to compute joint global orientation and position
        self.diff_motion._root_translation = self.root_pos_param[start: start + width]  # (width, 3)
        self.diff_motion._joint_rotation = joint_quat  # (width, num joint, 4)
        self.diff_motion._num_frames = joint_vec6d.shape[0]
        self.diff_motion.recompute_joint_global_info()

        # here we should compute body position by joint position..
        kine_body_pos, kine_body_quat = self.mocap_import.import_mocap_base_batch(self.diff_motion)  # (width, num body, 3), (width, num body, 4)
        # Note: here we need to compute body linear velocity by given local angular velocity...
        if self.args.also_optimize_velo:
            # ah, how to compute the body velocity by local joint angular velocity...?
            # convert angular velocity from reduced coordinate to maximal coordinate is difficult.
            # we can optimize the body velocity in maximal coordinate directly..
            if self.body_velo_param is not None:
                kine_body_vel: torch.Tensor = self.body_velo_param  # (width, num body, 3)
                kine_body_ang: torch.Tensor = self.body_angvel_param  # (width, num body, 3)
            else:
                kine_body_vel: Optional[torch.Tensor] = None
                kine_body_ang: Optional[torch.Tensor] = None
                # raise NotImplementedError("convert joint angular velocity from reduced coordinate to maximal coordinate is difficult. I have NO time to do this..")
        else:
            # Note: we can insert the velocity at the 0-th frame..
            kine_body_vel: torch.Tensor = torch.diff(kine_body_pos, dim=0) * fps # (width - 1, num body, 3)
            kine_body_ang: torch.Tensor = self.diff_motion.compute_angvel_frag(kine_body_quat, fps)  # (width - 1, num body, 3)
            kine_body_vel: torch.Tensor = torch.cat([kine_body_vel[0, None], kine_body_vel], dim=0)  # (width, num body, 3)
            kine_body_ang: torch.Tensor = torch.cat([kine_body_ang[0, None], kine_body_ang], dim=0)  # (width, num body, 3)

        kine_body_rotmat: torch.Tensor = DiffQuat.quat_to_matrix(kine_body_quat)  # (width, num body, 3)

        # should we also optimize the control torque..?
        # I think we can optimize after several epoches.
        # convert the pd target from 6d vector to quaternion.
        if self.pd_target_param is not None:
            pd_target_matrix: torch.Tensor = DiffQuat.vec6d_to_matrix(self.pd_target_param[start: start + width])
            pd_target_vec6d: torch.Tensor = pd_target_matrix[..., :2].contiguous()
            pd_target_quat: torch.Tensor = DiffQuat.quat_from_matrix(pd_target_matrix)
        else:
            pd_target_vec6d: Optional[torch.Tensor] = None
            pd_target_quat: Optional[torch.Tensor] = None

        return joint_6d, kine_body_pos, kine_body_rotmat, kine_body_quat, kine_body_vel, kine_body_ang, pd_target_vec6d, pd_target_quat


    def simu_forward_process_cio(
        self,
        index_t: int,
        width: int,
        epoch: int,
        kine_body_pos: torch.Tensor,  # (width, num body, 3)
        kine_body_rotmat: torch.Tensor,  # (width, num body, 3, 3)
        kine_body_quat: torch.Tensor,  # (width, num body, 4)
        kine_body_vel: torch.Tensor,  # (width, num body, 3)
        kine_body_ang: torch.Tensor,  # (width, num body, 3)
        pd_target_quat: torch.Tensor,  # (width, num joint, 4)
        contact_local_pos: torch.Tensor,  # (width, num body, 3), this doesn't require gradient.
        contact_label: torch.Tensor,  # (width, num body)
        contact_force: torch.Tensor,
        contact_torque: torch.Tensor,
    ):
        """
        Optimize by cio..
        1. common phys loss.
        2. CIO loss:
        - When contact label is large, the xz component of contact point should be close to 0.
        - when contact label is large, the y component of contact point should be close to 0.
        - When contact label is small, the contact force should be small.
        - Optional: The total control signal should be small (regular term)
        """
        # assert self.contact_force_param is not None and self.contact_torque_param is not None and self.contact_label_param is not None
        coef: float = float(width - 1) / (self.args.width - 1)
        fps: int = self.scene.sim_fps
        args: Namespace = self.args
        sim_body_pos_list, sim_body_mat_list, sim_body_velo_list, sim_body_omega_list = [], [], [], []
        body_c_id: np.ndarray = self.character.body_info.body_c_id
        for t in range(0, width - 1):
            diff_frame_import_from_tensor(
                self.curr_frame, kine_body_pos[t], kine_body_vel[t], kine_body_rotmat[t],
                kine_body_quat[t], kine_body_ang[t], self.scene.world, body_c_id)
            # 1. compute the linear velocity of each contact point..
            if False:
                contact_info = self.extractor.compute_contact_by_sub_body(
                    kine_body_pos.detach().numpy(),
                    kine_body_quat.detach().numpy(),
                    np.arange(0, self.num_body, dtype=np.int32)
                )
                diff_contact: DiffContactInfo = self.extractor.convert_to_diff_contact_single(*contact_info)
                self.diff_world.add_hack_local_contact(diff_contact)
                self.diff_world.curr_frame.remove_contact()
                contact_pos: torch.Tensor = diff_contact.contact_pos
                contact_velo: torch.Tensor = diff_contact.contact_velo

            # emm..the contact position is not differentiable..
            # the linear velocity of each contact point requires gradient..
            # or I can also compute loss by body velocity..(This will not work, for walking motion..)
            # we should add contact force and torque here..
            self.diff_world.body_force = contact_force[t]
            self.diff_world.body_tau = contact_torque[t]
            # add control signal here
            self.diff_world.curr_frame.stable_pd_control_fast(pd_target_quat[t])
            self.diff_world.step(do_contact=False)  # forward simulation using diff ode, without collision detection

            # save simulation state
            sim_body_pos_list.append(self.curr_frame.body_pos.view(1, self.num_body, 3))
            sim_body_mat_list.append(self.curr_frame.body_rot.view(1, self.num_body, 3, 3))
            sim_body_velo_list.append(self.curr_frame.body_velo.view(1, self.num_body, 3))
            sim_body_omega_list.append(self.curr_frame.body_omega.view(1, self.num_body, 3))

        sim_body_pos: torch.Tensor = torch.cat(sim_body_pos_list, dim=0)  # (width - 1, num body, 3)
        sim_body_mat: torch.Tensor = torch.cat(sim_body_mat_list, dim=0)  # (width - 1, num body, 3, 3)
        sim_body_velo: torch.Tensor = torch.cat(sim_body_omega_list, dim=0) # (width - 1, num body, 3)
        sim_body_omega: torch.Tensor = torch.cat(sim_body_omega_list, dim=0)  # (width - 1, num body, 3)

        # estimate position for each end effector..
        global_offset: torch.Tensor = kine_body_rotmat @ contact_local_pos
        contact_pos: torch.Tensor = kine_body_pos + global_offset  # TODO: + or - ?
        # velocity of contact point: body linear velocity + angular velocity \times global offset
        contact_velo: torch.Tensor = kine_body_vel + torch.cross(kine_body_ang, global_offset)
        # evaluate CIO loss
        # 1. When contact label is large, the xz component of contact point should be close to 0.
        cio_velo_loss: torch.Tensor = (args.w_cio_contact_velo) * torch.mean(contact_velo[..., [0, 2]] ** 2 * contact_label[..., None])
        # 2. when contact label is large, the y component of contact point should be close to 0.
        cio_pos_loss: torch.Tensor = (args.w_cio_contact_pos) * torch.mean(contact_pos[..., 1] ** 2 * contact_label)
        # 3. When contact label is small, the contact force should be small.
        inv_contact_label: torch.Tensor = 1.0 / (contact_label + 1e-4)
        cio_force_loss: torch.Tensor = torch.mean(contact_force[..., 1] ** 2 * inv_contact_label)
        # 4. each contact force should match friction cone limit.

        # compute phys loss here..
        pos_loss: torch.Tensor = (args.w_loss_pos) * self.phys_loss_func(sim_body_pos, kine_body_pos[1:]) if args.w_loss_pos > 0 else torch.as_tensor(0.0, dtype=torch.float64)
        rot_loss: torch.Tensor = (args.w_loss_rot) * self.phys_loss_func(sim_body_mat, kine_body_rotmat[1:]) if args.w_loss_rot > 0 else torch.as_tensor(0.0, dtype=torch.float64)

        # here, we should compute kinematic linear and angular velocity..
        # in ode simulation, x_{t+1} = x_{t} + h * v_{t+1}
        linvel_loss: torch.Tensor = (args.w_loss_velo) * self.phys_loss_func(sim_body_velo, kine_body_vel[1:]) if args.w_loss_velo > 0 else torch.as_tensor(0.0, dtype=torch.float64)
        angvel_loss: torch.Tensor = (args.w_loss_angvel) * self.phys_loss_func(sim_body_omega, kine_body_ang[1:]) if args.w_loss_angvel > 0 else torch.as_tensor(0.0, dtype=torch.float64)

        # Here we should conpute CIO loss..we can compute batchly.
        # I don't think original CIO loss will work, because contact happens at whole body..

        return pos_loss, rot_loss, linvel_loss, angvel_loss, cio_velo_loss, cio_pos_loss, cio_force_loss

    def simu_forward_process(
        self,
        index_t: int, width: int, epoch: int,
        kine_body_pos: torch.Tensor,
        kine_body_rotmat: torch.Tensor,
        kine_body_quat: torch.Tensor,
        kine_body_vel: torch.Tensor,
        kine_body_ang: torch.Tensor,
        pd_target_quat: torch.Tensor,
        contact_info_list: Optional[List[DiffContactInfo]] = None,
        do_evaluate: bool = False
    ):
        """
        Note:
        """
        # do forward simulation by a loop.
        # coef: float = float(width - 1) / (self.args.width - 1)
        fps: int = self.scene.sim_fps
        args: Namespace = self.args
        loss_contact_h: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)
        loss_contact_xz: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)
        # contact_force: torch.Tensor = torch.zeros((width - 1, 3), dtype=torch.float64)
        # contact_force: List[Optional[torch.Tensor]] = []
        # contact_torque: List[Optional[torch.Tensor]] = []
        # avg_contact_force: List[torch.Tensor] = []
        # avg_contact_torque: List[torch.Tensor] = []
        # dummy_vec3: torch.Tensor = torch.zeros((1, 3), dtype=torch.float64)
        if do_evaluate:
            eval_list = []
        else:
            eval_list = None

        body_c_id: np.ndarray = self.character.body_info.body_c_id
        total_pos_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)
        total_rot_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)
        total_velo_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)
        total_angvel_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        # here we should consider multiple forward step..
        num_forward: int = args.phys_loss_num_forward
        for t in range(0, width - num_forward):
            diff_frame_import_from_tensor(
                self.curr_frame, kine_body_pos[t], kine_body_vel[t], kine_body_rotmat[t],
                kine_body_quat[t], kine_body_ang[t], self.scene.world, body_c_id)
            if self.gt_tar_set is not None and comm_size == 1:  # for debug visualize..
                self.gt_tar_set.set_character_byframe(index_t + t, self.ref_character)
            # here we should add hacked contact, and initial inverse dynamics result..
            # At the begining of optimization, we can do collision detection here..
            # That is, both the hacked contact and real contact exists at the begining of optimization..
            sim_body_pos_list, sim_body_mat_list, sim_body_velo_list, sim_body_omega_list = [], [], [], []
            for forward_index in range(t, t + num_forward):
                if args.use_hack_contact:
                    if contact_info_list is not None:
                        hack_contact: DiffContactInfo = contact_info_list[forward_index]
                    else:
                        if not args.collision_detection_each_epoch:
                            hack_contact: DiffContactInfo = self.contact_info_list[index_t + forward_index]
                        else:
                            hack_contact = self.extractor.compute_contact_by_sub_body(
                                self.curr_frame.body_pos.detach().numpy().reshape((-1, 3)),
                                self.curr_frame.body_quat.detach().numpy().reshape((-1, 4)),
                                self.contact_mess[forward_index],
                            )
                            hack_contact: DiffContactInfo = self.extractor.convert_to_diff_contact_single(*hack_contact)
                    self.diff_world.add_hack_local_contact(hack_contact)

                # add the control signal.
                self.curr_frame.stable_pd_control_fast(pd_target_quat[forward_index])
                diff_contact: Optional[DiffContactInfo] = self.curr_frame.joint_info.diff_contact
                # The contact height should be zero
                # use_contact_loss: bool = diff_contact is not None and len(diff_contact.contact_pos) > 0 and False

                # if use_contact_loss and False:
                #     if args.w_contact_height > 0:
                #         contact_loss_t: torch.Tensor = (coef * args.w_contact_height) * torch.mean(diff_contact.contact_pos[..., 1, 0] ** 2)
                #         loss_contact_h += contact_loss_t
                #     else:
                #         contact_loss_t: torch.Tensor = (coef) * torch.mean(diff_contact.contact_pos[..., 1, 0].detach() ** 2)
                #         loss_contact_h += contact_loss_t
                #     if args.w_contact_horizontal_velo > 0:
                #         curr_contact_xz: torch.Tensor = (coef * args.w_contact_horizontal_velo) * torch.mean(diff_contact.contact_velo ** 2)
                #         loss_contact_xz += curr_contact_xz
                #     else:
                #         loss_contact_xz += coef * torch.mean(diff_contact.contact_velo[:, [0, 2]].detach() ** 2)

                self.diff_world.step(do_contact=True)  # forward simulation using diff ode
                # self.curr_frame.project_contact_on_floor(hack_contact, self.body_min_contact_h)
                # save the contact force here..
                if do_evaluate:  # body force and body torque.
                    eval_list.append((diff_contact.body0_force, diff_contact.body0_torque))

                # if diff_contact is not None and len(diff_contact) > 0:
                #     contact_force.append(diff_contact.body0_force)
                #     contact_torque.append(diff_contact.body0_torque)
                #     avg_contact_force.append(torch.mean(diff_contact.body0_force, dim=0).view(1, 3))
                #     avg_contact_torque.append(torch.mean(diff_contact.body0_torque, dim=0).view(1, 3))
                # else:
                #     contact_force.append(None)
                #     contact_torque.append(None)
                #     avg_contact_force.append(dummy_vec3)
                #     avg_contact_torque.append(dummy_vec3)

                sim_body_pos_list.append(self.curr_frame.body_pos.view(1, self.num_body, 3))
                sim_body_mat_list.append(self.curr_frame.body_rot.view(1, self.num_body, 3, 3))
                sim_body_velo_list.append(self.curr_frame.body_velo.view(1, self.num_body, 3))
                sim_body_omega_list.append(self.curr_frame.body_omega.view(1, self.num_body, 3))

            sim_body_pos: torch.Tensor = torch.cat(sim_body_pos_list, dim=0)  # (width - 1, num body, 3)
            sim_body_mat: torch.Tensor = torch.cat(sim_body_mat_list, dim=0)  # (width - 1, num body, 3, 3)
            sim_body_velo: torch.Tensor = torch.cat(sim_body_omega_list, dim=0) # (width - 1, num body, 3)
            sim_body_omega: torch.Tensor = torch.cat(sim_body_omega_list, dim=0)  # (width - 1, num body, 3)
            # The simulation result should be close to kinematic result
            if args.w_loss_pos > 0:
                pos_loss: torch.Tensor = (args.w_loss_pos) * self.phys_loss_func(sim_body_pos, kine_body_pos[t + 1: t + 1 + num_forward])
            else:
                pos_loss = torch.as_tensor(0.0, dtype=torch.float64)
            if args.w_loss_rot > 0:
                rot_loss: torch.Tensor = (args.w_loss_rot) * self.phys_loss_func(sim_body_mat, kine_body_rotmat[t + 1: t + 1 + num_forward])
            else:
                rot_loss = torch.as_tensor(0.0, dtype=torch.float64)

            if args.w_loss_velo > 0:
                linvel_loss: torch.Tensor = (args.w_loss_velo) * self.phys_loss_func(sim_body_velo, kine_body_vel[t + 1: t + 1 + num_forward])
            else:
                linvel_loss = torch.as_tensor(0.0, dtype=torch.float64)
            if args.w_loss_angvel > 0:
                angvel_loss: torch.Tensor = (args.w_loss_angvel) * self.phys_loss_func(sim_body_omega, kine_body_ang[t + 1: t + 1 + num_forward])
            else:
                angvel_loss = torch.as_tensor(0.0, dtype=torch.float64)

            total_pos_loss += pos_loss
            total_rot_loss += rot_loss
            total_velo_loss += linvel_loss
            total_angvel_loss += angvel_loss

        """
        evaluate virtual force and torque added on the root.
        We can compute the difference of momentum/ angular momentum between simulation result and kinematic result
        compute kinematic momentum and angular momentum
        if we compute forward path in sub-workers, maybe we should also evaluate the virtual force.
        actually, here we doesn't require gradient (only for evaluate).
        (Maybe we need to compare with other workers, such as physcap..)
        """
        if False:
            gravity: torch.Tensor = torch.from_numpy(self.character.body_info.sum_mass * self.scene.gravity_numpy).view(1, 3)
            # sim_lin_momentum, sim_ang_momentum = self.compute_momentum(sim_body_pos, sim_body_mat, sim_body_velo, sim_body_omega)
            kine_lin_momentum, kine_ang_momentum = self.compute_momentum(kine_body_pos, kine_body_rotmat, kine_body_vel, kine_body_ang)
            delta_kine_lin_mom = fps * torch.diff(kine_lin_momentum, dim=0)
            delta_kine_ang_mom = fps * torch.diff(kine_ang_momentum, dim=0)
            # note: we should subtract external contact force and torque here..
            loss_lin_momentum: torch.Tensor = \
                coef * torch.mean(torch.linalg.norm(delta_kine_lin_mom - avg_contact_force[1:] - gravity, dim=-1))
            loss_ang_momentum: torch.Tensor = coef * torch.mean(torch.linalg.norm(delta_kine_ang_mom - avg_contact_torque[1:], dim=-1))
            # print(comm_rank, coef, loss_lin_momentum, loss_ang_momentum)
        else:
            loss_lin_momentum: torch.Tensor = torch.as_tensor(0)
            loss_ang_momentum: torch.Tensor = torch.as_tensor(0)

        # loss between simulation result and kinematics motion at time t + 1  (coef * args.w_loss_pos)
        # here we should set loss function by args

        # we can compute hinge angle limit loss batchly, rather than compute by each frame.
        # for fast step..
        total_pos_loss /= width - num_forward
        total_rot_loss /= width - num_forward
        total_velo_loss /= width - num_forward
        total_angvel_loss /= width - num_forward

        return (
            total_pos_loss,
            total_rot_loss,
            total_velo_loss,
            total_angvel_loss,
            loss_contact_h,
            loss_contact_xz,
            loss_lin_momentum,
            loss_ang_momentum,
            eval_list
        )

    def contact_loss_kinematic(
        self,
        index_t: int,
        width: int,
        epoch: int,
        kine_body_pos: torch.Tensor,
        kine_body_velo: Optional[torch.Tensor] = None
    ):
        args: Namespace = self.args
        # here all of body height should >= 0
        # here we should use the intersection... contact label >= eps and initial height < eps
        body_min_height: torch.Tensor = kine_body_pos[..., 1] - self.body_min_contact_h.view(1, -1)
        contact_label: torch.Tensor = torch.from_numpy(self.contact_label[index_t: index_t + width]).clone()
        try:
            contact_label[body_min_height >= self.args.add_contact_height_eps] = 0
        except Exception as err:
            print(index_t, width, epoch, contact_label.shape, body_min_height.shape)
            raise err
        min_height_index = torch.where(contact_label > self.args.contact_eps)
        # body_min_height: torch.Tensor = self.init_body_pos[index_t: index_t + width, :, 1] - self.body_min_contact_h.view(1, -1)
        # min_height_index = torch.where(body_min_height < eps1)
        # we should introduce a smooth term here..
        # And we should not use velocity parameter here. we should use the delta of position
        if len(min_height_index[0]) > 0 and not self.at_contact_plan:
            # sub_height: torch.Tensor = body_min_height[min_height_index]
            # height_weight: torch.Tensor = soft_height_weight_func(sub_height, eps1)
            # velo_weight: torch.Tensor = soft_velo_weight_func(sub_height, eps1)
            height_weight = torch.ones_like(body_min_height[min_height_index])
            difference_velo: torch.Tensor = torch.diff(kine_body_pos, dim=0)
            difference_velo: torch.Tensor = torch.cat([difference_velo[:1], difference_velo], dim=0)

            # here we need to judge the weight. for pure kinematic optim, the weight is large.
            # for dyn optimization, for stablity, the weight is small.
            if epoch < args.optimize_root_epoch:
                contact_weight: float = args.w_contact_kine
            else:
                contact_weight: float = args.w_contact_dyn

            loss_contact_h: torch.Tensor = contact_weight * torch.mean(height_weight * body_min_height[min_height_index] ** 2)

            loss_contact_xz = contact_weight * torch.mean(height_weight[:, None] * difference_velo[min_height_index] ** 2)
            if kine_body_velo is not None:
                loss_contact_xz += contact_weight * torch.mean(height_weight[:, None] * kine_body_velo[min_height_index] ** 2)
            # print(f"compute contact xz loss")
        else:
            loss_contact_h = loss_contact_xz = torch.as_tensor(0.0, dtype=torch.float64)

        return loss_contact_h, loss_contact_xz

    def contact_loss_kinematic(self, index_t: int, width: int, kine_body_pos: torch.Tensor, kine_body_velo: Optional[torch.Tensor] = None):
        # here all of body height should >= 0
        # here we should use the intersection... contact label >= eps and initial height < eps
        body_min_height: torch.Tensor = kine_body_pos[..., 1] - self.body_min_contact_h.view(1, -1)
        contact_label = torch.from_numpy(self.contact_label[index_t: index_t + width]).clone()
        contact_label[body_min_height > 1e-1] = 0
        min_height_index = torch.where(contact_label > self.args.contact_eps)
        # body_min_height: torch.Tensor = self.init_body_pos[index_t: index_t + width, :, 1] - self.body_min_contact_h.view(1, -1)
        # min_height_index = torch.where(body_min_height < eps1)
        # we should introduce a smooth term here..
        # And we should not use velocity parameter here. we should use the delta of position
        if len(min_height_index[0]) > 0:
            # sub_height: torch.Tensor = body_min_height[min_height_index]
            # height_weight: torch.Tensor = soft_height_weight_func(sub_height, eps1)
            # velo_weight: torch.Tensor = soft_velo_weight_func(sub_height, eps1)
            height_weight = torch.ones_like(body_min_height[min_height_index])
            difference_velo: torch.Tensor = torch.diff(kine_body_pos, dim=0)
            difference_velo: torch.Tensor = torch.cat([difference_velo[:1], difference_velo], dim=0)
            loss_contact_xz = 5e3 * torch.sum(height_weight[:, None] * difference_velo[..., [0, 1, 2]][min_height_index] ** 2)
            if kine_body_velo is not None:
                loss_contact_xz += 5e3 * torch.sum(height_weight[:, None] * kine_body_velo[..., [0, 1, 2]][min_height_index] ** 2)

            loss_contact_h: torch.Tensor = 10 * torch.sum(height_weight * body_min_height[min_height_index] ** 2)
        else:
            loss_contact_h = loss_contact_xz = torch.as_tensor(0.0, dtype=torch.float64)

        return loss_contact_h, loss_contact_xz

    def forward_process(
        self,
        index_t: int,
        width: int,
        epoch: int,
        contact_info_list: Optional[List[DiffContactInfo]] = None,
        is_evaluate: bool = False
    ):
        # for debug..
        args: Namespace = self.args
        phys_param = save_phys_loss(args)
        w_pd_target_smooth = args.w_pd_target_smooth

        if epoch < args.phys_loss_epoch:
            args = disable_phys_loss(args)
            args.w_pd_target_smooth = 0

        if self.optim is not None:
            self.optim.zero_grad(set_to_none=True)

        joint_6d, kine_body_pos, kine_body_rotmat, kine_body_quat, kine_body_vel, kine_body_ang, pd_target_vec6d, pd_target_quat =\
            self.fk_forward_process()
        do_simu_loss: bool = (args.w_loss_pos > 0) or (args.w_loss_rot > 0) or (args.w_loss_velo > 0) or (args.w_loss_angvel > 0)
        self.eval_force_list = None
        if do_simu_loss:
            pos_loss, rot_loss, linvel_loss, angvel_loss, loss_contact_h, loss_contact_xz, loss_lin_momentum, loss_ang_momentum, eval_force = \
                self.simu_forward_process(
                index_t, width, epoch,
                kine_body_pos, kine_body_rotmat, kine_body_quat,
                kine_body_vel, kine_body_ang, pd_target_quat, contact_info_list,
                is_evaluate)
            if eval_force is not None:
                eval_force.append((eval_force[-1][0].detach().clone(), eval_force[-1][1].detach().clone()))
            self.eval_force_list = eval_force
        else:
            pos_loss = rot_loss = linvel_loss = angvel_loss = torch.as_tensor(0.0, dtype=torch.float64)
            # if epoch < args.optimize_root_epoch:
            #     # maybe we should not compute contact velocity loss when optimize the whole body kinematic motion..
            # else:
            #     loss_contact_h = loss_contact_xz = torch.as_tensor(0.0, dtype=torch.float64)
            # body with contact should not move..
            # in another word, body with minimal height should not move...
            loss_lin_momentum = loss_ang_momentum = torch.as_tensor(0.0, dtype=torch.float64)
        loss_contact_h, loss_contact_xz = self.contact_loss_kinematic(index_t, width, epoch, kine_body_pos, kine_body_vel)

        if index_t == args.index_t:
            close_slice = slice(0, width)
            close_coef: float = float(width) / args.width
            smooth_slice = slice(0, width)
            smooth_coef: float = (width - 1) / (args.width - 1)
        else:
            close_slice = slice(args.phys_loss_num_forward, width)
            close_coef: float = float(width - args.phys_loss_num_forward) / args.width
            smooth_slice = slice(args.phys_loss_num_forward - 1, width)
            smooth_coef: float = (width - args.phys_loss_num_forward + 1) / (args.width - 1)

        loss_result: Dict[str, Union[torch.Tensor, float]] = {}
        if self.gt_mocap is not None:  # Note that we need to divide these motion into several sub slice
            with torch.no_grad():  # evaluate mpjpe loss..
                # global_mpjpe_loss: torch.Tensor = close_coef * torch.mean(torch.sqrt((self.diff_motion.joint_position[close_slice] - self.gt_global_joint[index_t:][close_slice]) ** 2))
                # pred_j3ds, target_j3ds = self.diff_motion.joint_position[close_slice], self.gt_global_joint[index_t:][close_slice]
                # global_mpjpe_loss: torch.Tensor = close_coef * torch.mean(torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1))
                pred_motion = self.diff_motion.export_to_motion_data().sub_sequence(close_slice.start, close_slice.stop, copy=False)
                gt_sub_seq = self.gt_mocap.sub_sequence(index_t + close_slice.start, index_t + close_slice.stop, copy=False)
                facing_mpjpe = calc_motion_mpjpe(pred_motion, gt_sub_seq, True)
                global_mpjpe = calc_motion_mpjpe(pred_motion, gt_sub_seq, False)
                loss_result["facing_mpjpe_loss"] = close_coef * facing_mpjpe
                loss_result["global_mpjpe_loss"] = close_coef * global_mpjpe  # This term is not added to the total loss.
                # here maybe we can also compute MPJPE at local coordinate..? we can remove root translation and rotation here.

                # compute smooth loss (that is, difference for linear velocity between simulation and ground truth).
                # Here should be muliplied by coef for multi core task.
                # Note: here we should consider the velocity index..
                # global_joint_velo: torch.Tensor = torch.diff(self.diff_motion.joint_position, dim=0)
                # physcap_smooth_loss: torch.Tensor = smooth_coef * F.mse_loss(global_joint_velo, self.gt_global_joint_velo[index_t + 1:index_t + width])
                # loss_result["physcap_smooth_loss"] = physcap_smooth_loss

        # the result should be close to initial solution
        if args.w_root_close > 0:
            root_pos_close_loss: torch.Tensor = (args.w_root_close) * self.phys_loss_func(
                self.root_pos_param[close_slice],
                self.init_root_pos[index_t:][close_slice]
            )
        else:
            root_pos_close_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        if args.w_joint_close > 0:
            rot_close_loss: torch.Tensor = (args.w_joint_close) * self.phys_loss_func(
                joint_6d[close_slice],
                self.init_joint_vec6d[index_t:][close_slice]
            )
        else:
            rot_close_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        # the result should be smooth. TODO:
        if args.w_root_smooth > 0:
            smooth_root_pos_loss: torch.Tensor = (args.w_root_smooth) * torch.sum(torch.diff(self.root_pos_param[smooth_slice], dim=0) ** 2)
        else:
            smooth_root_pos_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        if args.w_joint_smooth > 0:
            smooth_rot_loss: torch.Tensor = (args.w_joint_smooth) * torch.sum(torch.diff(joint_6d[smooth_slice], dim=0) ** 2)
        else:
            smooth_rot_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        if args.w_pd_target_smooth > 0 and pd_target_vec6d is not None:
            pd_target_smooth_loss: torch.Tensor = (args.w_pd_target_smooth) * torch.sum(torch.diff(pd_target_vec6d[smooth_slice], dim=0) ** 2)
        else:
            pd_target_smooth_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        # the result should be close to 2d projection result
        # here we should consider confidence..
        # Note: we should build a map from bvh file to std-human joint order, or from bvh file to unified..
        if args.w_loss_2d > 0:
            joint_subset: torch.Tensor = self.diff_motion.joint_position[close_slice, self.character_to_bvh][:, stdhuman_with_root_to_unified_index]
            joint_camera: torch.Tensor = self.camera_torch.world_to_camera(joint_subset)
            joint_2d: torch.Tensor = self.camera_torch.project_to_2d_linear(joint_camera)
            loss_2d: torch.Tensor = (args.w_loss_2d) * \
                torch.sum(self.torch_confidence[index_t:][close_slice, :, None] * (joint_2d - self.torch_joint_pos_2d[index_t:][close_slice]) ** 2)
        else:
            loss_2d: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        if args.w_pd_target_close > 0 and pd_target_vec6d is not None:
            pd_target_close_loss: torch.Tensor = (args.w_pd_target_close) * self.phys_loss_func(pd_target_vec6d[close_slice], self.init_pd_target_vec6d[index_t:][close_slice])
        else:
            pd_target_close_loss: torch.Tensor = torch.as_tensor(0.0, dtype=torch.float64)

        tot_loss: torch.Tensor = pd_target_close_loss +\
            pos_loss + rot_loss + linvel_loss + angvel_loss + root_pos_close_loss + rot_close_loss + \
            smooth_root_pos_loss + smooth_rot_loss + pd_target_smooth_loss + loss_2d + loss_contact_h + loss_contact_xz

        loss_result.update({
            "pd_target_close_loss": pd_target_close_loss,
            "phys_pos_loss": pos_loss,  # loss between phys simu result and kinematic result
            "phys_rot_loss": rot_loss,
            "phys_linvel_loss": linvel_loss,
            "phys_angvel_loss": angvel_loss,
            "root_pos_close_loss": root_pos_close_loss,  # root position should be close to initial solution
            "rot_close_loss": rot_close_loss,  # joint rotation should be close to initial sulution
            "smooth_root_pos_loss": smooth_root_pos_loss,  # the root pos should be smooth
            "smooth_rot_loss": smooth_rot_loss,  # the rotation should be smooth
            "pd_target_smooth_loss": pd_target_smooth_loss,  # the pd target should be smooth
            "loss_2d": loss_2d,  # 2d projection loss
            "contact_h_loss": loss_contact_h,  # contact height loss
            "contact_xz_loss": loss_contact_xz,
            "loss_lin_momentum": loss_lin_momentum,  # actually, this component is not added to the total loss. only for evaluate.
            "loss_ang_momentum": loss_ang_momentum,  # actually, this component is not added to the total loss. only for evaluate.
            "tot_loss": tot_loss
        })

        load_phys_loss(args, *phys_param)
        args.w_pd_target_smooth = w_pd_target_smooth
        return loss_result

    def print_loss(self, epoch: int, ret: Dict[str, float], print_log: bool = False, use_parallel: bool = False):
        if print_log:
            message = str(
                f"epoch = {epoch}, global step = {self.global_step}\n"
                f"lr = {self.optim.param_groups[0]['lr']}"
            ) + ", \n".join([f"{key} = {value.item():.6f}" for key, value in ret.items()])
            print(message)
            logging.info(message)

        # here we should also check the gradient..
        if use_parallel:
            root_pos_param = self.main_root_pos_param
            joint_vec6d_param = self.main_joint_vec6d_param
            pd_target_param = self.main_pd_target_param
            root_vel_param = self.main_root_vel_param
            angvel_param = self.main_angvel_param
        else:
            root_pos_param = self.root_pos_param
            joint_vec6d_param = self.main_joint_vec6d_param
            pd_target_param = self.pd_target_param

        if root_pos_param.grad is not None:
            root_grad_abs: torch.Tensor = root_pos_param.grad.abs()
            root_grad_max: float = root_grad_abs.max().item()
            root_grad_mean: float = root_grad_abs.mean().item()
            joint_grad_abs: torch.Tensor = joint_vec6d_param.grad.abs()
            joint_grad_max: float = joint_grad_abs.max().item()
            joint_grad_mean: float = joint_grad_abs.mean().item()

            if print_log:
                message = str(
                    f"root pos grad max = {root_grad_max:.6f}, "
                    f"grad mean = {root_grad_mean:.6f}, \n"
                    f"joint rot grad max = {joint_grad_max:.6f}, "
                    f"joint rot grad mean = {joint_grad_mean:.6f}, "
                )
                print(message)
                logging.info(message)

            if self.writer is not None:
                self.writer.add_scalar("root grad max", root_grad_max, self.global_step)
                self.writer.add_scalar("root grad mean", root_grad_mean, self.global_step)
                self.writer.add_scalar("joint rot grad max", joint_grad_max, self.global_step)
                self.writer.add_scalar("joint rot grad mean", joint_grad_mean, self.global_step)

        if pd_target_param is not None and pd_target_param.grad is not None:
            pd_target_grad_abs: torch.Tensor = pd_target_param.grad.abs()
            pd_target_grad_max: float = pd_target_grad_abs.max().item()
            pd_target_grad_mean: float = pd_target_grad_abs.mean().item()

            if print_log:
                message = str(
                    f"pd target grad max = {pd_target_grad_max:.6f}, "
                    f"grad mean = {pd_target_grad_mean:.6f}, "
                )
                print(message)
                logging.info(message)

            if self.writer is not None:
                self.writer.add_scalar("pd target grad max", pd_target_grad_max, self.global_step)
                self.writer.add_scalar("pd target grad mean", pd_target_grad_mean, self.global_step)

        if self.writer is not None:
            for key, value in ret.items():
                self.writer.add_scalar(key, value, self.global_step)
            self.writer.add_scalar("lr", self.optim.param_groups[0]['lr'], self.global_step)

        self.global_step += 1

    def closure(self, epoch: int):  # This function is only called in single thread.
        # this method will not be used. we can use OptimizeEachFrameParallel.py directly.
        index_t: int = self.args.index_t
        width: int = self.args.width

        args: Namespace = self.args
        ret = self.forward_process(index_t, width, epoch)
        tot_loss_item: float = ret["tot_loss"].item()

        if tot_loss_item < self.best_loss:
            self.best_param = self.save_result()
            self.best_loss = tot_loss_item

        ret["tot_loss"].backward()
        if args.print_log_info:
            self.print_loss(epoch, ret)

        if args.pos_grad_clip is not None:  # clip the gradient.
            clip_grad_norm_(self.root_pos_param, args.pos_grad_clip)

        if args.rot_grad_clip is not None:
            clip_grad_norm_(self.joint_vec6d_param, args.rot_grad_clip)
            if self.pd_target_param.grad is not None:
                clip_grad_norm_(self.pd_target_param, args.rot_grad_clip)

        return ret

    def test_optimize_cfm_single(self):
        """
        Optimize cfm parameter for single frame..
        I think the gradient for cfm is really small..
        This doesn't work...
        """
        args: Namespace = self.args
        opt_frame = 3
        contact_info: DiffContactInfo = self.contact_info_list[opt_frame]
        contact_info.cfm = nn.Parameter(contact_info.cfm)
        self.root_pos_param = self.init_root_pos[opt_frame: opt_frame + 2].clone()  # (width, 3)
        self.joint_vec6d_param = self.init_joint_vec6d[opt_frame: opt_frame + 2].clone()  # (width, num joint, 3, 2)
        self.pd_target_param = self.init_pd_target_vec6d[opt_frame: opt_frame + 2].clone()  # (width, num joint, 3, 2)
        self.optim = torch.optim.AdamW([contact_info.cfm], lr=1)
        for epoch in range(1000):
            def closure():
                contact_info.cfm.data[contact_info.cfm.data < 1e-3] = 1e-3
                self.optim.zero_grad(True)
                loss_result = self.forward_process(opt_frame, 2, epoch)
                loss_result["tot_loss"].backward()
                with torch.no_grad():
                    print(contact_info.body0_index)
                    print(contact_info.cfm.reshape(-1, 3))
                    print(contact_info.cfm.grad.reshape(-1, 3))
                    print(loss_result["tot_loss"].item())
                return loss_result["tot_loss"]
            closure()
            self.optim.step()
            print()

    def test_one_piece(self):
        """
        Optimize for a single frame..
        """
        args: Namespace = self.args
        # optimize variable: reduced coordinate at time t and t+1
        # maybe we can also optimize the inverse dynamics..?
        # here we can use a flag..
        # loss:
        # 1. the simulation result is close to the reference
        # 2. the optimize variable is close to initial pose
        # 3. the frame t and t + 1 should be continuous
        # 4. the 2d projection loss should be small
        # rotate type: can be vector 6d

        # optimize these frames..
        self.init_parameter(args.index_t, args.width)
        for epoch_ in range(args.max_epoch):
            # if isinstance(self.optim, torch.optim.LBFGS):
            #    self.optim.step(lambda: self.closure(epoch_))
            # else:
            start_time = time.time()
            self.closure(epoch_)
            self.optim.step()
            print(time.time() - start_time)
            if epoch_ % 20 == 0:
                self.save_result(f"OptimizeEachFrame.ckpt{epoch_}")

    def handle_contact_label_by_body_height(self, root_pos: np.ndarray, joint_vec6d: np.ndarray):
        """
        here we can compute contact label by body height..
        1. compute body position at full coordinate
        2. get the contact label for each frame.
        3. compute new contact label
        """
        joint_quat: torch.Tensor = DiffQuat.vec6d_to_quat(torch.from_numpy(joint_vec6d))
        self.diff_motion._root_translation = torch.from_numpy(root_pos)
        self.diff_motion._joint_rotation = joint_quat  # (width, num joint, 4)
        self.diff_motion._num_frames = joint_quat.shape[0]
        self.diff_motion.recompute_joint_global_info()

        # here we should compute body position by joint position..
        # np_joint_quat: np.ndarray = joint_quat.detach().numpy()
        kine_body_pos, kine_body_quat = self.mocap_import.import_mocap_base_batch(self.diff_motion)  # (width, num body, 3), (width, num body, 4)
        np_body_pos: np.ndarray = kine_body_pos.detach().numpy()
        np_body_quat: np.ndarray = kine_body_quat.detach().numpy()
        num_frame: int = joint_quat.shape[0]
        diff_contact_list: List[Optional[DiffContactInfo]] = []
        new_contact_mess: List[List[int]] = []
        for frame in range(num_frame):
            body_height: np.ndarray = np_body_pos[frame, 1]
            body_index: np.ndarray = np.where(body_height < 0.08)[0]
            new_contact_mess.append(body_index.tolist())
            if len(body_index) == 0:
                diff_contact_list.append(None)
            else:
                contact_info = self.extractor.compute_contact_by_sub_body(np_body_pos[frame], np_body_quat[frame], body_index)
                diff_contact: Optional[DiffContactInfo] = self.extractor.convert_to_diff_contact_single(*contact_info)
                diff_contact_list.append(diff_contact)
        # set the diff contact list
        self.contact_mess = new_contact_mess
        self.contact_info_list = diff_contact_list

    def compute_contact_info_by_mess(self):
        for frame in range(self.args.width):
            self.contact_info_list[frame] = self.planning_list[frame].to_subcontact(self.contact_mess[frame])

    def extract_contact_message_for_save(self, eval_force_list: Optional[List]) -> List[Optional[SceneContactInfo]]:
        if eval_force_list is None:
            return None
        # convert the contact as Scene

        result_list: List[Optional[SceneContactInfo]] = []
        for frame, frame_force in enumerate(eval_force_list):
            diff_contact: Optional[DiffContactInfo] = self.contact_info_list[frame]
            if diff_contact is None or len(diff_contact) == 0:
                result_list.append(None)
                continue
            ode_contact_info: SceneContactInfo = diff_contact.export_scene_contact_info()
            if frame_force is None:
                result_list.append(result_list[-1])
            else:
                ode_contact_info.force = frame_force[0]
                ode_contact_info.torque = frame_force[1]
            # here we should insert the force and torque.

            result_list.append(ode_contact_info)

        return result_list

    @staticmethod
    def main():
        args = parse_args()
        opt = OptimizeEachFrame(args)
        opt.test_contact_planning()


def check_diff_quat_angvel():
    fname: str = os.path.join(fdir, "../../../Tests/CharacterData/lafan-mocap-100/walk1_subject1.bvh")
    motion: MotionData = BVHLoader.load(fname)
    diff_motion: PyTorchMotionData = PyTorchMotionData()
    diff_motion.build_from_motion_data(motion, dtype=torch.float64)
    diff_motion._root_translation = nn.Parameter(diff_motion._root_translation)
    diff_motion._joint_rotation = nn.Parameter(diff_motion._joint_rotation)
    diff_motion.recompute_joint_global_info() # the FK process has gradient.

    angvel: torch.Tensor = diff_motion.compute_angular_velocity()
    # check the gradient..
    sum_val = torch.sum(angvel)
    sum_val.backward()
    # print(diff_motion.root_translation.grad.shape)
    np_angvel = motion.compute_angular_velocity()
    print(np.max(np.abs(angvel.detach().numpy() - np_angvel)))
    exit(0)


# def test_soft_func():
#     import matplotlib.pyplot as plt
#     x = torch.linspace(-0.3, 0.3, 200)
#     x1 = soft_height_weight_func(x, 0.05)
#     x2 = soft_velo_weight_func(x, 0.05)
#     plt.plot(x.detach().numpy(), x1.detach().numpy())
#     plt.title("soft height")
#     plt.show()
#     plt.plot(x.detach().numpy(), x2.detach().numpy())
#     plt.title("soft velo")
#     plt.show()

# if __name__ == "__main__":
#     test_soft_func()

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
from collections import OrderedDict
import datetime
import gc
from mpi4py import MPI
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from typing import Optional, Dict, Any, Tuple, List
from tensorboardX import SummaryWriter

from .Common import NetworkType, build_mlp, joint_2d_type_dim
from .PoseFormer.PoseTransformer import PoseTransformer
from .SamconDataLoader import SamconDataLoader
from .SimulationLoss import prepare_data_parallel
from ..SamconMainWorkerBase import SamHlp, SamconWorkerBase

from ...Common.Helper import Helper
from ...Common.MathHelper import RotateType

from ...ODESim.CharacterWrapper import CharacterWrapper
from ...ODESim.ODECharacter import ODECharacter
from ...ODESim.ODEScene import ODEScene
from ...ODESim.Saver.CharacterToBVH import CharacterTOBVH

from ...DiffODE import DiffQuat
from ...Utils.Camera.CameraPyTorch import CameraParamTorch
from ...Utils.Camera.Human36CameraBuild import CameraParamBuilder
from ...Utils.Dataset.StdHuman import stdhuman_with_root_to_unified_index as to_unified_index

from ...pymotionlib.MotionData import MotionData
from ...pymotionlib.PyTorchMotionData import PyTorchMotionData


comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda")
fdir = os.path.dirname(__file__)


class TrainParallelHelper(nn.Module):
    """
    We can compute loss parallel on different graphics card by using DataParallel simply.
    """
    def __init__(
        self,
        network: nn.Module,
        args: Namespace,
        diff_motion: PyTorchMotionData,
        joint_to_child_body: List,
        child_body_rel_pos: torch.Tensor,
        mean_pos3d: torch.Tensor,
        std_pos3d: torch.Tensor,
    ) -> None:
        super().__init__()
        self.module = network
        self.args = args
        self.diff_motion = diff_motion
        self.num_joints = len(joint_to_child_body)
        self.num_bodies = self.num_joints + 1
        self.joint_to_body: List[int] = [0] + joint_to_child_body
        self.child_body_rel_pos: torch.Tensor = child_body_rel_pos
        self.mean_pos3d: torch.Tensor = mean_pos3d
        self.std_pos3d: torch.Tensor = std_pos3d

    def forward(
        self,
        x_gt: torch.Tensor,
        y_gt: torch.Tensor,
        camera_pos3d_gt: torch.Tensor,
        camera_2d_confidence: torch.Tensor,
        camera_f: torch.Tensor,
        camera_c: torch.Tensor,
        camera_trans: torch.Tensor,
        camera_rot: torch.Tensor
    ):
        args = self.args
        device = x_gt.device
        all_rotate_slice: Optional[slice] = args.out_slice["all_rotate_slice"]  # joint rotation and pd target rotation in output
        local_pos_slice: Optional[slice] = args.out_slice["local_pos_slice"]  # joint position in output
        contact_slice: Optional[slice] = args.out_slice["contact_slice"]  # contact label in output
        kine_in_rotate_offset: Optional[int] = args.out_slice["kine_in_rotate_offset"]  # kinematic pose in all_rotate_slice
        pd_target_in_rotate_offset: Optional[int] = args.out_slice["pd_target_in_rotate_offset"]
        rotate_type: RotateType = args.rotate_type
        frame_window: int = args.frame_window
        batch_size: int = x_gt.shape[0]
        pred_rot_shape: Tuple = (batch_size, -1) + args.rotate_shape
        self.diff_motion.clear()
        diff_motion: PyTorchMotionData = self.diff_motion.to_device(device)
        diff_motion_global: PyTorchMotionData = diff_motion.sub_sequnece(None, None, is_copy=True)

        y_pred: torch.Tensor = self.module(x_gt)
        y_pred_rot: torch.Tensor = y_pred[:, all_rotate_slice].reshape(pred_rot_shape)
        y_gt_rot: torch.Tensor = y_gt[:, all_rotate_slice].reshape(pred_rot_shape)
        if rotate_type == RotateType.Vec6d:
            y_pred_matrix: torch.Tensor = DiffQuat.vec6d_to_matrix(y_pred_rot)  # This is local rotation
            y_pred_rot: torch.Tensor = y_pred_matrix[..., :2].contiguous()
        else:
            raise NotImplementedError

        loss: torch.Tensor = args.loss_rotate_weight * F.mse_loss(y_gt_rot, y_pred_rot)
        pos_loss: torch.Tensor = torch.as_tensor(0.0)
        pred_root_pos: Optional[torch.Tensor] = None
        if args.also_output_local_rot:
            pred_root_pos: Optional[torch.Tensor] = y_pred[:, local_pos_slice].contiguous()
            pos_loss: torch.Tensor = args.loss_root_pos_weight * F.mse_loss(y_gt[:, local_pos_slice], pred_root_pos)
            loss += pos_loss

        contact_loss: torch.Tensor = torch.as_tensor(0.0)
        if args.also_output_contact_label:
            contact_loss: torch.Tensor = args.loss_contact_weight * F.mse_loss(y_gt[:, contact_slice], y_pred[:, contact_slice])
            loss += contact_loss

        # Here we can compute 3d joint position in camera space direcly.
        loss_fk: torch.Tensor = torch.as_tensor(0.0)
        loss_2d: torch.Tensor = torch.as_tensor(0.0)
        global_body_pos: Optional[torch.Tensor] = None
        global_body_quat: Optional[torch.Tensor] = None
        global_body_mat: Optional[torch.Tensor] = None
        if args.need_do_fk:
            # here we should convert rotation matrix to quaternion, for forward kinematics process.
            # we should also un-normalize the predicted position
            # However, fk requires many time..
            un_normalize_root_pos: torch.Tensor = self.std_pos3d.to(device) * pred_root_pos + self.mean_pos3d.to(device)
            kine_quat: torch.Tensor = DiffQuat.quat_from_matrix(y_pred_matrix[:, kine_in_rotate_offset:].contiguous())
            diff_motion._num_frames = batch_size
            diff_motion._num_joints = self.num_joints + 1
            diff_motion._root_translation = un_normalize_root_pos
            diff_motion._joint_rotation = kine_quat
            diff_motion.recompute_joint_global_info()
            joint_pos: torch.Tensor = diff_motion.joint_position
            joint_orientation: torch.Tensor = diff_motion.joint_orientation
            if args.loss_fk > 0:
                loss_fk: torch.Tensor = args.loss_fk * F.mse_loss(diff_motion.joint_position, camera_pos3d_gt)
                loss += loss_fk
            if args.loss_projection_2d > 0:
                position_unified: torch.Tensor = diff_motion.joint_position[:, to_unified_index]
                proj_pos2d: torch.Tensor = position_unified[..., :2] / torch.clamp(position_unified[..., 2:], 1e-2, 1e2)
                proj_pos2d: torch.Tensor = camera_f[:, None, :] * proj_pos2d + camera_c[:, None, :]
                x_gt: torch.Tensor = x_gt.view(batch_size, frame_window, -1, 2)
                camera_pos2d_gt: torch.Tensor = x_gt[:, frame_window // 2].contiguous()
                # note: here we should also consider the input confidence..
                loss_2d: torch.Tensor = args.loss_projection_2d * torch.mean(camera_2d_confidence[..., None] * (proj_pos2d - camera_pos2d_gt) ** 2)
                loss += loss_2d
            # convert to from camera coordinate to global coordinate
            # we can compute the global root position and translation, and do forward kinematics again
            if args.use_phys_loss:
                global_root_quat: torch.Tensor = CameraParamTorch.convert_rotation_to_world(kine_quat[:, 0, :], camera_rot)
                diff_motion_global._num_frames = batch_size
                diff_motion_global._num_joints = self.num_joints + 1
                diff_motion_global._root_translation = CameraParamTorch.camera_to_world_batch(un_normalize_root_pos, camera_trans, camera_rot, True)
                diff_motion_global._joint_rotation = torch.cat([global_root_quat[:, None], kine_quat[:, 1:, :]], dim=1)
                diff_motion_global.recompute_joint_global_info()
                y_pred_pd_target_mat: torch.Tensor = \
                    y_pred_matrix[:, pd_target_in_rotate_offset:pd_target_in_rotate_offset + self.num_joints].contiguous()
                y_pred_pd_target_quat: torch.Tensor = DiffQuat.quat_from_matrix(y_pred_pd_target_mat)
                dup_rel_pos: torch.Tensor = self.child_body_rel_pos.to(device).repeat(batch_size, 1, 1)
                child_body_offset: torch.Tensor = \
                    DiffQuat.quat_apply(diff_motion_global.joint_orientation.view(-1, 4), dup_rel_pos.view(-1, 3)).view(batch_size, self.num_joints + 1, 3)
                child_body_pos: torch.Tensor = diff_motion_global.joint_position - child_body_offset

                global_body_pos: torch.Tensor = child_body_pos[:, self.joint_to_body]
                global_body_quat: torch.Tensor = diff_motion_global.joint_orientation[:, self.joint_to_body]
                global_body_mat: torch.Tensor = DiffQuat.quat_to_matrix(global_body_quat)

        return loss, loss_fk, loss_2d, pos_loss, contact_loss, y_pred_rot, y_pred_pd_target_quat, global_body_pos, global_body_quat, global_body_mat, y_gt_rot

class Policy3dBase(CharacterWrapper):
    """
    e.g. the simulation fps (100 fps)
    e.g. the 2d fps of human 3.6 dataset is 50 fps, or we can downsample to 25 fps
    The structure of network is as follows.
    Input:
    - 2d key joint position in the future for 0.2 sec
    - current character state in heading frame
    - 2d error between current state and 2d key points
    - and etc

    Loss:
    - L2 loss between prediction and samcon result
    - Continuous loss of N piece of target pose
    - Regularization loss of output / network
    - 2d projection loss

    Total process:
    1. prepare training data
        - run 3d human pose estimation via kinematic based method, such as VIBE, MotioNet, and etc.
        - run duplicated samcon on 3d human pose estimation
        - fine-tune samcon result via trajectory optimization
    2. Train policy using simple MLP (or transformer, etc)
        - select training data batchly
        - NN forward
            - Compute loss between sim result and 3d hpe result (or samcon result) in child-workers
        - Maybe fine-tune or post-processing for training is required. (training on multi frames?)
    3. Evaluate policy

    """
    def __init__(
        self,
        samhlp: SamHlp,
        scene: Optional[ODEScene] = None,
        character: Optional[ODECharacter] = None,
        args: Optional[Namespace] = None
    ):
        super().__init__()
        self.param = self.args = args
        if args is not None:
            self.rotate_type = args.rotate_type
            loss_dict = {"mse_loss": F.mse_loss, "l1_loss": F.l1_loss}
            self.loss_func = loss_dict[args.loss_func]
        else:
            self.rotate_type = RotateType.Vec6d
            self.loss_func = F.mse_loss

        self.samhlp: SamHlp = samhlp
        self.conf: Dict[str, Any] = self.samhlp.conf
        log_path_name: str = self.samhlp.log_path_dname() + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.writer = SummaryWriter(log_path_name)
        self.writer_count: int = 0
        self.device = args.device

        if scene is None:
            scene: ODEScene = SamconWorkerBase.load_scene_with_conf(self.conf)

        self.scene: ODEScene = scene
        # Note: at training, the fps must be 100..
        self.scene.set_sim_fps(args.simulation_fps)
        print(f"load scene with sim fps = {self.scene.sim_fps}")
        if character is None:
            character = self.scene.character0
        self.character: ODECharacter = character
        self.out_num_joint: int = len(character.joints)
        self.to_bvh = CharacterTOBVH(self.character, self.scene.sim_fps)
        self.to_bvh.bvh_hierarchy_no_root()
        self.to_bvh.append_no_root_to_buffer()
        self.motion_hierarchy: MotionData = self.to_bvh.to_file(None, False)
        self.motion_hierarchy_no_end: MotionData = self.motion_hierarchy.remove_end_sites(copy=True)

        self.dump_path = os.path.join(fdir, "policy_train.ckpt")

        if args.pretrain_result:
            self.dump_load_path: str = args.pretrain_result
        else:
            self.dump_load_path: str = self.dump_path

        self.epoch: int = 0
        self.lr: float = self.param.lr

        print(f"use network type {str(self.args.network)}")

        self.input_num_joint = joint_2d_type_dim[self.args.joint_2d_type]

        # note: I think the network can also output root position (if the camera is not moving...)
        if self.args.network == NetworkType.MLP:
            self.network = build_mlp(
                self.param.frame_window, self.input_num_joint, self.out_num_joint, 512, self.rotate_type,
                self.param.also_output_pd_target,
                self.param.also_output_local_rot,
                self.param.also_output_contact_label,
                self.args.mlp_dropout,
                self.device
            )
        elif self.args.network == NetworkType.PoseFormer:
            self.network = PoseTransformer(
                self.param.frame_window, self.input_num_joint, self.out_num_joint, self.rotate_type,
                self.param.also_output_pd_target,
                self.param.also_output_local_rot,
                self.param.also_output_contact_label,
                device=self.device
            )
        else:
            raise NotImplementedError
        print(f"After build network")

        # if args.use_parallel and torch.cuda.device_count() > 1:  # train network with parallel GPUs
        #    self.network = nn.DataParallel(self.network)
        #    print(f"After build parallel network")
        # in pytorch document, DataParallel is slower than DistributedDataParallel,
        # because pytorch need to scatter the network weights at each batch.
        # as our network weight is small (~60MiB), I think use DataParallel simply is just OK

        self.optimizer: Optional[AdamW] = None
        self.data_length: Optional[np.ndarray] = None
        self.accum_data_length: Optional[np.ndarray] = None

        self.camera_numpy = CameraParamBuilder.build(np.float64)
        self.camera_torch = CameraParamTorch.build_dict_from_numpy(self.camera_numpy, torch.float32)

        self.diff_motion: PyTorchMotionData = PyTorchMotionData()
        self.diff_motion.build_from_motion_data(self.motion_hierarchy_no_end, torch.float32, self.device)

        self.samcon_dataloader: SamconDataLoader = SamconDataLoader(self.scene, self.character, self.args)
        if self.args.mode == "train":  # for eval mode, we need not to process training data.
            if not args.input_data_dir or not os.path.isdir(args.input_data_dir):
                args.input_data_dir = samhlp.save_folder_i_dname()
            if comm_size == 1 and False:
                self.samcon_dataloader.prepare_data(args.input_data_dir)
            else:
                self.samcon_dataloader.result_buffer = prepare_data_parallel(self.samcon_dataloader.get_data_fname_list(args.input_data_dir))
                gc.collect()
                print(f"before compute mean and std", flush=True)
                self.samcon_dataloader.normalize_result_buffer()
                print(f"after compute mean and std", flush=True)

    def init_optimizer(self, network: Optional[nn.Module] = None):
        if network is None:
            network = self.network
        self.optimizer = AdamW(network.parameters(), lr=self.param.lr, weight_decay=self.param.weight_decay)
        return self.optimizer

    def save_state(self, network: Optional[nn.Module] = None, optimizer: Optional[Optimizer] = None, index: Optional[str] = None):
        """
        we should also save random state here..
        """
        output_fname = self.dump_path
        if index:
            output_fname = output_fname + index
        if network is None:
            network = self.network
        if optimizer is None:
            optimizer = self.optimizer
        result = {
            "epoch": self.epoch,
            "lr": self.lr,
            "optimizer": optimizer.state_dict(),
            "network": network.state_dict(),
            "args": self.args,
            "writer_count": self.writer_count,
            "mean_pos3d": self.samcon_dataloader.mean_pos3d,  # for normalize the output for position..
            "std_pos3d": self.samcon_dataloader.std_pos3d
        }
        result.update(Helper.save_torch_state())
        torch.save(result, output_fname)
        print(f"save network weight at {self.epoch} to {output_fname}")

    def load_state(self):
        if os.path.exists(self.dump_load_path):
            print(f"load state from {self.dump_load_path}")
            result: Dict[str, Any] = torch.load(self.dump_load_path, map_location=self.device)
            if not self.args.not_load_lr:
                self.lr: float = result["lr"]
                if self.args.mode == "train":
                    self.optimizer.load_state_dict(result["optimizer"])
                self.epoch: int = result["epoch"]

            net_key: str = list(result["network"].keys())[0]
            key_off = 0
            while net_key[key_off:].startswith("module."):
                key_off += len("module.")
            if key_off > 0:
                new_dict = OrderedDict()
                for key, value in result["network"].items():
                    new_dict[key[key_off:]] = value
                result["network"] = new_dict
            load_net = self.network
            if isinstance(self.network, nn.DataParallel):
                load_net = load_net.module
            if isinstance(load_net, TrainParallelHelper):
                load_net = load_net.module
            if isinstance(load_net, nn.Sequential) or isinstance(load_net, PoseTransformer):
                load_net.load_state_dict(result["network"])
            else:
                raise ValueError

            if "writer_count" in result:
                self.writer_count: int = result["writer_count"]

            if "mean_pos3d" in result:
                self.samcon_dataloader.mean_pos3d = result["mean_pos3d"]

            if "std_pos3d" in result:
                self.samcon_dataloader.std_pos3d = result["std_pos3d"]

            # Helper.load_torch_state(result)
        else:
            print(f"{self.dump_load_path} not exist..ignore..")

    def normalize_pos3d(self, pos: torch.Tensor):
        return self.samcon_dataloader.normalize_pos3d(pos)

    def unnormalize_pos3d(self, pos: torch.Tensor):
        return self.samcon_dataloader.unnormalize_pos3d(pos)

    def compute_delta_angle(self, y_pred: torch.Tensor, y_gt: torch.Tensor):
        """
        delta angle between prediction rotation and ground truth rotation
        """
        with torch.no_grad():
            delta_angle = DiffQuat.compute_delta_angle(y_pred, y_gt, self.rotate_type)
            max_dangle = torch.max(delta_angle)
            mean_dangle = torch.mean(delta_angle)
            achieve_bad_ratio = torch.sum(delta_angle >= 0.9 * max_dangle) / delta_angle.numel()
            return max_dangle, mean_dangle, achieve_bad_ratio

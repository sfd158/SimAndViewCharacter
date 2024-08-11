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
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, List

from .Common import NetworkType
from .Policy3dBase import Policy3dBase, TrainParallelHelper
from .SimulationLoss import SimuLossParallel, stop_child_workers
from ...ODESim.ODECharacter import ODECharacter
from ...DiffODE import DiffQuat
from ...pymotionlib.PyTorchMotionData import PyTorchMotionData


def train(policy_base: Policy3dBase):
    """
    train the network
    """
    args: Namespace = policy_base.args
    print(f"start training", flush=True)

    contact_slice: Optional[slice] = args.out_slice["contact_slice"]  # contact label in output
    frame_window: int = args.frame_window
    character: ODECharacter = policy_base.character
    child_body_rel_pos: torch.Tensor = torch.as_tensor(
        np.concatenate([np.zeros((1, 3)), character.joint_info.get_child_body_relative_pos()]), dtype=torch.float32, device=args.device)
    policy_base.network = TrainParallelHelper(
        policy_base.network, args, policy_base.diff_motion,
        character.joint_to_child_body, child_body_rel_pos,
        policy_base.samcon_dataloader.mean_pos3d, policy_base.samcon_dataloader.std_pos3d
    )
    policy_base.network = nn.DataParallel(policy_base.network)
    policy_base.network.train()
    policy_base.init_optimizer()
    policy_base.load_state()

    while policy_base.epoch < policy_base.param.max_epoch:
        loss_list = []
        print(f"epoch = {policy_base.epoch}, lr = {policy_base.lr:.6f}", flush=True)
        for index, data in enumerate(policy_base.samcon_dataloader.random_iter()):
            policy_base.optimizer.zero_grad(set_to_none=True)
            x_gt, y_gt, camera_pos3d_gt, camera_2d_confidence, camera_f, camera_c, camera_trans, camera_rot, divide_index = data
            batch_size: int = x_gt.shape[0]
            if args.network == NetworkType.PoseFormer:
                # batch_size, num_frame, num_joint, in_channel
                x_gt: torch.Tensor = x_gt.view(batch_size, frame_window, -1, 2)
            loss, loss_fk, loss_2d, pos_loss, contact_loss, y_pred_rot, y_pred_pd_target_quat, global_body_pos, global_body_quat, global_body_mat, y_gt_rot = \
                policy_base.network(x_gt, y_gt, camera_pos3d_gt, camera_2d_confidence, camera_f, camera_c, camera_trans, camera_rot)
            if loss.numel() > 1:
                loss = torch.mean(loss)
                loss_fk = loss_fk.detach().cpu().mean()
                loss_2d = loss_2d.detach().cpu().mean()
                pos_loss = pos_loss.detach().cpu().mean()
                contact_loss = contact_loss.detach().cpu().mean()

            # we can compute physics loss between 2 frames here..
            # Note: we need to convert the rotation to from camera coordinate to world coordinate..
            # TODO: how to compute loss on GPU and CPU parallel..?
            avg_simu_loss = None
            if args.use_phys_loss:  # here we should convert joint order to body order..
                simu_loss, avg_simu_loss = SimuLossParallel.apply(
                    global_body_pos, global_body_mat, global_body_quat,
                    y_pred_pd_target_quat, y_gt[:, contact_slice].to(torch.long),
                    divide_index
                )
                loss += simu_loss

            loss.backward()

            if args.gradient_clip is not None:  # clip the gradient here.
                nn.utils.clip_grad_norm_(policy_base.network.parameters(), args.gradient_clip, norm_type=2)

            policy_base.optimizer.step()

            if index % 40 == 0:
                loss_item = loss.item()
                loss_list.append(loss_item)
                max_dangle, mean_dangle, achieve_bad_ratio = policy_base.compute_delta_angle(y_pred_rot, y_gt_rot)
                pos_loss_item, contact_loss_item = pos_loss.item(), contact_loss.item()
                max_dangle_item, mean_dangle_item = max_dangle.item(), mean_dangle.item()
                achieve_bad_ratio_item = achieve_bad_ratio.item()
                loss_fk_item, loss_2d_item = loss_fk.item(), loss_2d.item()
                print(
                    f"epoch = {policy_base.epoch}, index = {index}, loss = {loss_item:.4f}, "
                    f"pos = {pos_loss_item:.4f}, "
                    f"contact = {contact_loss_item:.4f}, "
                    f"fk = {loss_fk_item:.4f}, 2d = {loss_2d_item:.4f}, "
                    f"max dangle = {max_dangle_item:.4f}, mean dangle = {mean_dangle_item:.4f}, "
                    f"bad ratio = {achieve_bad_ratio_item:.4f}, ",
                    end="",
                    flush=True
                )
                if avg_simu_loss is not None:
                    for nodek, nodev in avg_simu_loss.items():
                        print(f"{nodek} = {nodev:.4f}, ", end="")
                        policy_base.writer.add_scalar(nodek, nodev, policy_base.writer_count)
                print(flush=True)

                policy_base.writer.add_scalar("loss", loss_item, policy_base.writer_count)
                policy_base.writer.add_scalar("pos loss", pos_loss_item, policy_base.writer_count)
                policy_base.writer.add_scalar("contact loss", contact_loss_item, policy_base.writer_count)
                policy_base.writer.add_scalar("fk loss", loss_fk_item, policy_base.writer_count)
                policy_base.writer.add_scalar("loss 2d", loss_2d_item, policy_base.writer_count)
                policy_base.writer.add_scalar("max dangle", max_dangle_item, policy_base.writer_count)
                policy_base.writer.add_scalar("mean dangle", mean_dangle_item, policy_base.writer_count)
                policy_base.writer_count += 1

            del loss, pos_loss, contact_loss

        policy_base.lr *= policy_base.param.lr_decay
        for param_group in policy_base.optimizer.param_groups:
            param_group['lr'] *= policy_base.param.lr_decay
        policy_base.epoch += 1
        avg_cost = sum(loss_list) / len(loss_list)
        print(f"epoch = {policy_base.epoch}, Avg cost = {avg_cost:.4f}")
        if policy_base.epoch % 1 == 0:
            policy_base.save_state()
            policy_base.save_state(index=str(policy_base.epoch))

    stop_child_workers()
    policy_base.save_state()
    print(f"after training process", flush=True)

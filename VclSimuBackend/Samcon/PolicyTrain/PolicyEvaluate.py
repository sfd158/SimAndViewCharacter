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
Here we should also evaluate the contact loss..
If the evaluate mode is training data, we should also compute the contact label accurate by a given threshold.
"""

from argparse import Namespace
from platform import platform
from mpi4py import MPI
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation
import subprocess
import torch
from typing import Any, Union, Optional, List, Tuple, Dict

from .Common import EvaluateMode, NetworkType, visualize_motion
from .PreProcess import pre_process_motion_method1
from .Policy3dBase import Policy3dBase
from ...Common.MathHelper import MathHelper
from ...DiffODE import DiffQuat
from ...pymotionlib import BVHLoader
from ...pymotionlib.MotionData import MotionData

from ...Utils.AlphaPose.Utils import load_alpha_pose
from ...Utils.Dataset.COCOUtils import coco_to_unified
from ...Utils.Dataset.Human36 import parse_human36_video_camera, std_human_to_human36m, pre_build_camera
from ...Utils.Camera.CameraNumpy import CameraParamNumpy
from ...Utils.MotionProjection import convert_rotation_to_world

from ...Utils.Dataset.StdHuman import stdhuman_with_root_names
from ...Utils.Dataset.Human36 import get_y_facing_coordinate
from ...Utils.Evaluation import calc_motion_mpjpe, mpjpe
from ...Utils.MotionUtils import motion_to_facing_coordinate


cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda")
fdir: str = os.path.dirname(__file__)
comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()


def evaluate_new(policy_base: Policy3dBase):
    args: Namespace = policy_base.args
    print(f"begin evaluate, save result at folder {args.output_data_dir}")

    policy_base.load_state()
    policy_base.network.eval()
    save_dpath: str = args.output_data_dir
    sim_fps: int = int(policy_base.scene.sim_fps)
    assert sim_fps % args.fps_2d == 0
    ratio: int = int(sim_fps // args.fps_2d)
    frame_window: int = args.frame_window
    cut_ratio = 100 // sim_fps
    num_body: int = len(policy_base.bodies)
    body_names: List[str] = policy_base.body_names()
    is_windows: bool = ("Windows" in platform()) and args.render_result

    def cat_func(arr: np.ndarray):
        return np.concatenate([arr[i:-ratio*frame_window+i, None] for i in range(0, ratio * frame_window, ratio)], axis=1)

    eval_mode = args.eval_mode
    eval_attr_fname = args.eval_attr_fname

    local_pos_slice: Optional[slice] = args.out_slice["local_pos_slice"]
    contact_slice: Optional[slice] = args.out_slice["contact_slice"]
    pd_target_slice: Optional[slice] = args.out_slice["pd_target_slice"]
    local_rot_slice: Optional[slice] = args.out_slice["local_rot_slice"]

    with torch.no_grad():
        joint_2d_pos: Optional[np.ndarray] = None
        total_x: Union[np.ndarray, torch.Tensor, None] = None
        total_y: Union[np.ndarray, torch.Tensor, None] = None
        camera_params: Optional[List[CameraParamNumpy]] = None
        eval_mocap_gt: Optional[MotionData] = None
        confidence: Optional[np.ndarray] = None
        if eval_mode == EvaluateMode.TrainData:  # the training data is 100 fps
            # we need to downsample here when fps == 50
            label: str = os.path.split(eval_attr_fname)[1][:2]
            input_buffer: List[Dict[str, Any]] = policy_base.samcon_dataloader.prepare_one_piece(eval_attr_fname, label, False, 0)  # in 100 fps
            joint_2d_pos: Optional[np.ndarray] = input_buffer[0]["joint_pos_2d"][:, frame_window // cut_ratio:-frame_window // cut_ratio]  # in 100 fps
            total_x, total_y = policy_base.samcon_dataloader.get_total_data(input_buffer[0])  # in 100 fps
            eval_mocap_gt: MotionData = visualize_motion(
                policy_base.to_bvh,
                input_buffer[0]["root_pos3d"],
                input_buffer[0]["root_quat"],
                MathHelper.quat_from_other_rotate(input_buffer[0]["joint_local_rot"], policy_base.rotate_type)
            ).sub_sequence(frame_window // cut_ratio, -frame_window // cut_ratio)
            if sim_fps == 50:  # down sample here
                joint_2d_pos: np.ndarray = joint_2d_pos[:, ::2]
                total_x: np.ndarray = total_x[:, ::2]
                total_y: np.ndarray = total_y[:, ::2]
                eval_mocap_gt = eval_mocap_gt.resample(sim_fps)
            print(f"total_x.shape = {total_x.shape}, total_y.shape = {total_y.shape}")
            camera_params = policy_base.camera_numpy[label][:1]
            confidence: np.ndarray = np.ones((joint_2d_pos.shape[0], total_x.shape[0], joint_2d_pos.shape[2]))
        elif eval_mode == EvaluateMode.BVH_MOCAP:
            # this is a test case. use a bvh mocap data to generate test data
            # use a simple camera model for test..
            eval_mocap_gt: MotionData = BVHLoader.load(args.eval_attr_fname)
            eval_mocap_gt: MotionData = eval_mocap_gt.resample(sim_fps)
            input_buffer: Dict[str, Any] = policy_base.samcon_dataloader.prepare_bvh_one_piece_base(eval_mocap_gt)  # use default camera param..
            joint_2d_pos: Optional[np.ndarray] = input_buffer["joint_pos_2d"]
            camera_params = input_buffer["camera_param"][:1]
            total_x, total_y = cat_func(joint_2d_pos[0]), None  # total_y is None
            if "confidences" in input_buffer:  # we need to consider simu fps here..
                confidence = input_buffer["confidences"][:, frame_window:-frame_window]
            joint_2d_pos = joint_2d_pos[:1, frame_window:-frame_window]
            eval_mocap_gt = eval_mocap_gt.sub_sequence(frame_window // cut_ratio, -frame_window // cut_ratio)
        elif eval_mode == EvaluateMode.Estimation_2d:  # Here, only a camera view is given.
            # note: we can resample the input data here..
            joint_2d_pos, confidence = load_alpha_pose(args.eval_attr_fname, args.start_frame, args.end_frame)
            if sim_fps != args.fps_2d:
                joint_2d_pos: np.ndarray = MathHelper.resample_joint_linear(joint_2d_pos, ratio, args.fps_2d)
                confidence: np.ndarray = MathHelper.resample_joint_linear(confidence, ratio, args.fps_2d)
            print(f"load alpha pose, {joint_2d_pos.shape}")
            subset: List[int] = coco_to_unified
            try:
                camera_params: List[CameraParamNumpy] = [parse_human36_video_camera(args.eval_attr_fname)]
            except ValueError as err:
                print(err)
                # actually, in test case, camera is not correct.
                # we need to optimize the root position and orientation
                # to match 2d projection..
                camera_params = pre_build_camera[f"S1"][:1]

            joint_2d_pos: np.ndarray = camera_params[0].normalize_screen_coordinates(joint_2d_pos)[None, :, subset, :]
            # actually, we can predict motion by nerual network...
            # Note: we cannot acquire camera parameter by 2d human pose estimation..
            # There are only 4 type of camera in Human3.6M dataset.
            # here we should divide the input data into continuous sliding window..
            total_x, total_y = cat_func(joint_2d_pos[0]), None

            joint_2d_pos: np.ndarray = joint_2d_pos[:, frame_window // cut_ratio:-frame_window // cut_ratio]
            confidence = confidence[None, frame_window // cut_ratio:-frame_window // cut_ratio, subset]
        else:
            raise NotImplementedError

        pred_result = {}
        output_gt_bvh_fname: Optional[str] = None
        if eval_mocap_gt is not None:
            output_gt_bvh_fname_raw: str = "eval-mocap-gt.bvh"
            output_gt_bvh_fname: str = os.path.join(save_dpath, output_gt_bvh_fname_raw)
            # here the eval_mocap_gt is resampled..
            # eval_mocap_gt: MotionData = eval_mocap_gt.sub_sequence(args.frame_window // cut_ratio, -args.frame_window // cut_ratio)
            BVHLoader.save(eval_mocap_gt, output_gt_bvh_fname)
            # visualize the eval mocap ground truth data
            if is_windows:
                subprocess.Popen(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", output_gt_bvh_fname])
            print(f"save eval ground truth mocap data to {output_gt_bvh_fname_raw}")
            pred_result["input_motion_gt"] = output_gt_bvh_fname_raw

        total_x: torch.Tensor = torch.as_tensor(total_x, dtype=torch.float32, device=policy_base.device)
        num_camera: int = len(camera_params)
        total_x: torch.Tensor = total_x.reshape(num_camera, total_x.shape[0] // num_camera, -1)
        if total_y is not None:  # The ground truth data..
            total_y: torch.Tensor = torch.as_tensor(total_y, dtype=torch.float32, device=policy_base.device)
            total_y: torch.Tensor = total_y.view(num_camera, total_y.shape[0] // num_camera, -1)

        pred_rot_shape: Tuple = (total_x.shape[1], -1) + args.rotate_shape
        for camera_index, camera in enumerate(camera_params):
            input_x: torch.Tensor = total_x[camera_index]
            if args.network == NetworkType.PoseFormer:
                # batch_size, num_frame, num_joint, in_channel
                input_x: torch.Tensor = input_x.view(input_x.shape[0], policy_base.param.frame_window, -1, 2)

            y_pred: torch.Tensor = policy_base.network(input_x)

            cam_confidence = confidence[camera_index]
            # Test case:
            # - only output kinematic pose
            # - only output inverse dynamics
            # - output both kinematic pose and inverse dynamics pose
            # for visualize, we can output the inverse dynamics result as bvh file
            y_pred_invdyn_numpy: Optional[np.ndarray] = None
            if args.also_output_pd_target:
                y_pred_invdyn: torch.Tensor = y_pred[:, pd_target_slice].contiguous().view(pred_rot_shape)
                y_pred_invdyn = DiffQuat.normalize_pred_rot(y_pred_invdyn, policy_base.rotate_type)
                y_pred_invdyn_quat: torch.Tensor = DiffQuat.convert_to_quat(y_pred_invdyn, policy_base.rotate_type)
                y_pred_invdyn_numpy: np.ndarray = y_pred_invdyn_quat.cpu().numpy().astype(np.float64)
                pred_invdyn_motion = policy_base.to_bvh.forward_kinematics(
                    np.zeros((input_x.shape[0], 3)), MathHelper.unit_quat_arr((input_x.shape[0], 4)), y_pred_invdyn_numpy
                )
                if False:
                    output_invdyn_bvh_fname = os.path.join(save_dpath, "eval-invdyn-predict.bvh")
                    BVHLoader.save(pred_invdyn_motion, output_invdyn_bvh_fname)
                    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", output_invdyn_bvh_fname])

            y_pred_local_quat: Optional[np.ndarray] = None
            pred_ref_motion: Optional[MotionData] = None
            if args.also_output_local_rot:
                y_pred_local: torch.Tensor = y_pred[:, local_rot_slice].contiguous().view(pred_rot_shape)
                y_pred_local: torch.Tensor = DiffQuat.normalize_pred_rot(y_pred_local, policy_base.rotate_type)
                y_pred_local_quat: torch.Tensor = DiffQuat.convert_to_quat(y_pred_local, policy_base.rotate_type)
                y_pred_local_quat: np.ndarray = y_pred_local_quat.cpu().numpy().astype(np.float64)
                y_pred_root_pos: torch.Tensor = y_pred[:, local_pos_slice]
                y_pred_root_pos: np.ndarray = policy_base.unnormalize_pos3d(y_pred_root_pos).cpu().numpy().astype(np.float64)
                if eval_mode != EvaluateMode.Estimation_2d:
                    y_pred_root_pos: np.ndarray = camera.camera_to_world(y_pred_root_pos)
                    # forward kinematics..
                    y_pred_local_quat[:, 0] = convert_rotation_to_world(y_pred_local_quat[:, 0], camera)
                pred_ref_motion: MotionData = policy_base.to_bvh.forward_kinematics(
                    y_pred_root_pos, y_pred_local_quat[:, 0], y_pred_local_quat[:, 1:])
                # pred_ref_motion = pre_process_method2(pred_ref_motion, joint_2d_pos[camera_index])
                if eval_mode == EvaluateMode.Estimation_2d:
                    pred_ref_motion.joint_translation[:, 0] = camera.camera_to_world(pred_ref_motion.joint_translation[:, 0])
                    pred_ref_motion._joint_rotation[:, 0] = convert_rotation_to_world(pred_ref_motion.joint_rotation[:, 0], camera)
                    pred_ref_motion.recompute_joint_global_info()
                # compute mpjpe eval loss using ground truth mocap data and predicted mocap data.
                # Here we can simply use mpjpe.
                # TODO: for human 3.6m dataset, we should use ground truth 3d pos, rather than retargeted motion
                if eval_mocap_gt is not None:
                    mpjpe_val = calc_motion_mpjpe(eval_mocap_gt, pred_ref_motion, True)
                    global_mpjpe_val = calc_motion_mpjpe(eval_mocap_gt, pred_ref_motion, False)
                    mpjpe_print_info = f"Evaluate file: {args.eval_attr_fname}, facing mpjpe = {mpjpe_val}, global mpjpe = {global_mpjpe_val}"
                    print(mpjpe_print_info, flush=True)
                    with open(os.path.join(save_dpath, "log.txt"), "w") as fout:
                        fout.write(mpjpe_print_info)

                if eval_mode == EvaluateMode.Estimation_2d:
                    # evaluate mpjpe using human3.6m ground truth position
                    # note: as character height in human3.6m data doesn't match
                    # human height of std-human model, we should scale the human height
                    # as we evaluate in facing coordinate, we can simply scale the root position..

                    # 1. resort the joint order as std-human.
                    # 2. compute head position
                    # 3. use 17 human3.6 subset to compute mpjpe in facing coordinate.
                    human36_pos_fname = ""
                    if human36_pos_fname:
                        facing_pred_motion: MotionData = motion_to_facing_coordinate(pred_ref_motion)
                        # compute head position (that is, we can estimate by head length..)
                        head_index = pred_ref_motion.joint_names.index("torso_head")
                        facing_head_pos = Rotation(facing_pred_motion.joint_orientation[:, head_index, :]).apply(
                            np.array([0.0, 0.1, 0.0])
                        ) + facing_pred_motion.joint_translation[:, head_index, :]
                        facing_gt_human36m = get_y_facing_coordinate(human36_pos_fname, convert_to_sub17=True)
                        new_order = [facing_pred_motion.joint_names.index(node) for node in stdhuman_with_root_names]
                        facing_joint_pos = facing_pred_motion.joint_position[:, new_order, :]
                        facing_pred_human36m = std_human_to_human36m(facing_joint_pos, facing_head_pos)
                        mpjpe_val = mpjpe(facing_gt_human36m, facing_pred_human36m)

            if args.also_output_contact_label:  # contact label.
                pred_contact_label: torch.Tensor = y_pred[:, contact_slice].detach()
                if args.normalize_contact_label:
                    # normalize by min_max
                    def normalize_contact_min_max() -> torch.Tensor:
                        pred_contact_max: torch.Tensor = torch.max(pred_contact_label, dim=-1, keepdim=True)[0]
                        pred_contact_min: torch.Tensor = torch.min(pred_contact_label, dim=-1, keepdim=True)[0]
                        pred_contact_div: torch.Tensor = pred_contact_max - pred_contact_min
                        pred_contact_div[pred_contact_div < 1e-5] = 1  # avoid divide by zero.
                        return (pred_contact_label - pred_contact_min) / pred_contact_div
                    pred_contact_label = normalize_contact_min_max()

                # evaluate the contact label by ground truth and given threshold..
                if eval_mode == EvaluateMode.TrainData:
                    def compute_contact_err(gt_contact_: torch.Tensor, pred_contact_: torch.Tensor):
                        threshold_: float = args.contact_label_eps
                        pred_contact_: torch.Tensor = pred_contact_.clone()
                        pred_contact_[pred_contact_ < threshold_] = 0
                        pred_contact_[pred_contact_ >= threshold_] = 1
                        delta_contact_: torch.Tensor = torch.abs(pred_contact_ - gt_contact_)
                        # compute the correct ratio..
                        contact_err_ratio_: torch.Tensor = torch.mean(delta_contact_)
                        return contact_err_ratio_

                    gt_contact: torch.Tensor = total_y[camera_index, :, contact_slice]
                    for body_index in range(num_body):
                        contact_err_i = compute_contact_err(gt_contact[:, body_index], pred_contact_label[:, body_index])
                        print(f"contact err {body_index}:{body_names[body_index]} = {contact_err_i.item()}")
                    total_contact_err = compute_contact_err(gt_contact, pred_contact_label)
                    print(f"total contact err = {total_contact_err.item()}")

                # print(f"contact predict error ratio = {contact_err_ratio.item()}")
                # # we need also compute contact error ratio for all the bodies..
                pred_contact_label: np.ndarray = pred_contact_label.cpu().numpy()
                pred_result["pred_contact_label"] = pred_contact_label

            # here, we can export the predicted local joint rotation
            pred_result.update({
                "invdyn_target": y_pred_invdyn_numpy,  # simulation fps == 100
                "camera_param": camera,
                "pos2d": joint_2d_pos[camera_index],
                "confidence": cam_confidence
            })

            print(y_pred_invdyn_numpy.shape, joint_2d_pos[camera_index].shape, cam_confidence.shape)

            if pred_ref_motion is not None:
                output_pred_bvh_fname_raw = "eval-mocap-predict.bvh"
                output_pred_bvh_fname = os.path.join(save_dpath, output_pred_bvh_fname_raw)
                pred_ref_motion = policy_base.to_bvh.insert_end_site(pred_ref_motion)
                # visualize in both window..
                # pred_ref_motion = smooth_motion_data(pred_ref_motion, GaussianBase(3), None, True, True)  # smooth the joint rotation
                BVHLoader.save(pred_ref_motion, output_pred_bvh_fname)
                print(f"save predict mocap data to {output_pred_bvh_fname_raw}")
                pred_result["pred_motion"] = output_pred_bvh_fname_raw
                if output_pred_bvh_fname is not None and output_gt_bvh_fname is not None and is_windows:
                    subprocess.Popen(["python", "-m", "VclSimuBackend.pymotionlib.editor",
                        "--bvh_fname", output_pred_bvh_fname, output_gt_bvh_fname])

            # output the evaluate result, for test samcon...
            policy_out_fname = os.path.join(save_dpath, f"network-output.bin")
            with open(policy_out_fname, "wb") as fout:
                pickle.dump(pred_result, fout)

            # process root height by contact label..
            if args.post_process_data:
                processed_fname = "pred-motion-method1.bvh"
                pre_process_motion_method1(policy_out_fname, 3, processed_fname, eval_mode == EvaluateMode.Estimation_2d)
                if is_windows:
                    cmd_list = ["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname",
                        os.path.join(save_dpath, processed_fname)]
                    if output_gt_bvh_fname is not None:
                        cmd_list.append(output_gt_bvh_fname)
                    subprocess.call(cmd_list)
            else:  # use a simple gaussian filter here..
                pass
            exit(0)

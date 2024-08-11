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

from argparse import Namespace, ArgumentParser
from enum import IntEnum
from mpi4py import MPI
import numpy as np
import os
import subprocess
import torch
import torch.nn as nn
from typing import Dict, Optional

from ...DiffODE.DiffHinge import DiffHingeInfo
from ...ODESim.Saver.CharacterToBVH import CharacterTOBVH
from ...Common.Helper import Helper
from ...Common.MathHelper import MathHelper, RotateType
from ...pymotionlib import BVHLoader
from ...pymotionlib.MotionData import MotionData


comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda")


class NormalizeMode(IntEnum):
    NO = 0
    MIN_MAX = 1
    MEAN_STD = 2
    MEAN_MIN_MAX = 3


class NetworkType(IntEnum):
    MLP = 0
    PoseFormer = 1  # same structure as PoseFormer


class EvaluateMode(IntEnum):
    TrainData = 0
    BVH_MOCAP = 1
    Estimation_2d = 2


class Joint2DType(IntEnum):
    StdHuman = 0  # Std Human has 19 real joints, and 1 empty root joints, total 20 joints
    StdHumanWithRoot = 1  # Std human real joints + root joint
    Coco17 = 2  # Alpha pose output has several types: coco 17 joints, and etc.
    OpenPose = 3  # Open Pose output
    Halpe26 = 4  # one of alpha pose output, 26 joints. not used in my code actually.
    MPII15 = 5  # we can view the hierarchy at https://pose.mpi-inf.mpg.de/
    Unified13 = 6  # we can use joint subset to compute 2d loss
    Human36 = 7


joint_2d_type_dim: Dict[Joint2DType, int] = {
    Joint2DType.StdHuman: 19,
    Joint2DType.Coco17: 17,
    Joint2DType.OpenPose: None,
    Joint2DType.Halpe26: 26,
    Joint2DType.MPII15: 15,
    Joint2DType.Unified13: 13,
    Joint2DType.Human36: 17
}

# class Loss2DNum(IntEnum):  # actually unused...
#    NumStdHuman = 0  # use original stdhuman joints to compute loss..
#    Num17 = 1  # use 17 joints to compute loss
#    Num14 = 2  # use 14 subset joints to compute loss


def get_output_dim(
    num_output_joint: int,  # not contains root joint..
    rotate_type: RotateType,
    also_output_pd_target: bool,
    also_output_local_rot: bool,
    also_output_contact_label: bool
) -> int:
    out_dim: int = 0
    rotate_size = MathHelper.get_rotation_dim(rotate_type)
    if also_output_pd_target:
        out_dim += num_output_joint * rotate_size
    if also_output_local_rot:  # also output root joint position
        out_dim += (num_output_joint + 1) * rotate_size + 3
    if also_output_contact_label:
        out_dim += num_output_joint + 1
    return out_dim


def build_mlp(
    num_window: int,
    num_input_joint: int,
    num_output_joint: int,
    hidden_dim: int,
    rotate_type: RotateType,
    also_output_pd_target: bool,
    also_output_local_rot: bool,
    also_output_contact_label: bool,
    dropout: float = 0.2,
    device = cpu_device
):
    """
    build a simple baseline network

    Args:
        num_window (int): [description]
        num_input_joint (int): [description]
        num_output_joint (int): [description]
        hidden_dim (int): [description]
        rotate_type (RotateType): [description]
        also_output_pd_target (bool): [description]
        also_output_local_rot (bool): [description]
        dropout (float, optional): [description]. Defaults to 0.2.
        device ([type], optional): [description]. Defaults to cpu_device.

    Returns:
        [type]: [description]
    """
    out_dim = get_output_dim(num_output_joint, rotate_type, also_output_pd_target, also_output_local_rot, also_output_contact_label)

    # I don't think there is a good method to initialize network weight of last layer
    # because output component must cover from -1 to 1 (contains 0)
    # so we cannot avoid the zero output of last layer by initialization..
    model = nn.Sequential(
        nn.Linear(num_window * num_input_joint * 2, hidden_dim),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim, bias=False),
    ).to(device)

    return model


def print_saved_args(dump_load_path: str):
    if os.path.exists(dump_load_path):
        result: Dict = torch.load(dump_load_path, map_location=cpu_device)
        print(result["args"])


def visualize_motion(
    to_bvh: CharacterTOBVH,
    root_pos: np.ndarray, root_quat: np.ndarray, joint_local_quat: np.ndarray, tmp_fname: Optional[str] = None) -> MotionData:
    if root_pos is None:
        root_pos = np.zeros((joint_local_quat.shape[0], 3))
    if root_quat is None:
        root_quat = MathHelper.unit_quat_arr((joint_local_quat.shape[0], 4))
    motion: MotionData = to_bvh.forward_kinematics(root_pos, root_quat, joint_local_quat)
    motion = to_bvh.insert_end_site(motion)
    motion.recompute_joint_global_info()

    if tmp_fname:
        BVHLoader.save(motion, tmp_fname)
        subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", tmp_fname])
    return motion


def get_out_slice(
    args: Namespace,
    num_joints: int = 19  # for std human model, there are 19 joints and 20 bodies.
) -> Dict[str, Optional[slice]]:
    start: int = 0
    rotate_dim: int = MathHelper.get_rotation_dim(args.rotate_type)

    all_rotate_slice = None
    if args.also_output_pd_target:
        ncount: int = rotate_dim * num_joints
        pd_target_slice = slice(start, start + ncount)
        all_rotate_slice = slice(start, start + ncount)
        start += ncount
        pd_target_in_rotate_offset = 0
    else:
        pd_target_slice = None
        pd_target_in_rotate_offset = None

    if args.also_output_local_rot:
        ncount: int = rotate_dim * (num_joints + 1)
        local_rot_slice = slice(start, start + ncount)
        all_rotate_slice = slice(start, start + ncount)

        if args.also_output_pd_target:
            kine_in_rotate_offset = num_joints
        else:
            kine_in_rotate_offset = 0

        start += ncount

        ncount: int = 3
        local_pos_slice = slice(start, start + ncount)
        start += ncount
    else:
        local_rot_slice = None
        local_pos_slice = None
        kine_in_rotate_offset = None

    if args.also_output_pd_target and args.also_output_local_rot:
        all_rotate_slice = slice(0, rotate_dim * (num_joints + num_joints + 1))

    if args.also_output_contact_label:
        ncount: int = num_joints + 1
        contact_slice = slice(start, start + ncount)
        start += ncount
    else:
        contact_slice = None

    result = {
        "all_rotate_slice": all_rotate_slice,
        "pd_target_slice": pd_target_slice,
        "local_rot_slice": local_rot_slice,
        "local_pos_slice": local_pos_slice,
        "contact_slice": contact_slice,
        "pd_target_in_rotate_offset": pd_target_in_rotate_offset,
        "kine_in_rotate_offset": kine_in_rotate_offset
    }

    return result


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="running mode")
    parser.add_argument("--frame_window", type=int, default=81, help="frame window")
    parser.add_argument("--network", type=str, choices=NetworkType._member_names_, default="PoseFormer")
    parser.add_argument("--rotate_type",
        type=str, choices=RotateType._member_names_, default="Vec6d", help="rotation type")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)

    parser.add_argument("--also_output_pd_target", action="store_true", default=True, help="output pd target in training")
    parser.add_argument("--also_output_local_rot", action="store_true", default=True, help="")
    parser.add_argument("--also_output_contact_label", action="store_true", default=True,
        help="also output contact label for each body."
        "For std-human model, there are 20 bodies, so the neural network also predicts 20 contact label.")

    parser.add_argument("--joint_2d_type", type=str, choices=Joint2DType._member_names_, default="Unified13")

    parser.add_argument("--loss_func", type=str, default="mse_loss", choices=["mse_loss", "l1_loss"])
    parser.add_argument("--loss_rotate_weight", type=float, default=5.0)
    parser.add_argument("--loss_root_pos_weight", type=float, default=2.0)
    parser.add_argument("--loss_contact_weight", type=float, default=0.5)
    parser.add_argument("--loss_fk", type=float, default=1, help="do forward kinematics to compute joint position loss")
    parser.add_argument("--loss_projection_2d", type=float, default=1,
        help="project the global 3d position to 2d in training process, with known camera parameter")
    parser.add_argument("--use_phys_loss", action="store_true", default=True, help="use physics loss with Diff-ODE")
    parser.add_argument("--w_phys_pos", type=float, default=5 * 1e-2)
    parser.add_argument("--w_phys_rot", type=float, default=1 * 1e-2)
    parser.add_argument("--w_phys_velo", type=float, default=1e-4 * 1e-2)
    parser.add_argument("--w_phys_omega", type=float, default=1e-4 * 1e-2)

    # in the code, we should save the camera parameter batchly.
    parser.add_argument("--need_do_fk", type=bool, help="need to do forward kinematics in training process.")

    parser.add_argument("--normalize_input_data", action="store_true",
        help="normalize the input data, actually, we need not to normalize if the input is in range [-1, 1]")
    parser.add_argument("--normalize_output_pos", action="store_true", default=True)
    parser.add_argument("--use_parallel", action="store_true", default=True)
    parser.add_argument("--use_mirror_data", action="store_true", default=True,
        help="left and right mirror for input character data.")

    parser.add_argument("--gradient_clip", type=float, default=10, help="we can use value 2 or 10 here.")  # clip the gradient at training
    parser.add_argument("--ignore_large_loss", type=float, default=None)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--noise2d", type=float, default=0.04,
        help="In MotioNet paper, the noise 2d is about 0.04. here the default value is None.")  # in MotioNet paper, the 2d noise is about 0.04
    parser.add_argument("--grad_accum_count", type=int, default=1,
        help="If GPU memory is not enough for training large neural network, we can divide the large batch size into"
        "several mini batches, then average gradients of mini batches.")

    parser.add_argument("--test_time_usage", action="store_true")
    parser.add_argument("--pretrain_result", type=str, default="")
    parser.add_argument("--partial_pretrain_result", type=str, default="")
    parser.add_argument("--not_load_lr", action="store_true")

    parser.add_argument("--eval_mode", type=str, default="BVH_MOCAP", choices=EvaluateMode._member_names_)
    parser.add_argument("--eval_attr_fname", type=str, default="")

    parser.add_argument("--mlp_dropout", type=float, default=0.4)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--fps_2d", type=int, default=50)
    parser.add_argument("--simulation_fps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096)  # the original paper use batch size of 1024. they says performance of lower batch size is not good.

    parser.add_argument("--resample_output", action="store_true", default=True)
    parser.add_argument("--smooth_eval_motion_pos", action="store_true", default=True)
    parser.add_argument("--smooth_eval_motion_rotate", action="store_true", default=True)
    parser.add_argument("--post_process_data", action="store_true", default=True)

    parser.add_argument("--only_use_human36_data", action="store_true")

    parser.add_argument("--input_data_dir", type=str, default="The input data directory")
    parser.add_argument("--output_data_dir", type=str, default="")

    parser.add_argument("--normalize_contact_label", action="store_true", default=True,
        help=f"Normalize the predicted contact label. (maybe we can normalize by min/max, or sigmoid..)")
    parser.add_argument("--contact_label_eps", type=float, default=0.6, help="threshold for contact.")

    parser.add_argument("--render_result", action="store_true", default=False)

    # convert str to enum
    args: Namespace = parser.parse_args()
    args.need_do_fk = (args.loss_fk > 0) or (args.loss_projection_2d > 0) or args.use_phys_loss  # do forward kinematics in training neural network

    args.eval_mode = EvaluateMode[args.eval_mode]
    args.joint_2d_type = Joint2DType[args.joint_2d_type]
    args.rotate_type = RotateType[args.rotate_type]
    args.rotate_shape = MathHelper.get_rotation_last_shape(args.rotate_type)
    args.network = NetworkType[args.network]

    if args.mode == "train":
        args.simulation_fps = 100

    if args.mode == "eval":  # not using noise in test case..
        args.noise2d = 1e-6

    if not torch.cuda.is_available():
        args.device = cpu_device
    else:
        args.device = torch.device(args.device)

    if comm_rank == 0 and args.mode == "train":
        print(args)

    args.out_slice = get_out_slice(args)
    args.smooth_eval_motion = args.smooth_eval_motion_pos or args.smooth_eval_motion_rotate

    # set random seed here
    Helper.set_torch_seed(args.random_seed)

    DiffHingeInfo.enable_limit = False # not use hinge angle limit here..

    if args.output_data_dir is None or len(args.output_data_dir) == 0:
        args.output_data_dir = os.path.abspath(fdir)
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir, exist_ok=True)

    return args

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
If you use 3090 graphics card, please use the newest version pytorch
othervize, it will cost a lot of time on method .to(device)
"""
import os
# cpu_num = 1
# os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import torch
# torch.set_num_threads(cpu_num)

from mpi4py import MPI
import numpy as np
from typing import List

from VclSimuBackend.DiffODE.config import ContactConfig

from VclSimuBackend.Samcon.PolicyTrain.Common import get_arguments, NetworkType, EvaluateMode
from VclSimuBackend.Samcon.PolicyTrain.Policy3dBase import Policy3dBase
from VclSimuBackend.Samcon.PolicyTrain.PolicyEvaluate import evaluate_new
from VclSimuBackend.Samcon.PolicyTrain.SimulationLoss import SimulationLoss, run_child_worker, simu_loss_init
from VclSimuBackend.Samcon.PolicyTrain.Train import train
from VclSimuBackend.Samcon.SamconMainWorkerBase import SamHlp

from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase

from VclSimuBackend.Utils.AlphaPose.Utils import load_alpha_pose
from VclSimuBackend.Utils.Dataset.COCOUtils import coco_to_unified
from VclSimuBackend.Utils.Dataset.Human36 import parse_human36_video_camera
from VclSimuBackend.Utils.Dataset.StdHuman import stdhuman_to_unified

from VclSimuBackend.pymotionlib import BVHLoader


comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
fdir = os.path.dirname(__file__)

def check_alphapose_result():
    args = get_arguments()
    args.mode = "eval"
    conf_fname = os.path.join(fdir, "../../../Tests/CharacterData/SamconConfig-duplicate.json")
    policy_3d_base = Policy3dBase(SamHlp(conf_fname, 0), args=args)

    bvh_fname = r"D:\song\documents\GitHub\ode-samcon\Tests\CharacterData\Human36Reheight\S1\Greeting-mocap-100.bvh"
    alphapose_fname = r"Z:\human3.6m_downloader-master\training\processing\alpha-pose-result\S1\Greeting.54138969.json"
    joint_2d_pos, confidence = load_alpha_pose(alphapose_fname)
    subset: List[int] = coco_to_unified
    camera_params = [parse_human36_video_camera(alphapose_fname)]
    joint_2d_pos: np.ndarray = camera_params[0].normalize_screen_coordinates(joint_2d_pos)
    joint_2d_pos: np.ndarray = joint_2d_pos[:, subset, :]
    confidence = confidence[:, subset]

    mocap = BVHLoader.load(bvh_fname)
    # mocap_joint_2d = camera_params[0].world_to_camera(mocap.joint_position)
    target = BVHToTargetBase(mocap, 100, policy_3d_base.character).init_target()
    unified_joint_pos: np.ndarray = stdhuman_to_unified(target.globally.pos)
    pos3d = camera_params[0].world_to_camera(unified_joint_pos)
    mocap_pos2d = camera_params[0].project_to_2d_linear(pos3d)[::2]

    print(mocap_pos2d.shape)


def main_backup():
    raise ValueError
    # for PoseFormer model, on 3090 GPU card with 27 frames as input, time cost is 0.008sec,
    # so maybe it can play in real time..
    # Note: the simple MLP model trends to overfit..
    # python TrainPolicyMain.py --mode eval --eval_mode BVH_MOCAP --eval_attr_fname D:\song\documents\GitHub\ode-scene\Tests\CharacterData\WalkF-mocap-100.bvh
    # python TrainPolicyMain.py --mode eval --eval_mode TrainData --eval_attr_fname D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Samcon\0\S5-Directions2-mocap-100.pickle

    args = get_arguments()
    # args.mode = "train"
    args.mode = "eval"
    # args.pretrain_result = None
    # args.pretrain_result = r"D:\song\documents\github\ode-scene\VclSimuBackend\Samcon\PolicyTrain\policy_train.ckpt"
    # args.pretrain_result = r"D:\song\document\GitHub\ode-develop\Tests\CharacterData\Samcon\0\policy_train.ckpt160"
    args.pretrain_result = r"G:\Samcon-Exps\learning-unified\Transformer81\policy_train.ckpt200"
    # args.pretrain_result = r"G:\Samcon-Exps\learning-unified\Transformer81\policy_train.ckpt140-only-human36-wrong-pos-train"
    # if args.mode == "eval" and not os.path.exists(args.pretrain_result):
    #    print(args.pretrain_result, "not exist")
    #    exit(0)
    if args.mode == "eval":
        args.noise2d = 1e-6

    # args.eval_attr_fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\WalkF-mocap-100.bvh"
    # args.eval_attr_fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Human36Reheight\S11\SittingDown1-mocap-100.bvh"
    # args.eval_attr_fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Human36Reheight\S11\Walking-mocap-100.bvh"
    # args.eval_mode = EvaluateMode.BVH_MOCAP
    args.eval_attr_fname = r"Z:\alphapose\AlphaPose\floor_10\alphapose-results.json"
    args.eval_mode = EvaluateMode.Estimation_2d
    args.output_data_dir = "wild-sitting-floor10"
    # args.start_frame = 1343
    # args.end_frame = 1715

    args.render_result = True
    # args.eval_attr_fname = r"G:\Samcon-Exps\current-results\S1-Directions-mocap-100.pickle"
    # args.eval_mode = EvaluateMode.TrainData
    args.noise2d = 1e-6

    args.network = NetworkType.PoseFormer
    # args.network = NetworkType.MLP

    conf_fname = os.path.join(fdir, "../../../Tests/CharacterData/SamconConfig-duplicate.json")
    policy_3d_base = Policy3dBase(SamHlp(conf_fname, 0), args=args)
    if args.mode == "train":
        train(policy_3d_base)
    elif args.mode == "eval":
        evaluate_new(policy_3d_base)
    else:
        raise NotImplementedError


def main():
    ContactConfig.use_ball_as_contact = True
    ContactConfig.use_inf_friction = False

    args = get_arguments()
    conf_fname = os.path.join(fdir, "../../../Tests/CharacterData/SamconConfig-duplicate.json")
    simu_loss_init(args, conf_fname)
    if comm_rank == 0:
        print(f"start main worker {comm_rank} / {comm_size}", flush=True)
        policy_3d_base: Policy3dBase = Policy3dBase(SamHlp(conf_fname, 0), args=args)
        if args.mode == "train":
            train(policy_3d_base)
        elif args.mode == "eval":
            evaluate_new(policy_3d_base)
        else:
            raise NotImplementedError
    else:
        if args.mode == "train":
            run_child_worker()
        else:
            pass  # we need to do nothing here now.


if __name__ == "__main__":
    main_backup()

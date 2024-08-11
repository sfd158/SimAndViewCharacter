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

from argparse import ArgumentParser, Namespace
import numpy as np
import os
import torch
import pickle
from typing import List

from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.Loader.JsonCharacterLoader import JsonCharacterLoader
from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH
from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import DirectTrajOptBVH, SamHlp, SamconWorkerFull
from VclSimuBackend.Samcon.SamconWorkerBase import WorkerInfo

from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo, generate_hack_contact
from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData


fdir = os.path.dirname(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_fname", type=str, default=os.path.join(fdir, "../../../Tests/CharacterData/SamconConfig-duplicate.json"))
    parser.add_argument("--start", type=int, default=4333)
    parser.add_argument("--end", type=int, default=4637)
    parser.add_argument("--samcon_result", type=str, default=r"G:\Samcon-Exps\current-results\lafan-pushAndStumble1_subject3.pickle")
    parser.add_argument("--ref_bvh_result", type=str, default=r"G:\lafan-mocap-100\pushAndStumble1_subject3.bvh")
    parser.add_argument("--use_hack_contact", action="store_true", default=False)
    parser.add_argument("--use_samcon_invdyn", action="store_true", default=False)
    # if the ref_bvh_result is None, we will use samcon result as reference motion
    # else, we will use bvh as reference motion.

    args: Namespace = parser.parse_args()
    # args.ref_bvh_result = r"D:\song\Documents\GitHub\ode-samcon\Tests\CharacterData\lafan-mocap-100\pushAndStumble1_subject3.bvh"
    # args.samcon_result = r"D:\lafan-pushAndStumble1_subject3.pickle"
    # args.ref_bvh_result = None
    return args


def track_samcon_data():
    args = parse_args()
    samhlp = SamHlp(args.config_fname)
    conf = samhlp.conf
    conf["inverse_dynamics"]["in_use"] = False
    conf["traj_optim"]["load_samcon_result"] = False
    scene: ODEScene = DirectTrajOptBVH.load_scene_with_conf(conf, None)
    character: ODECharacter = scene.character0
    # render = RenderWorld(scene.world)  # visualize in Long Ge's Framework

    if args.ref_bvh_result:
        # when we run samcon, the first 5 frames and the last 5 frames are dropped for inverse dynamics.
        ref_bvh_result: MotionData = BVHLoader.load(args.ref_bvh_result).sub_sequence(args.start - 5, args.end + 5)
    else:
        ref_bvh_result = None

    with open(args.samcon_result, "rb") as fin:
        samcon_result: List[BodyInfoState] = pickle.load(fin)[args.start:args.end]

    to_bvh = CharacterTOBVH(character, int(scene.sim_fps))
    to_bvh.build_hierarchy()
    for index, state in enumerate(samcon_result):
        character.load(state)
        to_bvh.append_no_root_to_buffer()
    mocap = to_bvh.to_file("test.bvh")
    mocap.recompute_joint_global_info()

    conf["filename"]["bvh"] = mocap if ref_bvh_result is None else ref_bvh_result

    contact_info_list: List[DiffContactInfo] = generate_hack_contact(scene, character, samcon_result)
    contact_pos_list: List[torch.Tensor] = [node.contact_pos.detach().clone() for node in contact_info_list]

    # note: we can also use samcon inverse dyanmics result here.. as we only need target pose in quaternion format.
    pd_target: np.ndarray = np.concatenate([node.pd_target[None, ...] for node in samcon_result])
    worker_info = WorkerInfo()
    worker = SamconWorkerFull(samhlp, worker_info, scene)
    main_worker = DirectTrajOptBVH(samhlp, worker_info, worker, scene, inv_dyn_target_quat=pd_target if args.use_samcon_invdyn else None)

    if args.use_hack_contact:
        worker.contact_info_list = contact_info_list
        worker.torch_loss.hack_contact_pos = contact_pos_list
        print("use hack contact..")

    # add visualize for hack..
    scene_dict = pickle.load(open(conf["filename"]["world"], "rb"))
    character_dict = scene_dict["CharacterList"]["Characters"][0]
    ref_character = JsonCharacterLoader(scene.world, scene.space).load(character_dict)
    ref_character.is_enable = False

    if ref_bvh_result is None:
        def ref_set_func(frame: int):
            ref_character.load(samcon_result[frame % len(samcon_result)])
    else:
        set_ref_tar = SetTargetToCharacter(ref_character, main_worker.target.pose)
        def ref_set_func(frame: int):
            set_ref_tar.set_character_byframe(frame)

    main_worker.in_sim_hack_func = ref_set_func
    worker.in_sim_hack_func = ref_set_func  # for visualize reference character using Long Ge's framework
    # render.start()
    main_worker.test_direct_trajopt()

    # now, remove contact hack, and train again..
    # worker.contact_info_list = None
    # worker.torch_loss.hack_contact_pos = None
    # main_worker.initial_lr = 1e-3
    # main_worker.test_direct_trajopt()


if __name__ == "__main__":
    track_samcon_data()

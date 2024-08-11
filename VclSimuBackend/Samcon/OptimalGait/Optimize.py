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
Optimize by transformer output.
We need also modify the ankle and toe rotation,
to make sure that if the foot collide with the floor.

if both heel and toe collides with the ground, the foot should be flatten.
Note that if the heel body contact label is 1,

sometimes there is no contact at the heel,
contact only occurs at the front part.

sometimes there is no contact at the front part,
contact only occurs at the behind part.
"""
from argparse import ArgumentParser, Namespace
import numpy as np
import os
import pickle
import torch
# torch.autograd.set_detect_anomaly(True)

from typing import Any, List, Tuple, Dict, Optional, Union
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.TargetPose import TargetPose
from VclSimuBackend.ODESim.UpdateSceneBase import UpdateSceneBase
from VclSimuBackend.ODESim.Loader.JsonCharacterLoader import JsonCharacterLoader
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import DirectTrajOptBVH, SamHlp, SamconWorkerFull
from VclSimuBackend.Samcon.SamconWorkerBase import LoadTargetPoseMode, WorkerInfo
from VclSimuBackend.Samcon.OptimalGait.ContactWithKinematic import ContactLabelExtractor

from VclSimuBackend.DiffODE import DiffQuat
from VclSimuBackend.DiffODE.DiffContact import generate_hack_contact, DiffContactInfo
from VclSimuBackend.DiffODE.DiffODEWorld import DiffODEWorld
from VclSimuBackend.DiffODE.DiffFrameInfo import DiffFrameInfo
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2

fdir = os.path.dirname(__file__)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config_fname", type=str, default=os.path.join(fdir, "../../../Tests/CharacterData/SamconConfig-duplicate.json"))
    parser.add_argument("--process_fname", type=str, default=os.path.join(fdir, "../../../Tests/CharacterData/Samcon/0/network-output.bin"))
    parser.add_argument("--contact_eps", type=float, default=0.6)
    args: Namespace = parser.parse_args()
    return args


def main(scene: Optional[ODEScene] = None):
    args = parse_args()
    samhlp: SamHlp = SamHlp(args.config_fname)
    conf = samhlp.conf
    conf["inverse_dynamics"]["in_use"] = False
    conf["traj_optim"]["load_samcon_result"] = False
    if scene is None:
        scene: ODEScene = DirectTrajOptBVH.load_scene_with_conf(conf, None)
    scene.use_soft_contact = True
    scene.soft_cfm = 1e-3
    scene.soft_cfm_tan = 1e-3
    scene.soft_erp = 0.2
    print(f"scene simulation fps = {scene.sim_fps}")
    character: ODECharacter = scene.character0
    worker_info = WorkerInfo()
    worker = SamconWorkerFull(samhlp, worker_info, scene)
    main_worker: DirectTrajOptBVH = DirectTrajOptBVH(samhlp, worker_info, worker, scene, character, LoadTargetPoseMode.PICKLE_TARGET)
    target: TargetPose = main_worker.target.pose

    # generate contact sequence
    extractor = ContactLabelExtractor(scene, character)
    contact_info = extractor.handle_mocap(target, False, main_worker.contact_label, args.contact_eps)
    print(f"After generate contact sequence")
    return main_worker, worker, extractor, contact_info
    # for debug, we should visualize these contacts in Unity..


def main_optimize(main_worker: DirectTrajOptBVH, worker: SamconWorkerFull, extractor: ContactLabelExtractor, contact_info):
    """
    Optimize the kinematics motion and given contact sequence.
    1. convert the contact to DiffODE
    2. compute contact local position, for computing the contact height loss
    3. optimize. I think if the initial inverse dynamics solution is given,
    it will be easier to optimize.
    """
    contact_info_list: List[DiffContactInfo] = extractor.convert_to_diff_contact(*contact_info, main_worker.target.pose)
    worker.contact_info_list = contact_info_list

    # for visualize using Long Ge's framework, we should add an additional reference character.
    scene_dict = pickle.load(open(main_worker.conf["filename"]["world"], "rb"))
    character_dict = scene_dict["CharacterList"]["Characters"][0]
    ref_character = JsonCharacterLoader(main_worker.scene.world, main_worker.scene.space).load(character_dict)
    ref_character.is_enable = False

    set_ref_tar = SetTargetToCharacter(ref_character, main_worker.target.pose)
    def ref_set_func(frame: int):
        set_ref_tar.set_character_byframe(frame)

    worker.in_sim_hack_func = ref_set_func
    render: RenderWorld = RenderWorld(main_worker.scene)
    render.start()

    main_worker.test_direct_trajopt()


class ContactGenerateUpdate(UpdateSceneBase):
    """
    For visualize the contact sequence.
    """
    def __init__(self, scene: Optional[ODEScene], main_worker, worker, extractor, contact_info):
        super().__init__(scene)
        self.main_worker: DirectTrajOptBVH = main_worker
        self.worker: SamconWorkerFull = worker
        self.extractor: ContactLabelExtractor = extractor
        self.contact_info = contact_info
        self.tot_frame: int = min(self.main_worker.target.num_frames, len(contact_info[0]))
        self.frame: int = 0

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        self.main_worker.tar_set.set_character_byframe(self.frame)
        # add hacked contact
        self.extractor.create_ode_contact_joint(self.contact_info[1][self.frame], self.contact_info[2][self.frame])
        self.frame = (self.frame + 1) % self.tot_frame


class ContactGenerateServer(ServerForUnityV2):
    """
    For visualize the contact sequence.
    """
    def __init__(self):
        super().__init__()

    def after_load_hierarchy(self):
        self.scene.set_sim_fps(100)
        main_ret = main(self.scene)
        self.update_scene: ContactGenerateUpdate = ContactGenerateUpdate(self.scene, *main_ret)


class PhysMotionCollideUpdate(UpdateSceneBase):
    """
    For test our collision detection. The input motion is physically plausiable (e.g. samcon result)
    """
    def __init__(self, scene: Optional[ODEScene], samcon_result: List[BodyInfoState]):
        super().__init__(scene)
        self.samcon_result = samcon_result
        self.tot_frame = len(self.samcon_result)
        self.extractor = ContactLabelExtractor(scene, self.character0)
        self.frame: int = 0

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        self.character0.load(self.samcon_result[self.frame])
        pos, quat = self.character0.get_body_pos(), self.character0.get_body_quat()
        ret_index, ret_pos, ret_label = self.extractor.compute_contact(pos, quat)
        self.extractor.create_ode_contact_joint(ret_pos, ret_label)
        self.frame = (self.frame + 1) % self.tot_frame


class PhysMotionCollideServer(ServerForUnityV2):
    """
    For test our collision detection. The input motion is physically plausiable (e.g. samcon result)
    """
    def __init__(self):
        super().__init__()
        fname = r"G:\Samcon-Exps\current-results\lafan-fallAndGetUp3_subject1.pickle"
        with open(fname, "rb") as fin:
            self.samcon_result: List[BodyInfoState] = pickle.load(fin)

    def after_load_hierarchy(self):
        self.scene.set_sim_fps(100)
        self.update_scene = PhysMotionCollideUpdate(self.scene, self.samcon_result)


if __name__ == "__main__":
    main_optimize(*main())
    # server = ContactGenerateServer()
    # server.run()

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
Visualize the contact label by neural network prediction in Unity..
It seems that in MLP, the walking contact is just OK.

Pipeline:
1. load character from Unity, load neural network prediction from file
2. set the contact label in Unity
"""

from ast import parse
import numpy as np
import os
import pickle
from typing import Dict, Optional, Any
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.TargetPose import TargetPose, SetTargetToCharacter
from VclSimuBackend.Samcon.SamHlp import SamHlp
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, UpdateSceneBase
from VclSimuBackend.Samcon.OptimalGait.OptimizeEachFrameParallel import OptimizeParallel, parse_args


fdir = os.path.dirname(__file__)
conf_name = os.path.join(fdir, "../CharacterData/SamconConfig-duplicate.json")


class ContactVisUpdate(UpdateSceneBase):
    def __init__(self, scene: Optional[ODEScene], target: TargetPose, contact_label: np.ndarray):
        super().__init__(scene)
        self.visualize_color = self.character0.body_info.visualize_color
        self.target = target
        self.contact_label: np.ndarray = contact_label
        self.num_frame = min(self.target.num_frames, contact_label.shape[0])
        self.frame = 0
        self.tar_set = SetTargetToCharacter(scene.character0, self.target)

        print(f"contact_label.shape = {contact_label.shape}")
        print(f"visualize_color = {type(self.visualize_color)}, len(visualize_color) = {len(self.visualize_color)}")

    def update_by_color(self, mess_dict: Optional[Dict[str, Any]] = None):
        self.tar_set.set_character_byframe(self.frame)
        self.visualize_color.clear()
        # print(
        #    f"label.max = {self.contact_label[self.frame].max()},"
        #    f"label.min = {self.contact_label[self.frame].min()}"
        # )
        for body_idx, label in enumerate(self.contact_label[self.frame]):
            # if label == 0:
            #    self.visualize_color.append(None)
            # else:
            #    self.visualize_color.append([1.0, 0.0, 0.0])
            self.visualize_color.append([float(label), 0.0, 0.0])
        self.frame = (self.frame + 1) % self.num_frame

    # def initial_diffode(self):
    #   pass

    def update_by_simulation(self):
        # Here we can visualize the contact by forward simulation..
        # After the hacked contact is generated, we can simply simulate in ODE
        pass

    def update(self, mess_dict):
        return self.update_by_color()


class ContactVisServer(ServerForUnityV2):
    def __init__(self):
        super().__init__()

    def after_load_hierarchy(self):
        fname = r"D:\song\Documents\GitHub\ode-develop\Tests\CharacterData\Samcon\0\network-output.bin"
        self.scene.set_sim_fps(100)
        samhlp = SamHlp(conf_name)
        motion, samcon_target, invdyn_target, confidence, contact_label, camera = samhlp.load_inv_dyn_from_pickle(self.scene, self.scene.character0, fname)
        self.update_scene = ContactVisUpdate(self.scene, samcon_target.pose, contact_label)


class OptimizationUpdate(UpdateSceneBase):
    def __init__(self, scene: Optional[ODEScene], optim: OptimizeParallel, simu_tar_set: SetTargetToCharacter):
        super().__init__(scene)
        self.optim = optim
        self.simu_tar_set = simu_tar_set
        self.visualize_color = self.simu_tar_set.character.body_info.visualize_color
        self.frame: int = 0
        self.num_frame = self.simu_tar_set.num_frames
        self.scene.extract_contact = True
        self.args = self.optim.args

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        # print(f"call update")
        self.simu_tar_set.set_character_byframe(self.frame)
        self.optim.gt_tar_set.set_character_byframe(self.args.index_t + self.frame)
        # do forward simulation to generate hacked contact.
        diff_contact = self.optim.contact_info_list[self.frame]
        if diff_contact is not None and False:
            diff_contact.generate_ode_contact(self.scene, self.simu_tar_set.character)
            self.scene.damped_simulate_once()
            self.optim.gt_tar_set.set_character_byframe(self.args.index_t + self.frame, self.simu_tar_set.character)

        self.visualize_color.clear()
        for body_idx, label in enumerate(self.optim.contact_label[self.frame]):
            # if label == 0:
            #    self.visualize_color.append(None)
            # else:
            #    self.visualize_color.append([1.0, 0.0, 0.0])
            self.visualize_color.append([float(label), 0.0, 0.0])
        self.frame = (self.frame + 1) % self.num_frame


class OptimizeUpdateServer(ServerForUnityV2):
    # Here we should use 2 characters.
    def __init__(self):
        super().__init__()
        self.init_instruction_buf = {"DupCharacterNames": [self.gt_str, self.sim_str]}

    def after_load_hierarchy(self):
        self.scene.set_sim_fps(100)
        optim = OptimizeParallel(parse_args(), self.scene)
        self.update_scene = OptimizationUpdate(self.scene, optim, optim.get_simu_target_set())


def main():
    server = OptimizeUpdateServer()
    server.run()


if __name__ == "__main__":
    main()

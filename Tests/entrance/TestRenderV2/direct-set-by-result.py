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

import copy
import pickle
from scipy.spatial.transform import Rotation
import torch
from typing import Optional, Dict, Any, List
from VclSimuBackend.ODESim.PDControler import DampedPDControler
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from VclSimuBackend.ODESim.UpdateScene.UpdateBVHWrapper import UpdateSceneBase
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2
from VclSimuBackend.Common.GetFileNameByUI import get_file_name_by_UI as get_from_UI
from VclSimuBackend.DiffODE.DiffPDControl import DiffDampedPDControler
from VclSimuBackend.DiffODE.Build import BuildFromODEScene


class UpdateSceneResult(UpdateSceneBase):
    def __init__(self, scene: ODEScene, path: List[BodyInfoState]):
        self.scene = scene
        self.path = path
        self.frame = 0
        self.pd_ctrl = DampedPDControler(self.character0)

        self.builder = BuildFromODEScene(self.scene)
        self.diff_world = self.builder.build()
        self.torch_stable_pd: DiffDampedPDControler = DiffDampedPDControler.build_from_joint_info(self.character0.joint_info)

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        self.drive_by_diffode_simu_hack()

    def drive_no_simu(self, mess_dict: Optional[Dict[str, Any]] = None):  # no simulation
        self.scene.str_info = f"{self.frame}/{len(self.path)}"
        self.character0.load(self.path[self.frame])
        self.scene.compute_collide_info()
        self.frame = (self.frame + 1) % len(self.path)

    def drive_by_ode_simu(self):
        self.scene.str_info = f"{self.frame}/{len(self.path)}"
        if self.frame == 0:
            self.character0.load(self.path[0])
        if self.frame < len(self.path) - 1:
            pd_target = self.path[self.frame + 1].pd_target
            if pd_target.shape[-1] == 3:
                pd_target = Rotation.from_rotvec(pd_target).as_quat()
            self.pd_ctrl.add_torques_by_quat(pd_target)
            self.scene.damped_simulate_once()
        self.frame = (self.frame + 1) % len(self.path)

    def drive_by_diffode_simu(self):
        self.scene.str_info = f"{self.frame}/{len(self.path)}"
        if self.frame == 0:
            self.diff_world.import_from_state(self.path[0])
        if self.frame < len(self.path) - 1:
            pd_target = self.path[self.frame + 1].pd_target
            if pd_target.shape[-1] == 3:
                pd_target = Rotation.from_rotvec(pd_target).as_quat()
            pd_target = torch.from_numpy(pd_target)
            self.diff_world.curr_frame.stable_pd_control_wrapper(self.torch_stable_pd, pd_target)
            self.diff_world.step()
        self.frame = (self.frame + 1) % len(self.path)

    def drive_by_diffode_simu_hack(self):
        self.scene.str_info = f"{self.frame}/{len(self.path)}"
        self.diff_world.import_from_state(self.path[self.frame])
        if self.frame < len(self.path) - 1:
            pd_target = self.path[self.frame + 1].pd_target
            if pd_target.shape[-1] == 3:
                pd_target = Rotation.from_rotvec(pd_target).as_quat()
            # simulate by ode
            self.pd_ctrl.add_torques_by_quat(pd_target)
            self.scene.damped_simulate_once()
            contact_info = copy.deepcopy(self.scene.contact_info)

            self.diff_world.import_from_state(self.path[self.frame])
            pd_target = torch.from_numpy(pd_target)
            self.diff_world.curr_frame.stable_pd_control_wrapper(self.torch_stable_pd, pd_target)
            self.diff_world.step()
            self.scene.contact_info = contact_info

        self.frame = (self.frame + 1) % len(self.path)


class ServerResult(ServerForUnityV2):
    def __init__(self, result_fname: str):
        super(ServerResult, self).__init__()
        with open(result_fname, "rb") as fin:
            self.path = pickle.load(fin)

    def after_load_hierarchy(self):
        self.update_scene = UpdateSceneResult(self.scene, self.path)


def main():
    result_fname = get_from_UI(initialdir='../../../../Resource')
    if not result_fname:
        return
    server = ServerResult(result_fname)
    server.run()


if __name__ == "__main__":
    main()

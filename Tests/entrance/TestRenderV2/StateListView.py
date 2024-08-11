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

import os
import pickle
from typing import List, Any, Dict, Optional
from VclSimuBackend.Common.GetFileNameByUI import get_file_name_by_UI
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.UpdateSceneBase import UpdateSceneBase
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from VclSimuBackend.ODESim.PDControler import DampedPDControler
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, ODEScene


class TestUpdate(UpdateSceneBase):
    def __init__(self, scene: ODEScene, character: ODECharacter, best_path: List[BodyInfoState]):
        super(TestUpdate, self).__init__(scene)
        self.best_path = best_path
        self.frame = 0
        if character is None:
            character = self.character0
        self.character = character
        self.damped_pd = DampedPDControler(self.character)

    def update_by_direct_set(self, mess_dict=None):
        self.scene.str_info = f"{self.frame}"
        self.character0.load(self.best_path[self.frame])
        self.scene.damped_simulate_once()  # only for visualize collision...
        self.character0.load(self.best_path[self.frame])
        self.frame = (self.frame + 1) % len(self.best_path)

    def update_by_track(self, mess_dict=None):
        self.scene.str_info = f"{self.frame}"
        state = self.best_path[self.frame]
        self.character0.load(state)
        pd_target = state.pd_target
        self.damped_pd.add_torques_by_quat(pd_target)
        self.scene.damped_simulate_once()
        self.frame = (self.frame + 1) % len(self.best_path)

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        self.update_by_track()


class TestServer(ServerForUnityV2):
    def __init__(self, best_path):
        super(TestServer, self).__init__()
        self.best_path = best_path

    def after_load_hierarchy(self):
        self.update_scene = TestUpdate(self.scene, self.scene.character0, self.best_path)


def main():
    fname = get_file_name_by_UI()
    if not os.path.exists(fname):
        return
    with open(fname, "rb") as fin:
        best_path = pickle.load(fin)
    if isinstance(best_path, dict):
        best_path = best_path["best_path"]
    server = TestServer(best_path)
    server.run()


if __name__ == "__main__":
    main()

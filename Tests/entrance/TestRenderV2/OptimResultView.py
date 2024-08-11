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
from VclSimuBackend.Samcon.SamconWorkerBase import SamHlp
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, UpdateSceneBase, ODEScene, ODECharacter
from VclSimuBackend.ODESim.PDControler import DampedPDControler


class OptimWithMultiStage(UpdateSceneBase):
    def __init__(self, scene: ODEScene, character: ODECharacter, dir_name: str):
        super().__init__(scene)
        self.best_path = None
        self.frame = 0
        if character is None:
            character = self.character0
        self.character = character
        self.damped_pd = DampedPDControler(self.character)
        self.frame = 0
        
        self.fname_list = self.get_fname_list(dir_name)
        self.fname_index = 0
        self.curr_file = ""

        self.load_pickle_file()

    def load_pickle_file(self):
        self.frame = 0
        self.curr_file = os.path.split(self.fname_list[self.fname_index])[1][:-7]
        self.curr_file = self.curr_file.replace("kstart", "s")
        self.curr_file = self.curr_file.replace("epoch", "e")

        with open(self.fname_list[self.fname_index], "rb") as fin:
            self.best_path = pickle.load(fin)
        self.fname_index += 1
        if self.fname_index == len(self.fname_list):
            self.fname_index = 0

    @staticmethod
    def get_fname_list(dir_name: str):
        ret_list = [fname.split("-") for fname in os.listdir(dir_name)
                    if fname.startswith("kstart") and fname.endswith(".pickle")]
        ret_list = [[node[0], int(node[1]), node[2], int(node[3].split(".")[0])] for node in ret_list]
        ret_list.sort(key=lambda x: (x[1], x[3]))
        ret_list = [os.path.join(dir_name, f"{node[0]}-{node[1]}-{node[2]}-{node[3]}.pickle") for node in ret_list]
        return ret_list

    def update(self, mess_dict=None):
        if self.world_signal is not None:
            pass

        self.scene.str_info = f"{self.curr_file}-f-{self.frame}"
        state = self.best_path[self.frame]
        self.character0.load(state)
        pd_target = state.pd_target
        self.damped_pd.add_torques_by_quat(pd_target)
        self.scene.damped_simulate_once()
        self.frame = (self.frame + 1) % len(self.best_path)
        self.frame += 1
        if self.frame == len(self.best_path):
            self.load_pickle_file()


class OptimViewServer(ServerForUnityV2):
    def __init__(self, dir_name: str):
        super().__init__()
        self.dir_name = dir_name

    def after_load_hierarchy(self):
        self.update_scene = OptimWithMultiStage(self.scene, self.scene.character0, self.dir_name)


def main():
    # dir_name = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Samcon\0"
    dir_name = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Samcon\trajopt-discussion-s1"
    server = OptimViewServer(dir_name)
    server.run()


if __name__ == "__main__":
    main()

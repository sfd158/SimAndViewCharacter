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
This is used for generate teaser in Unity
Input: character config file, body info state
Show character in Unity
"""
import numpy as np
import os
import pickle
from typing import Optional, Dict, Any, List
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, UpdateSceneBase, ODEScene, JsonSceneLoader
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter

fdir = os.path.dirname(__file__)

class TeaserServer(ServerForUnityV2):
    """
    Here we should modify the update function
    Not create too much characters..
    """
    def __init__(self):
        super().__init__()
        self.sub_state = self.generate_state()
        self.init_instruction_buf = {
            "DupCharacterNames": [f"ch{idx}" for idx, node in enumerate(self.sub_state)]
        }
        print(self.init_instruction_buf)

    @staticmethod
    def generate_state() -> List[BodyInfoState]:
        """
        Only for test
        """
        state = np.load("last_bodyinfo.npy", allow_pickle=True)
        sub_state = [i['bodyinfo'] for i in state[2:3220:20]]
        return sub_state

    def after_load_hierarchy(self):
        self.update_scene = None
        for ch_idx, state in enumerate(self.sub_state):
            self.scene.characters[ch_idx].load(state)


class BodyStateUpdateScene(UpdateSceneBase):
    def __init__(self, scene: Optional[ODEScene], sub_state):
        super().__init__(scene)
        self.sub_state = sub_state
        self.max_len = max([len(node) for node in sub_state])
        self.frame = 0

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        for ch_idx, state_list in enumerate(self.sub_state):
            if self.frame < len(state_list):
                self.scene.characters[ch_idx].load(state_list[self.frame])
            self.frame = (self.frame + 1) % self.max_len


class BodyStateServer(ServerForUnityV2):
    def __init__(self):
        super().__init__()
        self.sub_state = self.get_sub_state()
        self.init_instruction_buf = {
            "DupCharacterNames": [f"ch{idx}" for idx, node in enumerate(self.sub_state)]
        }

    @staticmethod
    def get_sub_state():
        state = np.load("last_bodyinfo.npy", allow_pickle=True)
        sub_state = [i['bodyinfo'] for i in state[2:3220:20]]
        return sub_state

    def after_load_hierarchy(self):
        self.update_scene = BodyStateUpdateScene(self.scene, self.sub_state)
        for ch_idx, state in enumerate(self.sub_state):
            self.scene.characters[ch_idx].load(state[0])



def main():
    server = TeaserServer()
    server.run()

if __name__ == "__main__":
    main()
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
Server for Unity Render.
Character 0: Simulation result
Character 1: Reference Motion (if exists)
Character 2: Inverse Dynamics Result (if exists)
"""
import enum
from typing import Dict

from VclSimuBackend.Samcon.SamconMainWorkerBase import SamHlp
from VclSimuBackend.Samcon.SamconTargetPose import SamconTargetPose
from VclSimuBackend.Samcon.SamconUpdateScene import SamconUpdateScene, ServerForUnityV2Samcon
from VclSimuBackend.ODESim.UpdateScene.UpdateBVHWrapper import UpdateSceneBVHDirectWrapper

from VclSimuBackend.Utils.CharacterRefViewer import CharacterRefServerBase, CharacterRefViewer
from VclSimuBackend.Utils.InverseDynamics import MotionInvDyn


class RefServerMode(enum.IntEnum):
    SIMULATE = 0
    DAMPED_SIMULATE = 1
    SAMCON_DAMPED_SIMULATE = 9


class CharacterRefServer(CharacterRefServerBase):
    def __init__(self, conf_fname: str, mode: RefServerMode = RefServerMode.SAMCON_DAMPED_SIMULATE):
        super(CharacterRefServer, self).__init__(conf_fname)
        self.samhlp = SamHlp(conf_fname, 0)
        self.mode = mode

    def after_load_hierarchy(self):
        self.update_scene = CharacterRefViewer(self.scene)

        if self.mode == RefServerMode.SIMULATE:
            pass
        elif self.mode == RefServerMode.DAMPED_SIMULATE:
            pass
        elif self.mode == RefServerMode.SAMCON_DAMPED_SIMULATE:
            best_path = self.samhlp.load_best_path_idx()
            sim_update = SamconUpdateScene.build_target(self.conf, self.scene, self.sim_character, best_path)
            # inv_dyn_target = sim_update.target
            self.update_scene.sim_update = sim_update
        else:
            raise ValueError

        ref_target = SamconTargetPose.load2(self.conf["filename"]["bvh"], self.ref_character, int(self.scene.sim_fps))
        ref_target = ref_target.sub_seq(*MotionInvDyn.ref_start_end())
        update_cls = UpdateSceneBVHDirectWrapper
        ref_update = update_cls(self.scene, character=self.ref_character, target=ref_target.pose)
        self.update_scene.ref_update = ref_update

        inv_dyn_target = MotionInvDyn.builder(self.scene, self.inv_dyn_character, self.conf).calc()
        self.update_scene.inv_dyn_update = update_cls(self.scene, character=self.inv_dyn_character, target=inv_dyn_target)


def main():
    fname = "../CharacterData/SamconConfig.json"
    server = CharacterRefServer(fname)
    server.run()


if __name__ == "__main__":
    main()

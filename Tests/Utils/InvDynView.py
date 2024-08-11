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
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.UpdateScene.UpdateBVHWrapper import UpdateSceneBVHDirectWrapper
from VclSimuBackend.Utils.InverseDynamics import MotionInvDyn
from VclSimuBackend.Utils.CharacterRefViewer import CharacterRefServerBase, CharacterRefViewer


def test_inv_dyn():
    loader = JsonSceneLoader()
    fname = "../../Tests/CharacterData/world-stdhuman.json"
    bvh_fname = "../../Tests/CharacterData/sfu/0005_Jogging001-mocap.bvh"
    # bvh_fname = "../../Tests/CharacterData/sfu/0005_Walking001-mocap.bvh"
    scene: ODEScene = loader.load_from_file(fname)
    invdyn = MotionInvDyn(scene, scene.character0, bvh_fname)
    print("After Create Motion Inv Dyn")
    invdyn.calc()  # get smoothed inverse dynamics target


class CharacterInvDynServer(CharacterRefServerBase):
    """
    Server for render simulation character, reference character, inverse dynamics character
    """
    def __init__(self, conf_fname: str):
        super(CharacterInvDynServer, self).__init__(conf_fname)

    def after_load_hierarchy(self):
        # load character from Unity Scene,
        # load target pose for ref-character and inv-dyn-character
        self.update_scene = CharacterRefViewer(self.scene)
        bvh_fname: str = self.conf["filename"]["bvh"]
        update_cls = UpdateSceneBVHDirectWrapper

        invdyn_bvh = self.conf["filename"].get("invdyn_bvh")
        if invdyn_bvh and os.path.isfile(invdyn_bvh):  # update from file
            self.update_scene.inv_dyn_update = update_cls(self.scene, invdyn_bvh, self.inv_dyn_character, do_smooth=False)
            print(f" load inverse dynamics target pose from {invdyn_bvh}")
        else:
            invdyn = MotionInvDyn(self.scene, self.inv_dyn_character, bvh_fname)
            invdyn_target = invdyn.calc()  # smoothed inverse dynamics target
            self.update_scene.inv_dyn_update = update_cls(self.scene, None, self.inv_dyn_character,
                                                          False, False, invdyn_target)

        loader = BVHToTargetBase(bvh_fname, self.scene.sim_fps, self.ref_character, False, *MotionInvDyn.ref_range())
        ref_target = loader.init_target()
        self.update_scene.ref_update = update_cls(self.scene, character=self.ref_character, target=ref_target)


def test_render():
    fname = "../CharacterData/SamconConfig.json"
    server = CharacterInvDynServer(fname)
    server.run()


if __name__ == "__main__":
    print(__file__)
    test_render()
    # test_inv_dyn()

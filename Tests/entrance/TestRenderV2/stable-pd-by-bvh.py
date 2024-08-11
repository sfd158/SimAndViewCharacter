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
from VclSimuBackend.ODESim.UpdateScene.UpdateBVHWrapper import UpdateSceneBVHStablePDWrapper
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, ODEScene
# from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH


class ServerStablePDControlBVH(ServerForUnityV2):
    def __init__(self, scene: ODEScene, bvh_fname: str):
        super(ServerStablePDControlBVH, self).__init__(scene)
        self.bvh_fname = bvh_fname

    def after_load_hierarchy(self):
        # for debug: export to bvh
        to_bvh = CharacterTOBVH(self.scene.character0)
        to_bvh.bvh_hierarchy_no_root()
        to_bvh.append_no_root_to_buffer()
        to_bvh.to_file("long-ge-hierarchy.bvh", True)
        self.update_scene = UpdateSceneBVHStablePDWrapper(self.scene, self.bvh_fname, 0)


def main():
    # bvh_fname = os.path.join(os.path.dirname(__file__), "../../CharacterData/sfu/0005_Walking001-mocap.bvh")
    # bvh_fname = "abc-full.bvh"
    scene = ODEScene()
    # render = RenderWorld(scene.world)
    # render.draw_background(0)
    # render.start()
    bvh_fname = r"D:\song\documents\WeChat Files\wxid_cy7nlkpekunr22\FileStorage\File\2021-10\retarget.bvh"
    server = ServerStablePDControlBVH(scene, bvh_fname)
    server.run()


if __name__ == "__main__":
    main()

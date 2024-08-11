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
Play bvh file, and set the arrow
"""
import numpy as np
import os
from typing import Optional, Dict, Any
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
from VclSimuBackend.ODESim.UpdateScene.UpdateBVHWrapper import UpdateSceneBVHDirectWrapper
from VclSimuBackend.ODESim.UpdateSceneBase import UpdateSceneBase
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2
from VclSimuBackend.Common.GetFileNameByUI import get_file_name_by_UI as get_from_UI
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.pymotionlib.MotionData import MotionData


fdir = os.path.dirname(__file__)
class UpdateLargeArrow(UpdateSceneBase):
    def __init__(self, scene: Optional[ODEScene], bvh: MotionData):
        super().__init__(scene)
        self.curr_frame = 0
        self.target = BVHToTargetBase(bvh, 100, scene.character0).init_target()
        self.tar_set = SetTargetToCharacter(self.character0, self.target)
        self.tot_frame = self.target.num_frames

        self.large_arrow = self.scene.characters[1].root_body

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        self.tar_set.set_character_byframe(self.curr_frame)
        vel = self.target.root_body.linvel[self.curr_frame]
        self.large_arrow.setQuaternionScipy(MathHelper.quat_between(np.array([0.0, 1.0, 0.0]), vel))
        self.large_arrow.PositionNumpy = self.character0.root_body.PositionNumpy
        self.curr_frame = (self.curr_frame + 1) % self.tot_frame


class LargeArrowServer(ServerForUnityV2):
    def __init__(self, scene: ODEScene, bvh_fname: str):
        super(LargeArrowServer, self).__init__(scene)
        self.bvh = BVHLoader.load(bvh_fname)

    def after_load_hierarchy(self):
        self.scene.set_sim_fps(100)
        self.update_scene = UpdateLargeArrow(self.scene, self.bvh)


def main():
    scene = ODEScene()
    bvh_fname = os.path.join(fdir, "../../CharacterData/sfu/0005_Jogging001-mocap-100.bvh")
    server = LargeArrowServer(scene, bvh_fname)
    server.run()


if __name__ == "__main__":
    main()

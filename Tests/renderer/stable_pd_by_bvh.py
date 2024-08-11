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
import atexit
import os
import time
import ModifyODE as ode
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.PDControler import DampedPDControler
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.Render import Renderer
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter

fdir = os.path.dirname(__file__)
scene_fname = os.path.join(fdir, '../CharacterData/stdhuman-400-kp-50-kd.pickle')
bvh_fname = os.path.join(fdir, '../CharacterData/StdHuman/WalkF-mocap.bvh')

SceneLoader = JsonSceneLoader()
scene = SceneLoader.load_from_pickle_file(scene_fname)
bvh2target = BVHToTargetBase(bvh_fname, 120, scene.character0)
target = bvh2target.init_target()
set_target = SetTargetToCharacter(scene.character0, target)
stable_pd = DampedPDControler(scene.character0)

set_target.set_character_byframe(0)
bvh_cnt = bvh2target.frame_cnt
renderObj = Renderer.RenderWorld(scene.world)
renderObj.set_joint_radius(0.05)
renderObj.start()

renderObj.look_at([3,3,3], [0,1,0], [0,1,0])
renderObj.track_body(scene.character0.bodies[0], False)

def exit_func():
    renderObj.kill()
atexit.register(exit_func)

i = 0
renderObj.start_record_video()
while i < 200:
    stable_pd.add_torques_by_quat(target.locally.quat[i])
    scene.damped_simulate(1)
    renderObj.pause(15)
    i += 1
    i = i % bvh_cnt

video = renderObj.end_record_video("test.mp4")

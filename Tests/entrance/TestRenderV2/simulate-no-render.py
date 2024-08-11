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
import time
import ModifyODE as ode
import numpy as np

from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.ODEScene import ContactType
from VclSimuBackend.ODESim.PDControler import DampedPDControler
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.pymotionlib import BVHLoader

fdir = os.path.dirname(__file__)

def main():
    loader = JsonSceneLoader()
    fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\StdHuman\world-leg3-long.json"
    # fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\lamp.json"
    scene = loader.load_from_file(fname)
    # Here we should export the bvh file
    # scene = loader.load_from_pickle_file(os.path.join(os.path.dirname(__file__), "../../CharacterData/stdhuman-400-kp-50-kd.pickle"))
    to_bvh = CharacterTOBVH(scene.character0, 100)
    to_bvh.build_hierarchy()
    to_bvh.append_no_root_to_buffer()
    motion = to_bvh.to_file()
    render = RenderWorld(scene.world)
    render.set_joint_radius(0.02)
    render.draw_hingeaxis(True)
    render.start()
    while True:
        # scene.damped_simulate_once()
        time.sleep(0.02)
        # input()


def main2():
    bvh_fname = os.path.join(fdir, "../../CharacterData/sfu-long-leg3/jogging.bvh")
    # bvh_fname = os.path.join(fdir, "../../CharacterData/Samcon/0/test.bvh.bvh")
    # "../../CharacterData/Samcon-Human.pickle"
    scene = JsonSceneLoader().load_from_file(os.path.join(fdir, "../../CharacterData/StdHuman/world-leg3-long.json"))
    scene.set_sim_fps(100)
    scene.set_gravity(0.0)
    scene.contact_type = ContactType.ODE_LCP
    motion = BVHLoader.load(bvh_fname)
    target = BVHToTargetBase(motion, scene.sim_fps, scene.character0).init_target()
    tar_set = SetTargetToCharacter(scene.character0, target)
    tar_set.set_character_byframe(0)
    stable_pd = DampedPDControler(scene.character0)
    render = RenderWorld(scene.world)
    render.start()
    render.draw_hingeaxis(1)
    render.draw_localaxis(1)
    render.set_axis_length(1)
    render.set_joint_radius(0)

    lankle_name = "lAnkle"
    rankle_name = "rAnkle"
    joint_names = scene.character0.get_joint_names()
    lankle_idx = joint_names.index(lankle_name)
    rankle_idx = joint_names.index(rankle_name)
    lankle: ode.BallJointAmotor = scene.character0.joints[lankle_idx]
    rankle: ode.BallJointAmotor = scene.character0.joints[rankle_idx]
    while True:
        for i in range(target.num_frames):
            # tar_set.set_character_byframe(i)
            # input()
            tar_local_q = target.locally.quat[i]
            stable_pd.add_torques_by_quat(tar_local_q)
            scene.damped_simulate_once()
            # l_angles = np.array(lankle.Angles) / np.pi * 180
            # r_angles = np.array(rankle.Angles) / np.pi * 180
            # print(l_angles, r_angles)
            time.sleep(0.01)


if __name__ == "__main__":
    main2()

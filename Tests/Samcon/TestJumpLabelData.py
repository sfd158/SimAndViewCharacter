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
import ModifyODE as ode
import numpy as np
import os
import pickle
import subprocess
import torch
from typing import List, Union, Tuple

from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH
from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import DirectTrajOptBVH, SamHlp, SamconWorkerFull
from VclSimuBackend.Samcon.SamconWorkerBase import WorkerInfo

from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo, generate_hack_contact
from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData


fdir = os.path.dirname(__file__)


def train():
    """
    1. not use inverse dynamics
    2. label contact manually in a single jump cycle (height < eps, and velocity < eps)
    3. use Long Ge's framework to render..
    """
    config_fname = os.path.join(fdir, "../CharacterData/SamconConfig-duplicate.json")
    samhlp = SamHlp(config_fname)
    conf = samhlp.conf
    conf["inverse_dynamics"]["in_use"] = False
    conf["traj_optim"]["load_samcon_result"] = False
    scene: ODEScene = DirectTrajOptBVH.load_scene_with_conf(conf, None)
    character: ODECharacter = scene.character0
    plane: ode.GeomObject = scene.environment.geoms[0]
    render = RenderWorld(scene.world)
    body_name_list = character.body_info.get_name_list()

    # get contact geometry position in global coordinate.
    def get_heel_geom(lr_name: str = "l"):
        heel_name = lr_name + "Foot"
        heel_index: int = body_name_list.index(heel_name)
        heel_body: ode.Body = character.bodies[heel_index]
        # heel_body_pos: np.ndarray = heel_body.PositionNumpy
        heel_geoms: List[Union[ode.GeomObject, ode.GeomSphere]] = list(heel_body.geom_iter())
        geom_names: List[str] = [node.name for node in heel_geoms]

        def get_ball_contact(ball_name: str) -> Tuple[ode.GeomSphere, np.ndarray]:
            ball_geom = heel_geoms[geom_names.index(ball_name)]
            return ball_geom, ball_geom.PositionNumpy - np.array([0.0, ball_geom.geomRadius, 0.0])

        ball0, ball0_contact = get_ball_contact("Ball0")  # heel (larger ball)
        ball1, ball1_contact = get_ball_contact("Ball1")  # (smaller ball)
        ball2, ball2_contact = get_ball_contact("Ball2")  # (smaller ball)
        return [(ball0, ball0_contact), (ball1, ball1_contact), (ball2, ball2_contact)]

    def get_toe_geom(lr_name: str = "l"):
        toe_name = lr_name + "Toes"
        toe_index: int = body_name_list.index(toe_name)
        toe_body: ode.Body = character.bodies[toe_index]
        toe_geom: ode.GeomCapsule = list(toe_body.geom_iter())[0]
        capsule_contact: np.ndarray = toe_geom.PositionNumpy - np.array([0.0, toe_geom.radius, 0.0])
        return [(toe_geom, capsule_contact),]

    l_heel_contact = get_heel_geom("l")
    l_toe_contact = get_toe_geom("l")
    l_heel_contact, l_toe_contact = l_heel_contact[:1], l_toe_contact + l_heel_contact[1:]

    r_heel_contact = get_heel_geom("r")
    r_toe_contact = get_toe_geom("r")
    r_heel_contact, r_toe_contact = r_heel_contact[:1], r_toe_contact + r_heel_contact[1:]

    bvh_fname = os.path.join(fdir, "../CharacterData/sfu/0005_JumpRope001-mocap-100.bvh")
    motion: MotionData = BVHLoader.load(bvh_fname)

    start_index = 67
    up_heel_index = 91 - start_index
    up_toe_index = 101 - start_index
    down_toe_index = 139 - start_index
    down_heel_index = 145 - start_index
    end_index = 151

    # visualize...
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", bvh_fname])
    motion: MotionData = motion.sub_sequence(start_index, end_index)
    conf["filename"]["bvh"] = motion

    # Now, we need to compute collision info
    y_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    def build_contact_base(geom_tuple):
        for geom_, pos in geom_tuple:
            contact = ode.Contact()
            contact.setContactGeomParams(pos, y_dir, 0.0, geom_, plane)
            joint = ode.ContactJoint(scene.world, scene.contact, contact)
            joint.attach(geom_.body, plane.body)

    def build_contact(build_l_heel: bool, build_l_toe: bool, build_r_heel: bool, build_r_toe: bool):
        if build_l_heel:
            build_contact_base(l_heel_contact)
        if build_l_toe:
            build_contact_base(l_toe_contact)
        if build_r_heel:
            build_contact_base(r_heel_contact)
        if build_r_toe:
            build_contact_base(r_toe_contact)
        diff_contact = DiffContactInfo.contact_extract(scene.contact.joints, len(character.joints), scene.contact_type)
        scene.contact.empty()
        return diff_contact

    toe_heel_diff_contact = build_contact(True, True, True, True)  # toe and heel have contact
    toe_diff_contact = build_contact(False, True, False, True)  # only toe have contact
    none_diff_contact = DiffContactInfo(scene.contact_type)  # no contact
    stage1 = [toe_heel_diff_contact.curr_copy() for i in range(0, up_heel_index)]
    stage2 = [toe_heel_diff_contact.curr_copy() for i in range(up_heel_index, up_toe_index)]
    stage3 = [none_diff_contact.curr_copy() for i in range(up_toe_index, down_toe_index)]
    stage4 = [toe_diff_contact.curr_copy() for i in range(down_toe_index, down_heel_index)]
    stage5 = [toe_heel_diff_contact.curr_copy() for i in range(down_heel_index, motion.num_frames)]
    total_stage: List[DiffContactInfo] = stage1 + stage2 + stage3 + stage4 + stage5
    print(f"total length of stage = {len(total_stage)}")
    contact_pos_list = [node.contact_pos.detach().clone() if node.contact_pos is not None else None for node in total_stage]

    worker_info = WorkerInfo()
    worker = SamconWorkerFull(samhlp, worker_info, scene)
    worker.contact_info_list = total_stage
    main_worker = DirectTrajOptBVH(samhlp, worker_info, worker, scene)
    worker.torch_loss.hack_contact_pos = contact_pos_list

    render.start()
    main_worker.test_direct_trajopt()

if __name__ == "__main__":
    train()

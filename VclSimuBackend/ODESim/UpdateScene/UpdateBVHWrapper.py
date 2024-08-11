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

import numpy as np
from typing import Optional, Dict, Any, Union
from ..ODECharacter import ODECharacter
from ..UpdateSceneBase import UpdateSceneBase, ODEScene
from ..TargetPose import SetTargetToCharacter
from ..BVHToTarget import BVHToTargetBase, TargetPose
from ..PDControler import DampedPDControler
from ...pymotionlib.MotionData import MotionData
from ...pymotionlib import BVHLoader


class UpdateSceneBVHDirectWrapper(UpdateSceneBase):
    def __init__(
        self,
        scene: ODEScene,
        bvh_target: Union[str, MotionData, TargetPose] = None,
        character: Optional[ODECharacter] = None,
        ignore_root_xz_pos: bool = False
    ):
        super(UpdateSceneBVHDirectWrapper, self).__init__(scene)
        if character is None:
            character = self.character0
        self.character = character
        if isinstance(bvh_target, str):
            bvh_target = BVHLoader.load(bvh_target)
        if isinstance(bvh_target, MotionData):
            bvh_target = BVHToTargetBase(bvh_target, scene.sim_fps, self.character, ignore_root_xz_pos).init_target()

        self.target: TargetPose = bvh_target
        self.set_tar = SetTargetToCharacter(self.character, self.target)
        self.frame: int = 0
        self.set_tar.set_character_byframe(self.frame)

    def check(self):
        # character's joint position should match target joint position
        anchor: np.ndarray = self.world.getBallAndHingeAnchor1(self.character.joint_info.joint_c_id).reshape((-1, 3))
        delta = self.target.globally.pos[self.frame] - anchor
        assert np.all(np.abs(delta) < 1e-6)

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):  # no simulation
        # for _ in self.scene.step_range():
        self.set_tar.set_character_byframe(self.frame % self.target.num_frames)
            # Do collision detection and visualize
            # self.scene.compute_collide_info()
        self.frame = (self.frame + 1) % self.target.num_frames
        print("bvh", self.frame)


class UpdateSceneBVHStablePDWrapper(UpdateSceneBase):
    def __init__(self, scene: ODEScene, bvh_fname: str, character_idx: int = 0, ignore_root_xz_pos: bool = False):
        super(UpdateSceneBVHStablePDWrapper, self).__init__(scene)
        self.character_idx = character_idx

        self.target = BVHToTargetBase(bvh_fname, scene.sim_fps, self.character, ignore_root_xz_pos).init_target()
        self.stable_pd = DampedPDControler(self.character)
        self.frame = 0
        # self.scene.use_implicit_damping()

    @property
    def character(self):
        return self.scene.characters[self.character_idx]

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        if self.frame == 0:
            self.character.load_init_state()

        tar_local_q = self.target.locally.quat[self.frame]
        # Simulate
        for _ in self.scene.step_range():
            self.stable_pd.add_torques_by_quat(tar_local_q)
            self.scene.damped_simulate(1)
            # self.scene.fast_simulate_once()
        self.frame = (self.frame + 1) % self.target.num_frames

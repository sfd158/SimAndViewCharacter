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
Viewer for Simulation character, reference character, inverse dynamics character
"""

from typing import Optional, Dict, Any
from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.ODESim.UpdateScene.UpdateBVHWrapper import UpdateSceneBVHDirectWrapper
from VclSimuBackend.ODESim.UpdateSceneBase import UpdateSceneBase, ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, SimpleUpdateScene


class CharacterRefViewer(UpdateSceneBase):
    """
    Update for Simulation character, reference character, inverse dynamics character
    """
    def __init__(self, scene: ODEScene, sim_update: Optional[UpdateSceneBase] = None):
        super(CharacterRefViewer, self).__init__(scene)
        self.sim_character: Optional[ODECharacter] = None  # character for simulation
        self.ref_character: Optional[ODECharacter] = None  # character for reference
        self.inv_dyn_character: Optional[ODECharacter] = None  # character for inverse dynamics
        # self.time_stamp = 0

        for ch in self.scene.characters:  # scan character list
            label = ch.label.lower()
            if label.endswith("simulation") or label.endswith("simu") or label.endswith("sim"):
                assert ch.is_kinematic is False
                self.sim_character = ch
            elif label.endswith("reference") or label.endswith("ref"):
                assert ch.is_kinematic is True
                ch.set_ode_space(None)  # close collision detection by set ref_character.geoms.space to None..
                ch.is_enable = False
                self.ref_character = ch
            elif label.endswith("inversedynamics") or label.endswith("inversedynamic"):
                assert ch.is_kinematic is True
                ch.set_ode_space(None)
                ch.is_enable = False
                self.inv_dyn_character = ch
            else:
                raise NotImplementedError

        if sim_update is None:
            sim_update = SimpleUpdateScene(self.scene)

        self.sim_update: Optional[UpdateSceneBase] = sim_update
        self.ref_update: Optional[UpdateSceneBVHDirectWrapper] = None
        self.inv_dyn_update: Optional[UpdateSceneBVHDirectWrapper] = None

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        # update self.sim_character by simulation
        # update self.ref_character by reference motion
        # update self.inv_dyn_character by inverse dynamics + PD/stable PD controller

        if self.sim_character is not None and self.sim_update is not None:
            self.sim_update.update(mess_dict)

        if self.ref_character is not None and self.ref_update is not None:
            self.ref_update.update(mess_dict)

        if self.inv_dyn_character is not None and self.inv_dyn_update is not None:
            self.inv_dyn_update.update(mess_dict)


class CharacterRefServerBase(ServerForUnityV2):
    """
    Basic server for rendering Simulation character, reference character, inverse dynamics character.
    """
    def __init__(self, conf_fname: str):
        super(CharacterRefServerBase, self).__init__()
        self.conf = Helper.conf_loader(conf_fname)
        self.update_scene: Optional[CharacterRefViewer] = None

    @property
    def sim_character(self) -> Optional[ODECharacter]:
        return self.update_scene.sim_character

    @property
    def ref_character(self) -> Optional[ODECharacter]:
        return self.update_scene.ref_character

    @property
    def inv_dyn_character(self) -> Optional[ODECharacter]:
        return self.update_scene.inv_dyn_character

    def after_load_hierarchy(self):  # Add in derived class.
        raise NotImplementedError

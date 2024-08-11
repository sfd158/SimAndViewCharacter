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
from typing import Any, List, Dict

from ...ODESim.ODECharacter import ODECharacter
from ...ODESim.ODEScene import ODEScene
from ...ODESim.ODESceneWrapper import ODESceneWrapper


class RemoveParse(ODESceneWrapper):

    def __init__(self, scene: ODEScene):
        super(RemoveParse, self).__init__(scene)

    def post_remove_characters(self, retain_index: List[int]):
        characters: List[ODECharacter] = []
        for i in retain_index:
            characters.append(self.characters[i])
        self.scene.characters = characters

    def post_remove_geoms(self, retain_index: List[int]):
        geoms: List[ode.GeomObject] = []
        for i in retain_index:
            geoms.append(self.environment.geoms[i])
        self.scene.environment.geoms = geoms

    def remove_characters(self, remove_character_id: List[int]):
        """
        Remove Characters by Character ID
        :param remove_character_id: list of character id to be removed
        :return:
        """
        remove_id_set = set(remove_character_id)
        retain_index: List[int] = []
        for i, character in enumerate(self.characters):
            if character.character_id in remove_id_set:
                character.clear()
            else:
                retain_index.append(i)
        self.post_remove_characters(retain_index)

    def remove_geoms(self, remove_geom_id: List[int]):
        """
        Remove Geometries in Environment by remove_geom_id
        :param remove_geom_id: list of geometry id to be removed
        :return:
        """
        remove_id_set = set(remove_geom_id)
        retain_index: List[int] = []
        for i, geom in enumerate(self.environment.geoms):
            if geom.instance_id in remove_id_set:
                geom.destroy_immediate()
            else:
                retain_index.append(i)
        self.post_remove_geoms(retain_index)

    def remove_ext_joints(self, remove_ext_id: List[int]):
        """

        :param remove_ext_id:
        :return:
        """
        pass

    def parse_remove_characters(self, mess: Dict[str, Any]):
        ch_ids = mess.get("CharacterID")
        if ch_ids:
            self.remove_characters(ch_ids)

    def parse_remove_env(self, mess: Dict[str, Any]):
        geom_ids = mess.get("GeomID")
        if geom_ids:
            self.remove_geoms(geom_ids)

    def parse_remove_ext_joint(self, mess: Dict[str, Any]):
        extjoint_ids = mess.get("ExtJointID")
        if extjoint_ids:
            self.remove_ext_joints(extjoint_ids)

    def parse(self, mess_dict: Dict[str, Any]):
        """
        Parse remove information from Unity client.
        :param mess_dict:
        :return:
        """
        ext_joint_info = mess_dict.get("ExtJointList")  # parse ext joints
        if ext_joint_info:
            self.parse_remove_ext_joint(ext_joint_info)

        env_info = mess_dict.get("Environment")
        if env_info:
            self.parse_remove_env(env_info)

        ch_info = mess_dict.get("CharacterList")
        if ch_info:
            self.parse_remove_characters(ch_info)

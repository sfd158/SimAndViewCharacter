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

from typing import Optional, Any, Dict, List
from ..ODESceneWrapper import ODEScene, ODESceneWrapper
from .DCharacterExport import JsonCharacterExport


class DSceneExport(ODESceneWrapper):
    def __init__(self, scene: ODEScene):
        super().__init__(scene)

    def export_world_attr(self) -> Dict[str, Any]:
        fixed_attr = {
            "SimulateFPS": int(self.scene.sim_fps),
            "UseHinge": True,
            "UseAngleLimit": True,
            "SelfCollision": self.scene.self_collision,
            "dWorldUpdateMode": None
        }
        change_attr = {
            "Gravity": self.scene.gravity_numpy.tolist(),
            "StepCount": None,
            "RenderFPS": None
        }

        result = {
            "FixedAttr": fixed_attr,
            "ChangeAttr": change_attr
        }

        return result

    def export_character_list(self) -> List[Dict[str, Any]]:
        result = []
        for character_index, character in enumerate(self.characters):
            exporter = JsonCharacterExport(character)
            character_result = exporter.export()
            result.append(character_result)
        return result

    def export_ext_joint_list(self):
        if self.scene.ext_joints is None:
            return {"Joints": []}

    def export_ext_force_list(self):
        return {"Forces": []}

    def export(self) -> Dict[str, Any]:
        world_attr = self.export_world_attr()
        character_list = self.export_character_list()
        ext_joint_list = self.export_ext_joint_list()
        ext_force_list = self.export_ext_force_list()
        result = {
            "WorldAttr": world_attr,
            "CharacterList": character_list,
            "ExtJointList": ext_joint_list,
            "ExtForceList": ext_force_list
        }
        return result

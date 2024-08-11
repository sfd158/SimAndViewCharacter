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

import enum
import json
import numpy as np
from typing import Any, Dict, List, Optional

from ...ODESim.ODECharacter import ODECharacter
from ...ODESim.ODESceneWrapper import ODEScene, ODESceneWrapper


class UnityDWorldUpdateMode(enum.IntEnum):
    # Only update via joint rotation in Unity
    ReducedCoordinate = 0

    # Update via global body position / orientation, and joint rotation in Unity
    # As Open Dynamics Engine (ODE) is a maximal coordinate simulator, we should use MaximalCoordinate mode here..
    MaximalCoordinate = 1


class ODEToUnity(ODESceneWrapper):
    """
    export ode scene for view in Unity
    """
    def __init__(self, scene: ODEScene):
        super(ODEToUnity, self).__init__(scene)
        self.unity_update_mode = UnityDWorldUpdateMode.MaximalCoordinate

    def environment_info(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def root_info(character: ODECharacter):
        res = {
            "Position": None,
            "Quaternion": character.root_body.getQuaternionScipy().tolist()
        }
        if character.joint_info.has_root:
            res["Position"] = character.root_joint.getAnchorNumpy().tolist()
        else:
            res["Position"] = character.root_body.PositionNumpy.tolist()

        return res

    @staticmethod
    def joint_info(character: ODECharacter) -> List[Dict[str, Any]]:
        pos: np.ndarray = character.joint_info.get_global_pos1()
        qs: np.ndarray = character.joint_info.child_qs()
        res = [{"JointID": joint_id,
                "Position": pos[joint_id].tolist(),
                "Quaternion": qs[joint_id].tolist()}
               for joint_id in range(len(character.joints))]
        return res

    def body_info(self, character: ODECharacter):
        # extract body color info.
        result = [{"BodyID": body_id} for body_id in range(len(character.bodies))]
        body_info = character.body_info
        if body_info.visualize_color is not None:
            for body_id, color in enumerate(body_info.visualize_color):
                if color is not None:
                    result[body_id]["Color"] = list(color)

        # TODO: export linear velocity and angular velocity in some debug mode..
        if self.unity_update_mode == UnityDWorldUpdateMode.ReducedCoordinate:
            pass
        elif self.unity_update_mode == UnityDWorldUpdateMode.MaximalCoordinate:
            pos: np.ndarray = body_info.get_body_pos()
            quat: np.ndarray = body_info.get_body_quat()
            vel: np.ndarray = body_info.get_body_velo()
            for sub_dict, body_pos, body_quat,body_vel in zip(result, pos, quat, vel):
                sub_dict.update({"Position": body_pos.tolist(), "Quaternion": body_quat.tolist(), "LinearVelocity": body_vel.tolist()})
        else:
            raise ValueError

        return result

    def character_info(self, character_idx: int):
        character: ODECharacter = self.characters[character_idx]
        res = {
            "CharacterID": character.character_id,
            "RootInfo": self.root_info(character),
            "JointInfo": self.joint_info(character),
            "BodyInfo": self.body_info(character)
        }
        return res

    def ext_joint_list_info(self):
        res = {
            "Joints": [{"JointID": ext_id, "Position": ext_joint.getAnchorNumpy().tolist()}
            for ext_id, ext_joint in enumerate(self.scene.ext_joints.joints)]
        }
        return res

    def contact_info(self) -> Optional[Dict[str, Any]]:
        if self.scene.contact_info is not None:
            res = {"Joints": [{"JointID": i, "Position": pos.tolist(), "Force":  force.tolist(), "ContactLabel": label}
                    for i, pos, force, label in self.scene.contact_info.out_iter()]}
        else:
            res = None

        self.scene.contact_info = None
        return res

    def arrow_list_info(self) -> Dict[str, List]:
        arrow_list = self.scene.arrow_list
        if arrow_list is None or len(arrow_list) == 0:
            return None
        res = [{"StartPos": node.start_pos.tolist(), "EndPos": node.end_pos.tolist(), "InUse": node.in_use} for node in arrow_list.arrows]
        return {"ArrowList" : res}

    def to_dict(self) -> Dict[str, Any]:
        """
        Export ODE Scene to Dict
        """
        res = {
            "Environment": self.environment_info(),
            "CharacterList": {
                "Characters":
                [self.character_info(i) for i in range(len(self.characters))]
            },
            "ExtJointList": self.ext_joint_list_info(),
            "ContactList": self.contact_info(),
            "HelperInfo": {"Message": self.scene.str_info}
        }
        arrow_info = self.arrow_list_info()
        if arrow_info is not None:
            res["ArrowList"] = arrow_info

        return res

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict())

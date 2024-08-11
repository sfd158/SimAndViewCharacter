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

import copy
import ModifyODE as ode
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional

from ..Loader.CharacterLoaderHelper import CharacterLoaderHelper
from ..ODECharacter import DRootInitInfo, ODECharacter


class MessDictScale:
    def __init__(self) -> None:
        pass

    @staticmethod
    def handle_value(key, value, scale: float):
        if key in ["Position", "Scale", "LinearVelocity"]:
            return scale * np.asarray(value)
        elif key in ["Kps", "Inertia"]:
            return (scale ** 4) * np.asarray(value)
        elif key == "Damping":  # 
            return (scale ** 4.5) * value
        elif key == "Mass":
            return (scale ** 3) * value
        elif isinstance(value, Dict):
            return MessDictScale.handle_dict(value, scale)
        elif isinstance(value, List):
            return MessDictScale.handle_list(value, scale)
        else:
            return value

    @staticmethod
    def handle_list(mess_list: List, load_scale: float):
        result: List = []
        for value in mess_list:
            result.append(MessDictScale.handle_value(None, value, load_scale))
        return result

    @staticmethod
    def handle_dict(mess_dict: Dict[str, Any], load_scale: float):
        result: Dict[str, Any] = {}
        for key, value in mess_dict.items():
            result[key] = MessDictScale.handle_value(key, value, load_scale)
        return result


class JsonCharacterLoader(CharacterLoaderHelper):
    def __init__(
        self,
        world: ode.World,
        space: ode.Space,
        use_hinge: bool = True,
        use_angle_limit: bool = True,
        ignore_parent_collision: bool = True,
        ignore_grandpa_collision: bool = True,
        load_scale: float = 1.0,
        update_body_pos_by_com: bool = False
    ):
        super(JsonCharacterLoader, self).__init__(world, space)
        self.use_hinge = use_hinge
        self.use_angle_limit = use_angle_limit
        self.ignore_parent_collision: bool = ignore_parent_collision
        self.ignore_grandpa_collision: bool = ignore_grandpa_collision
        self.load_scale: float = load_scale
        self.update_body_pos_by_com: bool = update_body_pos_by_com
        self.ignore_list: List[List[int]] = []

    def load_from_file(self, fname: str):
        """
        Load character from json file
        """
        with open(fname, "r") as f:
            mess_dict: Dict[str, Any] = json.load(f)
        return self.load(mess_dict)

    def load_bodies(self, json_bodies: List):
        """
        Load bodies in json
        """
        json_bodies.sort(key=lambda x: x["BodyID"])  # sort bodies by BodyID

        for json_body in json_bodies:
            self.add_body(json_body)
            ignore = json_body.get("IgnoreBodyID", [])
            self.ignore_list.append(ignore)

        self.parse_ignore_list()

    def parse_ignore_list(self):
        """
        ignore collision detection between some bodies
        """
        for body_idx, ignores in enumerate(self.ignore_list):
            res: List[int] = []
            for ignore_body_id in ignores:
                if ignore_body_id >= len(self.bodies):
                    logging.warning(f"{ignore_body_id} out of range. ignore.")
                    continue
                for geom in self.bodies[ignore_body_id].geom_iter():
                    res.append(geom.get_gid())

            for geom in self.bodies[body_idx].geom_iter():
                geom.extend_ignore_geom_id(res)

    def load_joints(self, json_joints: List):
        """
        load joints in json
        """
        json_joints.sort(key=lambda x: x["JointID"])  # sort joints by JointID
        self.joint_info.kds = np.zeros(len(json_joints))
        self.joint_info.weights = np.ones(len(json_joints))
        for json_joint in json_joints:
            self.add_joint(json_joint, self.use_hinge, self.use_angle_limit)

    def load_endjoints(self, json_endjoints: List):
        """
        Load end joints in json
        """
        joint_id, name, end_pos = [], [], []
        for json_endjoint in json_endjoints:
            joint_id.append(json_endjoint["ParentJointID"])
            name.append(json_endjoint["Name"])
            end_pos.append(np.array(json_endjoint["Position"]))
        try:
            body_id = [self.joint_to_child_body[i] for i in joint_id]
        except IndexError as e:
            raise e
        self.character_init.init_end_joint(name, body_id, np.array(end_pos))

    def load_pd_control_param(self, json_pd_param: Dict[str, Any]):
        """
        Load PD Control Param in json
        """
        kps = np.asarray(json_pd_param["Kps"])
        torque_lim = np.asarray(json_pd_param["TorqueLimit"])
        if len(kps) == 0 or len(torque_lim) == 0:
            return
        assert kps.size == len(self.joints)
        assert torque_lim.size == len(self.joints)
        self.joint_info.kps = kps
        self.joint_info.torque_limit = torque_lim

    def load_init_root_info(self, init_root_param: Dict[str, Any]):
        info = DRootInitInfo()
        info.pos = np.array(init_root_param["Position"])
        info.quat = np.array(init_root_param["Quaternion"])
        self.character.root_init_info = info

    def replace_names(self, mess_dict: Dict[str, Any]):
        """
        """
        joint_handle: List[Dict] = mess_dict["Joints"]
        body_handle: List[Dict] = mess_dict["Bodies"]
        end_handle: List[Dict] = mess_dict.get("EndJoints", None)
        joint_dict = {node["Name"]: index for index, node in enumerate(joint_handle)}
        body_dict = {node["Name"]: index for index, node in enumerate(body_handle)}
        if joint_handle is not None:
            for index, node in enumerate(joint_handle):
                if node["ParentBodyID"] in body_dict:
                    node["ParentBodyID"] = body_dict[node["ParentBodyID"]]
                elif node["ParentBodyID"] is None or node["ParentBodyID"] in ["", "None", "null"]:
                    node["ParentBodyID"] = -1

                if node["ChildBodyID"] in body_dict:
                    node["ChildBodyID"] = body_dict[node["ChildBodyID"]]
                elif node["ChildBodyID"] is None or node["ChildBodyID"] in ["", "None", "null"]:
                    node["ChildBodyID"] = -1
                
                if node["ParentJointID"] in body_dict:
                    node["ParentJointID"] = body_dict[node["ParentJointID"]]
                elif node["ParentJointID"] is None or node["ParentJointID"] in ["", "None", "null"]:
                    node["ParentJointID"] = -1
        
        if body_handle is not None:
            for index, node in enumerate(body_handle):
                if node["ParentJointID"] in joint_dict:
                    node["ParentJointID"] = joint_dict[node["ParentJointID"]]
                elif node["ParentJointID"] is None or node["ParentJointID"] in ["", "None", "null"]:
                    node["ParentJointID"] = -1

                if node["ParentBodyID"] in body_dict:
                    node["ParentBodyID"] = body_dict[node["ParentBodyID"]]
                elif node["ParentBodyID"] is None or node["ParentBodyID"] in ["", "None", "null"]:
                    node["ParentBodyID"] = -1
        
        if end_handle is not None:
            for index, node in enumerate(end_handle):
                if isinstance(node["ParentJointID"], str):
                    node["ParentJointID"] = joint_dict[node["ParentJointID"]]
        
        return mess_dict

    def load(self, mess_dict: Dict[str, Any]) -> ODECharacter:
        """
        Load ODE Character from json file
        """
        self.replace_names(mess_dict)
        self.ignore_parent_collision &= mess_dict.get("IgnoreParentCollision", True)
        self.ignore_grandpa_collision &= mess_dict.get("IgnoreGrandpaCollision", True)

        name: Optional[str] = mess_dict.get("CharacterName")
        if name:
            self.character.name = name

        label: Optional[str] = mess_dict.get("CharacterLabel")
        if label:
            self.character.label = label

        self.load_bodies(mess_dict["Bodies"])

        joints: List[Dict[str, Any]] = mess_dict.get("Joints")
        if joints:
            self.load_joints(joints)

        self.character_init.init_after_load(mess_dict["CharacterID"],
                                            self.ignore_parent_collision,
                                            self.ignore_grandpa_collision)

        self_colli: Optional[bool] = mess_dict.get("SelfCollision")
        if self_colli is not None:
            self.character.self_collision = self_colli

        kinematic: Optional[bool] = mess_dict.get("Kinematic")
        if kinematic is not None:
            self.character.is_kinematic = kinematic

        pd_param: Dict[str, List[float]] = mess_dict.get("PDControlParam")
        if pd_param:
            self.load_pd_control_param(pd_param)

        end_joints: List[Dict[str, Any]] = mess_dict.get("EndJoints")
        if end_joints:
            self.load_endjoints(end_joints)

        init_root: Dict[str, Any] = mess_dict.get("RootInfo")
        if init_root:
            self.load_init_root_info(init_root)

        # we can store the config at the character
        # for simply duplicate character.
        self.character.config_dict = copy.deepcopy(mess_dict)
        
        self.character.is_stdhuman_ = self.character.is_stdhuman()
        self.character.root_body = self.character.body_info.root_body = self.character.body_info._root_body
        # load smpl beta parameter..
        self.character.smpl_shape = mess_dict.get("smpl_shape")
        if self.character.smpl_shape is not None:
            self.character.smpl_shape = np.array(self.character.smpl_shape)
        return self.character

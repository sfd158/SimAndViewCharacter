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

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree
from ...Common.Helper import *
from ..ODEScene import *
from ..ODECharacterInit import ODECharacterInit, ODECharacter
from ..Loader.CharacterLoaderHelper import CharacterLoaderHelper

from scipy.spatial.transform.rotation import Rotation
import numpy as np

import typing
from typing import Dict

from ...Common.MathHelper import MathHelper


class XMLCharacterLoader(CharacterLoaderHelper):
    def __init__(self, scene: ODEScene, conf: Dict):
        raise ValueError("Should not used")
        super(XMLCharacterLoader, self).__init__(scene.world, scene.space)
        self.scene = scene
        self.conf = conf

        self.character_init: Optional[ODECharacterInit] = None
        self.body_idx_dict: Dict[str, int] = {}
        self.joint_idx_dict: Dict[str, int] = {}

    @property
    def f_conf(self):
        return self.conf["filename"]

    def parse_body(self, character_info: Element):
        for idx, body_info in enumerate(character_info.findall("Body")):
            InertiaGeometry = body_info.find("InertiaGeometry")
            if InertiaGeometry:
                density = float(body_info.find("PhysicsProperty").find("Density").text)
                inertia_type = InertiaGeometry.find("Type")
                length = [float(InertiaGeometry.find(i + "Length").text) for i in ["X", "Y", "Z"]]

                m = ode.Mass()
                m.setBox(density, *length)
                print(idx, inertia_type.text, body_info.find("Name").text, length, density, m.mass)
            continue
        exit(0)
        """
            body_frame = body_info.find("BodyFrame")
            xyz_axis = [body_frame.find(i) for i in ["X_Axis", "Y_Axis", "Z_Axis"]]
            body_rot: Rotation = Rotation.from_matrix(np.eye(3))
            if xyz_axis[0] is not None:
                xyz_axis_list = np.array([[float(i) for i in j.text.split(" ")] for j in xyz_axis])
                body_rot = Rotation.from_matrix(xyz_axis_list)

            physics_property = body_info.find("PhysicsProperty")

            xml_parent: Element = body_info.find("Parent")
            parent: Optional[int] = self.body_idx_dict[xml_parent.text] if xml_parent is not None else None

            json_body = {"BodyID": idx, "Name": body_info.find("Name").text,
                         "Density": float(physics_property.find("Density").text),
                         "Friction": float(physics_property.find("FrictionCoef").text), "ParentBodyID": parent,
                         "Position": np.array([float(i) for i in body_frame.find("Center").text.split(" ")]),
                         "Quaternion": body_rot.as_quat(), "Geoms": []}

            # Load Geom Info
            for geom_idx, colli_geometry in enumerate(body_info.findall("CollisionGeometry")):
                colli_body_frame = colli_geometry.find("BodyFrame")

                colli_xyz_axis = [colli_body_frame.find(i) for i in ["X_Axis", "Y_Axis", "Z_Axis"]]
                colli_rot: Rotation = Rotation.from_matrix(np.eye(3))
                if colli_xyz_axis[0] is not None:
                    colli_xyz_list = np.array([[float(i) for i in j.text.split(" ")] for j in colli_xyz_axis])
                    colli_rot = Rotation.from_matrix(colli_xyz_list)
                else:  # 2 Element do not have X, Y, Z Axis
                    pass

                geom_json = {"GeomID": geom_idx,
                             "Name": colli_geometry.find("Name").text,
                             "GeomType": colli_geometry.find("Type").text,
                             "Collidable": True,
                             "Position": np.array([float(i) for i in colli_body_frame.find("Center").text.split(" ")]),
                             "Quaternion": colli_rot.as_quat(),
                             "Scale": np.ones(3),
                             }

                if geom_json["GeomType"] == "Sphere":  # Rotation of Sphere are 0
                    geom_json["Scale"] = float(colli_geometry.find("Radius").text) * np.ones(3)
                elif geom_json["GeomType"] == "CCylinder":
                    colli_radius = float(colli_geometry.find("Radius").text)
                    colli_length = float(colli_geometry.find("Length").text)  # total_length - 2 * radius
                    geom_json["Scale"] = np.array([colli_radius, colli_length, 0])
                elif geom_json["GeomType"] == "Box":
                    geom_json["Scale"] = np.array(
                        [float(colli_geometry.find(i).text) for i in ["XLength", "YLength", "ZLength"]])
                else:
                    raise NotImplementedError

                json_body["Geoms"].append(geom_json)

            body = self.add_body(json_body)
            self.body_idx_dict[body.name] = idx
        """

    def parse_ignore_pair(self, character_info: Element):
        # Collision between some GeomObjects should be ignored.
        ignore_pair = character_info.find("IgnorePair")
        for ignore_mess in ignore_pair.iter("Pair"):
            ignore_pair_id = [self.body_idx_dict[i] for i in ignore_mess.text.split(" ")]
            ignore1_gids = self.bodies[ignore_pair_id[1]].getGeomIDNumpy()
            for geom0 in self.bodies[ignore_pair_id[0]].geom_iter():
                geom0.ignore_geom_id.extend(ignore1_gids)

    @staticmethod
    def _xml_root_info(character_info: Element) -> Optional[Element]:
        for idx, joint_info in enumerate(character_info.findall("Joint")):
            joint_name = joint_info.find("Name").text
            if joint_name == "RootJoint":
                return joint_info
        return None

    def parse_one_joint(self, joint_info: Element, xml_use_xyz=True):
        json_joint = {
            "JointID": len(self.joints),
            "Name": joint_info.find("Name").text,
            "JointType": joint_info.find("Type").text,
            "ChildBodyID": self.body_idx_dict[joint_info.find("Child").text],
            "Damping": 0.0,
            "TorqueLimit": 0.0,
            "Position": np.array([float(i) for i in joint_info.find("Position").text.split(" ")]),
            "Quaternion": MathHelper.unit_quat(),
            "AngleLoLimit": np.array([]),
            "AngleHiLimit": np.array([]),
        }
        parent = joint_info.find("Parent").text
        json_joint["ParentBodyID"] = self.body_idx_dict[parent] if not Helper.is_str_empty(parent) else -1
        if json_joint["JointType"] == "BallJoint":
            json_joint["EulerOrder"] = joint_info.find("EulerOrder").text

            angle_limits = []  # shape = (3, 2)
            for JointAngleLimit_iter in joint_info.findall("AngleLimit"):
                angle_limits.append([float(i) / 180.0 * np.pi for i in JointAngleLimit_iter.text.split(" ")])
            assert len(angle_limits) == 3
            if not xml_use_xyz:
                angle_limits = [angle_limits[ord(json_joint["EulerOrder"][eu]) - ord("X")] for eu in range(3)]
            json_joint["AngleLoLimit"] = np.array(angle_limits)[:, 0]
            json_joint["AngleHiLimit"] = np.array(angle_limits)[:, 1]

        elif json_joint["JointType"] == "HingeJoint":
            angle_limit = np.array([float(i) / 180.0 * np.pi for i in joint_info.find("AngleLimit").text.split(" ")])
            json_joint["AngleLoLimit"], json_joint["AngleHiLimit"] = angle_limit[0:1], angle_limit[1:2]
            json_joint["EulerOrder"] = joint_info.find("HingeAxis").text[0].upper()
        else:
            raise NotImplementedError

        self.add_joint(json_joint, self.conf["character"]["with_hinge_joint"], self.conf["character"]["with_limit"])

    def parse_joint(self, character_info: Element, load_root_joint: bool = False):
        xml_root_info = self._xml_root_info(character_info)
        if load_root_joint:
            if xml_root_info is not None:
                self.character_init.set_has_root()
                self.parse_one_joint(xml_root_info)
            else:
                self.character_init.add_root_joint()
                self.joint_idx_dict["RootJoint"] = len(self.joints)
                # print("XML dosen't have RootJoint. Add RootJoint at the Position of Root Body.")

        # parse ode joint
        for joint_info in character_info.findall("Joint"):
            if joint_info.find("Name").text == "RootJoint":
                continue
            self.joint_idx_dict[joint_info.find("Name").text] = len(self.joints)
            self.parse_one_joint(joint_info)

    def load_character(self, load_endjoint: bool = False, load_rootjoint: bool = False, character_id: int = 0):
        # tree: ElementTree = ET.parse(self.f_conf["character"])
        tree: ElementTree = ET.parse(r"D:\song\documents\GitHub\pfnn-learn\CharacterData\StdHuman_New.xml")
        character_info: Element = tree.getroot()
        self.parse_body(character_info)
        self.parse_ignore_pair(character_info)
        self.parse_joint(character_info, load_rootjoint)
        self.character.falldown_ratio = self.conf["character"]["falldown_ratio"]
        self.character_init.init_after_load(character_id)
        self.character_init.set_geom_clung()

        if load_endjoint:
            self.load_end_joints(character_info)
        # print("Load Body from %s. Body Count = %d." % (self.character_fname, len(self.body_info.bodies)))

    def load_joint_param(self, load_rootjoint: bool = False):
        tree = ET.parse(self.f_conf["joint"])
        joint_control_param = tree.getroot()

        self.joint_info.resize()
        # order in joint_param may not match order in character xml file..
        for joint_param in joint_control_param.findall("Joint"):
            joint_name = joint_param.find("Name").text
            if joint_name == "RootJoint" and not load_rootjoint:
                continue

            joint_idx = self.joint_idx_dict[joint_name]
            kd = float(joint_param.find("kd").text)
            self.joint_info.kps[joint_idx] = float(joint_param.find("kp").text)
            self.joint_info.kds[joint_idx] = kd
            self.joints[joint_idx].setSameKd(kd)  # kds can be equal in 3 dimension when using stable pd controler

            self.joint_info.torque_limit[joint_idx] = float(joint_param.find("TorqueLimit").text)

    def load(self, load_endjoint: bool = False, load_rootjoint: bool = False):
        self.character = ODECharacter(self.scene.world, self.scene.space)
        self.character_init = ODECharacterInit(self.character)
        self.load_character(load_endjoint, load_rootjoint)
        self.load_joint_param(load_rootjoint)
        self.scene.characters.append(self.character)

    def load_end_joints(self, character_info: Element):
        name, end_pos = [], []
        for idx, end_point_info in enumerate(character_info.findall("EndPoint")):
            end_point = end_point_info.text.split(" ")
            name.append(end_point[0])
            end_pos.append(np.array([float(i) for i in end_point[1:]]))

        parent_body_id = [self.body_idx_dict[i] for i in name]
        self.character_init.init_end_joint(name, parent_body_id, np.array(end_pos))

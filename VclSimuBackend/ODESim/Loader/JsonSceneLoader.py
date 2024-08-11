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
import json
import logging
import ModifyODE as ode
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation
from typing import Optional, Dict, Any, List, Union

from VclSimuBackend.ODESim.ArrowInfo import ArrowInfo, ArrowInfoList

from VclSimuBackend.ODESim.Loader.JsonCharacterLoader import JsonCharacterLoader
from VclSimuBackend.ODESim.Loader.CharacterLoaderHelper import CharacterLoaderHelper
from VclSimuBackend.ODESim.ODESceneWrapper import ODESceneWrapper
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter


class JsonRemovePart:

    def __init__(self, handle: Union[str, Dict]) -> None:
        if isinstance(handle, str):
            handle: Dict[str, Any] = json.load(open(handle, "r"))
        else:
            handle = copy.deepcopy(handle)
        self.other_info = {key: value for key, value in handle.items() if key != "CharacterList"}
        self.handle: List[Dict[str, Any]] = handle["CharacterList"]["Characters"][0]

        self.body_name_dict: Dict[str, Dict] = {node["Name"]: node for node in self.handle["Bodies"]}
        self.body_name_idict: Dict[int, str] = {node["BodyID"]: node["Name"] for node in self.handle["Bodies"]}
        self.joint_name_dict: Dict[str, Dict] = {node["Name"]: node for node in self.handle["Joints"]}
        self.joint_name_idict: Dict[int, str] = {node["JointID"]: node["Name"] for node in self.handle["Joints"]}

        self.children_dict: Dict[str, List[str]] = {node["Name"]: [] for node in self.handle["Joints"]}
        pd_param: Dict[str, List] = self.handle["PDControlParam"]

        for body_node in self.handle["Bodies"]:
            # remove parent and child index with string
            body_node["ParentJointID"] = self.joint_name_idict.get(body_node["ParentJointID"], None)
            body_node["ParentBodyID"] = self.body_name_idict.get(body_node["ParentBodyID"], None)
        for joint_node in self.handle["Joints"]:
            joint_node["ParentBodyID"] = self.body_name_idict.get(joint_node["ParentBodyID"], None)
            joint_node["ChildBodyID"] = self.body_name_idict.get(joint_node["ChildBodyID"], None)
            joint_node["ParentJointID"] = self.joint_name_idict.get(joint_node["ParentJointID"], None)
            if joint_node["ParentJointID"] is not None:
                self.children_dict[joint_node["ParentJointID"]].append(joint_node["Name"])
            jindex = joint_node["JointID"]
            joint_node["kp"] = pd_param["Kps"][jindex]
            joint_node["torque_limit"] = pd_param["TorqueLimit"][jindex]

        for end_node in self.handle["EndJoints"]:
            end_node["ParentJointID"] = self.joint_name_idict.get(end_node["ParentJointID"], None)

        # build parent dict
        self.end_name_dict: Dict[str, Dict] = {node["ParentJointID"]: node for node in self.handle["EndJoints"]}

    def move_node(self, node_name: str, offset: np.ndarray):
        assert node_name in self.joint_name_dict
        assert offset.shape == (3,)
        # move child body
        joint_node = self.joint_name_dict[node_name]
        child_bnode = self.body_name_dict[joint_node["ChildBodyID"]]
        child_bnode["Position"] = (np.array(child_bnode["Position"]) + offset).tolist()
        for geom_node in child_bnode["Geoms"]:
            geom_node["Position"] = (np.array(geom_node["Position"]) + offset).tolist()
        # move end joint
        if node_name in self.end_name_dict:
            end_joint = self.end_name_dict[node_name]
            end_joint["Position"] = (np.array(end_joint["Position"]) + offset).tolist()

        # handle child joints.
        for child_jname in self.children_dict[node_name]:
            self.move_node(child_jname)

    def remove_joint(
        self,
        remove_jname_list: Optional[List[str]] = None,
        remove_bname_list: Optional[List[str]] = None
    ):  
        # parent joint - parent_body - [current joint] - child body - child joint - child2 body
        if remove_jname_list is not None:
            for remove_jname in remove_jname_list:
                joint_info: Dict[str, Any] = self.joint_name_dict[remove_jname]
                parent_jinfo: Dict[str, Any] = self.joint_name_dict[joint_info["ParentJointID"]]
                child_binfo: Dict[str, Any] = self.body_name_dict[joint_info["ChildBodyID"]]
                parent_binfo = self.body_name_dict[joint_info["ParentBodyID"]]
                parent_binfo["Geoms"].extend(child_binfo["Geoms"])
                # handld child-child.
                for child_jname in self.children_dict[remove_jname]:
                    child_jinfo: Dict[str, Any] = self.joint_name_dict[child_jname]
                    child_jinfo["ParentJointID"] = parent_jinfo["Name"]
                    # consider the child-child body
                    child2_binfo = self.body_name_dict[child_jinfo]
                    child2_binfo["ParentJointID"] = parent_jinfo["Name"]
                    child2_binfo["ParentBodyID"] = parent_binfo["Name"]

                # handle end joint.
                if remove_jname in self.end_name_dict:
                    self.end_name_dict[remove_jname]["ParentJointID"] = parent_jinfo["Name"]

                # remove child body and current joint.
                del self.body_name_dict[child_binfo["Name"]]
                del self.joint_name_dict[remove_jname]

        # p2 joint - parent body - parent joint - [current body] - child joint - child body
        if remove_bname_list is not None:
            for remove_bname in remove_bname_list:
                binfo: Dict[str, Any] = self.body_name_dict[remove_bname]
                parent_jinfo: Dict[str, Any] = self.joint_name_dict[binfo["ParentJointID"]]
                parent_binfo: Dict[str, Any] = self.body_name_dict[binfo["ParentBodyID"]]
                parent2_jinfo: Dict[str, Any] = self.joint_name_dict[parent_jinfo["ParentJointID"]]
                # remove parent joint.
                joint_pos: np.ndarray = np.array(parent_jinfo["Position"])

                for ch_name in self.children_dict[parent_jinfo["Name"]]:
                    ch_jinfo: Dict[str, Any] = self.joint_name_dict[ch_name]
                    ch_binfo: Dict[str, Any] = self.body_name_dict[ch_jinfo["ChildBodyID"]]
                    ch_jinfo["ParentJointID"] = parent_jinfo["ParentJointID"]
                    ch_jinfo["ParentBodyID"] = parent_binfo["Name"]
                    ch_binfo["ParentBodyID"] = parent_binfo["Name"]

                    # move the child joint.
                    self.move_node(ch_name, -np.array(ch_jinfo["Position"]) + joint_pos)

                # handle end joint
                if parent_jinfo["Name"] in self.end_name_dict:
                    self.end_name_dict[remove_jname]["ParentJointID"] = parent2_jinfo["Name"]
                
                del self.body_name_dict[binfo["Name"]]
                del self.joint_name_dict[parent_jinfo["Name"]]
        
        # here we need to recalc joint and body index.
        for index, (key, node) in enumerate(self.body_name_dict.items()):
            node["BodyID"] = index
        for index, (key, node) in enumerate(self.joint_name_dict.items()):
            node["JointID"] = index
        
        return self.export_json()

    def export_json(self) -> Dict[str, Any]:
        # concat body name dict.
        ret_handle = {key: value for key, value in self.handle.items() if key != "Bodies" and key != "Joints"}
        ret_handle.update({
            "Bodies": list(self.body_name_dict.values()),
            "Joints": list(self.joint_name_dict.values())
        })
        ret_handle.update({
            "PDControlParam": {
                "Kps": [node["kp"] for node in ret_handle["Joints"]],
                "TorqueLimit": [node["torque_limit"] for node in ret_handle["Joints"]]
            }
        })
        ret = self.other_info
        ret["CharacterList"] = {"Characters": [ret_handle]}
        return ret

    def export_json_to_file(self, fname: str):
        with open(fname, "w") as fout:
            json.dump(self.export_json(), fout)
        print(f"Output to {fname}", flush=True)

    def modify_foot(self):
        """
        convert the foot
        """
        for lr in ["L", "R"]:
            body_dict: Dict[str, Any] = self.body_name_dict[f"{lr}_Ankle"]

            # compute the bounding box
            corner_list = []
            for geom in body_dict["Geoms"]:
                geom_pos: np.ndarray = np.array(geom["Position"])
                if geom["GeomType"] == "Sphere":
                    radius = geom["Scale"][0]
                    corner_list.append(geom_pos + np.array([radius, 0.0, 0.0]))
                    corner_list.append(geom_pos + np.array([-radius, 0.0, 0.0]))
                    corner_list.append(geom_pos + np.array([0.0, radius, 0.0]))
                    corner_list.append(geom_pos + np.array([0.0, -radius, 0.0]))
                    corner_list.append(geom_pos + np.array([0.0, 0.0, radius]))
                    corner_list.append(geom_pos + np.array([0.0, 0.0, -radius]))
                elif geom["GeomType"] == "Capsule":
                    radius: float = geom["Scale"][0]
                    length: float = 0.5 * geom["Scale"][1]
                    # emm..how to compute bounding box of capsule?
                    cap_rot: Rotation = Rotation(np.array(geom["Quaternion"]))
                    cap_offset: np.ndarray = geom_pos + cap_rot.apply(np.array([0.0, 0.0, 0.5 * length]))
                    if False:
                        corner_list.append(cap_offset + np.array([0.0, 0.0, radius]))
                        corner_list.append(cap_offset + np.array([0.0, 0.0, -radius]))
                        corner_list.append(cap_offset + np.array([0.0, -radius, 0.0]))

                        corner_list.append(-cap_offset + np.array([0.0, 0.0, radius]))
                        corner_list.append(-cap_offset + np.array([0.0, 0.0, -radius]))
                        corner_list.append(-cap_offset + np.array([0.0, -radius, 0.0]))
                else:
                    raise NotImplementedError
            corner_list: np.ndarray = np.concatenate([node[None] for node in corner_list], axis=0)
            bound_max: np.ndarray = np.max(corner_list, axis=0)
            bound_min: np.ndarray = np.min(corner_list, axis=0)
            bound_min[1] = max(0, bound_min[1])
            center_pos: np.ndarray = 0.5 * (bound_max + bound_min)
            center_pos[2] += 0.04
            shape: np.ndarray = np.abs(bound_max - bound_min)
            shape[0] *= 1.08
            shape[2] *= 1.14
            ret_geom: Dict[str, Any] = copy.deepcopy(body_dict["Geoms"][0])
            ret_geom["GeomType"] = "Cube"
            ret_geom["Quaternion"] = [0.0, 0.0, 0.0, 1.0]
            ret_geom["Scale"] = shape.tolist()
            ret_geom["Position"] = center_pos.tolist()
            body_dict["Geoms"] = [ret_geom]
            body_dict["Position"] = center_pos.tolist()

        if False:
            print(f"After modify foot.")
        return self.export_json()
            
    @staticmethod
    def test_func():
        fname = r"D:\song\documents\GitHub\ControlVAE\tmp-dir\smpl-world.json"
        remover = JsonRemovePart(fname)
        remover.remove_joint(["L_Foot_and_L_Ankle", "R_Foot_and_R_Ankle"])
        remover.export_json()
    
    def modify_upper_mass(self, ratio: float = 1.0) -> Dict[str, Any]:
        """
        """
        if ratio == 1.0:
            return self.export_json()

        upper_name_list = [
            "Pelvis",
            "Spine1",
            "Spine2",
            "Spine3",
            "Neck",
            "L_Collar",
            "R_Collar",
            "Head",
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow"
        ]

        for body_name in upper_name_list:
            body_dict: Dict[str, Any] = self.body_name_dict[body_name]
            body_dict["Density"] *= ratio
            body_dict["Mass"] *= ratio
            body_dict["Inertia"] = np.array(body_dict["Inertia"]).tolist()
        
        if False:
            print(f"modify mass of upper part", flush=True)
        return self.export_json()


class JsonSceneLoader(ODESceneWrapper):
    """
    Load Scene in json format generated from Unity
    """

    class AdditionalConfig:
        """
        Additional Configuration in loading json scene
        """
        def __init__(self):
            self.gravity: Optional[List[float]] = None
            self.step_count: Optional[int] = None
            self.render_fps: Optional[int] = None

            self.cfm: Optional[float] = None
            self.simulate_fps: Optional[int] = None
            self.use_hinge: Optional[bool] = None
            self.use_angle_limit: Optional[bool] = None
            self.self_collision: Optional[bool] = None

        def update_config_dict(self, mess: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if mess is None:
                mess = {}

            change_attr = mess.get("ChangeAttr", {})
            if self.gravity is not None:
                change_attr["Gravity"] = self.gravity
            if self.step_count is not None:
                change_attr["StepCount"] = self.step_count
            if self.render_fps is not None:
                change_attr["RenderFPS"] = self.render_fps

            fixed_attr = mess.get("FixedAttr", {})
            if self.cfm is not None:
                fixed_attr["CFM"] = self.cfm
            if self.simulate_fps is not None:
                fixed_attr["SimulateFPS"] = self.simulate_fps
            if self.use_hinge is not None:
                fixed_attr["UseHinge"] = self.use_hinge
            if self.use_angle_limit is not None:
                fixed_attr["UseAngleLimit"] = self.use_angle_limit
            if self.self_collision is not None:
                fixed_attr["SelfCollision"] = self.self_collision

            if len(fixed_attr) > 0:
                mess["FixedAttr"] = fixed_attr
            if len(change_attr) > 0:
                mess["ChangeAttr"] = change_attr

            return mess

    def __init__(self, scene: Optional[ODEScene] = None, is_running: bool = False):
        super(JsonSceneLoader, self).__init__(scene)
        if self.scene is None:
            self.scene = ODEScene()

        self.use_hinge: bool = True
        self.use_angle_limit: bool = True
        self.is_running = is_running

    def file_load(self, fname: str, config: Optional[AdditionalConfig] = None) -> ODEScene:
        if fname.endswith(".pickle"):
            return self.load_from_pickle_file(fname, config)
        elif fname.endswith(".json"):
            return self.load_from_file(fname, config)
        else:
            raise NotImplementedError

    def load_from_file(self, fname: str, config: Optional[AdditionalConfig] = None) -> ODEScene:
        with open(fname, "r") as f:
            mess_dict = json.load(f)
        return self.load_json(mess_dict, config)

    def load_from_pickle_file(self, fname: str, config: Optional[AdditionalConfig] = None) -> ODEScene:
        fname = os.path.abspath(fname)
        with open(fname, "rb") as f:
            mess_dict = pickle.load(f)
        logging.info(f"load from pickle file {fname}")
        return self.load_json(mess_dict, config)

    def load_environment(self, mess_dict: Dict[str, Any]):
        geom_info_list: List[Dict] = mess_dict["Geoms"]
        geom_info_list.sort(key=lambda x: x["GeomID"])
        helper = CharacterLoaderHelper(self.world, self.space, create_character=False)
        for geom_json in geom_info_list:
            geom, _ = helper.create_geom_object(geom_json, False)
            self.environment.geoms.append(geom)
            # if helper.geom_type.is_plane(geom_json["GeomType"]):  # assume there is only 1 plane
            #    self.environment.floor = geom
            geom.character_id = -1

        self.environment.get_floor_in_list()
        return self.environment

    def load_ext_joints(self, mess_dict: Dict[str, Any]):  # load constraint joints
        joints: List[Dict[str, Any]] = mess_dict["Joints"]
        joints.sort(key=lambda x: x["JointID"])
        for joint_json in joints:
            ext_joint = CharacterLoaderHelper.create_joint_base(self.world, joint_json, self.use_hinge)
            self.ext_joints.append_and_attach(ext_joint, joint_json["Character0ID"], joint_json["Body0ID"],
                                              joint_json["Character1ID"], joint_json["Body1ID"])
            CharacterLoaderHelper.post_create_joint(ext_joint, joint_json, self.use_angle_limit)

    def load_ext_forces(self, mess_dict: Dict[str, Any]):
        """
        Load external forces. such as force from mouse drag/push in Unity Scene
        """
        ch_dict = self.scene.get_character_id_map()  # key: character id. value: index in characterlist
        forces: List[Dict[str, Any]] = mess_dict["Forces"]
        for finfo in forces:
            character: ODECharacter = ch_dict[finfo["CharacterID"]]
            body: ode.Body = character.bodies[finfo["BodyID"]]
            pos: np.ndarray = np.asarray(finfo["Position"])
            force: np.ndarray = np.asarray(finfo["Force"])
            body.addForceAtPosNumpy(force, pos)

    def load_world_attr(self, mess_dict: Dict[str, Any]):
        change_attr = mess_dict.get("ChangeAttr")
        if change_attr:
            self.scene.set_gravity(change_attr["Gravity"])
            self.scene.set_render_fps(change_attr["RenderFPS"])
            step_cnt: Optional[int] = change_attr.get("StepCount")
            if step_cnt:
                self.scene.step_cnt = step_cnt

        fixed_attr: Optional[Dict[str, Any]] = mess_dict.get("FixedAttr")
        if fixed_attr and not self.is_running:
            if "SimulateFPS" in fixed_attr:
                self.scene.set_sim_fps(fixed_attr["SimulateFPS"])
            if "UseHinge" in fixed_attr:
                self.use_hinge = fixed_attr["UseHinge"]
            if "UseAngleLimit" in fixed_attr:
                self.use_angle_limit = fixed_attr["UseAngleLimit"]
            if "CFM" in fixed_attr:
                self.scene.world.CFM = fixed_attr["CFM"]
            if "SelfCollision" in fixed_attr:
                self.scene.self_collision = fixed_attr["SelfCollision"]

    def load_character_list(self, mess_dict: Dict[str, Any]):
        character_info_list: List[Dict[str, Any]] = mess_dict["Characters"]
        # character_info_list.sort(key=lambda x: x["CharacterID"])
        for character_info in character_info_list:
            # for debug..
            loader = JsonCharacterLoader(self.world, self.space, self.use_hinge, self.use_angle_limit)
            character = loader.load(character_info)
            character.scene = self.scene
            self.characters.append(character)

        return self.characters

    def load_arrow_info(self, info: Dict[str, List]):
        arrow_list = info["ArrowList"]
        if arrow_list is None:
            return
        self.scene.arrow_list = ArrowInfoList()
        for node in arrow_list:
            res = ArrowInfo()
            res.start_pos = np.array(node["StartPos"])
            res.end_pos = np.array(node["EndPos"])
            res.in_use = node["InUse"]
            self.scene.arrow_list.arrows.append(res)
        return self.scene.arrow_list

    def load_json(self, mess_dict: Dict[str, Any], config: Optional[AdditionalConfig] = None) -> ODEScene:
        world_attr: Optional[Dict[str, Any]] = mess_dict.get("WorldAttr")
        if config is not None:
            world_attr = config.update_config_dict(world_attr)

        if world_attr:
            self.load_world_attr(world_attr)

        if not self.is_running:  # for debug. TODO: remove this condition
            env_mess = mess_dict.get("Environment")
            if env_mess:
                self.load_environment(env_mess)

        characters_mess = mess_dict.get("CharacterList")
        if characters_mess:
            self.load_character_list(characters_mess)

        ext_joints = mess_dict.get("ExtJointList")
        if ext_joints:
            self.load_ext_joints(ext_joints)

        ext_force = mess_dict.get("ExtForceList")
        if ext_force:
            self.load_ext_forces(ext_force)

        arrow_info = mess_dict.get("ArrowList")
        if arrow_info:
            self.load_arrow_info(arrow_info)

        self.scene.resort_geoms()
        for ch in self.scene.characters:
            if ch.com_geom is not None:
                ch.com_geom.space = self.scene.debug_space
        return self.scene

if __name__ == "__main__":
    JsonRemovePart.test_func()

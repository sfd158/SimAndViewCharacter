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

build character from SMPL model
0. Visualize SMPL model
1. the character is in T pose, joint number matches SMPL model
2. the foot is modified. That is, 3 balls on foot, and 1 capsule on toe
3. mass and inertia is computed by mesh
4. visualize by Long Ge's framework or pyvista
5. we should export as json format, and load the character using ODE

Input: a smpl model parameter \theta
Output: ODECharacter
"""
from enum import IntEnum
import numpy as np
import os
import copy
import pyvista as pv
import math
import json
from typing import Optional, Dict, Tuple, Any, List

from ...SMPL.SMPL import SMPLModel
from ...SMPL.config import smpl_hierarchy, smpl_name_list, smpl_parent_info
from ...SMPL.config import smpl_parent_list, smpl_children_list, smpl_render_index
from ...SMPL.config import WorldAttr, Environment, basicCharacterInfo, PDControlParam
from ...SMPL.config import smpl_export_body_size, smpl_export_joint_size, smpl_joint_size
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.pymotionlib.Utils import flip_quaternion
from VclSimuBackend.SMPL.stdhuman2SMPL import (
    build_smpl_body_mirror_index,
    build_smpl_mirror_index,
    smpl_mirror_dict,
    smpl_body_mirror_dict
)


fdir = os.path.dirname(__file__)


def coor_tuple(coor: np.ndarray)-> tuple:
    return (coor[0], coor[1], coor[2])


class CreateMoveMode(IntEnum):
    normal = 0
    front = 1
    behind = 2


class SMPLBuildMode:
    mass_optimize: bool = True
    do_render: bool = False
    move_mode: str = "behind"
    use_ball_hand: bool = True
    back_offset: float = 0.025
    foot_enlarge: float = 0.0
    modify_toe_joint: bool = False


class SMPLCharacterBuild:
    """
    The created character is not totally same as original SMPL model.
    We can simply retarget SMPL result to our character model.

    Note: the SMPL model is y up (just same as ODE simulator)
    """

    default_zero_builder = None  # create character with hyper parameter = np.zeros(10)
    default_zero_json: Optional[Dict[str, Any]] = None  # export the default character to json.
    kps_160cm: Optional[Dict[str, float]] = None
    kds_160cm: Optional[Dict[str, float]] = None

    def __init__(
        self, smpl_theta: np.ndarray,
        mass_optimize: Optional[bool] = SMPLBuildMode.mass_optimize,
        do_render: bool = SMPLBuildMode.do_render,
        move_mode: str = SMPLBuildMode.move_mode,
        use_ball_hand: bool = SMPLBuildMode.use_ball_hand,
        back_offset: float = SMPLBuildMode.back_offset,
        foot_enlarge: float = SMPLBuildMode.foot_enlarge,
        modify_toe_joint: bool = SMPLBuildMode.modify_toe_joint
    ) -> None:
        self.smpl_param: np.ndarray = smpl_theta
        self.smpl = SMPLModel(smpl_theta, mass_optimize)
        self.plot: Optional[pv.Plotter] = None
        self.mass_optimize = mass_optimize
        self.do_render: bool = do_render

        self.render_opacity: float = 0.6
        self.joint_pos: np.ndarray = self.smpl.J.copy()  # modified joint position

        self.bodies: Optional[List[Dict[str, Any]]] = None
        self.joints: Optional[List[Dict[str, Any]]] = None
        self.endjoints: Optional[List[Dict[str, Any]]] = None

        self.move_mode: CreateMoveMode = CreateMoveMode[move_mode]
        self.use_ball_hand: bool = use_ball_hand
        self.back_offset: float = back_offset
        self.foot_enlarge: float = foot_enlarge
        self.modify_toe_joint: bool = modify_toe_joint

    @classmethod
    def create_default_builder(cls, do_render: bool = True):
        if cls.default_zero_json is not None:
            return

        builder = cls(np.zeros(10), do_render=do_render)
        builder.build()
        zero_json = builder.export_model_to_json(False)
        
        cls.default_zero_builder = builder
        kps_160cm = {node["Name"]: 400 for node in zero_json[0]["Joints"]}
        kds_160cm = {node["Name"]: 40 for node in zero_json[0]["Joints"]}
        # adjust kp and kds for toe and wrist joint..
        kps_160cm["L_Wrist_and_L_Elbow"] = 5
        kps_160cm["R_Wrist_and_R_Elbow"] = 5
        kps_160cm["L_Foot_and_L_Ankle"] = 10
        kps_160cm["R_Foot_and_R_Ankle"] = 10

        kds_160cm["L_Wrist_and_L_Elbow"] = 1
        kds_160cm["R_Wrist_and_R_Elbow"] = 1
        kds_160cm["L_Foot_and_L_Ankle"] = 1
        kds_160cm["R_Foot_and_R_Ankle"] = 1

        # adjust kp and kds for neck joint.
        kps_160cm["Neck_and_Spine3"] = 100
        kds_160cm["Neck_and_Spine3"] = 10
        kps_160cm["Head_and_Neck"] = 100
        kds_160cm["Head_and_Neck"] = 10

        # adjust kp and kds for knee joint..
        kps_160cm["L_Knee_and_L_Hip"] = 420
        kps_160cm["R_Knee_and_R_Hip"] = 420
        kps_160cm["L_Hip_and_Pelvis"] = 420
        kps_160cm["R_Hip_and_Pelvis"] = 420

        cls.kps_160cm = kps_160cm
        cls.kds_160cm = kds_160cm

        cls.default_zero_json = cls.compute_desired_kp_kd(zero_json)

        return builder

    @staticmethod
    def _axis_to_quaternion(direction: np.ndarray):
        # default axis of rotation (1,0,0)
        # cos(theta/2) + sin(theta/2)u
        if direction[0] < 0:
            direction = direction * -1
        direction = direction / np.linalg.norm(direction)
        axis = (direction + np.array([0.0, 0.0, 1.0])) * 0.5
        axis = axis / np.linalg.norm(axis)
        thre = 1.0 / math.sqrt(2)
        return [axis[0], axis[1], axis[2], 0.0]
        
    def _capsule_info(self, length: float, radius: float, center: np.ndarray, direction: np.ndarray, index: int, name: str):
        result = {
            "GeomID": index,
            "Name": "Geom_" + name + str(index),
            "GeomType": "Capsule",
            "Collidable": True,
            "Friction": 0.8,
            "Restitution": 1.0,
            "ClungEnv": False,
            "Position": list(center + self.global_offset),
            "Quaternion": self._axis_to_quaternion(direction),
            "Scale": [radius, max(length, 1e-3), 0]
        }
        return result

    def _cylinder_info(self, length: float, radius: float, center: np.ndarray, direction: np.ndarray, index: int, name: str):
        result = {
            "GeomID": index,
            "Name": "Geom_" + name + str(index),
            "GeomType": "Cylinder",
            "Collidable": True,
            "Friction": 0.8,
            "Restitution": 1.0,
            "ClungEnv": False,
            "Position": list(center + self.global_offset),
            "Quaternion": self._axis_to_quaternion(direction),
            "Scale": [radius, length, 0]
        }
        return result

    def _cube_info(self, center: np.ndarray, xlen: float, ylen: float, zlen: float, index: int, name: str):
        result = {
            "GeomID": index,
            "Name": "Geom_" + name + str(index),
            "GeomType": "Cube",
            "Collidable": True,
            "Friction": 0.8,
            "Restitution": 1.0,
            "ClungEnv": False,
            "Position": list(center + self.global_offset),
            "Quaternion": [0.0, 0.0, 0.0, 1.0],
            "Scale": [xlen, ylen, zlen]
        }
        return result

    def _sphere_info(self, radius: float, center: np.ndarray, index: int, name: str):
        result = {
            "GeomID": index,
            "Name": "Geom_" + name + str(index),
            "GeomType": "Sphere",
            "Collidable": True,
            "Friction": 0.8,
            "Restitution": 1.0,
            "ClungEnv": False,
            "Position": list(center + self.global_offset),
            "Quaternion": [0.0, 0.0, 0.0, 1.0],
            "Scale": [radius, radius, radius]
        }
        return result
    
    def _render_capsule(
        self, length: float, radius: float,
        center_pos: np.ndarray, direction: np.ndarray, index: int
    ) -> None:
        if not self.do_render:
            return
        if length < 0:
            radius: float = radius + length / 2
            length = 0

        if index in smpl_render_index:
            cylinder = pv.Cylinder(center_pos, coor_tuple(direction), radius, length)
            ball0 = pv.Sphere(radius, center_pos + 0.5 * length * direction)
            ball1 = pv.Sphere(radius, center_pos - 0.5 * length * direction)
            self.plot.add_mesh(cylinder, opacity=self.render_opacity)
            self.plot.add_mesh(ball0, opacity=self.render_opacity)
            self.plot.add_mesh(ball1, opacity=self.render_opacity)
    
    def _render_Sphere(self, radius: float, center_pos: np.ndarray, index: int) -> None:
        if index in smpl_render_index:
            ball = pv.Sphere(radius, center_pos)
            self.plot.add_mesh(ball, opacity=self.render_opacity)

    def _body_info(self, index: int) -> dict:
        return {
            "BodyID" : index,
            "Name": smpl_name_list[index],
            "MassMode" : "MassValue",
            "Density" : 1000.0,
            "Mass" : self.mass[index],
            "InertiaMode" : "InertiaValue",
            "Inertia" : list(self.inertia[index].reshape(-1)),
            "ParentJointID" : index - 1,
            "ParentBodyID" : smpl_parent_list[index],
            "Position" : list(self.center_mass[index] + self.global_offset),
            "Quaternion" : [0.0, 0.0, 0.0, 1.0],
            "LinearVelocity" : [0.0, 0.0, 0.0],
            "AngularVelocity" : [0.0, 0.0, 0.0],
            "Geoms" : [],
            "IgnoreBodyID": []  # TODO: we should add ignore collision..
        }

    def _ball_joint_info(self, index: int, axis: str, angle_low, angle_high, weight: float = 1):
        parent_index: int = smpl_parent_list[index]
        result = {
            "JointID": index - 1,
            "Name": smpl_name_list[index] + '_and_' + smpl_name_list[parent_index],
            "JointType": "BallJoint",
            "Damping": 50.0,
            "Weight": weight,
            "Position": list(self.joint_pos[index] + self.global_offset),
            "Quaternion": [0.0, 0.0, 0.0, 1.0],
            "AngleLoLimit": angle_low,
            "AngleHiLimit": angle_high,
            "EulerOrder": axis,
            "EulerAxisLocalRot": [0.0, 0.0, 0.0, 1.0],
            "ParentBodyID": parent_index,
            "ChildBodyID": index,
            "ParentJointID": parent_index - 1
        }
        return result

    def _hinge_joint_info(self, index: int, axis : int, angle_low : float, angle_high : float, weight : float = 1):
        parent_index: int = smpl_parent_list[index]
        result = {
            "JointID": index - 1,
            "Name": smpl_name_list[index] + '_and_' + smpl_name_list[parent_index],
            "JointType": "HingeJoint",
            "Damping": 50.0,
            "Weight": weight,
            "Position": list(self.joint_pos[index] + self.global_offset),
            "Quaternion": [0.0, 0.0, 0.0, 1.0],
            "AngleLoLimit": [angle_low],
            "AngleHiLimit": [angle_high],
            "EulerOrder": axis,
            "EulerAxisLocalRot": [0.0, 0.0, 0.0, 1.0],
            "ParentBodyID": parent_index,
            "ChildBodyID": index,
            "ParentJointID": parent_index - 1
        }
        return result
    
    def _endjoint_info(self, index : int, position : np.ndarray):
        result = {
            "ParentJointID": index - 1,
            "Name": "EndJoint_" + smpl_name_list[index],
            "Position": list(position)
        }
        return result

    def _build_arm_base(self, lr_flag: str, name: str, export_json: bool = False, add_fix_hand: bool = False):
        arm_index: int = smpl_hierarchy[lr_flag + name]
        arm_point_id: np.ndarray = self.smpl.get_subpart_place(arm_index)
        point_cloud: np.ndarray = self.smpl.verts[arm_point_id]
        point_cloud_min: np.ndarray = np.min(point_cloud, axis=0)
        point_cloud_max: np.ndarray = np.max(point_cloud, axis=0)
        bounding_box: np.ndarray = point_cloud_max - point_cloud_min
        radius: float = 0.9 * 0.25 * (bounding_box[1] + bounding_box[2])
        length: float = bounding_box[0] - 2 * radius

        center_pos: np.ndarray = np.mean(point_cloud, axis=0)
        center_pos[0] = (np.max(point_cloud[..., 0]) + np.min(point_cloud[..., 0])) * 0.5

        if export_json:
            result = self._body_info(arm_index)
            result["Geoms"] = [self._capsule_info(length, radius, center_pos, np.array([1, 0, 0]), 0, lr_flag + name)]
            if add_fix_hand:
                hand_center_pos = center_pos + np.array([0.5 * length + radius, 0.0, 0.0])
                result["Geoms"].append(self._sphere_info(0.8 * radius, hand_center_pos, 1, lr_flag + "_Hand"))

            self.bodies[arm_index] = result
            if "Elbow" in name:
                self.joints[arm_index - 1] = self._hinge_joint_info(arm_index, "Y", 0, 150)
            else:
                self.joints[arm_index - 1] = self._ball_joint_info(arm_index, "ZXY", [-170.0, -80.0, -170.0], [170.0, 80.0, 170.0])
        
        if self.do_render:
            self._render_capsule(length, radius, center_pos, np.array([1, 0, 0]), arm_index)

    def _build_upper_arm(self, lr_flag: str, export_json: bool = False):
        self._build_arm_base(lr_flag, "_Shoulder", export_json)

    def _build_lower_arm(self, lr_flag: str, export_json: bool = False):
        self._build_arm_base(lr_flag, "_Elbow", export_json)

    def _build_shoulder(self, lr_flag: str, export_json: bool = False):
        """
        Note: we can also set the shoulder as ball
        """
        weight: np.ndarray = np.array((1, 0.6, 0.75))
        self._build_shoulder_base(lr_flag + "_Collar", weight, export_json)

    def _build_shoulder_base(self, name: str, weight: np.ndarray, export_json: bool = False):

        arm_index: int = smpl_hierarchy[name]
        chest_index: int = smpl_hierarchy[smpl_name_list[9]] # Index for Chest
        arm_point_id: np.ndarray = self.smpl.get_subpart_place(arm_index)
        chest_point_id: np.ndarray = self.smpl.get_subpart_place(chest_index)
        arm_point_cloud: np.ndarray = self.smpl.verts[arm_point_id]
        chest_point_cloud: np.ndarray = self.smpl.verts[chest_point_id]
        point_cloud_min: np.ndarray = np.min(arm_point_cloud, axis=0)
        point_cloud_max: np.ndarray = np.max(arm_point_cloud, axis=0)
        bounding_box: np.ndarray = point_cloud_max - point_cloud_min
        radius: float = 0.9 * 0.25 * (bounding_box[1] + bounding_box[2])
        length: float = bounding_box[0] - 2 * radius

        arm_center_pos: np.ndarray = np.mean(arm_point_cloud, axis=0)
        chest_center_pos: np.ndarray = np.mean(chest_point_cloud, axis=0)
        center_pos: np.ndarray = arm_center_pos * weight + chest_center_pos * (1 - weight)
        center_pos[0] = (np.max(arm_point_cloud[..., 0]) + np.min(arm_point_cloud[..., 0])) * 0.5

        if export_json:
            result = self._body_info(arm_index)
            result["Geoms"] = [self._capsule_info(length, radius, center_pos, np.array([1, 0, 0]), 0, name)]
            self.bodies[arm_index] = result
            self.joints[arm_index - 1] = self._ball_joint_info(arm_index, "XYZ", [-45.0, -45.0, -45.0], [45.0, 45.0, 45.0])
        
        if self.do_render:
            self._render_capsule(length, radius, center_pos, np.array([1, 0, 0]), arm_index)

    def _build_hand(self, lr_flag: str, export_json: bool = False):
        """
        In original SMPL model, there is a sub joint for finger.
        Maybe we can remove this joint.
        """
        hand_name: str = lr_flag + "_Wrist"
        finger_name: str = lr_flag + "_Hand"
        hand_index: int = smpl_hierarchy[hand_name]
        finger_index: int = smpl_hierarchy[finger_name]
        # here we should select index to visualize the hand part.
        hand_point_id: np.ndarray = self.smpl.get_subpart_place(hand_index)
        finger_point_id: np.ndarray = self.smpl.get_subpart_place(finger_index)
        point_id: np.ndarray = np.concatenate([hand_point_id, finger_point_id], axis=0)
        # for debug
        finger_point_cloud: np.ndarray = self.smpl.verts[finger_point_id]
        point_cloud: np.ndarray = self.smpl.verts[point_id]

        # get length of hand body
        x_len: float = np.max(point_cloud[..., 0]) - np.min(point_cloud[..., 0])  # use total part
        y_len: float = np.max(finger_point_cloud[..., 1]) - np.min(finger_point_cloud[..., 1])
        z_len: float = np.max(finger_point_cloud[..., 2]) - np.min(finger_point_cloud[..., 2])

        # get mass and inertia of hand body.
        # the simple method is use box to instead
        # however, this is not accurate, but I think it is OK..

        # get position of hand body
        # Actually there are always more vertices in finger, center_pos seems too close to the finger
        # So Maybe the in x-axis, the mean of minimum & maximun is better
        center_pos: np.ndarray = np.mean(point_cloud, axis=0)
        center_pos[0] = (np.max(point_cloud[..., 0]) + np.min(point_cloud[..., 0])) * 0.5
        if self.use_ball_hand:
            if "L" in lr_flag:
                center_pos[0] -= 0.25 * x_len
            elif "R" in lr_flag:
                center_pos[0] += 0.25 * x_len
            center_pos[1] += 0.1 * y_len
            center_pos[2] += 0.05 * z_len

        endjoint_pos = center_pos
        if "L" in lr_flag:
            endjoint_pos = endjoint_pos + np.array([0.5 * x_len, 0.0, 0.0])
        if "R" in lr_flag:
            endjoint_pos = endjoint_pos + np.array([-0.5 * x_len, 0.0, 0.0])

        if export_json:
            result = self._body_info(hand_index)
            if self.use_ball_hand:
                result["Geoms"] = [self._sphere_info((y_len + z_len) / 4, center_pos, 0, lr_flag + hand_name)]
            else:
                result["Geoms"] = [self._cube_info(center_pos, x_len, y_len, z_len, 0, lr_flag + hand_name)]

            self.bodies[hand_index] = result
            self.joints[hand_index - 1] = self._ball_joint_info(hand_index, "XYZ", [-10.0, -10.0, -90.0], [90.0, 10.0, 90.0])
            self.endjoints.append(self._endjoint_info(hand_index, endjoint_pos))
        
        if self.do_render:  # debug mode
            if (hand_index in smpl_render_index) or (finger_index in smpl_render_index):
                box = pv.Cube(center_pos, x_len, y_len, z_len)
                self.plot.add_mesh(box, opacity=self.render_opacity)
                circ = pv.Sphere(0.02, endjoint_pos)
                self.plot.add_mesh(circ, color="green")

    def _build_head(self, export_json: bool = False):
        """
        In original SMPL model, there is a head joint.
        The body position can be placed at heat joint.
        head body can be viewed as a capsule
        """
        head_index: int = smpl_hierarchy["Head"]
        head_point_id: np.ndarray = self.smpl.get_subpart_place(head_index)
        point_cloud: np.ndarray = self.smpl.verts[head_point_id]
        point_cloud_max: np.ndarray = np.max(point_cloud, axis=0)
        point_cloud_min: np.ndarray = np.min(point_cloud, axis=0)
        x_len: float = point_cloud_max[0] - point_cloud_min[0]
        y_len: float = point_cloud_max[1] - point_cloud_min[1]
        z_len: float = point_cloud_max[2] - point_cloud_min[2]
        radius: float = 0.9 * 0.25 * (x_len + z_len)
        capsule_len: float = y_len - 2 * radius
        center_pos = np.array([0.0, np.mean(point_cloud[..., 1], axis=0), 0.0])
        endjoint_pos = center_pos + np.array([0.0, capsule_len * 0.5 + radius, 0.0])

        if export_json:
            result = self._body_info(head_index)
            result["Geoms"] = [self._capsule_info(capsule_len, radius, center_pos, np.array([0, 1, 0]), 0, "Head")]
            self.bodies[head_index] = result
            self.joints[head_index - 1] = self._ball_joint_info(head_index, "XYZ", [-50.0, -50.0, -50.0], [50.0, 50.0, 50.0])
            self.endjoints.append(self._endjoint_info(head_index, endjoint_pos))

        if self.do_render:
            self._render_capsule(capsule_len, radius, center_pos, np.array([0, 1, 0]), head_index)
            circ = pv.Sphere(0.02, endjoint_pos)
            self.plot.add_mesh(circ, color="green")

    def _build_neck(self, export_json: bool = False):
        """
        TODO: Add offset for neck..
        The neck can be viewed as capsule
        """
        neck_index: int = smpl_hierarchy["Neck"]
        neck_parent_name = smpl_parent_info["Neck"]
        neck_parent: int = smpl_hierarchy[neck_parent_name]
        neck_point_id: np.ndarray = self.smpl.get_subpart_place(neck_index)
        point_cloud: np.ndarray = self.smpl.verts[neck_point_id]
        # maybe we need to rotate leg point cloud for computing bounding box...?
        # emm..I don't think so. the offset is not large, so simple bounding box is just ok..
        # for simple, we can just compute bounding box..
        point_cloud_min: np.ndarray = np.min(point_cloud, axis=0)
        point_cloud_max: np.ndarray = np.max(point_cloud, axis=0)
        xyz_len = 0.8 * (point_cloud_max - point_cloud_min)
        radius: float = 0.25 * (xyz_len[0] + xyz_len[2])
        length: float = xyz_len[1] - 2 * radius
        center_pos = np.mean(point_cloud, axis=0)
        # center_pos = np.array([0.0, np.mean(point_cloud[:, 1]), 0.0])
        # center_pos = np.mean(point_cloud)
        # for simple, we just ignore the z offset..
        if export_json:
            result = self._body_info(neck_index)
            result["Geoms"] = [self._sphere_info(radius, center_pos, 0, "Neck")]
            self.bodies[neck_index] = result
            self.joints[neck_index - 1] = self._ball_joint_info(neck_index, "XYZ", [-50.0, -50.0, -50.0], [50.0, 50.0, 50.0])
        
        if self.do_render:
            self._render_Sphere(radius, center_pos, neck_index)

    def _build_upper_leg(self, lr_flag: str, export_json: bool = False):
        """
        The upper leg can be viewed as capsule
        Here we should add offset to hip joint
        """
        self._build_leg_base(lr_flag, "_Hip", 0.8, export_json=export_json)

    def _build_lower_leg(self, lr_flag: str, export_json: bool = False):
        self._build_leg_base(lr_flag, "_Knee", export_json=export_json)

    def _build_leg_base(self, lr_flag: str, name: str, radius_ratio: float = 0.8, export_json: bool = False):
        """
        The lower leg can be viewed as capsule
        """
        is_hip: bool = "Hip" in name
        is_knee: bool = "Knee" in name
        leg_name: str = lr_flag + name
        leg_index: int = smpl_hierarchy[leg_name]
        child_index: int = smpl_children_list[leg_index][0]

        joint_pos: np.ndarray = self.smpl.J[leg_index].copy()
        child_pos: np.ndarray = self.smpl.J[child_index].copy()

        main_axis: np.ndarray = joint_pos - self.smpl.J[child_index]
        main_axis[0] = 0
        unit_main_axis: np.ndarray = main_axis / np.linalg.norm(main_axis)

        # lower_leg_len: float = knee_pos[1] - ankle_pos[1]
        # compute radius of capsule geometry
        # here we can just use simple bounding box...
        low_leg_ids: np.ndarray = self.smpl.get_subpart_place(leg_index)
        point_cloud: np.ndarray = self.smpl.verts[low_leg_ids]
        max_low_leg: np.ndarray = np.max(point_cloud, axis=0)
        min_low_leg: np.ndarray = np.min(point_cloud, axis=0)
        bounding_box: np.ndarray = max_low_leg - min_low_leg
        radius = radius_ratio * 0.25 * (bounding_box[0] + bounding_box[2])
        length = bounding_box[1] - 2 * radius
        center: np.ndarray = (max_low_leg + min_low_leg) * 0.5
        #np.mean(point_cloud, axis=0)

        if export_json:
            result = self._body_info(leg_index)
            result["Geoms"] = [self._capsule_info(length, radius, center, unit_main_axis, 0, leg_name)]
            self.bodies[leg_index] = result
            if "Hip" in name:
                self.joints[leg_index - 1] = self._ball_joint_info(leg_index, "XYZ", [-140.0, -80.0, -170.0], [80.0, 80.0, 170.0])
            else:
                self.joints[leg_index - 1] = self._hinge_joint_info(leg_index, "X", 0.0, 170.0)
        
        if self.do_render:
            self._render_capsule(length, radius, center, unit_main_axis, leg_index)
        
    def _build_foot(self, lr_flag: str, export_json: bool = False):
        """
        The foot can be viewed as 3 ball + 1 capsule
        The heel can be viewed as a single capsule
        """
        if export_json:
            foot_index: int = smpl_hierarchy[lr_flag + "_Ankle"]
            toe_index: int = smpl_hierarchy[lr_flag + "_Foot"]
            foot_joint_pos: np.ndarray = self.smpl.J[foot_index]
            toe_joint_pos: np.ndarray = self.smpl.J[toe_index]
            # szh: here we should move the toe joint position
            foot_point_ids: np.ndarray = self.smpl.get_subpart_place(foot_index)##, 0.6)
            toe_point_ids: np.ndarray = self.smpl.get_subpart_place(toe_index)

            # compute heel
            foot_point_cloud: np.ndarray = self.smpl.verts[foot_point_ids]
            foot_center: np.ndarray = np.mean(foot_point_cloud, axis=0)
            foot_max: np.ndarray = np.max(foot_point_cloud, axis=0)
            foot_min: np.ndarray = np.min(foot_point_cloud, axis=0)
            foot_box: np.ndarray = 1.05 * (foot_max - foot_min)

            # compute toe
            toe_point_cloud: np.ndarray = self.smpl.verts[toe_point_ids]
            toe_center: np.ndarray = np.mean(toe_point_cloud, axis=0)
            toe_max: np.ndarray = np.max(toe_point_cloud, axis=0)
            toe_min: np.ndarray = np.min(toe_point_cloud, axis=0)
            toe_box: np.ndarray = (toe_max - toe_min) * 1.05

            # Here we can enlarge the foot.
            if self.foot_enlarge > 0:
                pass

            foot_toe_offset: np.ndarray = toe_center - foot_center
            foot_toe_length: float = np.linalg.norm(foot_toe_offset)
            unit_offset: np.ndarray = foot_toe_offset / foot_toe_length

            foot_result = self._body_info(foot_index)
            center: np.ndarray = foot_center + unit_offset * foot_box[1] * 0.55
            foot_result["Geoms"].append(self._capsule_info(1.5 * foot_box[1] , 0.3 * toe_box[1], center + np.array([0.0, 0.05 * foot_box[1], 0.0]), unit_offset, 0, f"{lr_flag}Ankle0"))
            foot_toe_offset[1] = 0
            foot_toe_length: float = np.linalg.norm(foot_toe_offset)
            unit_offset: np.ndarray = foot_toe_offset / foot_toe_length
            orthogonal_unit_offset: np.array = np.array([unit_offset[2], 0, -unit_offset[0]])

            center: np.ndarray = foot_center - unit_offset * foot_box[1] * 0.3
            foot_result["Geoms"].append(self._sphere_info(0.3 * foot_box[1], center, 1, f"{lr_flag}Ankle0"))

            def build_ball_a():
                center: np.ndarray = toe_center + unit_offset * foot_toe_length * 0.10 + orthogonal_unit_offset * toe_box[0] * 0.28
                foot_result["Geoms"].append(self._sphere_info(0.35 * toe_box[1], center, len(foot_result["Geoms"]), f"{lr_flag}Ankle0"))
            
            def build_ball_b():
                center: np.ndarray = toe_center + unit_offset * foot_toe_length * 0.10 - orthogonal_unit_offset * toe_box[0] * 0.28
                foot_result["Geoms"].append(self._sphere_info(0.35 * toe_box[1], center, len(foot_result["Geoms"]), f"{lr_flag}Ankle0"))

            if lr_flag == "L":
                build_ball_a()
                build_ball_b()
            else:
                build_ball_b()
                build_ball_a()

            self.bodies[foot_index] = foot_result

            toe_result = self._body_info(toe_index)
            center: np.ndarray = toe_center + unit_offset * foot_toe_length * 0.35
            toe_result["Geoms"].append(self._capsule_info(0.6 * toe_box[0], 0.35 * toe_box[1], center, orthogonal_unit_offset, 0, f"{lr_flag}Toe"))
            self.bodies[toe_index] = toe_result

            new_toe_joint_pos = toe_center + 0.5 * (-unit_offset * foot_toe_length * 0.10 + unit_offset * foot_toe_length * 0.10)
            self.joint_pos[toe_index] = new_toe_joint_pos
            self.joints[foot_index - 1] = self._ball_joint_info(foot_index, "XYZ", [-70.0, -45.0, -30.0], [90.0, 45.0, 30.0], weight = 0.2)
            self.joints[toe_index - 1] = self._hinge_joint_info(toe_index, "X", -10.0, 45.0, weight = 0.2)
            endjoint_pos: np.ndarray = toe_center + unit_offset * (foot_toe_length * 0.10 + toe_box[1] * 0.35)
            self.endjoints.append(self._endjoint_info(toe_index, endjoint_pos))

        # if self.do_render:  # TODO
        #     for (length, radius, center, direction) in tuple_list:
        #         self._render_capsule(length, radius, center, direction, foot_index)
        #     circ = pv.Sphere(0.02, endjoint_pos)
        #     self.plot.add_mesh(circ, color="green")
        
    def _build_root(self, export_json: bool = False):
        """
        The root body can be viewed as ball or capsule
        scale for Y dimension is smaller because the Root contains the private area(butt and something else)
        """
        self._build_capsulelike_spine_base("Pelvis", np.array([0.9, 0.7, 0.9]), export_json)
        
    def _build_capsulelike_spine_base(self, name: str, scale: np.ndarray, export_json: bool = False):
        # For Pelvis & Spine1 & Spine2
        
        spine_index: int = smpl_hierarchy[name]
        spine_point_ids: np.ndarray = self.smpl.get_subpart_place(spine_index)
        point_cloud: np.ndarray = self.smpl.verts[spine_point_ids]
        min_point_cloud: np.ndarray = np.min(point_cloud, axis=0)
        max_point_cloud: np.ndarray = np.max(point_cloud, axis=0)
        center: np.ndarray = (min_point_cloud + max_point_cloud) * 0.5
        mean_diff_point_cloud: np.ndarray = np.mean(np.abs(point_cloud - center), axis=0)
        xyz_len = (max_point_cloud - min_point_cloud) * scale
        radius = 0.45 * xyz_len[1]
        length = xyz_len[0] - 2 * radius
        unit_main_axis = np.array((1.0, 0.0, 0.0))
        main_axis = unit_main_axis * length

        if export_json:
            result = self._body_info(spine_index)
            result["Geoms"] = [self._capsule_info(length, radius, center, unit_main_axis, 0, name)]
            self.bodies[spine_index] = result
            self.joints[spine_index - 1] = self._ball_joint_info(spine_index, "XYZ", [-80.0, -60.0, -60.0], [80.0, 60.0, 60.0])

        if self.do_render:
            self._render_capsule(length, radius, center, unit_main_axis, spine_index)

    # Add by Zhenhua Song
    @classmethod
    def compute_desired_kp_kd(cls, mess_dict: List):
        """
        Scale kp parameter following the paper:
        Adapting Simulated Behaviors For New Characters.
        in this paper, stiffness ~ L^2, damping ~ L^{5/2}
        here L means the length of leg.
        """
        h_160cm: float = 0.931
        root_h: float = mess_dict[0]["Bodies"][0]["Position"][1]
        mess_dict[0]["Joints"].sort(key=lambda x: x["JointID"])
        joints = mess_dict[0]["Joints"]
        kp_ratio: float = (root_h / h_160cm) ** 4
        kd_ratio: float = (root_h / h_160cm) ** 4.5
        kps = []
        for node in joints:
            jname = node["Name"]
            kp: float = kp_ratio * cls.kps_160cm[jname]
            kd: float = kd_ratio * cls.kds_160cm[jname]
            node["Damping"] = kd
            kps.append(kp)
        pd_param = mess_dict[0]["PDControlParam"]
        pd_param["Kps"] = kps
        pd_param["TorqueLimit"] = copy.deepcopy(kps)
        return mess_dict

    def _build_vert_capsulelike_spine_base(self, name: str, scale: np.ndarray, export_json: bool = False):
        # For Spine3
        spine_index: int = smpl_hierarchy[name]
        spine_point_ids: np.ndarray = self.smpl.get_subpart_place(spine_index)
        point_cloud: np.ndarray = self.smpl.verts[spine_point_ids]
        min_point_cloud: np.ndarray = np.min(point_cloud, axis=0)
        max_point_cloud: np.ndarray = np.max(point_cloud, axis=0)
        center: np.ndarray = (min_point_cloud + max_point_cloud) * 0.5
        xyz_len = (max_point_cloud - min_point_cloud) * scale
        radius = 0.45 * xyz_len[2]
        length = xyz_len[1] - 2 * radius
        unit_main_axis = np.array((0.0, 1.0, 0.0))

        if export_json:
            result = self._body_info(spine_index)
            result["Geoms"] = [self._capsule_info(length, radius, center, unit_main_axis, 0, name)]
            self.bodies[spine_index] = result
            self.joints[spine_index - 1] = self._ball_joint_info(spine_index, "XYZ", [-80.0, -60.0, -60.0], [80.0, 60.0, 60.0])

        if self.do_render:
            self._render_capsule(length, radius, center, unit_main_axis, spine_index)
    
    def _build_spine1(self, export_json: bool = False):
        """
        The chest body can be viewed as a ball or capsule
        """
        self._build_capsulelike_spine_base("Spine1", np.array([0.8, 0.9, 0.9]), export_json)

    def _build_spine2(self, export_json: bool = False):
        self._build_capsulelike_spine_base("Spine2", np.array([0.7, 0.9, 0.9]), export_json)
        #smaller scale in X dimension because a smaller capsule

    def _build_spine3(self, export_json: bool = False):
        self._build_vert_capsulelike_spine_base("Spine3", np.array([0.9, 0.7, 0.7]), export_json)

    def build(self):
        """
        we can convert the smpl model into json format here.
        """
        if self.do_render:
            self.plot = pv.Plotter()
            self.smpl.visualize_pyvista_impl(self.plot)
            # render Axis.
            self.plot.add_arrows(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

        self._build_root(False)
        self._build_spine1(False)
        self._build_spine2(False)
        self._build_spine3(False)
        self._build_neck(False)
        self._build_head(False)

        for index in ["L", "R"]:
            self._build_shoulder(index, False)
            self._build_upper_arm(index, False)
            self._build_lower_arm(index, False)
            self._build_hand(index, False)
            self._build_upper_leg(index, False)
            self._build_lower_leg(index, False)
            self._build_foot(index, False)

        self.mass, self.center_mass, self.inertia = self.smpl.initialize_mass(self.plot if self.do_render else None)
        if self.do_render:
            self.smpl.visualize_joint_pyvista_impl(self.plot, self.joint_pos)
            self.plot.show()

        return self

    def export_scene_to_json(self):
        json_output = {
            "WorldAttr" : WorldAttr,
            "Environment": Environment,
            "CharacterList" : {
                "Characters" : self.export_model_to_json()
            },
            "ExtJointList": {"Joints": []},
            "ExtForceList": {"Forces": []}
        }

        return json_output

    def export_scene_to_json_file(self, fname: str):
        result = self.export_scene_to_json()
        with open(fname, "w") as fout:
            json.dump(result, fout, indent=4)
        return result

    def export_model_to_json(self, adjust_kp_kd: bool = True):
        self.bodies = [{} for _ in range(smpl_export_body_size)]
        self.joints = [{} for _ in range(smpl_export_joint_size)]
        self.global_offset = [0.0, -np.min(self.smpl.verts, axis = 0)[1], 0.0]
        self.endjoints = []

        self._build_root(True)
        self._build_spine1(True)
        self._build_spine2(True)
        self._build_spine3(True)
        self._build_neck(True)
        self._build_head(True)

        for index in ["L", "R"]:
            self._build_shoulder(index, True)
            self._build_upper_arm(index, True)
            self._build_lower_arm(index, True)
            self._build_hand(index, True)
            self._build_upper_leg(index, True)
            self._build_lower_leg(index, True)
            self._build_foot(index, True)

        if self.mass_optimize:
            for index in range(1, smpl_export_body_size):
                oppo_idx: int = index - 1
                if smpl_name_list[index][1:] != smpl_name_list[oppo_idx][1:]:
                    continue
                ave_mass: float = (self.mass[index] + self.mass[oppo_idx]) / 2
                for i in [oppo_idx, index]:
                    ratio: float = ave_mass / self.mass[i]
                    self.bodies[i]["Mass"] = self.bodies[i]["Mass"] * ratio
                    self.bodies[i]["Inertia"] = list(np.array(self.bodies[i]["Inertia"]) * ratio)
        
        characters = []
        characters.append(copy.deepcopy(basicCharacterInfo))
        characters[0]["Bodies"] = self.bodies
        characters[0]["Joints"] = self.joints
        # Add by Zhenhua Song. Modify the end joints.
        for node in self.endjoints:
            node["Position"] = list(np.array(node["Position"]) + self.global_offset)

        characters[0]["EndJoints"] = self.endjoints
        characters[0]["PDControlParam"] = PDControlParam
        characters[0]["smpl_theta"] = self.smpl_param.tolist()

        self.mirror_json_character()

        # Modify the kp and kd
        if adjust_kp_kd:
            self.compute_desired_kp_kd(characters)
    
        return characters

    @staticmethod
    def _mirror_avg_pos(node: Dict[str, Any], mirror_node: Dict[str, Any], root_pos: np.ndarray):
        rel_pos: np.ndarray = np.array(node["Position"]) - root_pos
        mirror_rel_pos: np.ndarray = np.array(mirror_node["Position"]) - root_pos
        avg_pos_x = 0.5 * (rel_pos[0] - mirror_rel_pos[0])
        avg_pos_y = 0.5 * (rel_pos[1] + mirror_rel_pos[1])
        avg_pos_z = 0.5 * (rel_pos[2] + mirror_rel_pos[2])
        node["Position"] = (root_pos + np.array([avg_pos_x, avg_pos_y, avg_pos_z])).tolist()
        mirror_node["Position"] = (root_pos + np.array([-avg_pos_x, avg_pos_y, avg_pos_z])).tolist()
        return node, mirror_node

    @staticmethod
    def _mirror_avg_info(node: Dict[str, Any], mirror_node: Dict[str, Any], key: str):
        avg_value = 0.5 * (np.array(node[key]) + np.array(mirror_node[key])) # .tolist()
        if avg_value.size > 1:
            avg_value = avg_value.tolist()
        else:
            avg_value = float(avg_value)
        node[key] = avg_value
        mirror_node[key] = avg_value

    @staticmethod
    def _mirror_pos_info(joints: List[Dict[str, Any]], joint_mirror_list: np.ndarray, root_pos: np.ndarray):
        joint_vis: np.ndarray = np.zeros(len(joints), dtype=np.int32)
        for joint_index, mirror_index in enumerate(joint_mirror_list):
            if joint_vis[joint_index] or joint_vis[mirror_index]:
                continue
            joint_vis[joint_index] = joint_vis[mirror_index] = 1
            SMPLCharacterBuild._mirror_avg_pos(joints[joint_index], joints[mirror_index], root_pos)
        return joints

    @staticmethod
    def _mirror_inertia(node, mirror_node):
        # The correct way is rotate the geom, and compute the mean inertia, and rotate back
        # This implementation is just OK and works.
        inertia = np.array(node["Inertia"])
        mirror_inertia = np.array(mirror_node["Inertia"])
        avg_abs = 0.5 * (np.abs(inertia) + np.abs(mirror_inertia))
        node["Inertia"] = (np.sign(inertia) * avg_abs).tolist()
        mirror_node["Inertia"] = (np.sign(mirror_inertia) * avg_abs).tolist()
        return node, mirror_node

    @staticmethod
    def _mirror_geom_rotation(node: Dict[str, Any], mirror_node: Dict[str, Any]):
        """
        The rotation of geom should also be aligned.
        """
        quat: np.ndarray = np.array(node["Quaternion"])
        m_quat: np.ndarray = np.array(mirror_node["Quaternion"])
        unit_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        unit_x: np.ndarray = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(quat - unit_quat) < 1e-7 and np.linalg.norm(m_quat - unit_quat) < 1e-7:
            return
        align_quat: np.ndarray = flip_quaternion(m_quat, unit_x, False).reshape(4)
        if np.linalg.norm(align_quat * np.sign(align_quat[3]) - quat * np.sign(quat[3])) < 1e-7:
            return
        print("This will not happen???")
        mean_quat = MathHelper.slerp(quat, align_quat, 0.5)
        node["Quaternion"] = mean_quat.tolist()
        mirror_node["Quaternion"] = flip_quaternion(mean_quat, unit_x, False).tolist()
        return node, mirror_node

    def mirror_json_character(self):
        # Add by Zhenhua Song: here we should make sure left part and right part are same..
        for body_index, body_node in enumerate(self.bodies):
            assert body_index == body_node["BodyID"], "body index is not aligned"
        for joint_index, joint_node in enumerate(self.joints):
            assert joint_index == joint_node["JointID"], "joint index is not aligned"
        body_names: List[str] = [node["Name"] for node in self.bodies]  # len == 22
        joint_names: List[str] = [node["Name"] for node in self.joints]  # len == 21
        end_names: List[str] = [node["Name"] for node in self.endjoints]  # len == 5
        body_name_dict = {node: index for index, node in enumerate(body_names)}
        joint_name_dict = {node: index for index, node in enumerate(joint_names)}
        body_mirror_list: np.ndarray = build_smpl_body_mirror_index(body_names)
        joint_mirror_list: np.ndarray = build_smpl_mirror_index(joint_names)
        end_mirror_keys = {
            "EndJoint_Head": "EndJoint_Head",
            "EndJoint_L_Wrist": "EndJoint_R_Wrist",
            "EndJoint_R_Wrist": "EndJoint_L_Wrist",
            "EndJoint_L_Foot": "EndJoint_R_Foot",
            "EndJoint_R_Foot": "EndJoint_L_Foot"
        }
        end_mirror_list = np.array([end_names.index(end_mirror_keys[node]) for node in end_names])
        # TODO: check load end joint in TargetPose.
        root_x: float = self.bodies[0]["Position"][0]
        if root_x != 0:
            for body_index, body_node in enumerate(self.bodies):
                body_node["Position"][0] -= root_x
                for geom_index, geom_node in enumerate(body_node["Geoms"]):
                    geom_node["Position"][0] -= root_x
            for joint_index, joint_node in enumerate(self.joints):
                joint_node["Position"][0] -= root_x
            for end_index, end_node in enumerate(self.endjoints):
                end_node["Position"][0] -= root_x

        root_pos: np.ndarray = np.array(self.bodies[0]["Position"])
        body_vis: np.ndarray = np.zeros(len(self.bodies), dtype=np.int32)
        for body_index, mirror_index in enumerate(body_mirror_list):
            if body_index == mirror_index or body_vis[body_index] or body_vis[mirror_index]:
                continue
            body_vis[body_index] = body_vis[mirror_index] = 1
            body_node = self.bodies[body_index]
            mirror_body_node = self.bodies[mirror_index]
            self._mirror_avg_info(body_node, mirror_body_node, "Mass")
            self._mirror_avg_pos(body_node, mirror_body_node, root_pos)
            self._mirror_inertia(body_node, mirror_body_node)

            # we should also mirror the size, position of geoms, and rotation of geoms.
            for geom_index in range(len(body_node["Geoms"])):
                body_geom = body_node["Geoms"][geom_index]
                mirror_body_geom = mirror_body_node["Geoms"][geom_index]
                assert body_geom["GeomType"] in ["Sphere", "Capsule", "Cube"], "Geom Type not supported."
                self._mirror_avg_info(body_geom, mirror_body_geom, "Scale")
                self._mirror_avg_pos(body_geom, mirror_body_geom, root_pos)
                self._mirror_geom_rotation(body_geom, mirror_body_geom)

        # for body_index, body_node in enumerate(self.bodies):
        #     if len(body_node["Geoms"]) == 1:
        #         body_geom = body_node["Geoms"][0]
        #         body_node["Positon"] = (0.5 * (
        #             np.array(body_node["Position"]) + np.array(body_geom["Position"]))).tolist()
        #         body_geom["Position"] = body_node["Positon"]

        # modify the foot and toe.
        def get_avg_h(lr_flag_: str):
            ankle = self.bodies[smpl_hierarchy[f"{lr_flag_}_Ankle"]]
            toe = self.bodies[smpl_hierarchy[f"{lr_flag_}_Foot"]]
            avg_h = [node["Position"][1] - node["Scale"][0] for node in ankle["Geoms"] if node["GeomType"] == "Sphere"]
            avg_h.append(toe["Geoms"][0]["Position"][1] - toe["Geoms"][0]["Scale"][0])
            return np.mean(avg_h)

        avg_h_l = get_avg_h("L")
        avg_h_r = get_avg_h("R")

        self._mirror_pos_info(self.joints, joint_mirror_list, root_pos)
        self._mirror_pos_info(self.endjoints, end_mirror_list, root_pos)

        self._adjust_toe_mass(body_name_dict)
        self._adjust_back_mass(body_name_dict, joint_name_dict)

        # Add geom ignore index.
        ignore_list = {
            "Neck": ["L_Collar", "R_Collar"],
            "L_Collar": ["Neck", "R_Collar"],
            "R_Collar": ["L_Collar", "Neck"],
        }
        for body_name, ignore_names in ignore_list.items():
            body_idx = body_name_dict[body_name]
            self.bodies[body_idx]["IgnoreBodyID"] = [body_name_dict[node] for node in ignore_names]

    def _adjust_toe_mass(self, body_name_dict: Dict[str, int]):
        """
        The inertia is proportional to the mass
        """
        def adjust_bnode(bname: str, ratio: float):
            bnode = self.bodies[body_name_dict[bname]]
            bnode["Density"] *= ratio
            bnode["Mass"] *= ratio
            bnode["Inertia"] = (ratio * np.array(bnode["Inertia"])).tolist()

        for lr in ["L", "R"]:  # here we should also modify mass of hand
            adjust_bnode(f"{lr}_Foot", 1.5)
            adjust_bnode(f"{lr}_Wrist", 1.5)

        adjust_bnode("Head", 0.6)
        adjust_bnode("Neck", 1.5)
    
    def _adjust_back_mass(self, body_name_dict: Dict[str, int], joint_name_dict: Dict[str, int]):
        """
        Adjust mass position
        """
        # adjust joints.
        center_jnames = [key for key, value in smpl_mirror_dict.items()
            if key == value and key != "RootJoint"]
        for jname in center_jnames:
            joint = self.joints[joint_name_dict[jname]]
            assert joint["Name"] == jname
            joint["Position"][0] = 0
        
        # adjust bodies.
        center_bnames = [key for key, value in smpl_body_mirror_dict.items() if key == value]
        for cname in center_bnames:
            body = self.bodies[body_name_dict[cname]]
            assert body["Name"] == cname
            if len(body["Geoms"]) == 1:
                body["Geoms"][0]["Position"][0] = 0
            body["Position"][0] = 0

        # adjust center of mass.
        back_names = ["Spine1", "Spine2", "Spine3"]
        adjust_val = self.back_offset
        for cname in back_names:
            body = self.bodies[body_name_dict[cname]]
            if self.move_mode == CreateMoveMode.behind:
                body["Position"][2] -= adjust_val
            elif self.move_mode == CreateMoveMode.front:
                body["Position"][2] += adjust_val


SMPLCharacterBuild.create_default_builder(do_render=False)

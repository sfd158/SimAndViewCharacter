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
from typing import Tuple
from scipy.spatial.transform import Rotation
from ...Common.MathHelper import MathHelper
from .UnityBase import PrimitiveType, UnityInfoBaseType
from ...ODESim.ODEScene import *


class ODEToUnity:
    def __init__(self, scene: ODEScene, with_joints: bool = True):
        self.scene = scene
        self.unity_base = UnityInfoBaseType()
        self.rot_z2y: Rotation = Rotation(MathHelper.quat_between(np.array([0, 0, 1]), np.array([0, 1, 0])))
        self.rot_y2z: Rotation = self.rot_z2y.inv()

        self.with_joints = with_joints
        self.joint_radius = 0.025

    @property
    def world(self):
        return self.scene.world

    @property
    def space(self):
        return self.scene.space

    # Input is 1-Dim
    @staticmethod
    def geom_type_from_ode_to_unity(ode_type: np.ndarray) -> np.ndarray:
        """
        convert ode Geom Type to Unity PrimitiveType
        """
        res = np.zeros_like(ode_type, dtype=np.uint64)
        res[ode_type == ode.GeomTypes.Sphere] = PrimitiveType.Sphere.value
        res[ode_type == ode.GeomTypes.Box] = PrimitiveType.Cube.value
        res[ode_type == ode.GeomTypes.Capsule] = PrimitiveType.Capsule.value
        res[ode_type == ode.GeomTypes.Cylinder] = PrimitiveType.Cylinder.value
        res[ode_type == ode.GeomTypes.Plane] = PrimitiveType.Plane.value
        return res

    def get_hierarchy_info(self):
        """
        Create Body and Joint in Unity for Rendering
        """
        self.unity_base.clear()

        geom_info: Tuple = self.space.getAllGeomInfos(True, True)
        self.unity_base.create_id = geom_info[0]
        unity_type = self.geom_type_from_ode_to_unity(geom_info[1])
        self.unity_base.create_type = unity_type
        self.unity_base.create_pos = geom_info[2]
        self.unity_base.create_quat = geom_info[3]
        self.unity_base.create_scale = geom_info[4]  # Get Create Scale
        self.unity_base.create_name = geom_info[5]  # Get Create Name

        # Get Create Child Quaternion
        child_quat = MathHelper.unit_quat_arr((len(self.unity_base), 4))
        child_quat[unity_type == PrimitiveType.Cylinder.value] = self.rot_y2z.as_quat()
        child_quat[unity_type == PrimitiveType.Capsule.value] = self.rot_y2z.as_quat()
        child_quat[unity_type == PrimitiveType.Plane] = self.rot_y2z.as_quat()
        self.unity_base.create_child_quat = child_quat.flatten()

        if self.with_joints:
            self.create_joint_info()

        # Create Color
        self.unity_base.gen_ran_color()

        return self.unity_base.to_json_dict()

    def get_update_info(self):
        """
        Update Body and Joint Position and Rotation in Unity for Rendering
        """
        self.unity_base.clear()

        geom_info: Tuple = self.space.getPlaceableGeomInfos()
        # return geom id, type(ode), pos, quat(scipy)
        self.unity_base.modify_id = geom_info[0]
        self.unity_base.modify_pos = geom_info[2]
        self.unity_base.modify_quat = geom_info[3]

        if self.with_joints:
            self.update_joint_info()

        return self.unity_base.to_json_dict()
        # TODO: Add Create Info

    def create_joint_info(self):
        """
        Create Joint in Unity for rendering
        """
        ball_and_hinge_info: Tuple[np.ndarray, np.ndarray] = self.world.getBallAndHingeInfos()  # id, pos
        ball_and_hinge_id = ball_and_hinge_info[0]
        ball_and_hinge_pos = ball_and_hinge_info[1]
        self.unity_base.create_id = np.concatenate([self.unity_base.create_id, ball_and_hinge_id])
        create_type = np.zeros_like(ball_and_hinge_id, dtype=np.uint64)
        create_type[:] = PrimitiveType.Sphere.value
        self.unity_base.create_type = np.concatenate([self.unity_base.create_type, create_type])
        self.unity_base.create_pos = np.concatenate([self.unity_base.create_pos, ball_and_hinge_pos])
        self.unity_base.create_quat = np.concatenate([self.unity_base.create_quat,
                                                      MathHelper.unit_quat_arr((ball_and_hinge_id.size, 4)).flatten()])
        scale = np.zeros_like(ball_and_hinge_pos)
        scale[:] = self.joint_radius
        self.unity_base.create_scale = np.concatenate([self.unity_base.create_scale, scale])
        self.unity_base.create_child_quat = np.concatenate(
            [self.unity_base.create_child_quat,
             MathHelper.unit_quat_arr((ball_and_hinge_id.size, 4)).flatten()
             ])
        self.unity_base.create_name.extend(["joint" for _ in range(ball_and_hinge_info[0].size)])

    def update_joint_info(self):
        """
        Update Joint Position in Unity for rendering
        """
        ball_info: Tuple[np.ndarray, np.ndarray] = self.world.getBallAndHingeInfos()
        ball_id = ball_info[0]
        ball_pos = ball_info[1]
        self.unity_base.modify_id = np.concatenate([self.unity_base.modify_id, ball_id])
        self.unity_base.modify_pos = np.concatenate([self.unity_base.modify_pos, ball_pos])
        self.unity_base.modify_quat = np.concatenate([self.unity_base.modify_quat,
                                                      MathHelper.unit_quat_arr((ball_id.size, 4)).flatten()])


if __name__ == "__main__":
    a = np.array([PrimitiveType.Cube.value, PrimitiveType.Sphere.value, PrimitiveType.Cube.value,
                  PrimitiveType.Cylinder.value, PrimitiveType.Capsule.value])
    b = np.array([ode.GeomTypes.Box, ode.GeomTypes.Sphere, ode.GeomTypes.Box,
                  ode.GeomTypes.Cylinder, ode.GeomTypes.Capsule])

    print(a, b)
    print(ODEToUnity.geom_type_from_ode_to_unity(b))

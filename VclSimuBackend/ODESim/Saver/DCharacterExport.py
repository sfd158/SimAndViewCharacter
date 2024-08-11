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
TODO: This file should be tested..
Export ODECharacter as Unity format..
"""
import ModifyODE as ode
from scipy.spatial.transform import Rotation
from typing import Optional, Any, Dict, List
from ..ODECharacter import ODECharacter
from ..CharacterWrapper import CharacterWrapper


class TwoDictSameCheck:
    def __init__(self):
        pass

    @staticmethod
    def check_two_list(result0: List, result1: List, eps: float = 1e-4):
        assert len(result0) == len(result1)
        result0.sort()
        result1.sort()
        for value0, value1 in zip(result0, result1):
            TwoDictSameCheck.sub_check(value0, value1, eps)

    @staticmethod
    def sub_check(value0, value1, eps: float = 1e-4):
        assert type(value0) == type(value1)
        if isinstance(value0, List):
            TwoDictSameCheck.check_two_list(value0, value1)
        elif isinstance(value0, int) or isinstance(value0, str):
            assert value0 == value1
        elif isinstance(value0, float):
            assert abs(value0 - value1) < eps
        elif isinstance(value0, Dict):
            TwoDictSameCheck.check_two_dict(value0, value1)
        else:
            raise NotImplementedError(f"type(value0) == {type(value0)}")

    @staticmethod
    def check_two_dict(result0: Dict, result1: Dict, eps: float = 1e-4):
        for key in result0.keys():
            value0 = result0[key]
            value1 = result1[key]
            TwoDictSameCheck.sub_check(value0, value1, eps)


class JsonCharacterExport(CharacterWrapper):
    """
    Export Character as json format
    """
    def __init__(self, character: ODECharacter) -> None:
        super().__init__(character)
        self.character.load_init_state()

    def export_body_list(self) -> List[Dict[str, Any]]:
        """
        Export all bodies
        """
        result = []
        for body_index, body in enumerate(self.bodies):
            body_result = self.export_body(body_index, body)
            result.append(body_result)
        return result

    def export_joint_list(self) -> List[Dict[str, Any]]:
        """
        export all joints.
        """
        result = []
        for joint_index, joint in enumerate(self.joints):
            joint_result = self.export_joint(joint_index, joint)
            result.append(joint_result)
        return result

    def export_end_joint_list(self):
        result = []
        return result

    def export_pd_param(self):
        """
        Export PD Control Param
        """
        result = {

        }
        return result

    @staticmethod
    def export_geom_list(geom_list: List[ode.GeomObject]):
        pass

    def export_body(self, body_index: int, body: ode.Body) -> Dict[str, Any]:
        geom_list = list(body.geom_iter())

        result = {
            "BodyID": body.instance_id,
            "Name": body.name,
            "Density": None,
            "ParentJointID": None,
            "ParentBodyID": None,
            "Position": body.PositionNumpy.tolist(),
            "Quaternion": body.getQuaternionScipy().tolist(),
            "LinearVelocity": body.LinearVelNumpy.tolist(),
            "AngularVelocity": body.getAngularVelNumpy().tolist(),
            "Geoms": JsonCharacterExport.export_geom_list(geom_list),
            "IgnoreBodyID": None
        }
        return result

    @staticmethod
    def export_joint_type(joint: ode.Joint):
        """
        joint type string
        """
        if isinstance(joint, ode.HingeJoint):
            return "HingeJoint"
        elif isinstance(joint, ode.BallJointBase):
            return "BallJoint"
        else:
            raise NotImplementedError

    @staticmethod
    def export_joint_angle_limit(joint: ode.Joint):
        if isinstance(joint, ode.HingeJoint):
            lo_limit, hi_limit = ode.HingeJoint.AngleLoStop
        elif isinstance(joint, ode.BallJoint):  # There is no Angle Limit..
            lo_limit = [-180.0] * 3
            hi_limit = [180.0] * 3
        elif isinstance(joint, ode.BallJointAmotor):
            lo1, hi1 = joint.getAngleLimit1()
            lo2, hi2 = joint.getAngleLimit2()
            lo3, hi3 = joint.getAngleLimit3()
            lo_limit = [lo1, lo2, lo3]
            hi_limit = [hi1, hi2, hi3]
        else:
            raise NotImplementedError

        return lo_limit, hi_limit

    def export_joint(self, joint_index: int, joint: ode.Joint) -> Dict[str, Any]:
        """
        export joint info
        """
        # joint quaternion: inv(parent body quat) * child body quat
        child_body: ode.Body = joint.body1
        parent_body: Optional[ode.Body] = joint.body2

        result = {
            "JointID": joint.instance_id,
            "Name": joint.name,
            "JointType": self.export_joint_type(joint),
            "Weight": self.joint_info.weights[joint_index],
            "Position": joint.getAnchorNumpy(),
            "Quaternion": child_body.getQuaternionScipy(),
            "AngleLoLimit": None,
            "AngleHiLimit": None,
            "EulerOrder": joint.euler_order,
            "EulerAxisLocalRot": None,
            "ParentBodyID": -1 if parent_body is None else parent_body.instance_id,
            "ChildBodyID": child_body.instance_id,
            "ParentJointID": self.joint_info.pa_joint_id[joint_index]
        }
        return result

    @staticmethod
    def export_geom_scale(geom: ode.GeomObject) -> List[float]:
        """
        export geom scale
        """
        if isinstance(geom, ode.GeomBox):
            return list(geom.geomLength)
        elif isinstance(geom, ode.GeomSphere):
            radius: float = geom.geomRadius
            return [radius] * 3
        elif isinstance(geom, ode.GeomCapsule):
            radius, length = geom.geomRadiusAndLength
            return [radius, length, radius]
        else:
            raise NotImplementedError("Geom Type not supported")

    def export_geom(self, geom: ode.GeomObject) -> Dict[str, Any]:
        """
        export geom info
        """
        result = {
            "GeomID": geom.instance_id,
            "Name": geom.name,
            "GeomType": None,
            "Collidable": geom.collidable,
            "Friction": geom.friction,
            "Restitution": None,
            "ClungEnv": geom.clung_env,
            "Position": geom.PositionNumpy.tolist(),
            "Quaternion": geom.QuaternionScipy.tolist(),
            "Scale": JsonCharacterExport.export_geom_scale(geom)
        }
        return result

    def export(self) -> Dict[str, Any]:
        """
        export character
        """
        body_list = self.export_body_list()
        joint_list = self.export_joint_list()
        end_list = self.export_end_joint_list()
        result = {
            "Bodies": body_list,
            "Joints": joint_list,
            "EndJoints": end_list
        }

        return result

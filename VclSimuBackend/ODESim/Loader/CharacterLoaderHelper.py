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

from enum import IntEnum
import numpy as np
import ModifyODE as ode
from typing import Any, Tuple, Optional, Dict, List, Union
from scipy.spatial.transform import Rotation
from ...Common.MathHelper import MathHelper
from ..ODECharacterInit import ODECharacterInit, ODECharacter, CharacterWrapper, JointInfoInit


class DRigidBodyMassMode(IntEnum):
    """
    Compute the mass of rigid body by density or given mass value
    """
    Density = 0
    MassValue = 1


class DRigidBodyInertiaMode(IntEnum):
    """
    Compute the inertia of rigid body by density or given inertia value
    """
    Density = 0
    InertiaValue = 1


class GeomType:
    """
    parse geometry type
    """
    def __init__(self):
        self._sphere: Tuple[str] = ("sphere", )
        self._capsule: Tuple[str, str] = ("capsule", "ccylinder")
        self._box: Tuple[str, str] = ("box", "cube")
        self._plane: Tuple[str] = ("plane",)

    def is_sphere(self, geom_type: str):
        return geom_type.lower() in self._sphere

    def is_capsule(self, geom_type: str):
        return geom_type.lower() in self._capsule

    def is_box(self, geom_type: str):
        return geom_type.lower() in self._box

    def is_plane(self, geom_type: str):
        return geom_type.lower() in self._plane

    @property
    def sphere_type(self) -> str:
        return self._sphere[0]

    @property
    def capsule_type(self) -> str:
        return self._capsule[0]

    @property
    def box_type(self) -> str:
        return self._box[0]

    @property
    def plane_type(self) -> str:
        return self._plane[0]


class CharacterLoaderHelper(CharacterWrapper):
    """
    Our character model is defined at world coordinate.
    """
    def __init__(self, world: ode.World, space: ode.SpaceBase, update_body_pos_by_com: bool = False, create_character: bool = True):
        super(CharacterLoaderHelper, self).__init__()
        if create_character:
            self.character = ODECharacter(world, space)
        else:
            self.character: Optional[ODECharacter] = None
        self.geom_type = GeomType()
        self.character_init = ODECharacterInit(self.character)
        self.default_friction: float = 0.8
        self.update_body_pos_by_com: bool = update_body_pos_by_com
        self._world = world
        self._space = space

    @property
    def world(self):
        return self._world
    
    @property
    def space(self):
        return self._space

    def set_character(self, character: Optional[ODECharacter] = None):
        self.character = character
        self.character_init.character = character

    def create_geom_object(
        self,
        json_geom: Dict[str, Any],
        calc_mass: bool = True,
        default_density: float = 1000.0,
        friction: Optional[float] = None
    ) -> Tuple[ode.GeomObject, Optional[ode.Mass]]:
        """
        create geometry object
        """
        geom_type: str = json_geom["GeomType"]
        geom_scale = np.array(json_geom["Scale"])

        mass_geom = ode.Mass() if calc_mass else None
        if self.geom_type.is_sphere(geom_type):
            geom_radius: float = geom_scale[0]
            assert geom_radius > 0, json_geom["Name"]
            geom = ode.GeomSphere(self.space, geom_radius)
            if calc_mass:
                mass_geom.setSphere(default_density, geom_radius)
        elif self.geom_type.is_capsule(geom_type):
            geom_radius, geom_length = geom_scale[0], geom_scale[1]
            assert geom_radius >= 0 and geom_length >= 0, json_geom["Name"]
            geom = ode.GeomCCylinder(self.space, geom_radius, geom_length)
            if calc_mass:
                mass_geom.setCapsule(default_density, 3, geom_radius, geom_length)
        elif self.geom_type.is_box(geom_type):
            assert geom_scale[0] > 0 and geom_scale[1] > 0 and geom_scale[2] > 0, json_geom["Name"]
            geom = ode.GeomBox(self.space, geom_scale)
            if calc_mass:
                mass_geom.setBox(default_density, *geom_scale)
        elif self.geom_type.is_plane(geom_type):
            # convert position and quaternion to n.x*x+n.y*y+n.z*z = dist
            normal_vec = Rotation(json_geom["Quaternion"]).apply(MathHelper.up_vector())
            dist = np.dot(np.asarray(json_geom["Position"]), normal_vec).item()
            geom = ode.GeomPlane(self.space, normal_vec, dist)
            if calc_mass:
                raise ValueError("Plane Geom Object dosen't have mass.")
        else:
            raise NotImplementedError(geom_type)

        geom.name = json_geom["Name"]
        geom.collidable = json_geom["Collidable"]
        # print(geom.name, geom.collidable)

        if friction is None:
            friction = self.default_friction
        geom.friction = json_geom["Friction"] if "Friction" in json_geom else friction

        clung_env = json_geom.get("ClungEnv")
        if clung_env is not None:
            geom.clung_env = clung_env

        geom.instance_id = json_geom["GeomID"]

        if not self.geom_type.is_plane(geom_type):
            geom.PositionNumpy = np.asarray(json_geom["Position"])
            geom.QuaternionScipy = np.asarray(json_geom["Quaternion"])

        return geom, mass_geom

    def add_body(self, json_body: Dict[str, Any]) -> ode.Body:
        """
        @param: recompute_body_pos:
        return: ode.Body
        """
        assert json_body["BodyID"] == len(self.bodies)

        body = ode.Body(self.world)
        body.instance_id = json_body["BodyID"]
        geom_info_list: List = json_body["Geoms"]
        geom_info_list.sort(key=lambda x: x["GeomID"])

        body_density: float = json_body["Density"]
        if body_density == 0.0:
            body_density = 1.0
        create_geom: List[ode.GeomObject] = []
        body.PositionNumpy = np.array(json_body["Position"])
        def geom_creator():
            gmasses: List[ode.Mass] = []
            gcenters = []
            grots = []

            for json_geom in geom_info_list:
                geom, mass_geom = self.create_geom_object(
                    json_geom, True, body_density,
                    json_body["Friction"] if "Friction" in json_body else None
                )

                create_geom.append(geom)
                gmasses.append(mass_geom)
                gcenters.append(np.array(json_geom["Position"]))
                grots.append(Rotation(np.asarray(json_geom["Quaternion"])))

            mass_total_ = self.character_init.compute_geom_mass_attr(
                body, create_geom, gmasses, gcenters, grots, self.update_body_pos_by_com)
            return mass_total_

        mass_total = geom_creator()

        mass_mode: DRigidBodyMassMode = DRigidBodyMassMode[json_body.get("MassMode", "Density")]
        inertia_mode: DRigidBodyInertiaMode = DRigidBodyInertiaMode[json_body.get("InertiaMode", "Density")]
        # load fixed inertia of this body, rather than compute by geometry
        if inertia_mode == DRigidBodyInertiaMode.InertiaValue:
            mass_total.inertia = np.asarray(json_body["Inertia"])

        # load fixed mass value of this body, rather than compute by geometry
        if mass_mode == DRigidBodyMassMode.Density:
            if json_body["Density"] == 0.0:
                mass_total = ode.Mass()
        elif mass_mode == DRigidBodyMassMode.MassValue:
            mass_total.mass = json_body["Mass"]
        else:
            raise ValueError(f"mass_mode = {mass_mode}, which is not supported")

        body.setQuaternionScipy(np.asarray(json_body["Quaternion"]))

        lin_vel = json_body.get("LinearVelocity")  # initial linear velocity
        if lin_vel is not None and len(lin_vel) > 0:
            body.LinearVelNumpy = np.asarray(lin_vel)

        ang_vel = json_body.get("AngularVelocity")  # initial angular velocity
        if ang_vel:
            body.setAngularVelNumpy(np.asarray(ang_vel))

        self.character_init.append_body(body, mass_total, json_body["Name"], json_body["ParentBodyID"])
        return body

    def joint_attach(self, joint: ode.Joint, joint_pos: np.ndarray, joint_parent: int, joint_child: int):
        """
        attach bodies to joint
        """
        if joint_parent == -1:
            joint.attach_ext(self.bodies[joint_child], ode.environment)
        else:
            joint.attach_ext(self.bodies[joint_child], self.bodies[joint_parent])
        # if type(joint) == ode.FixedJoint:
        #    return
        joint.setAnchorNumpy(np.asarray(joint_pos))

    @staticmethod
    def calc_hinge_axis(euler_order: str, axis_mat: Optional[np.ndarray] = None) -> np.ndarray:
        if axis_mat is None:
            axis_mat = np.eye(3)
        return axis_mat[ord(euler_order[0].upper()) - ord('X')]

    @staticmethod
    def set_ball_limit(
        joint: ode.BallJointAmotor,
        euler_order: str,
        angle_limits: Union[List, np.ndarray],
        raw_axis: Optional[np.ndarray] = None
    ):
        return JointInfoInit.set_ball_joint_limit(joint, euler_order, angle_limits, raw_axis)

    # Not Attached
    @staticmethod
    def create_joint_base(world: ode.World, json_joint: Dict[str, Any], load_hinge: bool = True, use_ball_limit: bool = True):
        if json_joint["JointType"] == "BallJoint":
            if "Character0ID" in json_joint or "Character1ID" in json_joint:
                joint = ode.BallJointAmotor(world)
            elif json_joint["ParentBodyID"] == -1 or json_joint["Name"] == "RootJoint":
                joint = ode.BallJoint(world)
            else:
                if use_ball_limit:
                    joint = ode.BallJointAmotor(world)
                else:
                    joint = ode.BallJoint(world)

        elif json_joint["JointType"] == "HingeJoint":
            if load_hinge:
                joint = ode.HingeJoint(world)
            else:
                joint = ode.BallJointAmotor(world)
        elif json_joint["JointType"] == "FixedJoint":
            joint = ode.FixedJoint(world)
        else:
            raise NotImplementedError

        joint.name = json_joint["Name"]
        joint.euler_order = json_joint["EulerOrder"]
        if "Damping" in json_joint:
            joint.setSameKd(json_joint["Damping"])

        joint.instance_id = json_joint["JointID"]
        return joint

    @staticmethod
    def post_create_joint(
        joint: Union[ode.BallJointAmotor, ode.HingeJoint, ode.BallJoint, ode.FixedJoint],
        json_joint: Dict[str, Any],
        load_limits: bool = True
    ):
        axis_q: Optional[List[float]] = json_joint.get("EulerAxisLocalRot")
        axis_q: np.ndarray = np.asarray(MathHelper.unit_quat() if not axis_q else axis_q)
        axis_mat = Rotation(axis_q).as_matrix()
        # joint.euler_axis = axis_mat

        if type(joint) == ode.BallJointAmotor:
            if load_limits and json_joint["JointType"] == "BallJoint":
                angle_lim = np.vstack([json_joint["AngleLoLimit"], json_joint["AngleHiLimit"]]).T
                CharacterLoaderHelper.set_ball_limit(joint, json_joint["EulerOrder"], angle_lim, axis_mat)
        elif type(joint) == ode.HingeJoint:
            hinge_axis = CharacterLoaderHelper.calc_hinge_axis(json_joint["EulerOrder"], axis_mat)
            joint.setAxis(hinge_axis)
            if load_limits:
                angle_lim = np.deg2rad([json_joint["AngleLoLimit"][0], json_joint["AngleHiLimit"][0]])
                joint.setAngleLimit(angle_lim[0], angle_lim[1])
        elif type(joint) == ode.BallJoint:  # need to do nothing here
            pass
        elif type(joint) == ode.FixedJoint:
            joint.setFixed()
        else:
            raise NotImplementedError
        return joint

    def create_joint(self, json_joint: Dict[str, Any], load_hinge: bool = True, load_limits: bool = True) -> ode.Joint:
        """

        :param json_joint:
        :param load_hinge:
        :param load_limits:
        :return:
        """
        joint = self.create_joint_base(self.world, json_joint, load_hinge, load_limits)

        if json_joint["ParentBodyID"] == -1 or json_joint["Name"] == "RootJoint":
            self.joint_info.root_idx = json_joint["JointID"]
            self.joint_info.has_root = True

        self.joint_attach(joint, json_joint["Position"], json_joint["ParentBodyID"], json_joint["ChildBodyID"])
        if json_joint["JointType"] == "HingeJoint" and not load_hinge:  # if load_hinge == False, ignore amotor limit
            return joint
        else:
            return self.post_create_joint(joint, json_joint, load_limits)

    def add_joint(self, json_joint: Dict[str, Any], load_hinge: bool = True, load_limits: bool = True) -> ode.Joint:
        """
        parse joint info
        :param json_joint: joint in json format
        :param load_hinge:
        :param load_limits:
        :return: joint
        """
        assert json_joint["JointID"] == len(self.joints)
        joint = self.create_joint(json_joint, load_hinge, load_limits)

        damping = json_joint.get("Damping")
        if damping:
            self.joint_info.kds[json_joint["JointID"]] = damping
            joint.setSameKd(damping)

        self.joint_info.weights[json_joint["JointID"]] = json_joint.get("Weight", 1.0)
        self.joint_to_child_body.append(json_joint["ChildBodyID"])
        self.joint_info.joints.append(joint)

        return joint

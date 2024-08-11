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
from typing import List, Optional
from scipy.spatial.transform import Rotation
import ModifyODE as ode

from .CharacterWrapper import CharacterWrapper, ODECharacter
from .JointInfoInit import JointInfoInit


class ODECharacterInit(CharacterWrapper):
    def __init__(self, character: ODECharacter):
        super(ODECharacterInit, self).__init__(character)
        self.joint_init: Optional[JointInfoInit] = JointInfoInit(self.joint_info) if character is not None else None

    def init_after_load(self, character_id: int = 0,
                        ignore_parent_collision: bool = True,
                        ignore_grandpa_collision: bool = True,
                        ):
        """
        initialize character after loading from (xml) configure file
        """
        self.character.character_id = character_id
        self.joint_init.init_after_load()
        self.body_info.init_after_load(ignore_parent_collision, ignore_grandpa_collision)
        self.calc_map_body_joint()
        self.calc_joint_parent_idx()
        self.calc_height()
        # self.set_geom_clung()
        self.set_geom_character_id(character_id)
        self.set_geom_max_friction()
        self.set_geom_index()
        self.calc_joint_parent_body_c_id()
        self.calc_joint_child_body_c_id()
        self.character.save_init_state()
        if self.space is not None:
            self.space.ResortGeoms()

    def calc_map_body_joint(self):
        """

        """
        # self.character.joint_to_child_body = [-1] * len(self.joint_info)
        self.character.child_body_to_joint = [-1] * len(self.body_info)
        self.character.joint_to_parent_body = [-1] * len(self.joint_info)  # used in PDController
        # self.character.parent_body_to_joint = [-1] * len(self.body_info)

        # for j_idx, ch_body_name in enumerate(self.joint_info.child_body_name):  # joint->child_body
        #    self.joint_to_child_body[j_idx] = self.body_info.body_idx_dict[ch_body_name]
        # print(self.joint_to_child_body)
        for j_idx, b_idx in enumerate(self.joint_to_child_body):
            self.child_body_to_joint[b_idx] = j_idx

        for j_idx, b_idx in enumerate(self.joint_to_child_body):
            self.joint_to_parent_body[j_idx] = self.body_info.parent[b_idx]

    # Calc parent joint's id
    def calc_joint_parent_idx(self):
        """
        Calc parent joint id of each joint.
        requirement:
        :return:
        """
        self.joint_info.pa_joint_id = [-1] * len(self.joint_info)
        for j_idx, b_idx in enumerate(self.joint_to_child_body):
            if b_idx == -1:
                continue
            pa_b_idx = self.body_info.parent[b_idx]
            if pa_b_idx == -1:  # Root Body
                continue
            self.joint_info.pa_joint_id[j_idx] = self.child_body_to_joint[pa_b_idx]

    def init_end_joint_pa_joint_id(self, init_c_id: bool = True):
        """
        Calc parent joint id of each end joint.
        requirement: self.end_joint.pa_body_id, self.child_body_to_joint
        :param init_c_id:
        :return:
        """
        self.end_joint.pa_joint_id = np.array([self.child_body_to_joint[i] for i in self.end_joint.pa_body_id])
        if init_c_id:
            self.end_joint.pa_joint_c_id = np.array([
                self.joints[i].get_jid()
                for i in self.end_joint.pa_joint_id], dtype=np.uint64)

    def init_end_joint(self, names: List[str], parent_body_ids: List[int], end_pos: np.ndarray):
        """
        initialize end joints
        """
        # name
        self.end_joint.name = names
        self.end_joint.weights = np.ones(len(names))

        # parent body id
        self.end_joint.pa_body_id = np.array(parent_body_ids)
        # np.array([self.body_info.body_idx_dict[i] for i in names])
        self.end_joint.pa_body_c_id = np.array([self.bodies[i].get_bid()
                                                for i in self.end_joint.pa_body_id], dtype=np.uint64)

        # parent body quaternion
        body_quat = self.world.getBodyQuatScipy(self.end_joint.pa_body_c_id).reshape((-1, 4))
        rot_inv: Rotation = Rotation(body_quat).inv()

        # position relative to parent body
        self.end_joint.init_global_pos = end_pos
        self.end_joint.jtob_init_global_pos = end_pos - self.world.getBodyPos(
            self.end_joint.pa_body_c_id).reshape((-1, 3))
        self.end_joint.jtob_init_local_pos = rot_inv.apply(self.end_joint.jtob_init_global_pos)

        # parent joint id
        self.init_end_joint_pa_joint_id(True)

        # global position of parent joint (anchor 1)
        pa_joint_global_pos = self.world.getBallAndHingeAnchor1(self.end_joint.pa_joint_c_id).reshape((-1, 3))

        # position relative to parent joint in world coordinate
        self.end_joint.jtoj_init_global_pos = end_pos - pa_joint_global_pos

        # position relative to parent joint in local coordinate
        self.end_joint.jtoj_init_local_pos = rot_inv.apply(self.end_joint.jtoj_init_global_pos)

    def calc_height(self) -> float:
        """
        compute character's height by AABB bounding box
        """
        aabb: np.ndarray = self.body_info.get_aabb()
        self.character.height = aabb[3] - aabb[2]
        return self.character.height

    def set_has_root(self):
        self.joint_info.root_idx = len(self.joint_info)
        self.joint_info.has_root = True

    def add_root_joint(self):
        # load Root Joint as Ball Joint
        joint = ode.BallJoint(self.world)
        # assume that body 0 is the root body..
        root_body = self.body_info.root_body
        joint.setAnchor(root_body.PositionNumpy)
        joint.attach(root_body, ode.environment)
        joint.name = "RootJoint"
        self.set_has_root()
        self.joint_info.joints.append(joint)

    @staticmethod
    def compute_geom_mass_attr(
        body: ode.Body,
        create_geom: List[ode.GeomObject],
        gmasses: List[ode.Mass],
        gcenters: List[np.ndarray],
        grots: List[Rotation],
        update_body_pos_by_com: bool = False
    ):
        # Body's Position is com
        # Calc COM and set body's position
        ms: np.ndarray = np.array([i.mass for i in gmasses])
        com: np.ndarray = (ms[np.newaxis, :] @ np.asarray(gcenters) / np.sum(ms)).flatten()
        if update_body_pos_by_com:
            body.PositionNumpy = com
        mass_total = ode.Mass()
        for g_idx, geom in enumerate(create_geom):
            geom.body = body
            geom.setOffsetWorldPositionNumpy(gcenters[g_idx])
            geom.setOffsetWorldRotationNumpy(grots[g_idx].as_matrix().flatten())
            # Rotation of body is 0, so setOffsetWorldRotation and setOffsetRotation is same.

            geom_inertia = ode.Inertia()
            geom_inertia.setFromMassClass(gmasses[g_idx])
            geom_inertia.RotInertia(grots[g_idx].as_matrix().flatten())
            geom_inertia.TransInertiaNumpy(-gcenters[g_idx] + com)
            mass_total.add(geom_inertia.toMass())
        return mass_total

    def append_body(self, body: ode.Body, mass_total: ode.Mass, name: str, parent: Optional[int]):
        """
        param:
        body: ode.Body,
        mass_total: total mass of body
        name: body's name
        idx: body's index
        parent: body's parent's index
        """
        body.setMass(mass_total)
        body.name = name
        self.body_info.parent.append(parent if parent is not None else -1)
        self.body_info.bodies.append(body)

    def set_geom_character_id(self, character_id: int = 0):
        """
        set character_id of each ode GeomObject.
        used in collision detection: To judge whether one character is collided with other character..
        """
        for body in self.bodies:
            for geom in body.geom_iter():
                geom.character_id = character_id
                geom.character = self.character  # character attr in geom is weakref

    def set_geom_max_friction(self, coef: float = 3.0):
        self.character.set_geom_max_friction(coef)

    def set_geom_index(self):
        cnt = 0
        for body in self.bodies:
            for g in body.geom_iter():
                geom: ode.GeomObject = g
                geom.geom_index = cnt
                cnt += 1

    def calc_joint_parent_body_c_id(self):
        """

        """
        self.joint_info.parent_body_index = np.zeros(len(self.joints), dtype=np.uint64)
        self.joint_info.parent_body_c_id = np.zeros(len(self.joints), dtype=np.uint64)
        for idx, joint in enumerate(self.joints):
            pa_body_index = self.character.joint_to_parent_body[idx]
            if pa_body_index != -1:
                pa_body = self.bodies[pa_body_index]
            else:
                pa_body = None

            assert joint.body2 == pa_body, joint.name
            self.joint_info.parent_body_index[idx] = pa_body_index
            self.joint_info.parent_body_c_id[idx] = pa_body.get_bid() if pa_body is not None else 0

        # print(self.joint_info.parent_body_c_id)
        # set root joint's parent body c id to NULL
        if self.joint_info.has_root:
            self.joint_info.parent_body_c_id[self.joint_info.root_idx] = 0

    def calc_joint_child_body_c_id(self):
        """

        """
        self.joint_info.child_body_index = np.zeros(len(self.joints), dtype=np.uint64)
        self.joint_info.child_body_c_id = np.zeros(len(self.joints), dtype=np.uint64)
        # joint should always have child body.
        for j_idx, b_idx in enumerate(self.joint_to_child_body):
            joint = self.joint_info.joints[j_idx]
            child_body = self.bodies[b_idx]
            assert child_body == joint.body1
            self.joint_info.child_body_index[j_idx] = b_idx
            self.joint_info.child_body_c_id[j_idx] = child_body.get_bid()

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
from typing import List, Optional, Tuple, Iterable
import numpy as np
from ..Common.Helper import Helper
from ..Common.MathHelper import MathHelper
from .BodyInfoState import BodyInfoState
from MotionUtils import facing_decompose_rotation_single_fast, quat_apply_single_fast, quat_inv_single_fast


def my_concatenate(tup: Iterable[np.ndarray], axis=0):
    a, b = tup
    if np.size(b) == 0 or b is None:
        return a
    if np.size(a) == 0 or a is None:
        return b
    return np.concatenate([a,b], axis=axis)


class BodyInfo:

    __slots__ = ("world", "space", "bodies", "body_c_id", "parent", "children", "mass_val", "sum_mass", "root_body_id",
                 "initial_inertia", "visualize_color", "root_body")

    def __init__(self, world: ode.World, space: ode.SpaceBase):
        self.world: ode.World = world  # world character belongs to
        self.space: ode.SpaceBase = space  # used for collision detection

        self.bodies: List[ode.Body] = []  # load from file
        self.body_c_id: Optional[np.ndarray] = None  # pointer for bodies. dtype == np.uint64. calc in init_after_load()

        # self.parent_joint_index: Optional[np.ndarray] = None
        self.parent: List[int] = []  # parent body's index. load from file in ODECharacterInit.py
        self.children: List[List[int]] = []  # children body's index.

        self.mass_val: Optional[np.ndarray] = None  # calc in init_after_load()
        self.sum_mass: float = 0.0  # calc in init_after_load()
        self.initial_inertia: Optional[np.ndarray] = None  # calc in init_after_load()

        # The default index of root body is 0...
        self.root_body_id: int = 0

        self.visualize_color: Optional[List] = None
        self.root_body = None

    def get_subset(self, remain_body_index: List[int] = None):
        result = BodyInfo(self.world, self.space)
        result.bodies = [self.bodies[index] for index in remain_body_index]
        if self.body_c_id is not None:
            result.body_c_id = np.ascontiguousarray(self.body_c_id[remain_body_index])
        # actually, we should not compute com by subset..
        result.root_body_id = self.root_body_id
        result.visualize_color = self.visualize_color
        return result

    def copy_body_info(self, space: Optional[ode.SpaceBase] = None):
        # TODO
        result = BodyInfo(self.world, space)
        for body_idx, body in enumerate(self.bodies):
            pass
        result.parent = copy.deepcopy(self.parent)
        result.mass_val = copy.deepcopy(self.mass_val)
        result.sum_mass = self.sum_mass
        result.root_body_id = self.root_body_id
        return result

    def __add__(self, other):
        b_len = len(self.bodies)
        self.bodies += other.bodies
        self.body_c_id = my_concatenate([self.body_c_id, other.body_c_id])

        tmp = other.parent.copy()
        tmp = [ p_id + b_len if p_id >=0 else p_id for p_id in tmp]
        self.parent += tmp

        self.mass_val = my_concatenate([self.mass_val, other.mass_val])
        self.sum_mass += other.sum_mass

        for body in other.bodies:
            x = body.offset_instance_id

            if body.instance_id == body.offset_instance_id:
                body.offset = b_len
        return self

    @property
    def body0(self) -> Optional[ode.Body]:
        """
        Get the 0-th body of the character
        """
        return self.bodies[0] if len(self.bodies) > 0 else None

    @property
    def body1(self) -> Optional[ode.Body]:
        """
        Get the 1-th body of the character
        """
        return self.bodies[1] if len(self.bodies) > 1 else None

    @property
    def _root_body(self) -> ode.Body:
        return self.bodies[self.root_body_id]

    def get_name_list(self) -> List[str]:
        """
        get names for all the bodies
        """
        return [body.name for body in self.bodies]

    def calc_body_c_id(self) -> np.ndarray:
        """
        get pointer for all of bodies. shape == (num_body,).  dtype == np.uint64
        """
        self.body_c_id = self.world.bodyListToNumpy(self.bodies)
        return self.body_c_id

    def init_after_load(self, ignore_parent_collision: bool = True,
                        ignore_grandpa_collision: bool = True):
        self.calc_geom_ignore_id(ignore_parent_collision, ignore_grandpa_collision)
        self.calc_body_c_id()
        self.mass_val: Optional[np.ndarray] = np.array([i.mass_val for i in self.bodies])
        self.sum_mass: Optional[np.ndarray] = np.sum(self.mass_val).item()
        self.initial_inertia: Optional[np.ndarray] = self.calc_body_init_inertia()
        self.visualize_color: Optional[List] = [None] * len(self.bodies)

        # here we can compute childrens..
        self.children = [[] for _ in range(len(self.bodies))]
        for i, p in enumerate(self.parent):
            if p == -1:
                continue
            self.children[p].append(i)

    def calc_body_init_inertia(self) -> np.ndarray:
        """
        Compute the initial inertia for all of bodies
        """
        inertia: np.ndarray = np.zeros((len(self), 3, 3), dtype=np.float64)
        for idx, body in enumerate(self.bodies):
            inertia[idx, :, :] = body.init_inertia.reshape((3, 3))
        return np.ascontiguousarray(inertia)

    def calc_body_init_inertia_inv(self) -> np.ndarray:
        """
        Compute the inverse of initial inertia for all of bodies
        """
        inertia_inv: np.ndarray = np.zeros((len(self), 3, 3))
        for idx, body in enumerate(self.bodies):
            inertia_inv[idx, :, :] = body.init_inertia_inv.reshape((3, 3))
        return np.ascontiguousarray(inertia_inv)

    def __len__(self) -> int:
        """
        length of self.bodies
        """
        return len(self.bodies)

    def get_body_contact_mu(self) -> np.ndarray:
        res: np.ndarray = np.zeros(len(self), dtype=np.float64)
        for idx, body in enumerate(self.bodies):
            res[idx] = list(body.geom_iter())[0].friction
        return res

    # Get Relative Position of parent body in global coordinate
    def get_relative_global_pos(self) -> np.ndarray:
        assert self.bodies

        pos_res = np.zeros((len(self), 3))
        pos_res[0, :] = self.bodies[0].PositionNumpy
        # parent of root body is -1

        for idx in range(1, len(self)):
            body = self.bodies[idx]
            pa_body = self.bodies[self.parent[idx]]
            pos_res[idx, :] = body.PositionNumpy - pa_body.PositionNumpy

        return pos_res

    def get_body_pos_at(self, index: int) -> np.ndarray:
        """
        Get position of index-th body. shape == (3,).  dtype == np.float64
        """
        return self.bodies[index].PositionNumpy

    def get_body_velo_at(self, index: int) -> np.ndarray:
        """
        get linear velocity of index-th body. shape == (3,).  dtype == np.float64
        """
        return self.bodies[index].LinearVelNumpy

    def get_body_quat_at(self, index: int) -> np.ndarray:
        """
        get quaternion of index-th body. shape == (3,).  dtype == np.float64
        """
        return self.bodies[index].getQuaternionScipy()

    def get_body_rot_mat_at(self, index: int) -> np.ndarray:
        """
        get rotation matrix of index-th body. shape == (9,).  dtype == np.float64
        """
        return self.bodies[index].getRotationNumpy()

    def get_body_angvel_at(self, index: int) -> np.ndarray:
        """
        get angular velocity of index-th body. shape == (3,).  dtype == np.float64
        """
        return self.bodies[index].getAngularVelNumpy()

    def get_body_pos(self) -> np.ndarray:
        """
        Get all body's position
        return np.ndarray in shape (num body, 3)
        """
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyPos(self.body_c_id).reshape((-1, 3))

    def get_body_velo(self) -> np.ndarray:
        """
        Get all body's linear velocity
        return np.ndarray in shape (num body, 3)
        """
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyLinVel(self.body_c_id).reshape((-1, 3))

    def get_body_ang_velo(self) -> np.ndarray:
        """
        get all body's angular velocity
        in shape (num body, 3)
        """
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyAngVel(self.body_c_id).reshape((-1, 3))

    # get all body's quaternion
    def get_body_quat(self) -> np.ndarray:
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyQuatScipy(self.body_c_id).reshape((-1, 4))

    # get all body's rotation
    def get_body_rot(self) -> np.ndarray:
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyRot(self.body_c_id).reshape((-1, 3, 3))

    # get all body's force
    def get_body_force(self) -> np.ndarray:
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyForce(self.body_c_id).reshape((-1, 3))

    # get all body's torque
    def get_body_torque(self) -> np.ndarray:
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyTorque(self.body_c_id).reshape((-1, 3))

    def get_body_inertia(self) -> np.ndarray:
        """
        shape == (num_body, 3, 3)
        """
        assert self.body_c_id.dtype == np.uint64
        return self.world.getBodyInertia(self.body_c_id).reshape((-1, 3, 3))

    def set_body_pos(self, pos: np.ndarray):
        assert pos.size == self.body_c_id.size * 3
        self.world.loadBodyPos(self.body_c_id, pos.flatten().astype(np.float64))

    def set_body_velo(self, velo: np.ndarray):
        assert velo.size == self.body_c_id.size * 3
        self.world.loadBodyLinVel(self.body_c_id, velo.flatten().astype(np.float64))

    def set_body_quat(self, quat: np.ndarray):
        self.world.loadBodyQuat(self.body_c_id, quat.flatten())

    def set_body_quat_rot(self, quat: np.ndarray, rot: np.ndarray):
        assert quat.size == self.body_c_id.size * 4
        assert rot.size == self.body_c_id.size * 9
        q = quat.flatten().astype(np.float64)
        r = rot.flatten().astype(np.float64)
        self.world.loadBodyQuatAndRotNoNorm(self.body_c_id, q, r)

    def set_body_ang_velo(self, omega: np.ndarray):
        """
        set body angular velocity
        """
        assert omega.size == 3 * self.body_c_id.size
        self.world.loadBodyAngVel(self.body_c_id, omega.flatten().astype(np.float64))

    def add_body_force(self, force: np.ndarray):
        assert force.size == self.body_c_id.size * 3
        self.world.addBodyForce(self.body_c_id, force)

    def add_body_torque(self, torque: np.ndarray):
        assert torque.size == self.body_c_id.size * 3
        self.world.addBodyTorque(self.body_c_id, torque)

    # calc center of mass in world coordinate
    def calc_center_of_mass(self) -> np.ndarray:  # TODO: test this function
        return self.world.compute_body_com(self.body_c_id)

    # calc CoM by BodyInfoState
    def calc_com_by_body_state(self, state: BodyInfoState) -> np.ndarray:
        pos: np.ndarray = state.pos.reshape((-1, 3))
        return np.matmul(self.mass_val[np.newaxis, :], pos).reshape(3) / self.sum_mass

    def calc_com_and_facing_com_by_body_state(self, state: BodyInfoState) -> Tuple[np.ndarray, np.ndarray]:
        """
        return: com, facing com
        """
        com: np.ndarray = self.calc_com_by_body_state(state)
        qy: np.ndarray = facing_decompose_rotation_single_fast(state.quat.reshape((-1, 4))[self.root_body_id])
        root_pos: np.ndarray = MathHelper.vec_axis_to_zero(state.pos.reshape((-1, 3))[self.root_body_id], 1)
        return com, quat_apply_single_fast(quat_inv_single_fast(qy), com - root_pos)

    def calc_facing_com_by_body_state(self, state: BodyInfoState) -> np.ndarray:
        """
        return: np.ndarray in shape (3,)
        TODO: check with ODE Character
        """
        _, facing_com = self.calc_com_and_facing_com_by_body_state(state)
        return facing_com

    def calc_velo_com(self) -> np.ndarray:
        """
        Calc Velocity of Center of Mass in World Coordinate
        """
        return np.matmul(self.mass_val[np.newaxis, :], self.get_body_velo()).reshape(-1) / self.sum_mass

    # Divide Rotation of Root into Ry * Rxz. Return Ry.
    def calc_facing_quat(self) -> np.ndarray:
        """
        return: in shape (4,)
        """
        # replace with fast implementation..
        return facing_decompose_rotation_single_fast(self.root_body.getQuaternionScipy())

    def calc_facing_quat_inv(self) -> np.ndarray:
        return quat_inv_single_fast(
            facing_decompose_rotation_single_fast(self.root_body.getQuaternionScipy())
        )

    # Calc CoM Momentum
    def calc_sum_mass_pos(self) -> np.ndarray:
        return np.matmul(self.mass_val[np.newaxis, :], self.get_body_pos()).reshape(-1)

    # Calc momentum velocity in world coordinate
    def calc_momentum(self) -> np.ndarray:
        return np.matmul(self.mass_val[np.newaxis, :], self.get_body_velo()).reshape(-1)

    def calc_angular_momentum(self, com: Optional[np.ndarray] = None) -> np.ndarray:
        inertia: np.ndarray = self.initial_inertia
        dcm: np.ndarray = self.get_body_rot()
        inertia: np.ndarray = dcm @ inertia @ dcm.transpose((0, 2, 1))
        ang_vel: np.ndarray = self.get_body_ang_velo()
        angular_momentum: np.ndarray = (inertia @ (ang_vel[..., None])).reshape(ang_vel.shape)

        if com is None:
            com: np.ndarray = self.calc_center_of_mass()
        com_to_body: np.ndarray = self.get_body_pos() - com[None, :]

        linear_momentum: np.ndarray = self.mass_val[:, None] * self.get_body_velo()
        try:
            angular_momentum: np.ndarray = angular_momentum + np.cross(com_to_body, linear_momentum, axis=-1)
        except:
            pass
        angular_momentum: np.ndarray = np.sum(angular_momentum, axis=0)  # (frame, 3)

        return angular_momentum

    def get_geom_rot(self) -> List[List[np.ndarray]]:
        return [[geom.QuaternionScipy for geom in body.geom_iter()] for body in self.bodies]

    def get_geom_pos(self):
        return [[geom.PositionNumpy for geom in body.geom_iter()] for body in self.bodies]

    def calc_geom_ignore_id(self, ignore_parent_collision: bool = True,
                            ignore_grandpa_collision: bool = True):
        """
        Calc ignore id of each geoms in character. ignore collision detection between body and its parent & grandparent
        :return:
        """
        if not ignore_parent_collision and not ignore_grandpa_collision:
            return

        for idx, body in enumerate(self.bodies):
            ignore_geom_id = []
            if self.parent[idx] != -1:
                pa_body_idx = self.parent[idx]
                for pa_geom in self.bodies[pa_body_idx].geom_iter():
                    if ignore_parent_collision:
                        ignore_geom_id.append(pa_geom.get_gid())

                    if ignore_grandpa_collision:
                        grandpa_idx = self.parent[pa_body_idx]
                        if grandpa_idx != -1:
                            for grandpa_geom in self.bodies[grandpa_idx].geom_iter():
                                ignore_geom_id.append(grandpa_geom.get_gid())

            if len(ignore_geom_id) > 0:
                for geom in body.geom_iter():
                    geom.extend_ignore_geom_id(ignore_geom_id)

    # Get AABB bounding box of bodies and geoms.
    def get_aabb(self) -> np.ndarray:
        """
        Get AABB bounding box of bodies and geoms.
        """
        return ode.SpaceBase.get_bodies_aabb(self.body_c_id)

    def get_aabb_batch(self):
        return ode.SpaceBase.get_batch_aabb

    def clear(self):
        for body in self.bodies:
            for geom in body.geom_iter():
                geom.destroy_immediate()  # destroy all of geometries
            body.destroy_immediate()

        self.bodies.clear()
        self.body_c_id: Optional[np.ndarray] = None

        self.parent.clear()

        self.mass_val: Optional[np.ndarray] = None
        self.sum_mass = 0.0

        self.root_body_id: int = 0
        self.visualize_color: Optional[List] = None
        return self

    def get_mirror_index(self) -> List[int]:
        body_names = self.get_name_list()
        return Helper.mirror_name_list(body_names)

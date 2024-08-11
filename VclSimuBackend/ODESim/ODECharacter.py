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
import ModifyODE as ode
from mpi4py import MPI
import platform
from typing import Set, List, Optional, Tuple, Iterable, Union, Dict, Any

from ..Common.MathHelper import MathHelper
from .JointInfo import JointInfos
from .BodyInfo import BodyInfo
from .BodyInfoState import BodyInfoState
from .EndJointInfo import EndJointInfo

from MotionUtils import (
    quat_to_matrix_fast,
    quat_apply_forward_fast,
    quat_apply_single_fast,
    quat_inv_single_fast,
    quat_apply_forward_one2many_fast,
    quat_multiply_forward_fast,
    decompose_rotation_single_fast
)

# from MotionUtils import RefStateCompute

is_windows = "Windows" in platform.platform()
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

class DRootInitInfo:
    def __init__(self) -> None:
        self.pos: Optional[np.ndarray] = None
        self.quat: Optional[np.ndarray] = None

    def clear(self):
        self.pos: Optional[np.ndarray] = None
        self.quat: Optional[np.ndarray] = None


class ODECharacter:
    stdhuman_body_name: List[str] = [
        'pelvis',
        'lowerBack',
        'torso',
        'rUpperLeg',
        'lUpperLeg',
        'rLowerLeg',
        'lLowerLeg',
        'rFoot',
        'lFoot',
        'rToes',
        'lToes',
        'head',
        'rClavicle',
        'lClavicle',
        'rUpperArm',
        'lUpperArm',
        'rLowerArm',
        'lLowerArm',
        'rHand',
        'lHand'
    ]

    stdhuman_body_name_set = set(stdhuman_body_name)

    def __init__(self, world: ode.World, space: ode.SpaceBase):
        self.name: str = "character"  # character name
        self.label: str = ""  # character label
        self.world: ode.World = world  # world in open dynamics engine
        self.space: ode.SpaceBase = space  # for collision detection
        self.character_id: int = 0  # The id of character
        self.root_init_info: Optional[DRootInitInfo] = None
        self.joint_info = JointInfos(self.world)  # joint information
        self.body_info = BodyInfo(self.world, self.space)  # rigid body information
        self.end_joint = EndJointInfo(self.world)  # End Joint in xml Human file. Just like end site of bvh file.

        self.joint_to_child_body: List[int] = []  # idx is joint id, joint_to_child_body[idx] is body id
        self.child_body_to_joint: List[int] = []  # idx is body id, child_body_to_joint[idx] is joint id

        self.joint_to_parent_body: List[int] = [] # idx is joint id, joint_to_parent_body[idx] is body id

        self.init_body_state: Optional[BodyInfoState] = None  # initial body state

        self.height: float = 0.0  # height of character

        self.simulation_failed: bool = False
        self.fall_down: bool = False  # Fall Down Flag. Will be set in collision callback
        self.falldown_ratio = 0.0  # if com <= falldown_ratio * initial_com, we can say the character fall down
        self._self_collision: bool = True  # self collision detaction

        self._is_enable: bool = True
        self._is_kinematic: bool = False

        self.curr_frame_index = 0

        self.config_dict: Optional[Dict[str, Any]] = None
        self.scene = None
        self.root_body = None
        self.is_stdhuman_ = None  # judge is std human

        # smpl parameter of character.
        self.smpl_shape: Optional[np.ndarray] = None

        self.com_geom: Optional[ode.GeomBox] = None
        if is_windows:
            self.com_geom: ode.GeomBox = ode.GeomBox(None, (0.08, 1.0, 0.08))
            self.com_geom.render_user_color = np.array([1.0, 0.0, 0.0])
            self.com_geom.disable()

    def __del__(self):
        print(f"deconstruct env at {mpi_rank}", flush=True)

    @property
    def bodies(self) -> List[ode.Body]:
        return self.body_info.bodies

    # get list of joints of rigid body
    @property
    def joints(self) -> List[ode.Joint]:
        return self.joint_info.joints

    # get the root rigid body
    @property
    def _root_body(self) -> ode.Body:
        return self.body_info.root_body

    @property
    def root_body_pos(self) -> np.ndarray:
        return self.root_body.PositionNumpy

    @property
    def root_body_quat(self) -> np.ndarray:
        return self.root_body.getQuaternionScipy()

    @property
    def root_joint(self) -> Optional[ode.Joint]:
        return self.joint_info.root_joint

    def set_character_id(self, new_id: int):
        self.character_id = new_id
        return self

    # get global position of index-th rigid body
    def get_body_pos_at(self, index: int) -> np.ndarray:
        return self.body_info.get_body_pos_at(index)

    # get global velocity of index-th rigid body
    def get_body_velo_at(self, index: int) -> np.ndarray:
        return self.body_info.get_body_velo_at(index)

    # 
    def get_body_quat_at(self, index: int) -> np.ndarray:
        return self.body_info.get_body_quat_at(index)

    def get_body_rot_mat_at(self, index: int) -> np.ndarray:
        return self.body_info.get_body_rot_mat_at(index)

    def get_body_angvel_at(self, index: int) -> np.ndarray:
        return self.body_info.get_body_angvel_at(index)

    def get_body_name_list(self) -> List[str]:
        return self.body_info.get_name_list()

    def get_body_pos(self) -> np.ndarray:
        return self.body_info.get_body_pos()

    def get_body_velo(self) -> np.ndarray:
        return self.body_info.get_body_velo()

    def get_body_mat(self) -> np.ndarray:
        raise NotImplementedError

    # get quaternion of all of bodies
    def get_body_quat(self) -> np.ndarray:
        return self.body_info.get_body_quat()

    # get angular velocity of all of bodies
    def get_body_ang_velo(self) -> np.ndarray:
        return self.body_info.get_body_ang_velo()

    # get position of all of bodies
    def set_body_pos(self, pos: np.ndarray):
        self.body_info.set_body_pos(pos)

    def set_body_velo(self, velo: np.ndarray):
        self.body_info.set_body_velo(velo)

    # def set_body_quat(self, quat: np.ndarray):
    #    self.body_info.set_body_quat_rot

    def set_body_ang_velo(self, ang_velo: np.ndarray):
        self.body_info.set_body_ang_velo(ang_velo)

    def get_aabb(self) -> np.ndarray:
        """
        get character aabb
        """
        return self.body_info.get_aabb()

    @property
    def has_end_joint(self) -> bool:
        return self.end_joint is not None and len(self.end_joint) > 0

    @property
    def joint_weights(self) -> Optional[np.ndarray]:
        return self.joint_info.weights

    @property
    def end_joint_weights(self) -> Optional[np.ndarray]:
        return self.end_joint.weights if self.end_joint is not None else None

    @property
    def is_kinematic(self) -> bool:
        return self._is_kinematic

    @is_kinematic.setter
    def is_kinematic(self, value: bool):
        if self._is_kinematic == value:
            return
        if value:
            for body in self.bodies:
                body.setKinematic()
        else:
            for body in self.bodies:
                body.setDynamic()
        self._is_kinematic = value

    # self collision of this character
    @property
    def self_collision(self) -> bool:
        return self._self_collision

    @self_collision.setter
    def self_collision(self, value: bool):
        if self._self_collision == value:
            return
        for body in self.bodies:
            for _geom in body.geom_iter():
                geom: ode.GeomObject = _geom
                geom.character_self_collide = int(value)
        self._self_collision = value

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    @is_enable.setter
    def is_enable(self, value: bool):  # TODO: Test. How to handle ext joints?
        if self._is_enable == value:
            return
        
        self._is_enable = value
        if value:
            for body in self.bodies:
                body.enable()
                for geom in body.geom_iter():
                    geom.enable()
            for joint in self.joints:
                joint.enable()
        else:
            for body in self.bodies:
                body.disable()
                for geom in body.geom_iter():
                    geom.disable()
            for joint in self.joints:
                joint.disable()

    def set_ode_space(self, space: Optional[ode.SpaceBase]):
        """
        set space of each geometry in character.
        """
        for body in self.bodies:
            for g_iter in body.geom_iter():
                geom: ode.GeomObject = g_iter
                geom.space = space
        self.space = space

    def set_root_pos(self, pos: np.ndarray):
        # maybe can write in cython..
        raise NotImplementedError

    def save_init_state(self) -> BodyInfoState:
        """
        Save init state
        :return: initial state
        """
        if self.init_body_state is None:
            self.init_body_state = self.save()
        return self.init_body_state

    def load_init_state(self) -> None:
        """
        load initial state
        """
        if self.init_body_state is not None:
            self.load(self.init_body_state)

    def save(self) -> BodyInfoState:
        """
        Save to BodyInfoState
        """
        if self.body_info.body_c_id is None:
            self.body_info.calc_body_c_id()
            # body id is created from body list.
            # so, if the order of body list is fixed, the order of body id is fixed.
        body_state = BodyInfoState()
        body_state.save(self.world, self.body_info.body_c_id)
        return body_state

    def load(self, body_state: BodyInfoState):
        """
        Load BodyInfoState
        """
        if self.body_info.body_c_id is None:
            self.body_info.calc_body_c_id()
        body_state.load(self.world, self.body_info.body_c_id)

    # get joint name list.
    def get_joint_names(self, with_root: bool = False) -> List[str]:
        result = self.joint_info.joint_names()
        if with_root:
            result = ["RootJoint"] + result
        return result

    def get_raw_anchor(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        joint's body1 raw anchor, joint's body2 raw anchor
        """
        return self.world.getBallAndHingeRawAnchor(self.joint_info.joint_c_id)

    # get position of end effectors in facing coordinate
    def character_local_end_pos(self, root_inv: Optional[np.ndarray] = None) -> np.ndarray:
        end_a = self.end_joint.get_global_pos()
        if root_inv is None:
            root_inv = quat_inv_single_fast(self.body_info.root_body.getQuaternionScipy())
        root_pos = self.bodies[0].PositionNumpy
        root_pos[1] = 0
        return quat_apply_forward_one2many_fast(root_inv[None, :], end_a - root_pos)

    # get position of com
    def character_local_com(self, root_inv: Optional[np.ndarray] = None, com=None) -> np.ndarray:
        if root_inv is None:
            root_inv = quat_inv_single_fast(self.body_info.root_body.getQuaternionScipy())
        if com is None:
            com = self.body_info.calc_center_of_mass()
        root_pos = self.bodies[0].PositionNumpy
        root_pos[1] = 0
        return quat_apply_single_fast(root_inv, com - root_pos)

    def character_local_balance(self, root_inv: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if root_inv is None:
            root_inv = quat_inv_single_fast(self.body_info.root_body.getQuaternionScipy())
        local_end: np.ndarray = self.character_local_end_pos(root_inv)
        local_com: np.ndarray = self.character_local_com(root_inv)
        local_balance: np.ndarray = local_end - local_com[None, :]
        return local_com, local_end, local_balance

    def character_facing_coor_end_pos(self, facing_rot_inv: Optional[np.ndarray] = None) -> np.ndarray:
        end_a = self.end_joint.get_global_pos()
        if facing_rot_inv is None:
            facing_rot_inv = quat_inv_single_fast(self.body_info.calc_facing_quat())
        root_pos = self.bodies[0].PositionNumpy
        root_pos[1] = 0
        return quat_apply_forward_one2many_fast(facing_rot_inv[None, :], end_a - root_pos)

    def character_facing_coor_com(self, facing_rot_inv: Optional[np.ndarray] = None, com: Optional[np.ndarray] = None) -> np.ndarray:
        """
        character's CoM in facing coordinate
        """
        if facing_rot_inv is None:
            facing_rot_inv = quat_inv_single_fast(self.body_info.calc_facing_quat())
        if com is None:
            com: np.ndarray = self.body_info.calc_center_of_mass()
        root_pos: np.ndarray = self.bodies[0].PositionNumpy
        root_pos[1] = 0
        return quat_apply_single_fast(facing_rot_inv, com - root_pos)

    def character_facing_coor_com_velo(self) -> np.ndarray:
        """
        character's CoM's velocity in facing coordinate
        """
        ry_rot_inv: np.ndarray = quat_inv_single_fast(self.body_info.calc_facing_quat())
        com_velo: np.ndarray = self.body_info.calc_momentum() / self.body_info.sum_mass
        root_velo: np.ndarray = MathHelper.vec_axis_to_zero(self.root_body.LinearVelNumpy, 1)
        return quat_apply_single_fast(ry_rot_inv, com_velo - root_velo)

    def character_facing_coord_angular_momentum(self) -> np.ndarray:
        """
        character's angular momentum in facing coordinate
        """
        ry_rot_inv: np.ndarray = quat_inv_single_fast(self.body_info.calc_facing_quat())
        angular_momentum = self.body_info.calc_angular_momentum()
        return quat_apply_single_fast(ry_rot_inv, angular_momentum)

    def calc_kinetic_energy(self) -> np.ndarray:
        """
        1/2*m*v^2 + 1/2*w^T*I*w
        """
        velo: np.ndarray = self.body_info.get_body_velo()
        omega: np.ndarray = self.body_info.get_body_ang_velo()
        mass: np.ndarray = self.body_info.mass_val
        inertia: np.ndarray = self.body_info.get_body_inertia()
        v2 = np.sum(velo ** 2, axis=-1)  # in shape (num body,)
        eng1 = np.sum(mass * v2)
        eng2 = omega.reshape((-1, 1, 3)) @ inertia @ omega.reshape((-1, 3, 1))
        eng2 = np.sum(eng2)
        res: np.ndarray = 0.5 * (eng1 + eng2)
        return res

    def cat_root_child_body_value(self, root_value: np.ndarray, child_body_value: np.ndarray, dtype=np.float64):
        """
        cat value for root body and child body
        root_value.shape == (batch size, num value)
        child_body.shape == (batch size, num body - 1, num value)
        """
        assert not self.joint_info.has_root
        assert root_value.ndim == 2 and child_body_value.ndim == 3 and child_body_value.shape[1] == len(self.bodies) - 1
        assert root_value.shape[0] == child_body_value.shape[0] and root_value.shape[-1] == child_body_value.shape[-1]
        res: np.ndarray = np.zeros((root_value.shape[0], len(self.bodies), root_value.shape[-1]), dtype=dtype)
        res[:, self.body_info.root_body_id, :] = root_value
        res[:, self.joint_to_child_body, :] = child_body_value
        return np.ascontiguousarray(res)

    def init_root_body_pos(self) -> np.ndarray:
        """
        initial root position
        """
        return self.init_body_state.pos.reshape((-1, 3))[self.body_info.root_body_id].copy()

    def init_root_quat(self) -> np.ndarray:
        """
        initial root quaternion
        """
        return self.init_body_state.quat.reshape((-1, 4))[self.body_info.root_body_id].copy()

    # set the render color in draw stuff
    def set_render_color(self, color: np.ndarray):
        for body in self.bodies:
            for geom_ in body.geom_iter():
                geom: ode.GeomObject = geom_
                geom.render_by_default_color = 0
                geom.render_user_color = color

    def enable_all_clung_env(self) -> None:
        for body in self.bodies:
            for geom in body.geom_iter():
                geom.clung_env = True

    def disable_all_clung_env(self) -> None:
        for body in self.bodies:
            for geom in body.geom_iter():
                geom.clung_env = False

    def set_clung_env(self, body_names: Iterable[str], value: bool = True):
        names: Set[str] = set(body_names)
        for body in self.bodies:
            if body.name in names:
                for geom in body.geom_iter():
                    geom.clung_env = value

    # set the max friction for each geometry when contact mode is max force
    def set_geom_max_friction(self, coef: float = 3.0):
        value: float = coef * self.body_info.sum_mass * np.linalg.norm(self.world.getGravityNumpy())
        for body in self.bodies:
            for geom in body.geom_iter():
                geom.max_friction = value
        return value

    def set_geom_mu_param(self, mu: float = 0.8):
        for body in self.bodies:
            for geom in body.geom_iter():
                geom.friction = mu

    # clear the character 
    def clear(self):
        self.joint_info.clear()
        self.body_info.clear()
        self.end_joint.clear()
        self.joint_to_child_body.clear()
        self.child_body_to_joint.clear()
        self.joint_to_parent_body.clear()
        self.init_body_state: Optional[BodyInfoState] = None
        self.fall_down: bool = False

        return self

    def check_root(self):
        """
        check root joint and root body
        :return:
        """
        if self.bodies and self.root_body is None:
            raise ValueError("There should be a root body")

        if self.root_joint is not None:
            if self.root_joint.getNumBodies() != 1:
                raise ValueError("root joint should only has 1 child")
            if self.root_joint.body1 != self.root_body and self.root_joint.body2 != self.root_body:
                raise ValueError("Root Body should be attached to root joint.")

    # move the character to new position
    # new_pos.shape == (3,)
    def move_character(self, new_pos: np.ndarray) -> None:
        assert new_pos.shape == (3,)
        delta_pos: np.ndarray = (new_pos - self.root_body.PositionNumpy).reshape((1, 3))
        old_pos = self.body_info.get_body_pos()
        new_pos = delta_pos + old_pos
        self.body_info.set_body_pos(new_pos)

    def move_character_by_delta(self, delta_pos: np.ndarray) -> None:
        assert delta_pos.shape ==  (3,)
        old_pos = self.body_info.get_body_pos()
        new_pos = delta_pos + old_pos
        self.body_info.set_body_pos(new_pos)

    def rotate_character(self):
        pass

    def copy_character(self, space: Optional[ode.SpaceBase] = None):
        # TODO
        new_character = ODECharacter(self.world, space)
        return new_character

    @staticmethod
    def rotate_body_info_state_y_axis(state: BodyInfoState, angle: float, use_delta_angle: bool = False) -> BodyInfoState:
        """
        rotate the BodyInfoState by y axis
        return: BodyInfoState

        For position, move to the original position, and rotate, then move back
        For rotation, rotate directly. note that rotation matrix should be recomputed.
        For linear velocity and angular velocity, rotate directly.
        Test:
        After rotate, the simulation result should match
        """
        result: BodyInfoState = state.copy().reshape()
        num_body = result.pos.shape[0]
        delta_pos: np.ndarray = result.pos[None, 0].copy()
        delta_pos[0, 1] = 0
        root_quat = result.quat[0]
        facing_quat = decompose_rotation_single_fast(root_quat, np.array([0.0, 1.0, 0.0]))
        facing_angle = 2 * np.arctan2(facing_quat[1], facing_quat[3])
        if use_delta_angle:
            delta_angle = angle
        else:
            delta_angle = angle - facing_angle
        delta_quat = np.array([0.0, np.sin(0.5 * delta_angle), 0.0, np.cos(0.5 * delta_angle)])
        delta_quat = np.ascontiguousarray(np.tile(delta_quat, (num_body, 1)))
        result.pos = quat_apply_forward_fast(delta_quat, result.pos - delta_pos).reshape(-1)
        result.quat = quat_multiply_forward_fast(delta_quat, result.quat).reshape(-1)
        result.rot = quat_to_matrix_fast(result.quat).reshape(-1)
        result.linear_vel = quat_apply_forward_fast(delta_quat, result.linear_vel).reshape(-1)
        result.angular_vel = quat_apply_forward_fast(delta_quat, result.angular_vel).reshape(-1)
        return result

    def rotate_y_axis(self, angle: float, use_delta_angle: bool = False):
        next_state: BodyInfoState = self.rotate_body_info_state_y_axis(self.save(), angle, use_delta_angle)
        self.load(next_state)
        return next_state

    def is_stdhuman(self) -> bool:
        """
        Hack. judge if character is std-human model.
        """
        return set(self.get_body_name_list()) == self.stdhuman_body_name_set

    def get_tpose_root_h(self):
        return float(self.init_body_state.pos[1])

    def build_ref_state_handle(
        self,
        root_pos: np.ndarray, 
        joint_rot: Optional[np.ndarray],
        joint_orient: Optional[np.ndarray],
        done: np.ndarray,
        original_h: float,
        dt: float
    ):
        # make sure the input joint sequence matches the joint in this character.
        assert root_pos.dtype == np.float64, "dtype of root_pos should be np.float64"
        # assert joint_rot.dtype == np.float64, "dtype of joint_rot should be np.float64"
        assert done.dtype == np.int32, "dtype of done should be np.int32"
        
        self.load_init_state()
        target_h: float = float(self.init_body_state.pos[1])
        # here we should make sure these joints are aligned with child body.
        child_body_index: np.ndarray = np.concatenate(
            [np.array([0]), self.joint_info.child_body_index], dtype=np.int32)
        joint_parent: np.ndarray = np.concatenate(
            [np.array([-1]), 1 + np.array(self.joint_info.pa_joint_id)], dtype=np.int32)

        joint_offset: np.ndarray = np.zeros((joint_parent.shape[0], 3), np.float64)
        joint_pos = np.concatenate([self.init_body_state.pos[None, :3], self.joint_info.get_global_anchor1()], axis=0)
        for joint_idx, parent_idx in enumerate(joint_parent):
            if joint_idx == 0:
                continue
            joint_offset[joint_idx] = joint_pos[joint_idx] - joint_pos[parent_idx]
        body_rel_pos = np.concatenate(
            [np.zeros((1, 3)), self.joint_info.get_child_body_relative_pos()], axis=0)

        result = RefStateCompute(
            root_pos,  # (batch, 3)
            joint_rot,  # (batch, nj, 4)
            joint_orient,
            done.reshape(-1, 1),  # (batch, 1)
            original_h,  # float
            target_h,  # float
            joint_parent,  # (nj,)
            child_body_index,  # (nj,)
            np.ascontiguousarray(joint_offset),  # (nj, 3), float64
            -body_rel_pos,  # (nj, 3), float64
            dt  # float
        )
        return result

    def update_com_drawstuff(self):
        pass

    def compute_root_inertia(self) -> np.ndarray:
        """
        The Inertia around root at T-pose
        return shape: (3, 3)
        """
        self.load_init_state()
        root_pos: np.ndarray = self.root_body.PositionNumpy
        inertia: np.ndarray = np.zeros((9,))
        
        for body_idx, body in enumerate(self.bodies):
            mass: float = body.mass_val
            init_inertia: np.ndarray = body.init_inertia
            if body_idx == 0:
                inertia += init_inertia
            else:
                de: np.ndarray = body.PositionNumpy - root_pos
                trans_i: np.ndarray = ode.translate_inertia(init_inertia, mass, de[0], de[1], de[2])
                inertia += trans_i
        
        return inertia.reshape((3, 3))

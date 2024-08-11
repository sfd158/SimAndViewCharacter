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

import enum
import numpy as np
import ModifyODE as ode  # please run pip install -e . at the folder ModifyODESrc to install ode package
from scipy.spatial.transform import Rotation
from typing import Optional, Union, List, Iterable, Iterator, Tuple, Dict

from .ArrowInfo import ArrowInfoList
from .config import debug_mode
from .Environment import Environment
from .ExtJointList import ExtJointList
from .ODECharacter import ODECharacter

from MotionUtils import quat_apply_single_fast
import MotionUtils
from VclSimuBackend.Common.MathHelper import MathHelper


class SceneContactLocalInfo:
    """

    """
    def __init__(self) -> None:
        self.global_pos: Optional[np.ndarray] = None
        self.local_pos: Optional[np.ndarray] = None
        self.normal: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.body1_cid: Optional[np.ndarray] = None

    def set_value(
            self,
            global_pos: np.ndarray,
            local_pos: np.ndarray,
            normal: np.ndarray,
            depth: np.ndarray,
            body1_cid: np.ndarray,
            to_continuous: bool = True
    ):
        self.global_pos = np.ascontiguousarray(global_pos) if to_continuous else global_pos
        self.local_pos = np.ascontiguousarray(local_pos) if to_continuous else local_pos
        self.normal = np.ascontiguousarray(normal) if to_continuous else normal
        self.depth = np.ascontiguousarray(depth) if to_continuous else normal
        self.body1_cid = np.ascontiguousarray(body1_cid) if to_continuous else normal

    def get_global_pos(self, world: ode.World) -> np.ndarray:
        if self.global_pos is None:
            body1_pos: np.ndarray = world.getBodyPos(self.body1_cid)
            body1_quat: np.ndarray = world.getBodyQuatScipy(self.body1_cid)
            self.global_pos: Optional[np.ndarray] = Rotation(body1_quat).apply(self.local_pos) + body1_pos
            
        return self.global_pos

    def clear(self):
        self.global_pos: Optional[np.ndarray] = None
        self.local_pos: Optional[np.ndarray] = None
        self.normal: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.body1_cid: Optional[np.ndarray] = None

        return self


class SceneContactInfo:
    """
    Contact Info Extractor, for visualize in Unity..
    """
    __slots__ = (
        "pos",
        "force",
        "torque",
        "geom1_name",
        "geom2_name",
        "body1_index",
        "body2_index",
        "contact_label",
        "body_contact_force"
    )

    def __init__(
        self,
        pos: Union[np.ndarray, List, None] = None,  # contact position (in global coordinate)
        force: Union[np.ndarray, List, None] = None,  # contact force (in global coordinate)
        geom1_name: Optional[List[str]] = None,  # name of geometry 1
        geom2_name: Optional[List[str]] = None,  # name of geometry 2
        contact_label: Union[np.ndarray, List[float], None] = None,  #
        body_contact_force: Union[np.ndarray, None] = None # sum of contact force on each body
    ):
        self.pos: Union[np.ndarray, List, None] = pos
        self.force: Union[np.ndarray, List, None] = force
        self.torque: Union[np.ndarray, List, None] = None
        # note: the contact torque is unused, because rendering the contact torque is hard..

        self.geom1_name: Optional[List[str]] = geom1_name
        self.geom2_name: Optional[List[str]] = geom2_name

        self.body1_index: Optional[List[int]] = None
        self.body2_index: Optional[List[int]] = None

        self.contact_label: Union[np.ndarray, List[float], None] = contact_label
        self.body_contact_force: Union[np.ndarray, None] = body_contact_force

    def __len__(self) -> int:
        if self.pos is None:
            return 0
        else:
            return len(self.pos)

    def merge_force_by_body1(self):
        """
        merge the total force by body1 index..
        There is only one character in the scene
        """
        # 1. divide the contact into several groups
        body1_index: np.ndarray = np.asarray(self.body1_index)
        unique_body1: np.ndarray = np.unique(body1_index)
        forces: np.ndarray = np.asarray(self.force)
        if self.torque is not None:
            torques: Optional[np.ndarray] = np.asarray(self.torque)
        else:
            torques: Optional[np.ndarray] = None

        ret_force = self.force.copy()
        for sub_body in unique_body1:
            try:
                sub_contact = np.where(body1_index == sub_body)[0]
                # divide the sub force..
                sub_force = forces[sub_contact]
                sub_force_len: np.ndarray = np.linalg.norm(sub_force, axis=-1)
                sub_force_len = (sub_force_len / np.sum(sub_force_len)).reshape((-1, 1))  # in shape (sum contact, 1)
                sub_force_avg: np.ndarray = np.mean(sub_force, axis=0).reshape((1, 3))  # in shape (1, 3,)
                divide_force: np.ndarray = sub_force_len * sub_force_avg
                ret_force[sub_contact] = divide_force

                # divide the sub torque..
                if torques is not None and False:
                    sub_torque: np.ndarray = torques[sub_contact]
                    sub_torque_len: np.ndarray = np.linalg.norm(sub_torque, axis=-1)
                    sub_torque_len: np.ndarray = (sub_torque_len / np.sum(sub_torque_len)).reshape((-1, 1))
                    sub_torque_avg: np.ndarray = np.mean(sub_torque, axis=0).reshape((1, 3))
                    divide_torque: np.ndarray = sub_torque_len * sub_torque_avg
                    self.torque: np.ndarray = divide_torque
            except Exception as err:
                raise err

        self.force = ret_force
        return self

    def set_value(
        self,
        pos: Optional[np.ndarray],
        force: Optional[np.ndarray],
        geom1_name: Optional[List[str]],
        geom2_name: Optional[List[str]],
        contact_label: Union[np.ndarray, List[float], None] = None,
        body_contact_force: Union[np.ndarray, None] = None
    ):
        self.pos: Optional[np.ndarray] = pos
        self.force: Optional[np.ndarray] = force
        self.geom1_name = geom1_name
        self.geom2_name = geom2_name
        self.contact_label = contact_label
        self.body_contact_force = body_contact_force

    def clear(self):
        self.pos = None
        self.force = None
        if self.geom1_name is not None:
            del self.geom1_name[:]
        if self.geom2_name is not None:
            del self.geom2_name[:]
        self.contact_label = None
        self.body_contact_force = None

    def check_delta(self, other):
        if self.pos is None and other.pos is None:
            return True
        if len(self) != len(other):
            print(f"Contact count not match. self is {len(self)}, other is {len(other)}")
            return False
        res = np.all(self.pos - other.pos == 0) and np.all(self.force - other.force == 0)
        if res is False:
            print(f"self.pos {self.pos}, other.pos {other.pos}")
            print(f"self.force {self.force}, other.force {other.force}")
        return res

    def out_iter(self):
        """
        get 
        """
        if self.contact_label is None:
            self.contact_label = np.ones(len(self))
        if isinstance(self.contact_label, np.ndarray):
            self.contact_label = self.contact_label.tolist()
        try:
            return zip(range(len(self)), self.pos, self.force, self.contact_label) if len(self) > 0 else zip((), (), (), ())
        except Exception as err:
            print(self.pos, self.force, self.contact_label)
            raise err


# TODO: add some other type, such as lemke, dart, and etc.
"""
ODE_LCP: use normal LCP model for contact
MAX_FORCE_ODE_LCP: for contact, f <= F_max (a constant value), rather than f <= u F_n
BALL: use ball joint for contact
"""
class ContactType(enum.IntEnum):
    ODE_LCP = 0
    MAX_FORCE_ODE_LCP = 1
    BALL = 2  # NOTE: Ball Contact doesn't work at all!


"""
DAMPED_STEP / DAMPED_FAST_STEP: use stable PD control & damping in forward simulation (larger simulation step is required)
STEP / FAST_STEP: use PD control in forward simulation (smaller simulation step is required)
"""
class SimulationType(enum.IntEnum):
    DAMPED_STEP = 0
    DAMPED_FAST_STEP = 1
    STEP = 2
    FAST_STEP = 3


class ODEScene:
    default_gravity: List[float] = [0.0, -9.8, 0.0]

    def __init__(
        self,
        render_fps: int = 60,
        sim_fps: int = 120,
        gravity: Union[Iterable, float, None] = None,
        friction: float = 0.8,
        bounce: float = 0.0,
        self_collision: bool = True,
        contact_type: ContactType = ContactType.MAX_FORCE_ODE_LCP,
        contact_count: int = 4,
        extract_contact: bool = True,
        hand_scene: bool = False
    ):
        self.render_dt, self.render_fps = 0.0, 0  # Frame Time in Unity
        self.sim_dt, self.sim_fps = 0.0, 0.0  # Simulate Step in ODE
        self.step_cnt: int = 1
        self.set_render_fps(render_fps)
        self.set_sim_fps(sim_fps)
        self.bounce, self.friction = bounce, friction

        self.world: Optional[ode.World] = None  # use for forward simulation
        self.space: Optional[ode.SpaceBase] = None  # use for collision detection

        self._self_collision: bool = self_collision  # enable self collision for characters
        self._contact_type: ContactType = contact_type  #

        self._use_soft_contact: bool = False
        self._soft_cfm: Optional[float] = None
        self.soft_cfm_tan: Optional[float] = None

        self._soft_erp: Optional[float] = None

        self.contact: ode.JointGroup = ode.JointGroup()  # contact joints used in contact detection
        self.characters: List[ODECharacter] = []  # store all of characters
        self.environment: Environment = Environment(self.world, self.space)
        self.arrow_list = ArrowInfoList()  # for render arrow in Unity

        self.ext_joints: Optional[ExtJointList] = None
        self._contact_count: int = contact_count
        self.contact_info: Optional[SceneContactInfo] = None  # for unity render
        self.contact_local_info: Optional[SceneContactLocalInfo] = None  # for hack in DiffODE

        self.extract_contact: bool = extract_contact

        # in zyl's hand scene mode
        self.hand_scene: bool = hand_scene
        self.r_character_id = None
        self.l_character_id = None
        self.obj_character_id = None

        self.str_info: str = ""  # text information for Unity client
        self.clear(gravity)

        self.simu_func = self.damped_simulate_once
        self.simu_type = SimulationType.DAMPED_STEP

        self.debug_space = ode.Space()
        self.visualize_contact = None
        self.visualize_point: Optional[List[ode.GeomObject]] = None

        if False:
            self.create_contact_visualize()
            self.create_point_visualize()
        # self.character0.com_geom.space = self.debug_space

    def create_point_visualize(self):
        self.visualize_point = []
        for i in range(20):
            vis_point = ode.GeomSphere(self.debug_space, 0.02)
            vis_point.render_by_default_color = 0
            vis_point.render_user_color = np.array([0.0, 1.0, 0.0])
            vis_point.disable()
            self.visualize_point.append(vis_point)
        
    def create_contact_visualize(self):
        """
        Visualize contact in draw stuff.
        """
        self.visualize_contact = []
        for i in range(20):
            vis_box = ode.GeomBox(self.debug_space, [1e-3, 1e-3, 1e-3])
            vis_box.disable()
            self.visualize_contact.append(vis_box)

    def update_contact_visualize(self, ratio=1000.0):
        if self.visualize_contact is None:
            return
        n_contact = len(self.contact_info)
        for index, pos, force, label in self.contact_info.out_iter():
            geom = self.visualize_contact[index]
            len_f = np.sum(force ** 2)
            if len_f == 0:
                geom.setLengths([1e-3, 1e-3, 1e-3])
            else:
                len_f = np.sqrt(len_f)
                geom.setLengths([1e-2, len_f / ratio, 1e-2])
                unit_f = force / len_f
                new_quat = MathHelper.quat_between(np.array([0.0, 1.0, 0.0]), unit_f)
                geom.QuaternionScipy = new_quat
                center_pos = pos + force / ratio / 2
                geom.PositionNumpy = center_pos

        for index in range(n_contact, len(self.visualize_contact)):
            geom = self.visualize_contact[index]
            geom.PositionNumpy = np.zeros(3)
            geom.setLengths([1e-3, 1e-3, 1e-3])

    @property
    def contact_type(self) -> ContactType:
        return self._contact_type

    @contact_type.setter
    def contact_type(self, value):
        self._contact_type = value
        self.world.use_max_force_contact = bool(self._contact_type == ContactType.MAX_FORCE_ODE_LCP)

    @property
    def soft_erp(self):
        """
        erp value for soft contact
        """
        return self._soft_erp

    @soft_erp.setter
    def soft_erp(self, value: float):
        self._soft_erp = value
        self.world.soft_erp = value

    @property
    def soft_cfm(self) -> float:
        """
        cfm value for soft contact
        """
        return self._soft_cfm

    @soft_cfm.setter
    def soft_cfm(self, value: float):
        self._soft_cfm = value
        self.world.soft_cfm = value

    @property
    def use_soft_contact(self):
        return self._use_soft_contact

    @use_soft_contact.setter
    def use_soft_contact(self, value: bool):
        self._use_soft_contact = value
        self.world.use_soft_contact = value

    # add by Yulong Zhang, use in hand scene
    def set_hand_character_id(self):
        if self.hand_scene:
            self.r_character_id = self.characters[0].character_id
            self.l_character_id = self.characters[1].character_id
            self.obj_character_id = self.characters[2].character_id

    def set_simulation_type(self, sim_type: SimulationType) -> SimulationType:
        if self.simu_type == sim_type:
            return sim_type
        if sim_type == SimulationType.DAMPED_STEP:
            self.simu_func = self.damped_simulate_once
            self.disable_implicit_damping()
        elif sim_type == SimulationType.DAMPED_FAST_STEP:  # TODO: maybe we can test this case.
            self.simu_func = self.fast_simulate_once
            self.use_implicit_damping()
        elif sim_type == SimulationType.STEP:
            self.simu_func = self.simulate_once
            self.disable_implicit_damping()
        elif sim_type == SimulationType.FAST_STEP:
            self.simu_func = self.fast_simulate_once
            self.disable_implicit_damping()
        else:
            raise NotImplementedError

        self.simu_type = sim_type
        print(f"set sim type to {self.simu_type}")
        return self.simu_type

    def step_range(self) -> range:
        return range(self.step_cnt)

    def use_implicit_damping(self):
        for character in self.characters:
            for joint in character.joint_info.joints:
                joint.enable_implicit_damping()

    def disable_implicit_damping(self):
        for character in self.characters:
            for joint in character.joint_info.joints:
                joint.disable_implicit_damping()

    def set_gravity(self, gravity: Union[Iterable, float, None] = None):
        if gravity is None:
            g = self.default_gravity
        elif isinstance(gravity, float):
            g = [0, gravity, 0]
        elif isinstance(gravity, Iterable):
            g = list(gravity)
        else:
            raise NotImplementedError

        self.world.setGravity(g)

    def copy_character(self, old_character: ODECharacter, space: Optional[ode.SpaceBase] = None):
        pass

    @property
    def gravity_numpy(self) -> np.ndarray:
        """
        Get the gravity. default gravity is [0, -9.8, 0]
        """
        return self.world.getGravityNumpy()

    @property
    def self_collision(self) -> bool:
        """
        consider self collision detection is enabled for each character
        """
        return self._self_collision

    @self_collision.setter
    def self_collision(self, value: bool):
        for character in self.characters:
            character.self_collision = value
        self._self_collision = value
        self.world.self_collision = value

    def build_world_and_space(self, gravity: Union[Iterable, float, None] = None):
        self.world = ode.World()
        self.set_gravity(gravity)
        # self.space = ode.Space()  # simple space. using AABB for collision detection.
        self.space = ode.HashSpace()
        return self.world, self.space

    @property
    def floor(self) -> Optional[ode.GeomPlane]:  # TODO: add floor id in unity
        return self.environment.floor

    # Get the first character in the scene
    @property
    def character0(self) -> Optional[ODECharacter]:
        return self.characters[0] if self.characters else None

    def get_character_id_map(self) -> Dict[int, ODECharacter]:
        return {character.character_id: character for character in self.characters}

    def set_render_fps(self, render_fps: int):
        self.render_dt = 1.0 / render_fps  # Frame Time in Unity
        self.render_fps = render_fps
        self.step_cnt = self.sim_fps // self.render_fps if self.render_fps > 0 else 1

    def set_sim_fps(self, sim_fps: int):
        self.sim_dt = 1.0 / sim_fps
        self.sim_fps = sim_fps
        self.step_cnt = self.sim_fps // self.render_fps if self.render_fps > 0 else 1

    def create_floor(self) -> ode.GeomPlane:
        """
        Create floor geometry
        """
        return self.environment.create_floor()

    def reset(self):
        """
        reset each character to initial state
        """
        for character in self.characters:
            character.load_init_state()
        return self

    @staticmethod
    def set_falldown_flag(geom1: ode.GeomObject, geom2: ode.GeomObject):
        if geom1.is_environment and not geom2.clung_env:
            geom2.character.fall_down = True
        if geom2.is_environment and not geom1.clung_env:
            geom1.character.fall_down = True

    @property
    def contact_count(self) -> int:  # TODO: support contact count in Unity, load contact param in json file
        return self._contact_count  # in ode.dCollide

    @contact_count.setter
    def contact_count(self, value: int):
        self._contact_count = value
        self.world.max_contact_num = value

    def contact_save(self) -> SceneContactInfo:
        """
        save contact position, force.
        render in Unity.
        """
        if len(self.contact) > 0:
            pos = np.zeros((len(self.contact), 3), dtype=np.float64)
            force = np.zeros((len(self.contact), 3), dtype=np.float64)
            geom1_name, geom2_name = [], []
            if self.hand_scene is False:
                for idx, contact_joint in enumerate(self.contact.joints):
                    joint: ode.ContactJoint = contact_joint
                    contact: ode.Contact = joint.contact
                    pos[idx, :] = contact.contactPosNumpy  # contact position
                    force[idx, :] = joint.FeedBackForce()  # contact force
                    geom1_name.append(joint.contactGeom1.name)
                    geom2_name.append(joint.contactGeom2.name)
                if self.contact_info is None:
                    self.contact_info = SceneContactInfo(pos, force, geom1_name, geom2_name)
                else:
                    self.contact_info.set_value(pos, force, geom1_name, geom2_name)
            else:
                body_contact_force_r = np.zeros((len(self.characters[0].bodies), 3), dtype=np.float64)
                body_contact_force_l = np.zeros((len(self.characters[1].bodies), 3), dtype=np.float64)
                body_contact_label_r = np.zeros(len(self.characters[0].bodies), dtype=np.bool8)
                body_contact_label_l = np.zeros(len(self.characters[1].bodies), dtype=np.bool8)
                for idx, contact_joint in enumerate(self.contact.joints):
                    joint: ode.ContactJoint = contact_joint
                    contact: ode.Contact = joint.contact
                    pos[idx, :] = contact.contactPosNumpy  # contact position
                    force[idx, :] = joint.FeedBackForce()  # contact force
                    geom1_name.append(joint.contactGeom1.name)
                    geom2_name.append(joint.contactGeom2.name)
                    body1: ode.Body = joint.contactGeom1.body
                    body2: ode.Body = joint.contactGeom2.body
                    if body1 is None or body2 is None:
                        continue
                    if joint.contactGeom1.character_id == self.r_character_id and joint.contactGeom2.character_id == self.obj_character_id:
                        body_contact_label_r[body1.instance_id] = True
                        body_contact_force_r[body1.instance_id] += force[idx]
                    elif joint.contactGeom1.character_id == self.l_character_id and joint.contactGeom2.character_id == self.obj_character_id:
                        body_contact_label_l[body1.instance_id] = True
                        body_contact_force_l[body1.instance_id] += force[idx]
                    elif joint.contactGeom2.character_id == self.r_character_id and joint.contactGeom1.character_id == self.obj_character_id:
                        body_contact_label_r[body2.instance_id] = True
                        body_contact_force_r[body2.instance_id] += force[idx]
                    elif joint.contactGeom2.character_id == self.l_character_id and joint.contactGeom1.character_id == self.obj_character_id:
                        body_contact_label_l[body2.instance_id] = True
                        body_contact_force_l[body2.instance_id] += force[idx]
                body_contact_label = np.concatenate((body_contact_label_r, body_contact_label_l), axis=0)
                body_contact_force = np.concatenate((body_contact_force_r, body_contact_force_l), axis=0)
                if self.contact_info is None:
                    self.contact_info = SceneContactInfo(pos, force, geom1_name, geom2_name, contact_label=body_contact_label, body_contact_force=body_contact_force)
                else:
                    self.contact_info.set_value(pos, force, geom1_name, geom2_name, contact_label=body_contact_label, body_contact_force=body_contact_force)
        else:
            if self.contact_info is not None:
                self.contact_info.clear()

        return self.contact_info

    def contact_local_save(self) -> SceneContactLocalInfo:
        """
        we need only save position in body 1 coordinate
        we can remain global normal vector
        """
        if self.contact_local_info is not None:
            self.contact_local_info.clear()

        len_contact: int = len(self.contact)
        if len(self.contact) > 0:
            global_pos_ret: np.ndarray = np.zeros((len_contact, 3), dtype=np.float64)
            local_pos_ret: np.ndarray = np.zeros((len_contact, 3), dtype=np.float64)
            normal_ret: np.ndarray = np.zeros_like(local_pos_ret)
            depth_ret: np.ndarray = np.zeros(len_contact)
            body1_cid_ret: np.ndarray = np.zeros(len_contact, dtype=np.uint64)
            for index, contact_joint in enumerate(self.contact.joints):
                joint: ode.ContactJoint = contact_joint
                contact: ode.Contact = joint.contact
                contact_pos: np.ndarray = contact.contactPosNumpy
                body1: ode.Body = joint.body1
                assert body1 is not None
                global_pos_ret[index] = contact_pos
                contact_local_pos: np.ndarray = body1.getPosRelPointNumpy(contact_pos)
                local_pos_ret[index] = contact_local_pos
                normal: np.ndarray = contact.contactNormalNumpy
                normal_ret[index] = normal
                depth_ret[index] = contact.contactDepth
                body1_cid_ret[index] = body1.get_bid()
            if self.contact_local_info is None:
                self.contact_local_info = SceneContactLocalInfo()
            self.contact_local_info.set_value(global_pos_ret, local_pos_ret, normal_ret, depth_ret, body1_cid_ret)

        return self.contact_local_info

    def contact_basic(self, geom1: ode.GeomObject, geom2: ode.GeomObject) -> Optional[List[ode.Contact]]:
        if geom1.character_id == geom2.character_id and (not self.self_collision or not geom1.character.self_collision):
            return None

        # self.set_falldown_flag(geom1, geom2)
        if geom1.body is None and geom2.body is not None:
            geom1, geom2 = geom2, geom1
        contacts: List[ode.Contact] = ode.collide(geom1, geom2, self._contact_count)

        return contacts

    def _generate_contact_joint(self, geom1: ode.GeomObject, geom2: Optional[ode.GeomObject], contacts: List[ode.Contact]):
        if geom2 is not None:
            if geom1.body is None and geom2.body is not None:
                geom1, geom2 = geom2, geom1

        # Create contact joints. Add contact joint position and contact force in Unity
        if self.contact_type == ContactType.ODE_LCP:
            mu: float = min(geom1.friction, geom2.friction if geom2 is not None else geom1.friction)
            for c in contacts:
                c.bounce = self.bounce
                c.mu = mu
                if self.use_soft_contact:
                    if self.soft_cfm is not None and self.soft_erp is not None:
                        c.enable_soft_cfm_erp(self.soft_cfm, self.soft_erp)
                    if self.soft_cfm_tan is not None:
                        c.enable_contact_slip(self.soft_cfm_tan)
                j = ode.ContactJoint(self.world, self.contact, c)  # default mode is ode.ContactApprox1
                j.setFeedback(self.extract_contact)
                j.attach(geom1.body, geom2.body)
        elif self.contact_type == ContactType.MAX_FORCE_ODE_LCP:
            max_fric: float = min(geom1.max_friction, geom2.max_friction if geom2 is not None else geom1.max_fraction)  # TODO: set max fric of plane..
            for c in contacts:
                c.bounce = 0.0
                c.mu = max_fric

                # default world erp in ODE is 0.2
                # in Samcon implement of Libin Liu,
                # default dSoftCFM = 0.007;
                # default dSoftERP = 0.8;

                if self.use_soft_contact:
                    if self.soft_cfm is not None and self.soft_erp is not None:
                        c.enable_soft_cfm_erp(self.soft_cfm, self.soft_erp)
                    if self.soft_cfm_tan is not None:
                        c.enable_contact_slip(self.soft_cfm_tan)

                j: ode.ContactJointMaxForce = ode.ContactJointMaxForce(self.world, self.contact, c)
                if self.use_soft_contact:
                    j.joint_cfm = self.soft_cfm
                    j.joint_erp = self.soft_erp
                j.setFeedback(self.extract_contact)
                j.attach(geom1.body, geom2.body)

                if debug_mode:
                    print(f"max force ode LCP: mu = {c.mu}")

        elif self.contact_type == ContactType.BALL:
            raise ValueError("Do not create ball joint here.")
            for c in contacts:
                j = ode.BallJoint(self.world, self.contact)
                j.setFeedback(True)
                j.attach(geom1.body, geom2.body)
                j.setAnchorNumpy(c.contactPosNumpy)
        else:
            raise ValueError

    def near_callback(self, args, geom1: ode.GeomObject, geom2: ode.GeomObject):
        contacts = self.contact_basic(geom1, geom2)
        if not contacts:
            return
        self._generate_contact_joint(geom1, geom2, contacts)

    def _compute_collide_callback(self, args, geom1: ode.GeomObject, geom2: ode.GeomObject):
        contacts = self.contact_basic(geom1, geom2)
        if not contacts:
            return
        for c in contacts:
            self.contact_info.pos.append(c.contactPosNumpy)
            self.contact_info.force.append(c.contactNormalNumpy * c.contactDepth)
            self.contact_info.geom1_name.append(geom1.name)
            self.contact_info.geom2_name.append(geom2.name)

    def compute_collide_info(self) -> SceneContactInfo:
        for character in self.characters:
            character.fall_down = False

        self.contact_info = SceneContactInfo([], [], [], [])
        self.space.collide(None, self._compute_collide_callback)
        self.resort_geoms()
        if self.contact_info.pos:
            self.contact_info.pos = np.concatenate([i[None, :] for i in self.contact_info.pos], axis=0)
        if self.contact_info.force:
            self.contact_info.force = np.concatenate([i[None, :] for i in self.contact_info.force], axis=0)

        return self.contact_info

    def extract_body_contact_label(self) -> np.ndarray:
        """
        extract contact label (0/1). here we need not to create contact joints.
        """
        # assert len(self.characters) == 1
        contact_label = np.zeros(len(self.character0.bodies), dtype=np.int32)
        def callback(args, geom1: ode.GeomObject, geom2: ode.GeomObject):
            if geom1.character_id == geom2.character_id and (not self.self_collision or not geom1.character.self_collision):
                return
            body1: ode.Body = geom1.body
            if body1 is not None:
                contact_label[body1.instance_id] = 1
            body2: ode.Body = geom2.body
            if body2 is not None:
                contact_label[body2.instance_id] = 1

        self.space.collide(None, callback)
        self.space.ResortGeoms()

        return contact_label

    # collision detection
    def pre_simulate_step(self) -> ode.JointGroup:
        for character in self.characters:
            character.fall_down = False
        self.space.collide((self.world, self.contact), self.near_callback)
        return self.contact

    def post_simulate_step(self):
        self.contact.empty()  # clear contact joints after simulation
        self.space.ResortGeoms()

    def resort_geoms(self):
        # in Open Dynamics Engine, the order of geometries are changed after a step of forward simulation.
        # make sure the order of geometries are not changed.
        self.space.ResortGeoms()

    def step_fast_collision(self):  # do collision detection in cython, not python
        self.world.step_fast_collision(self.space, self.sim_dt)
        self.space.ResortGeoms()

    def damped_step_fast_collision(self):  # do collision detection in cython, not python
        self.world.damped_step_fast_collision(self.space, self.sim_dt)
        # for debug, compute center of mass, and 

    def simulate_once(self):
        self.pre_simulate_step()
        self.world.step(self.sim_dt)  # This will change geometry order in ode space
        if self.extract_contact:
            self.contact_save()  # save contact info
        self.post_simulate_step()

    def simulate(self, n: int = 0):
        cnt = n if n > 0 else self.step_cnt
        for _ in range(cnt):
            self.simulate_once()

    def damped_simulate_once(self):
        # self.world.damped_step_fast_collision(self.space, self.sim_dt)
        self.pre_simulate_step()
        self.world.dampedStep(self.sim_dt)  # This will change geometry order in ode space
        if self.extract_contact:
            self.contact_save()
            self.update_contact_visualize()
        self.post_simulate_step()

    def fast_simulate_once(self):  # use quick step in ode engine
        self.pre_simulate_step()
        # This will change geometry order in ode space
        self.world.quickStep(self.sim_dt)
        if self.extract_contact:
            self.contact_save()
            self.update_contact_visualize()
        self.post_simulate_step()

    def damped_simulate(self, n: int = 0):
        cnt = n if n > 0 else self.step_cnt
        for _ in range(cnt):
            self.damped_simulate_once()

    def simulate_no_collision(self, n: int = 0):
        cnt = n if n > 0 else self.step_cnt
        for _ in range(cnt):
            self.world.step(self.sim_dt)
            self.space.ResortGeoms()  # make sure simulation result is same

    def damped_simulate_no_collision_once(self):
        self.world.dampedStep(self.sim_dt)  # This will change geometry order in ode space
        self.space.ResortGeoms()

    def damped_simulate_no_collision(self, n: int = 0):
        cnt = n if n > 0 else self.step_cnt
        for _ in range(cnt):
            self.damped_simulate_no_collision_once()

    def clear(self, gravity: Union[Iterable, float, None] = None):
        """
        clear the scene
        """
        if self.environment is not None:
            self.environment.clear()

        if self.characters is not None:
            self.characters.clear()

        self.build_world_and_space(self.default_gravity if gravity is None else gravity)

        if self.ext_joints is not None:
            self.ext_joints.clear()

        self.environment = Environment(self.world, self.space)
        self.ext_joints = ExtJointList(self.world, self.characters)
        self.contact_info = SceneContactInfo()

        self.str_info = ""  # use for print or show information in console or Unity

        return self

    def build_aux_controller(self):
        raise ValueError("This not work.")
        ret = MotionUtils.AuxControlHandle(
            self.gravity_numpy,
            self.character0.body_info.mass_val,
            self.character0.body_info.initial_inertia,
            np.concatenate([np.zeros((1, 3)), self.character0.joint_info.get_child_body_relative_pos()], axis=0),
            np.array(self.character0.body_info.parent, dtype=np.int32),
            use_gravity=1,
            use_coriolis=0
        )
        return ret

    def set_mu_param(self, mu: float):
        for ch in self.characters:
            ch.set_geom_mu_param(mu)
        for geom in self.environment.geoms:
            geom.friction = mu
        return self

    def update_view_com(self):
        """
        """
        for ch in self.characters:
            # compute center of mass
            com: np.ndarray = ch.body_info.calc_center_of_mass()
            length: float = ch.com_geom.getLengths()[1]
            ch.com_geom.PositionNumpy = com + np.array([0.0, -0.5 * length, 0.0])

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
Use Kinematics result to estimate contact point.

As kinematic result is not physically plausiable, the result may be noisy sometimes.


if contact label is 1 for some body,
we can create a contact point for this body.

if we predict contact label by neural network, the output value is continuous.
We can get contact label by a threshold, that is, when the prediction of contact label >= eps,
we can say there is a contact on the body.

I don't know the accurate of predicted contact label and kinematic result..
We need more experiments.
"""

import copy
import ModifyODE as ode
from mpi4py import MPI
import numpy as np
import os
from scipy.spatial.transform import Rotation
from typing import Optional, List, Tuple, Union

from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo
from VclSimuBackend.ODESim.ODEScene import ContactType, SceneContactInfo

from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase, TargetPose
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.Render.Renderer import RenderWorld  # render using Long Ge's framework

fdir = os.path.dirname(__file__)
comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()


class ContactLabelExtractor:
    """
    Actually, collision detection by Open Dynamics Engine is noisy,
    and when the input data is not physically plausiable,

    TODO: We should also set a eps, that is, if the height < eps, we can say collision occurs.

    When create contact joint from body contact label, we should consider left heel and right heel separetely.
    because there are many geometry on left and right heel.

    We can do collision from body contact label in this way:
    1. create contact joint for bodies except left and right heel with contact label.
    2. compute bounding box for left and right heel, and move the left and right heel body on the floor.
    3. do real collision detection for left and right heel.
    4. move the contact back to original position

    In optimization process, we should add contact dynamically when body height is too small..
    When create hack contact joints, we should
    """

    box_mask: np.ndarray = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ], dtype=np.float64)

    l_heel_name: str = "lFoot"
    r_heel_name: str = "rFoot"

    def __init__(self, scene: ODEScene, character: ODECharacter) -> None:
        self.scene: ODEScene = scene
        self.character: ODECharacter = character
        self.body_names: List[str] = self.character.get_body_name_list()
        self.n_bodies: int = len(self.character.bodies)
        # self.height_eps: float = height_eps
        self.root_body_index: int = 0
        self.l_heel_body_index: int = self.body_names.index(self.l_heel_name)
        self.r_heel_body_index: int = self.body_names.index(self.r_heel_name)
        self.body_min_contact_h: Optional[np.ndarray] = np.zeros(self.n_bodies, dtype=np.float64)

        self.body_parents: List[int] = character.body_info.parent.copy()
        self.body_children: List[List[int]] = copy.deepcopy(character.body_info.children)
        self.body_neighbour: List[List[int]] = []
        # compute near bodies..
        for body_index in range(self.n_bodies):
            parent = self.body_parents[body_index]
            children = self.body_children[body_index]
            near_body = children.copy()
            if parent != -1:
                near_body.append(parent)
            self.body_neighbour.append(near_body)

        # compute body length..
        self.character.load_init_state()
        body_pos = self.character.get_body_pos()
        self.body_distance: np.ndarray = MathHelper.array_distance(body_pos, body_pos)

        # store body with a box geometry
        self.box_index: Optional[np.ndarray] = None
        self.box_length: Optional[np.ndarray] = None
        self.box_max_length: Optional[np.ndarray] = None
        self.num_box: Optional[int] = None

        # stores body with a ball geometry
        self.ball_index: Optional[np.ndarray] = None
        self.ball_radius: Optional[np.ndarray] = None
        # for std human model, ball location will not always match body position
        self.ball_offset: Optional[np.ndarray] = None
        self.num_ball: Optional[int] = None

        # stores body with a capsule geometry
        self.capsule_index: Optional[np.ndarray] = None
        self.capsule_radius: Optional[np.ndarray] = None
        self.capsule_length: Optional[np.ndarray] = None
        self.capsule_axis: Optional[np.ndarray] = None
        self.num_capsule: Optional[int] = None

        self.max_contact_num: int = 0
        self.box_label_offset: int = 0
        self.ball_label_offset: int = 0
        self.capsule_label_offset: int = 0

        self.box_label_arange: Optional[np.ndarray] = None
        self.ball_label_arange: Optional[np.ndarray] = None
        self.capsule_label_arange: Optional[np.ndarray] = None

        self.box_geom_arange: Optional[np.ndarray] = None
        self.ball_geom_arange: Optional[np.ndarray] = None
        self.capsule_geom_arange: Optional[np.ndarray] = None

        self.body_to_geom: Optional[List[List[int]]] = None

        self.ode_box_geoms: Optional[List[ode.GeomBox]] = None
        self.ode_ball_geoms: Optional[List[ode.GeomSphere]] = None
        self.ode_capsule_geoms: Optional[List[ode.GeomCapsule]] = None
        self.total_ode_geoms: Optional[List[ode.GeomObject]] = None

        self.box_geom_offset: int = 0
        self.ball_geom_offset: int = 0
        self.capsule_geom_offset: int = 0

        self.compute_index()  # initialize..

    def clear(self):
        # store body with a box geometry
        self.box_index: Optional[np.ndarray] = None
        self.box_length: Optional[np.ndarray] = None
        self.box_max_length: Optional[np.ndarray] = None
        self.num_box: Optional[int] = None

        # stores body with a ball geometry
        self.ball_index: Optional[np.ndarray] = None
        self.ball_radius: Optional[np.ndarray] = None
        # for std human model, ball location will not always match body position
        self.ball_offset: Optional[np.ndarray] = None
        self.num_ball: Optional[int] = None

        # stores body with a capsule geometry
        self.capsule_index: Optional[np.ndarray] = None
        self.capsule_radius: Optional[np.ndarray] = None
        self.capsule_length: Optional[np.ndarray] = None
        self.capsule_axis: Optional[np.ndarray] = None
        self.num_capsule: Optional[int] = None

        self.max_contact_num: int = 0
        self.box_label_offset: int = 0
        self.ball_label_offset: int = 0
        self.capsule_label_offset: int = 0

        self.box_label_arange: Optional[np.ndarray] = None
        self.ball_label_arange: Optional[np.ndarray] = None
        self.capsule_label_arange: Optional[np.ndarray] = None

        self.box_geom_arange: Optional[np.ndarray] = None
        self.ball_geom_arange: Optional[np.ndarray] = None
        self.capsule_geom_arange: Optional[np.ndarray] = None

        self.body_to_geom: Optional[List[List[int]]] = None

        self.ode_box_geoms: Optional[List[ode.GeomBox]] = None
        self.ode_ball_geoms: Optional[List[ode.GeomSphere]] = None
        self.ode_capsule_geoms: Optional[List[ode.GeomCapsule]] = None
        self.total_ode_geoms: Optional[List[ode.GeomObject]] = None

    def compute_index(self):
        self.character.load_init_state()  # load initial state for T pose
        box_index: List[int] = []  # [18, 19] for two hands in std-human model
        box_length: List[np.ndarray] = []  # box length
        box_max_length: List[float] = []

        # lowerBack, torso have only 1 ball geom
        # root body has 3 ball geom
        # heel body has 3 ball geom, and 1 capsule geom (which can be ignored)
        ball_index: List[int] = []  # [1, 2, 0, 0, 0, 8, 8, 8, 7, 7, 7] for std-human model
        ball_radius: List[float] = []
        ball_offset: List[np.ndarray] = []

        # most of bodies in std-human model are capsules
        capsule_index: List[int] = []  # [3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17] for std-human model
        capsule_radius: List[float] = []
        capsule_length: List[float] = []
        capsule_axis: List[np.ndarray] = []

        self.ode_box_geoms: Optional[List[ode.GeomBox]] = []
        self.ode_ball_geoms: Optional[List[ode.GeomSphere]] = []
        self.ode_capsule_geoms: Optional[List[ode.GeomCapsule]] = []

        z_axis: np.ndarray = np.array([0.0, 0.0, 1.0])

        # here we should compute aabb for each body..
        for body_idx, body in enumerate(self.character.bodies):
            aabb: np.ndarray = body.get_aabb()
            min_bound: float = min(aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4])
            self.body_min_contact_h[body_idx] = min_bound

        # we should also consider root body and heel body for balls.
        for body_idx, body in enumerate(self.character.bodies):
            geoms: List[ode.GeomObject] = list(body.geom_iter())
            if len(geoms) > 1:
                continue
            # print(f"body index = {body_idx}") # for debug
            geom: Union[ode.GeomSphere, ode.GeomBox, ode.GeomCCylinder] = geoms[0]
            if isinstance(geom, ode.GeomSphere):
                ball_index.append(body_idx)
                ball_radius.append(geom.geomRadius)
                offset_pos: np.ndarray = geom.PositionNumpy - body.PositionNumpy
                ball_offset.append(offset_pos[None, :])
                self.ode_ball_geoms.append(geom)
            elif isinstance(geom, ode.GeomBox):
                box_index.append(body_idx)
                box_len: np.ndarray = np.array(geom.geomLength).reshape((1, 3))
                box_length.append(box_len)
                box_max_length.append(np.max(box_len))
                self.ode_box_geoms.append(geom)
            elif isinstance(geom, ode.GeomCCylinder):
                capsule_index.append(body_idx)
                radius, length = geom.geomRadiusAndLength
                capsule_radius.append(radius)
                capsule_length.append(length)
                # in open dynamics engine, the default capsule axis is along z axis.
                geom_rot: Rotation = Rotation(geom.QuaternionScipy)
                axis: np.ndarray = geom_rot.apply(z_axis)
                capsule_axis.append(axis[None, :])
                self.ode_capsule_geoms.append(geom)
            else:
                raise ValueError

        def handle_multi_ball(body_index_: int):
            body_: ode.Body = self.character.bodies[body_index_]
            for geom_ in body_.geom_iter():
                if isinstance(geom_, ode.GeomSphere):
                    ball_index.append(body_index_)
                    ball_radius.append(geom_.geomRadius)
                    offset_pos_: np.ndarray = geom_.PositionNumpy - body_.PositionNumpy
                    ball_offset.append(offset_pos_[None, :])
                    self.ode_ball_geoms.append(geom_)

        # There are 3 balls in root geom
        handle_multi_ball(0)

        # for left foot and right foot..
        body_name_list = self.character.body_info.get_name_list()
        handle_multi_ball(body_name_list.index("lFoot"))
        handle_multi_ball(body_name_list.index("rFoot"))

        # concat to np.ndarray
        self.box_index: Optional[np.ndarray] = np.array(box_index)  # [18, 19] for std human
        self.box_length: Optional[np.ndarray] = np.concatenate(box_length, axis=0)
        self.box_max_length: Optional[np.ndarray] = np.array(box_max_length)
        self.num_box: int = self.box_index.shape[0]  # 2 for std human

        self.ball_index: Optional[np.ndarray] = np.array(ball_index)  # [1, 2, 0, 0, 0, 8, 8, 8, 7, 7, 7] for std human
        self.ball_radius: Optional[np.ndarray] = np.array(ball_radius)
        self.ball_offset: Optional[np.ndarray] = np.concatenate(ball_offset, axis=0)  # [ 3,  4,  5,  6,  9, 10, 11, 12, 13, 14, 15, 16, 17] for std human
        self.num_ball: int = self.ball_index.shape[0]  # 11 for std human

        self.capsule_index: Optional[np.ndarray] = np.array(capsule_index)  # [ 3,  4,  5,  6,  9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.capsule_radius: Optional[np.ndarray] = np.array(capsule_radius)
        self.capsule_length: Optional[np.ndarray] = np.array(capsule_length)
        self.capsule_axis: Optional[np.ndarray] = np.concatenate(capsule_axis)
        self.num_capsule: int = self.capsule_index.shape[0]  # 13 for std human

        self.box_label_offset: int = 0  # This means potential contact place, NOT corresbounding geometry!
        self.ball_label_offset: int = 8 * self.box_index.shape[0]  # collision will occurs at 8 corners for each box geom, 16 for std-human
        self.capsule_label_offset: int = self.ball_label_offset + self.ball_index.shape[0]  # 27 for std human
        self.max_contact_num: int = self.capsule_label_offset + 2 * self.capsule_index.shape[0]  # 53 for std human

        self.box_label_arange: Optional[np.ndarray] = self.box_label_offset + np.arange(0, 8 * self.box_index.shape[0])  # [0..15] for std-human
        self.ball_label_arange: Optional[np.ndarray] = self.ball_label_offset + np.arange(0, self.ball_index.shape[0])  # [16..26] for std-human
        self.capsule_label_arange: Optional[np.ndarray] = self.capsule_label_offset + np.arange(0, 2 * self.capsule_index.shape[0])  # [27..52] for std-human

        self.total_ode_geoms: List = self.ode_box_geoms + self.ode_ball_geoms + self.ode_capsule_geoms  # len == 26 for std-human
        self.box_geom_offset: int = 0
        self.ball_geom_offset: int = self.num_box
        self.capsule_geom_offset: int = self.num_box + self.num_ball

        self.box_geom_arange: np.ndarray = self.box_geom_offset + np.arange(0, self.num_box, dtype=np.int32)  # [0, 1] for std-human
        self.ball_geom_arange: np.ndarray = self.ball_geom_offset + np.arange(0, self.num_ball, dtype=np.int32)  # [2..12] for std-human
        self.capsule_geom_arange: np.ndarray = self.capsule_geom_offset + np.arange(0, self.num_capsule, dtype=np.int32)  # [13..25] for std-human

        # for std-human, self.contact_to_geom_index == 
        # 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,
        # 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 13, 14, 14, 15, 15, 16,
        # 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25
        self.contact_to_geom_index: Optional[np.ndarray] = np.zeros(self.max_contact_num, dtype=np.int32)  # len == 53, map the conatct to geometry
        for box_num in range(self.num_box):
            self.contact_to_geom_index[self.box_label_offset + box_num * 8:][:8] = self.box_geom_offset + box_num
        self.contact_to_geom_index[self.ball_label_offset:][:self.num_ball] = self.ball_geom_offset + np.arange(0, self.num_ball, dtype=np.int32)
        for cap_num in range(self.num_capsule):
            self.contact_to_geom_index[self.capsule_label_offset + cap_num * 2:][:2] = self.capsule_geom_offset + cap_num
        if comm_size == 1:
            print(
                f"Build Contact With Kinematic: "
                f"len(ode_box) = {len(self.ode_box_geoms)}, len(ode_ball) = {len(self.ode_ball_geoms)},"
                f"len(ode_capsule) = {len(self.ode_capsule_geoms)}"
            )

        self.body_to_geom: List[List] = [[] for _ in range(self.n_bodies)]  # len == 20 for std-human
        # for std-human, self.body_to_geom == 
        # [[4, 5, 6], [2], [3], [13], [14], [15], [16], [10, 11, 12], [7, 8, 9], [17], [18], [19], [20], [21], [22], [23], [24], [25], [0], [1]]
        for box_index, node in enumerate(self.box_index):
            self.body_to_geom[node].append(self.box_geom_offset + box_index)
        for ball_index, node in enumerate(self.ball_index):
            self.body_to_geom[node].append(self.ball_geom_offset + ball_index)
        for capsule_index, node in enumerate(self.capsule_index):
            self.body_to_geom[node].append(self.capsule_geom_offset + capsule_index)

        return self

    def contact_index_to_geom(self, index: int) -> ode.GeomObject:
        return self.total_ode_geoms[self.contact_to_geom_index[index]]

    def compute_box_contact_base(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        geom_index: Optional[np.ndarray] = None
    ):
        """
        when use hack contact, the geom position may not match..
        we can use this way: first, estimate minimal height, then move body position, then do collision detection, then move back
        TODO: Test
        """
        box_index: np.ndarray = self.box_index[geom_index] if geom_index is not None else self.box_index
        box_length: np.ndarray = self.box_length[geom_index] if geom_index is not None else self.box_length
        box_max_length: np.ndarray = self.box_max_length[geom_index] if geom_index is not None else self.box_max_length

        box_body_pos: np.ndarray = body_pos[box_index, :]
        if geom_index is None: # and np.all(box_body_pos[:, 1] - box_max_length > 0.1):
            return None

        # (num box, 3), (8 ,3) -> (num box, 8, 3)
        # we should rotate body length
        offset: np.ndarray = 0.5 * box_length[:, None, :] * self.box_mask[None, ...]  # (num box, 8, 3)
        global_offset: np.ndarray = np.zeros_like(offset)
        for ncount, box_idx in enumerate(box_index):
            rot: Rotation = Rotation(body_quat[box_idx])
            global_offset[ncount] = rot.apply(offset[ncount])

        corner_pos: np.ndarray = box_body_pos[:, None, :] + global_offset
        corner_shape: Tuple = corner_pos.shape

        if geom_index is not None:
            min_corner_pos: np.ndarray = np.min(corner_pos[..., 1], axis=-1, keepdims=True) + 1e-8
            if min_corner_pos.size == 1:
                min_corner_pos = min_corner_pos.flatten()
            corner_pos[..., 1] -= min_corner_pos
        else:
            min_corner_pos: Optional[np.ndarray] = None

        corner_pos: np.ndarray = corner_pos.reshape((-1, 3))

        # here we should compute y component of each contact
        flag: np.ndarray = np.where(corner_pos[:, 1] < 0)[0]
        if len(flag) == 0:
            return None
        # note: we should also return contact body index..
        index: np.ndarray = np.repeat(box_index, 8)

        if geom_index is not None:
            corner_pos = corner_pos.reshape(corner_shape)
            corner_pos[..., 1] += min_corner_pos
            corner_pos = corner_pos.reshape(-1, 3)

            # we need also re map the flag index..
            ret_label = np.arange(0, 8 * self.num_box).reshape((self.num_box, 8))[geom_index].flatten()[flag] + self.box_label_offset
            # print(corner_pos[flag])
            return index[flag], corner_pos[flag], ret_label
        else:
            return index[flag], corner_pos[flag], flag + self.box_label_offset

    def compute_box_contact(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        here we should also compute the contact label..

        return: contact body index, contact global position, contact label index
        """
        return self.compute_box_contact_base(body_pos, body_quat)

    def compute_capsule_contact_base(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        geom_index: Optional[np.ndarray] = None
    ):
        capsule_index: np.ndarray = self.capsule_index[geom_index] if geom_index is not None else self.capsule_index
        capsule_length: np.ndarray = self.capsule_length[geom_index] if geom_index is not None else self.capsule_length
        capsule_axis: np.ndarray = self.capsule_axis[geom_index] if geom_index is not None else self.capsule_axis
        capsule_radius: np.ndarray = self.capsule_radius[geom_index] if geom_index is not None else self.capsule_radius

        capsule_body_pos: np.ndarray = body_pos[capsule_index, :]
        capsule_body_rot: Rotation = Rotation(body_quat[capsule_index, :])
        down_y_vec: np.ndarray = np.array([[0.0, -1.0, 0.0]])

        # now, we should consider offset
        offset: np.ndarray = capsule_body_rot.apply(0.5 * capsule_length[:, None] * capsule_axis)

        ball_pos0: np.ndarray = capsule_body_pos + offset
        ball_pos1: np.ndarray = capsule_body_pos - offset

        if geom_index is not None:
            minimal_pos: np.ndarray = np.minimum(ball_pos0[..., 1], ball_pos1[..., 1]) + 1e-8
            ball_pos0[..., 1] -= minimal_pos
            ball_pos1[..., 1] -= minimal_pos
        else:
            minimal_pos: Optional[np.ndarray] = None

        # check if the ball has contact
        flag0: np.ndarray = np.where(ball_pos0[:, 1] < capsule_radius)[0]
        flag1: np.ndarray = np.where(ball_pos1[:, 1] < capsule_radius)[0]
        base_index: np.ndarray = np.arange(0, 2 * self.num_capsule).reshape((self.num_capsule, 2))
        def get_contact(ball_pos_: np.ndarray, flag_: np.ndarray, l_r: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            tmp_ret: np.ndarray = ball_pos_[flag_] + capsule_radius[flag_, None] @ down_y_vec
            if geom_index is not None:
                tmp_ret[..., 1] += minimal_pos[flag_]
                return capsule_index[flag_], tmp_ret, base_index[geom_index, l_r][flag_] + self.capsule_label_offset
            else:
                return capsule_index[flag_], tmp_ret, base_index[:, l_r][flag_] + self.capsule_label_offset

        c0 = get_contact(ball_pos0, flag0, 0) if flag0 is not None else None
        c1 = get_contact(ball_pos1, flag1, 1) if flag1 is not None else None
        if c0 is not None and c1 is not None:
            return [np.concatenate([c0[i], c1[i]]) for i in range(3)]
        elif c0 is not None:
            return c0
        elif c1 is not None:
            return c1
        else:
            return None

    def compute_capsule_contact(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        for body with capsule geometry,
        we need also consider geometry orientation.
        height of q * (0.5 * init_length * init_axis) * q^{-1} <= radius.

        Note: toe is in shape of capsule.
        """
        return self.compute_capsule_contact_base(body_pos, body_quat)

    def compute_ball_min_height(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        contact_flag: np.ndarray
    ) -> np.ndarray:
        ball_index = self.ball_index[contact_flag] if contact_flag is not None else self.ball_index
        ball_offset = self.ball_offset[contact_flag] if contact_flag is not None else self.ball_offset
        ball_geom_pos: np.ndarray = body_pos[ball_index, :] + Rotation(body_quat[ball_index, :]).apply(ball_offset)
        return np.min(ball_geom_pos[..., 1])

    def compute_ball_contact_base(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        contact_flag: Optional[np.ndarray] = None
    ):
        """
        body_pos: np.ndarray in shape (num body, 3)
        body_quat: np.ndarray in shape (num body, 4)
        contact_flag: subset of ball geoms.

        return: ball body index, contact position, collision index
        """
        ball_index: np.ndarray = self.ball_index[contact_flag] if contact_flag is not None else self.ball_index
        ball_offset: np.ndarray = self.ball_offset[contact_flag] if contact_flag is not None else self.ball_offset
        ball_radius: np.ndarray = self.ball_radius[contact_flag] if contact_flag is not None else self.ball_radius

        ball_geom_pos: np.ndarray = body_pos[ball_index, :] + Rotation(body_quat[ball_index, :]).apply(ball_offset)

        down_y_vec: np.ndarray = np.array([[0.0, -1.0, 0.0]])
        if contact_flag is None:  # real collision detection
            ball_height: np.ndarray = ball_geom_pos[:, 1]
            contact_flag: np.ndarray = np.where(ball_height <= ball_radius)[0]
            if len(contact_flag) == 0:
                return None
            contact_pos: np.ndarray = ball_geom_pos[contact_flag] + ball_radius[contact_flag, None] @ down_y_vec
            return ball_index[contact_flag], contact_pos, contact_flag + self.ball_label_offset
        else:
            contact_pos: np.ndarray = ball_geom_pos + ball_radius[:, None] @ down_y_vec
            return ball_index, contact_pos, contact_flag + self.ball_label_offset

    def compute_ball_contact(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        There is only a frame.
        TODO: we should consider a eps for contact height..
        """
        return self.compute_ball_contact_base(body_pos, body_quat)

    @staticmethod
    def _append_contact_to_list(ret_index: List, ret_pos: List, ret_label: List, contact_ret: Tuple):
        ret_index.append(contact_ret[0])
        ret_pos.append(contact_ret[1])
        ret_label.append(contact_ret[2])

    def compute_contact_by_sub_body(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        sub_body_index: np.ndarray,
        process_heel: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        body_pos: np.ndarray in shape (num body, 3)
        body_quat: np.ndarray in shape (num body, 4)
        sub_body_index: np.ndarray
        """
        # 1. compute the inter-section between sub_body and current box..
        if len(sub_body_index) == 0:
            return None

        process_l_heel: bool = self.l_heel_body_index in sub_body_index
        process_r_heel: bool = self.r_heel_body_index in sub_body_index
        process_root: bool = self.root_body_index in sub_body_index

        if (not process_l_heel) and (not process_r_heel) and (not process_root):
            process_heel: bool = False  # we need not to process heel here.

        l_heel_geom, r_heel_geom = self.body_to_geom[self.l_heel_body_index], self.body_to_geom[self.r_heel_body_index]
        root_geom: List = self.body_to_geom[self.root_body_index]

        if process_heel:  # when process heel seperately, we should set geom for heel to empty.
            self.body_to_geom[self.l_heel_body_index] = []
            self.body_to_geom[self.r_heel_body_index] = []
            self.body_to_geom[self.root_body_index] = []

        ret_index, ret_pos, ret_label = [], [], []
        sub_geom: np.ndarray = np.array(sum([self.body_to_geom[node] for node in sub_body_index], []))
        sub_ball: np.ndarray = np.intersect1d(sub_geom, self.ball_geom_arange)  # geom index.
        if len(sub_ball) > 0:
            sub_ball.sort()
            idx: np.ndarray = sub_ball - self.ball_geom_offset
            ball_contact = self.compute_ball_contact_base(body_pos, body_quat, idx)
            self._append_contact_to_list(ret_index, ret_pos, ret_label, ball_contact)

        sub_box: np.ndarray = np.intersect1d(sub_geom, self.box_geom_arange)
        if len(sub_box) > 0:
            sub_box.sort()
            idx: np.ndarray = sub_box - self.box_geom_offset
            box_contact = self.compute_box_contact_base(body_pos, body_quat, idx)
            self._append_contact_to_list(ret_index, ret_pos, ret_label, box_contact)

        sub_capsule: np.ndarray = np.intersect1d(sub_geom, self.capsule_geom_arange)
        if len(sub_capsule) > 0:
            sub_capsule.sort()
            idx: np.ndarray = sub_capsule - self.capsule_geom_offset
            capsule_contact = self.compute_capsule_contact_base(body_pos, body_quat, idx)
            self._append_contact_to_list(ret_index, ret_pos, ret_label, capsule_contact)

        if process_heel:
            new_body_pos: np.ndarray = body_pos.copy()
            # Note: here we should also consider the contact label..
            l_heel_min = self.compute_ball_min_height(body_pos, body_quat, np.array(l_heel_geom) - self.ball_geom_offset) + 1e-8
            r_heel_min = self.compute_ball_min_height(body_pos, body_quat, np.array(r_heel_geom) - self.ball_geom_offset) + 1e-8
            root_min = self.compute_ball_min_height(body_pos, body_quat, np.array(root_geom) - self.ball_geom_offset) + 1e-8
            new_body_pos[self.l_heel_body_index, 1] -= l_heel_min
            new_body_pos[self.r_heel_body_index, 1] -= r_heel_min
            new_body_pos[self.root_body_index, 1] -= root_min

            # here we should do real collision detection..
            # we can do collsion detection for all ball geoms, and use the subset for left heel and right heel
            real_ball_contact = self.compute_ball_contact(new_body_pos, body_quat)
            # Note: we moved the body to floor before, so there will always be collisions here.
            if real_ball_contact is None:
                raise ValueError("Actually, this will not happen in my code.")
            if process_l_heel:  # here we should move the contact position back to original position..
                heel_subset: np.ndarray = np.where(real_ball_contact[0] == self.l_heel_body_index)[0]
                if len(heel_subset) > 0:
                    sub_idx, sub_pos, sub_label = [real_ball_contact[ti][heel_subset] for ti in range(3)]
                    sub_pos[..., 1] += l_heel_min
                    self._append_contact_to_list(ret_index, ret_pos, ret_label, (sub_idx, sub_pos, sub_label))

            if process_r_heel:
                heel_subset: np.ndarray = np.where(real_ball_contact[0] == self.r_heel_body_index)[0]
                if len(heel_subset) > 0:
                    sub_idx, sub_pos, sub_label = [real_ball_contact[ti][heel_subset] for ti in range(3)]
                    sub_pos[..., 1] += r_heel_min
                    self._append_contact_to_list(ret_index, ret_pos, ret_label, (sub_idx, sub_pos, sub_label))

            if process_root:
                root_subset: np.ndarray = np.where(real_ball_contact[0] == self.root_body_index)[0]
                if len(root_subset) > 0:
                    sub_idx, sub_pos, sub_label = [real_ball_contact[ti][root_subset] for ti in range(3)]
                    sub_pos[..., 1] += root_min
                    self._append_contact_to_list(ret_index, ret_pos, ret_label, (sub_idx, sub_pos, sub_label))

            # reset the original body geom index..
            self.body_to_geom[self.l_heel_body_index] = l_heel_geom
            self.body_to_geom[self.r_heel_body_index] = r_heel_geom
            self.body_to_geom[self.root_body_index] = root_geom

        if len(ret_index) == 0:
            return None

        ret_index_arr: np.ndarray = np.concatenate(ret_index, axis=0)
        ret_pos_arr: np.ndarray = np.concatenate(ret_pos, axis=0)
        ret_label_arr: np.ndarray = np.concatenate(ret_label, axis=0)

        return ret_index_arr, ret_pos_arr, ret_label_arr

    def compute_contact(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        compute contact with ball, capsule, and box
        """
        ret_index, ret_pos, ret_label = [], [], []
        box_contact = self.compute_box_contact(body_pos, body_quat)
        if box_contact is not None:
            self._append_contact_to_list(ret_index, ret_pos, ret_label, box_contact)

        ball_contact = self.compute_ball_contact(body_pos, body_quat)
        if ball_contact is not None:
            self._append_contact_to_list(ret_index, ret_pos, ret_label, ball_contact)

        capsule_contact = self.compute_capsule_contact(body_pos, body_quat)
        if capsule_contact is not None:
            self._append_contact_to_list(ret_index, ret_pos, ret_label, capsule_contact)

        if len(ret_index) == 0:
            return None

        ret_index_arr: np.ndarray = np.concatenate(ret_index, axis=0)
        ret_pos_arr: np.ndarray = np.concatenate(ret_pos, axis=0)
        ret_label_arr: np.ndarray = np.concatenate(ret_label, axis=0)

        return ret_index_arr, ret_pos_arr, ret_label_arr

    def compute_contact_by_label(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        body_label: np.ndarray,
        contact_eps: float = 0.6,
        no_contact_height: Optional[float] = None
    ):
        """
        As neural network will predict contact 0-1 label,
        we need to generate contact by 0-1 label.

        To achieve code reuse, we can compute body position by contact label.
        That is, if the contact label is larger than contact_eps, we will create a contact point.

        we need to consider different geometry type..

        The result is in global coordinate

        We should also consider that if the body height >= eps, we should remove the contact here..
        The eps may be large..
        """
        assert body_label.shape == (self.n_bodies,)
        body_index: np.ndarray = np.where(body_label > contact_eps)[0]
        if len(body_index) == 0:
            return None
        # here we should consider that: if body height >= no_contact_height, we should not create hacked contact here.
        if no_contact_height is not None:
            select_index: np.ndarray = np.where(body_pos[body_index, 1] <= no_contact_height)[0]
            if select_index is None:
                return None
            body_index: np.ndarray = body_index[select_index]
            # print(f"body index = {body_index}")  # for debug here..
        return self.compute_contact_by_sub_body(body_pos, body_quat, body_index)

    def extend_contact(
        self,
        contact_label: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        get full contact flag..
        although this requires more memory, but it will cost less time..
        """
        result: np.ndarray = np.zeros(self.max_contact_num, dtype=np.int32)
        if contact_label is not None:
            result[contact_label] = 1
        return result

    def compute_0_label(
        self,
        contact_label: Optional[np.ndarray]
    ) -> np.ndarray:
        result: np.ndarray = np.ones(self.max_contact_num, dtype=np.int32)
        if contact_label is None:
            result[contact_label] = 0
        return result

    def contact_0_label(
        self,
        contact_label: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        result: Optional[np.ndarray] = np.arange(0, self.max_contact_num, dtype=np.int32)
        if contact_label is not None:
            result: Optional[np.ndarray] = np.setdiff1d(result, contact_label)
        if len(result) == 0:
            result: Optional[np.ndarray] = None
        return result

    @staticmethod
    def smooth_contact_mess(contact_mess: List[List[int]]) -> List[List[int]]:
        # Note: we should smooth the contact for multiple frames..
        contacts = [np.asarray(node, dtype=np.int32) if node is not None else np.array([], dtype=np.int32) for node in contact_mess]
        num_frames: int = len(contacts)
        # insert for gap 1 frame
        for frame in range(1, num_frames - 1):
            common1: np.ndarray = np.intersect1d(contacts[frame - 1], contacts[frame + 1])
            insert1: np.ndarray = common1[np.where(~np.in1d(common1, contacts[frame]))[0]]
            contacts[frame] = np.sort(np.union1d(contacts[frame], insert1))

        # insert for gap 2 frame.
        # That is, when contact exists at time t-1, not exists at t and t+1, but exists at t + 2
        # we should create contact..
        for frame in range(1, num_frames - 2):
            common = np.intersect1d(contacts[frame - 1], contacts[frame + 2])
            if len(common) == 0:
                continue
            contacts[frame] = np.union1d(contacts[frame], common)
            contacts[frame + 1] = np.union1d(contacts[frame + 1], common)

        return [node.tolist() for node in contacts]

    def smooth_contact(
        self,
        body_index_list: List[Optional[np.ndarray]],
        contact_pos_list: List[Optional[np.ndarray]],
        contact_label_list: List[Optional[np.ndarray]],
        enable_insert_two: bool = True,
        remove_orphan_contact: bool = True,
        tar_set: Optional[SetTargetToCharacter] = None,
        debug_print: bool = False
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        - if collision not occurs at time t, but occurs at time t-1 and t+1,
        we can add a contact point. the position can be the average position of t-1 and t+1

        - if collision not occurs at time t and t+1, but occurs at time t-1 and t+2,
        we can add a contact point. the position can be linearly interpolated by time t-1 and t+2

        - if collision occurs at time t, but not occurs at time t-1 and t+1,
        we can remove this contact point

        for loop in python is too slow.. maybe I can rewrite it in cython..
        We can also visualize using Long Ge's framework
        """
        num_frame: int = len(contact_label_list)
        assert len(contact_pos_list) == num_frame and len(contact_label_list) == num_frame
        contact_0_label: List[Optional[np.ndarray]] = [self.contact_0_label(node) for node in contact_label_list]

        print(f"Start add contact")

        def add_contact_multi(num_center: int):
            for frame in range(1, num_frame - num_center):
                if tar_set is not None:  # Visualize using Long Ge's framework
                    tar_set.set_character_byframe(frame)

                one_prev_frame: Optional[np.ndarray] = contact_label_list[frame - 1]
                one_next_n_frame: Optional[np.ndarray] = contact_label_list[frame + num_center]

                def need_continue():
                    if one_prev_frame is None or one_next_n_frame is None:
                        return True
                    for idx in range(num_center):
                        if contact_label_list[frame + idx] is None:
                            return True
                    return False

                if need_continue():
                    continue

                common_index: np.ndarray = np.intersect1d(one_prev_frame, one_next_n_frame)
                for idx in range(num_center):
                    common_index: np.ndarray = np.intersect1d(common_index, contact_0_label[frame + idx])

                if len(common_index) == 0:  # common label.
                    continue

                # here we should create contact, and the position can be the average of prev and next frame.
                if debug_print:
                    print(f"insert contact at {frame}, the common index are {common_index}")
                prev_subset: np.ndarray = np.where(np.in1d(one_prev_frame, common_index))[0]
                prev_pos: np.ndarray = contact_pos_list[frame - 1][prev_subset]

                next_n_subset: np.ndarray = np.where(np.in1d(one_next_n_frame, common_index))[0]
                next_n_pos: np.ndarray = contact_pos_list[frame + num_center][next_n_subset]

                err: np.ndarray = np.linalg.norm(prev_pos - next_n_pos)
                if err > 0.1:
                    if debug_print:
                        print(f"contact position doesn't match, err = {err:.5f}, ignore..")
                    continue

                new_body_index: np.ndarray = body_index_list[frame - 1][prev_subset]

                for idx in range(0, num_center):
                    # insert position into current array..
                    avg_pos: np.ndarray = ((num_center - 1.0 - idx) / num_center) * prev_pos + ((1.0 + idx) / num_center) * next_n_pos
                    curr_pos: Optional[np.ndarray] = contact_pos_list[frame + idx]
                    if curr_pos is None:
                        contact_pos_list[frame + idx] = avg_pos
                    else:
                        contact_pos_list[frame + idx] = np.concatenate([curr_pos, avg_pos], axis=0)

                    # insert body index into current array
                    curr_body_index: Optional[np.ndarray] = body_index_list[frame + idx]
                    if curr_body_index is None:
                        body_index_list[frame + idx] = new_body_index
                    else:
                        body_index_list[frame + idx] = np.concatenate([curr_body_index, new_body_index], axis=0)

                    # insert contact label into current array
                    curr_one_frame: Optional[np.ndarray] = contact_label_list[frame + idx]
                    if curr_one_frame is None:
                        contact_label_list[frame + idx] = common_index
                    else:
                        contact_label_list[frame + idx] = np.concatenate([curr_one_frame, common_index], axis=0)

                    # modify the 0 contact label..
                    curr_zero_frame: Optional[np.ndarray] = contact_0_label[frame + idx]
                    if curr_zero_frame is None:
                        raise ValueError("Actually, this will not happen in my code.")
                    contact_0_label[frame + idx] = np.setdiff1d(curr_zero_frame, common_index)

        def remove_contact():
            print(f"\nStart remove orphan contacts")
            for frame in range(1, num_frame - 1):
                if tar_set is not None:  # Visualize using Long Ge's framework
                    tar_set.set_character_byframe(frame)

                one_frame: Optional[np.ndarray] = contact_label_list[frame]
                zero_prev_frame: Optional[np.ndarray] = contact_0_label[frame - 1]
                zero_next_frame: Optional[np.ndarray] = contact_0_label[frame + 1]
                if one_frame is None or zero_prev_frame is None or zero_next_frame is None:
                    continue
                common_index: np.ndarray = np.intersect1d(one_frame, zero_prev_frame)
                common_index: np.ndarray = np.intersect1d(common_index, zero_next_frame)
                if len(common_index) == 0:
                    continue

                if debug_print:
                    print(f"At frame {frame}, remove orphan contact at {common_index}")
                # compute the index.
                keep_place: np.ndarray = np.where(~(np.in1d(one_frame, common_index)))[0]
                if len(keep_place) == 0:
                    body_index_list[frame] = None
                    contact_pos_list[frame] = None
                    contact_label_list[frame] = None
                    # here we should also maintain the zero index..
                    contact_0_label[frame] = np.arange(0, self.max_contact_num, dtype=np.int32)
                else:
                    body_index_list[frame] = body_index_list[frame][keep_place]
                    contact_pos_list[frame] = contact_pos_list[frame][keep_place]
                    contact_label_list[frame] = contact_label_list[frame][keep_place]
                    contact_0_label[frame] = np.union1d(contact_0_label[frame], common_index)

        add_contact_multi(1)
        # if collision not occurs at time t and t+1, but occurs at time t-1 and t+2,
        # we can add a contact point. the position can be linearly interpolated by time t-1 and t+2
        if enable_insert_two:
            add_contact_multi(2)

        # for robust, we can also consider
        if remove_orphan_contact:
            remove_contact()

        print(f"After smooth the contact sequence")
        # contact body index, contact position (global), contact label
        return body_index_list, contact_pos_list, contact_label_list

    def handle_mocap(
        self,
        target: Union[str, MotionData, TargetPose],
        do_smooth: bool = False,
        body_contact_label: Optional[np.ndarray] = None,
        contact_eps: float = 0.6,
        no_contact_height: Optional[float] = None
    ):
        """
        return: body_index_list, contact_pos_list, contact_label_list
        """
        character: ODECharacter = self.character

        if isinstance(target, str):
            target: MotionData = BVHLoader.load(target)
        if isinstance(target, MotionData):
            target: TargetPose = BVHToTargetBase(target, target.fps, character).init_target()

        set_tar: SetTargetToCharacter = SetTargetToCharacter(character, target)
        body_index_list, contact_pos_list, contact_label_list = [], [], []
        num_frame = target.num_frames if body_contact_label is None else min(target.num_frames, body_contact_label.shape[0])

        for frame in range(num_frame):
            set_tar.set_character_byframe(frame)
            body_pos, body_quat = character.body_info.get_body_pos(), character.body_info.get_body_quat()
            if body_contact_label is None:
                contact_res = self.compute_contact(body_pos, body_quat)
            else:
                contact_res = self.compute_contact_by_label(body_pos, body_quat, body_contact_label[frame], contact_eps, no_contact_height)
            if contact_res is None:
                contact_res = None, None, None
            body_index_list.append(contact_res[0])
            contact_pos_list.append(contact_res[1])
            contact_label_list.append(contact_res[2])

        # here we should also smooth the input contact sequence
        if do_smooth:
            return self.smooth_contact(body_index_list, contact_pos_list, contact_label_list, True, True, set_tar)
        else:
            return body_index_list, contact_pos_list, contact_label_list

    def create_ode_contact_joint(
        self,
        contact_pos: np.ndarray,
        contact_label: np.ndarray,
        contact_depth: float = 0.0,
        save_contact_info: bool = True
    ) -> Optional[ode.JointGroup]:
        self.scene.contact.empty()
        if contact_label is None or len(contact_label) == 0:
            return None
        floor_geom: ode.GeomObject = self.scene.floor
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # here we should record the contact normal, position

        if save_contact_info:
            self.scene.contact_info = SceneContactInfo([], [], [], [])

        for index, contact_place in enumerate(contact_label):
            geom: ode.GeomObject = self.contact_index_to_geom(contact_place)  # here has bug...
            ode_contact: ode.Contact = ode.Contact()
            ode_contact.contactPosNumpy = contact_pos[index]
            ode_contact.contactNormalNumpy = up_vec
            ode_contact.contactDepth = contact_depth  # we doesn't requires accurate depth.
            ode_contact.contactGeom1 = geom
            ode_contact.contactGeom2 = floor_geom
            self.scene._generate_contact_joint(geom, floor_geom, [ode_contact])
            # print(f"contact_place = {contact_place}, body id = {geom.body.instance_id}")

            if save_contact_info:
                self.scene.contact_info.pos.append(contact_pos[index])
                self.scene.contact_info.force.append(up_vec)
                self.scene.contact_info.geom1_name.append(geom.name)
                self.scene.contact_info.geom2_name.append(floor_geom.name)

        self.scene.contact_info.pos = np.concatenate([node[None, :] for node in self.scene.contact_info.pos], axis=0)
        self.scene.contact_info.force = np.concatenate([node[None, :] for node in self.scene.contact_info.force], axis=0)
        return self.scene.contact

    def convert_to_diff_contact_single(self, body_index: np.ndarray, contact_pos: np.ndarray, contact_label: np.ndarray) -> DiffContactInfo:
        num_joints: int = len(self.character.joints)
        self.create_ode_contact_joint(contact_pos, contact_label)
        ret: DiffContactInfo = DiffContactInfo.contact_extract(self.scene.contact.joints, num_joints, ContactType.MAX_FORCE_ODE_LCP, True)
        self.scene.contact.empty()
        return ret

    def convert_to_diff_contact(
        self,
        body_index_list: List,
        contact_pos_list: List,
        contact_label_list: List,
        target_pose: Union[TargetPose, MotionData]
    ) -> List[Optional[DiffContactInfo]]:
        """
        convert the pre computed contact sequence as List[DiffContactInfo] for each frame
        Note: we should set the character to mocap data..
        """
        if isinstance(target_pose, MotionData):
            target_pose = BVHToTargetBase(target_pose).init_target()
        set_tar = SetTargetToCharacter(self.character, target_pose)
        num_frame: int = len(body_index_list)
        result = []
        num_joints: int = len(self.character.joints)
        for frame in range(num_frame):
            # body_index = body_index_list[frame]
            contact_pos = contact_pos_list[frame]
            contact_label = contact_label_list[frame]
            if contact_pos is None:
                result.append(None)
                continue

            # create contact joint. Note: there are only contact between floor and bodies
            # we can create contact joint in ode, and then destroy them..
            set_tar.set_character_byframe(frame)
            self.create_ode_contact_joint(contact_pos, contact_label)
            ret = DiffContactInfo.contact_extract(self.scene.contact.joints, num_joints, ContactType.MAX_FORCE_ODE_LCP, True)
            result.append(ret)

        self.scene.contact.empty()
        return result

    def preprocess_contact_mess(self, contact_mess: List[List[int]], target_pose: TargetPose) -> List[List[int]]:
        set_tar = SetTargetToCharacter(self.character, target_pose)
        num_frame = len(contact_mess)
        num_body = self.n_bodies
        result: List[List[int]] = []
        for frame in range(num_frame):
            mess = contact_mess[frame]
            if mess is None:
                result.append(None)
                continue
            set_tar.set_character_byframe(frame)
            # here we can compute the delta height pair..
            body_h: np.ndarray = self.character.get_body_pos()[:, 1]
            delta_h: np.ndarray = np.asarray(np.tile(body_h, (num_body, 1)) - np.tile(body_h, (num_body, 1)).T)
            curr_label = np.zeros(num_body, dtype=np.int32)
            curr_label[mess] = 1
            body_h_list = [(index, height) for index, height in enumerate(body_h)]
            body_h_list.sort(key=lambda x: x[1])

            new_contact_mess: np.ndarray = np.zeros(num_body, dtype=np.int32)
            min_h = body_h_list[0][1]
            for body_index, body_height in body_h_list:
                if body_height > min_h + 0.2:
                    break
                if curr_label[body_index] == 0:
                    continue

                # here we should avoid the case that:
                # both parent and child contains contact, but height distance of parent and child is large..
                neighbour = np.array(self.body_neighbour[body_index])
                neighbour_dh: np.ndarray = -delta_h[body_index, neighbour]
                less_idx = np.where(neighbour_dh >= 0)[0]
                if len(less_idx) > 0:
                    less_height = neighbour_dh[less_idx]
                    less_h_body = neighbour[less_idx]
                    less_ratio = less_height / self.body_distance[body_index, less_h_body]
                    contact_body: np.ndarray = less_h_body[less_ratio >= 0.3]
                    if len(contact_body) > 0:
                        # here we need also judge the contact velocity..
                        # maybe when velocity is large, we should not add contact..
                        new_contact_mess[contact_body] = 1
                        continue
                else:
                    pass
                new_contact_mess[body_index] = 1
            result.append(np.where(new_contact_mess == 1)[0].tolist())

        return result


def test():
    """
    we can test contact sequence by bvh mocap data
    or, maybe we should visualize by Unity..?
    That is, do collision detection by kinematic state
    """
    scene_fname = os.path.join(fdir, "../../../Tests/CharacterData/Samcon-Human.pickle")
    scene: ODEScene = JsonSceneLoader().load_from_pickle_file(scene_fname)
    scene.set_sim_fps(100)
    character: ODECharacter = scene.character0
    extracter: ContactLabelExtractor = ContactLabelExtractor(scene, character)

    mocap_fname: str = os.path.join(fdir, "../../../Tests/CharacterData/lafan-mocap-100/fallAndGetUp1_subject1.bvh")
    mocap: MotionData = BVHLoader.load(mocap_fname)
    # target: TargetPose = BVHToTargetBase(mocap, scene.sim_fps, character).init_target()
    # set_tar = SetTargetToCharacter(character, target)
    # here we can visualize using Long Ge's framework
    render = RenderWorld(scene)
    render.start()
    # frame = 900
    # set_tar.set_character_byframe(frame)
    # extracter.compute_contact(character.body_info.get_body_pos(), character.body_info.get_body_quat())

    extracter.handle_mocap(mocap)


if __name__ == "__main__":
    test()

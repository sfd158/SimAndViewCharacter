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
from typing import Optional, Union, List
from ..ODESim.ODECharacter import ODECharacter
from ..ODESim.CharacterWrapper import CharacterWrapper


class Joint2dComSimple(CharacterWrapper):
    """
    we can compute the com by averaging of (left shoulder, right shoulder, hip position..)
    This only works for std-human model
    """
    left_shoulder_name = ""
    right_shoulder_name = ""

    def __init__(self, character: ODECharacter) -> None:
        super().__init__(character)
        assert not self.joint_info.has_root
        joint_names = ["RootJoint"] + self.joint_info.joint_names()
        self.hip_index = 0  # assume hip is character root body
        self.left_shoulder = joint_names.index(self.left_shoulder_name)
        self.right_shoulder = joint_names.index(self.right_shoulder_name)
        self.select_index = [self.hip_index, self.left_shoulder, self.right_shoulder]
        self.select_index.sort()

    def calc(self, joint_pos_2d: np.ndarray):
        result: np.ndarray = np.mean(joint_pos_2d[..., self.select_index, :], axis=-2)
        return result



class Joint2dComIgnoreHandToe:
    """
    as we cannot estimate hand and toe position only use 2d joint information,
    and mass of hand/toe is small,
    we can ignore hand and toe.
    """
    _debug = False

    def __init__(self):
        self.curr_joint_in_use: Optional[np.ndarray] = None
        self.curr_parent_in_use: Optional[np.ndarray] = None
        self.mass_in_use: Optional[np.ndarray] = None
        self.total_mass: Union[np.ndarray, float, None] = None

    def build(self, character: ODECharacter):
        assert not character.joint_info.has_root
        # note:
        # the parent joint exists, the body position can be computed by (parent index + current index)
        # we need also get body mass.

        # as there is no root joint, root body position can be directly get.
        joint_names: List[str] = ["RootJoint"] + character.joint_info.joint_names()

        parent_joint = np.array([-1] + [node + 1 for node in character.joint_info.pa_joint_id])
        tot_joint: int = parent_joint.size
        root_as_parent: np.ndarray = np.where(parent_joint == 0)[0]
        joint_in_use: np.ndarray = np.arange(1, tot_joint, dtype=np.int32)

        # hand body and toe body will be ignored. however, as their mass is little, so they can be ignored.
        self.curr_joint_in_use: np.ndarray = np.setdiff1d(joint_in_use, root_as_parent)
        self.curr_parent_in_use: np.ndarray = parent_joint[self.curr_joint_in_use]

        # if parent joint == 0, as the root body position is considered, we need not to compute position between this joint and parent
        self.curr_joint_in_use: np.ndarray = np.concatenate([np.array([0]), self.curr_joint_in_use], dtype=np.int32)
        self.curr_parent_in_use: np.ndarray = np.concatenate([np.array([0]), self.curr_parent_in_use], dtype=np.int32)

        # as there are more than 1 branchs of some bodies
        parent_body_index: np.ndarray = np.concatenate([np.array([0]), character.joint_info.parent_body_index], dtype=np.int32)  # (total joint,)
        parent_body_in_use: np.ndarray = parent_body_index[self.curr_joint_in_use]
        total_body_mass: np.ndarray = character.body_info.mass_val.copy()  # (num body,)

        node_count = np.zeros_like(total_body_mass, dtype=np.int32)
        for node in parent_body_in_use:
            node_count[node] += 1
        node_count[node_count == 0] = 1
        total_body_mass /= node_count

        self.mass_in_use: np.ndarray = total_body_mass[parent_body_in_use].copy()  # (num)
        self.total_mass = np.sum(self.mass_in_use)

        # we need to check if body and joint name are correct.
        if self._debug:
            body_names = character.body_info.get_name_list()
            for count_index, joint_index in enumerate(self.curr_joint_in_use):
                parent_joint_index: int = self.curr_parent_in_use[count_index]
                parent_body_index: int = parent_body_in_use[count_index]
                print(
                    f"joint index = {joint_index}, joint name = {joint_names[joint_index]}, "
                    f"parent index = {parent_joint_index}, parent joint name = {joint_names[parent_joint_index]}, "
                    f"parent body = {parent_body_index}, parent body name = {body_names[parent_body_index]}"
                )
            print(f"Total mass = {self.total_mass:.4f}, original total mass = {character.body_info.sum_mass}")

        return self

    def calc(self, joint_pos_2d: np.ndarray) -> np.ndarray:
        """
        input shape: ..., joint (contains root), 2
        output: com in 2d coordinate
        """
        parent_pos: np.ndarray = joint_pos_2d[..., self.curr_parent_in_use, :]
        child_pos: np.ndarray = joint_pos_2d[..., self.curr_joint_in_use, :]
        body_pos: np.ndarray = 0.5 * (parent_pos + child_pos)
        mass_shape = (1,) * (body_pos.ndim - 2) + (-1, 1)
        mass_in_use: np.ndarray = self.mass_in_use.reshape(mass_shape)
        sum_mass: np.ndarray = np.sum(mass_in_use * body_pos, axis=-2) / self.total_mass
        return sum_mass

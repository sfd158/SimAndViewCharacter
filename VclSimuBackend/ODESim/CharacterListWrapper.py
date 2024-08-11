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

from .ODECharacter import *
from typing import Optional, Union
from copy import copy,deepcopy
from itertools import chain


class CharacterListWrapper:
    """
    Wrapper of ODE Character
    """
    def __init__(self, character_list: List[ODECharacter] = None):
        self.character_list: List[ODECharacter] = character_list

        self.body_info_cache = None
        self.joint_info_cache = None

    @property
    def body_info(self) -> BodyInfo:
        if self.body_info_cache is not None:
            return self.body_info_cache
        body_info = BodyInfo(self.character_list[0].world, self.character_list[0].space)
        body_info: BodyInfo = body_info + copy(self.character_list[0].body_info)
        for character in self.character_list[1:]:
            body_info = body_info + character.body_info
        self.body_info_cache = body_info
        return body_info

    @property
    def joint_info(self) -> JointInfos:
        if self.joint_info_cache is not None:
            return self.joint_info_cache
        info = JointInfos(self.character_list[0].world)
        info = info + self.character_list[0].joint_info
        for character in self.character_list[1:]:
            info = info + character.joint_info
        self.joint_info_cache = info
        return info

    def joint_names(self) -> List[str]:
        return [item for joint in self.joint_info for item in joint.joint_names()]

    @property
    def end_joint(self) -> List[EndJointInfo]:
        return [character.end_joint for character in self.character_list]

    @property
    def world(self) -> ode.World:
        return self.character_list[0].world

    @property
    def space(self) -> ode.SpaceBase:
        return self.character_list[0].space

    @property
    def bodies(self) -> List[ode.Body]:
        return [body for character in self.character_list for body in character.bodies]

    @property
    def joints(self) -> List[Union[ode.Joint, ode.BallJoint, ode.BallJointAmotor, ode.HingeJoint]]:
        return [joint for character in self.character_list for joint in character.joints]

    @property
    def root_body(self) -> ode.Body:
        return self.character_list[0].root_body

    @property
    def root_joint(self) -> Optional[ode.Joint]:
        return self.character_list[0].root_joint

    @property
    def joint_to_child_body(self) -> List[int]:
        return [item for character in self.character_list for item in character.joint_to_child_body]

    @property
    def child_body_to_joint(self) -> List[int]:
        return [item for character in self.character_list for item in character.child_body_to_joint]

    @property
    def joint_to_parent_body(self) -> List[int]:
        return [item for character in self.character_list for item in character.joint_to_parent_body]

    @property
    def has_end_joint(self) -> bool:
        return bool(sum([character.has_end_joint for character in self.character_list]))

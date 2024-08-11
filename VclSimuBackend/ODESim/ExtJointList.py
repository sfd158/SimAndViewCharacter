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

from typing import Optional, List

import ModifyODE as ode
from .JointInfo import JointInfosBase
from .ODECharacter import ODECharacter


class ExtJointInfo:
    def __init__(self, character0_id: int = 0, body0_id: int = 0,
                 character1_id: int = 0, body1_id: int = 0):
        self.character0_id: int = character0_id
        self.body0_id: int = body0_id

        self.character1_id: int = character1_id
        self.body1_id: int = body1_id


# Ext Joint as constraints
class ExtJointList(JointInfosBase):
    def __init__(self, world: ode.World, characters: Optional[List[ODECharacter]] = None):
        super(ExtJointList, self).__init__(world)
        self.characters: Optional[List[ODECharacter]] = characters
        self.infos: List[ExtJointInfo] = []

    def clear(self):
        super(ExtJointList, self).clear()
        self.infos.clear()

        return self

    def attach_joint(self, joint: ode.Joint, info: ExtJointInfo):
        ch0 = [i for i in self.characters if i.character_id == info.character0_id]
        ch1 = [i for i in self.characters if i.character_id == info.character1_id]

        body0: ode.Body = ch0[0].bodies[info.body0_id]
        body1: ode.Body = ch1[0].bodies[info.body1_id]
        joint.attach_ext(body0, body1)

    def append_and_attach(self, joint: ode.Joint, character0_id: int = 0, body0_id: int = 0,
                          character1_id: int = 0, body1_id: int = 0):
        info = ExtJointInfo(character0_id, body0_id, character1_id, body1_id)
        self.joints.append(joint)
        self.infos.append(info)
        self.attach_joint(joint, info)

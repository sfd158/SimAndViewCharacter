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

import ModifyODE as ode
import numpy as np
from typing import Optional, List
from .JointInfo import JointInfos


class JointInfoWrapper:
    """
    Wrapper of JointInfos
    """
    def __init__(self, joint_info: Optional[JointInfos] = None):
        self.joint_info: Optional[JointInfos] = joint_info

    @property
    def world(self) -> ode.World:
        return self.joint_info.world

    @property
    def joints(self) -> List[ode.Joint]:
        return self.joint_info.joints

    @joints.setter
    def joints(self, value: Optional[ode.Joint]):
        self.joint_info.joints = value

    def __len__(self) -> int:
        return len(self.joint_info)

    def joint_names(self) -> List[str]:
        return self.joint_info.joint_names()

    @property
    def pa_joint_id(self) -> List[int]:
        return self.joint_info.pa_joint_id

    @property
    def kps(self) -> Optional[np.ndarray]:
        return self.joint_info.kps

    @property
    def kds(self) -> Optional[np.ndarray]:
        return self.joint_info.kps

    @property
    def euler_axis_local(self) -> Optional[np.ndarray]:
        return self.joint_info.euler_axis_local

    @euler_axis_local.setter
    def euler_axis_local(self, value: np.ndarray):
        self.joint_info.euler_axis_local = value

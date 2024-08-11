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
from typing import List, Optional
import numpy as np
from scipy.spatial.transform import Rotation
from MotionUtils import quat_apply_forward_fast


class EndJointInfo:
    def __init__(self, world: ode.World):
        self.world: ode.World = world
        self.name: List[str] = []
        self.init_global_pos: Optional[np.ndarray] = None
        self.pa_body_id: Optional[np.ndarray] = None  # parent body id. (head, l-r hand, l-r foot)
        self.pa_body_c_id: Optional[np.ndarray] = None

        # Position of joint relative to its parent body in world(global) coordinate
        self.jtob_init_global_pos: Optional[np.ndarray] = None

        # Position of joint relative to its parent body in body coordinate
        self.jtob_init_local_pos: Optional[np.ndarray] = None

        # Parent Joint ID
        self.pa_joint_id: Optional[np.ndarray] = None
        # Parent Joint C ID
        self.pa_joint_c_id: Optional[np.ndarray] = None

        # Positions relative to parent joint in world (global) coordinate
        self.jtoj_init_global_pos: Optional[np.ndarray] = None

        # Position relative to parent joint in pajoint frame
        self.jtoj_init_local_pos: Optional[np.ndarray] = None

        self.weights: Optional[np.ndarray] = None  # weight for calc loss

    def __len__(self) -> int:
        return len(self.name)

    def resize(self):
        self.init_global_pos = np.zeros((len(self), 3))
        self.jtob_init_global_pos = np.zeros((len(self), 3))
        self.jtob_init_local_pos = np.zeros((len(self), 3))
        self.weights = np.ones(len(self))

        return self

    def get_global_pos(self) -> np.ndarray:
        """
        Get End Joint's Global Position
        """
        body_quat: np.ndarray = self.world.getBodyQuatScipy(self.pa_body_c_id).reshape((-1, 4))
        body_pos: np.ndarray = self.world.getBodyPos(self.pa_body_c_id).reshape((-1, 3))
        return quat_apply_forward_fast(body_quat, self.jtob_init_local_pos) + body_pos

    def clear(self):
        self.name.clear()
        self.init_global_pos: Optional[np.ndarray] = None
        self.pa_body_id: Optional[np.ndarray] = None
        self.pa_body_c_id: Optional[np.ndarray] = None
        self.jtob_init_global_pos: Optional[np.ndarray] = None
        self.jtob_init_local_pos: Optional[np.ndarray] = None
        self.pa_joint_id: Optional[np.ndarray] = None
        self.pa_joint_c_id: Optional[np.ndarray] = None
        self.jtoj_init_global_pos: Optional[np.ndarray] = None
        self.jtoj_init_local_pos: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

        return self

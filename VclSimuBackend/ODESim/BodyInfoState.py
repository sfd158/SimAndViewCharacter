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
import numpy as np
from typing import Optional


"""
save the state for rigid bodies
"""
class BodyInfoState:
    __slots__ = ("pos", "rot", "quat", "linear_vel", "angular_vel", "pd_target", "force", "torque") # , "contact_info")

    def __init__(self):
        self.pos: Optional[np.ndarray] = None
        self.rot: Optional[np.ndarray] = None
        self.quat: Optional[np.ndarray] = None
        self.linear_vel: Optional[np.ndarray] = None
        self.angular_vel: Optional[np.ndarray] = None

        self.pd_target: Optional[np.ndarray] = None  # Target pose for pd controller
        self.force: Optional[np.ndarray] = None
        self.torque: Optional[np.ndarray] = None

        # self.contact_info = None

    # reshape the rigid body
    def reshape(self):
        if self.pos is not None:
            self.pos = self.pos.reshape((-1, 3))
        if self.rot is not None:
            self.rot = self.rot.reshape((-1, 3, 3))
        if self.quat is not None:
            self.quat = self.quat.reshape((-1, 4))
        if self.linear_vel is not None:
            self.linear_vel = self.linear_vel.reshape((-1, 3))
        if self.angular_vel is not None:
            self.angular_vel = self.angular_vel.reshape((-1, 3))
        if self.pd_target is not None:
            self.pd_target = self.pd_target.reshape((-1, 4))
        if self.force is not None:
            self.force = self.force.reshape((-1, 3))
        if self.torque is not None:
            self.torque = self.torque.reshape((-1, 3))
        return self

    def set_value(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        quat: np.ndarray,
        linvel: np.ndarray,
        angvel: np.ndarray,
        pd_target: Optional[np.ndarray],
        ):
        self.pos = pos.reshape(-1).astype(np.float64)
        self.rot = rot.reshape(-1).astype(np.float64)
        self.quat = quat.reshape(-1).astype(np.float64)
        self.linear_vel = linvel.reshape(-1).astype(np.float64)
        self.angular_vel = angvel.reshape(-1).astype(np.float64)
        if pd_target is not None:
            self.pd_target = pd_target.astype(np.float64)

        self.to_continuous()
        return self

    def __del__(self):
        del self.pos
        del self.rot
        del self.quat
        del self.linear_vel
        del self.angular_vel
        if self.pd_target is not None:
            del self.pd_target
        if self.force is not None:
            del self.force

    def check_failed(self) -> bool:
        if np.any(np.abs(self.pos) > 10000):
            return True
        if np.any(np.abs(self.rot) > 10000):
            return True
        if np.any(np.abs(self.quat) > 10000):
            return True
        if np.any(np.abs(self.linear_vel) > 10000):
            return True
        if np.any(np.abs(self.angular_vel) > 10000):
            return True
        return False

    def clear(self):
        self.pos: Optional[np.ndarray] = None
        self.rot: Optional[np.ndarray] = None
        self.quat: Optional[np.ndarray] = None
        self.linear_vel: Optional[np.ndarray] = None
        self.angular_vel: Optional[np.ndarray] = None

        self.pd_target: Optional[np.ndarray] = None
        self.force: Optional[np.ndarray] = None
        self.torque: Optional[np.ndarray] = None

        # self.contact_info = None

    def is_empty(self):
        return self.pos is None and self.rot is None \
               and self.quat is None and self.linear_vel is None and self.angular_vel is None

    def __len__(self):
        return self.pos.shape[0] // 3 if self.pos is not None else 0

    def calc_delta(self, o):
        d_pos: np.ndarray = np.max(np.abs(self.pos - o.pos))
        d_rot: np.ndarray = np.max(np.abs(self.rot - o.rot))
        d_quat_ode: np.ndarray = np.max(np.abs(self.quat - o.quat))
        d_lin_vel: np.ndarray = np.max(np.abs(self.linear_vel - o.linear_vel))
        d_ang_vel: np.ndarray = np.max(np.abs(self.angular_vel - o.angular_vel))

        return d_pos, d_rot, d_quat_ode, d_lin_vel, d_ang_vel

    def check_delta(self, o):
        d_pos, d_rot, d_quat_ode, d_lin_vel, d_ang_vel = self.calc_delta(o)

        try:
            assert d_pos == 0
            assert d_rot == 0
            assert d_quat_ode == 0
            assert d_lin_vel == 0
            assert d_ang_vel == 0
        except AssertionError:
            bug_info = (
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                f"dpos = {d_pos}, "
                f"drot = {d_rot}, "
                f"d_quat_ode = {d_quat_ode}, "
                f"d_lin_vel = {d_lin_vel}, "
                f"d_ang_vel = {d_ang_vel}"
            )
            import logging
            logging.info(bug_info)
            print(bug_info)

    def save(self, world: ode.World, body_c_id: np.ndarray):
        _, self.pos, self.quat, self.rot, self.linear_vel, self.angular_vel = world.getBodyInfos(body_c_id)

    def load(self, world: ode.World, body_c_id: np.ndarray):
        world.loadBodyInfos(body_c_id, self.pos, self.quat, self.rot, self.linear_vel, self.angular_vel,
                            self.force, self.torque)

    def copy(self):
        
        res = BodyInfoState()
        if self.pos is not None:
            res.pos = self.pos.copy()
        if self.rot is not None:
            res.rot = self.rot.copy()
        if self.quat is not None:
            res.quat = self.quat.copy()
        if self.linear_vel is not None:
            res.linear_vel = self.linear_vel.copy()
        if self.angular_vel is not None:
            res.angular_vel = self.angular_vel.copy()
        if self.pd_target is not None:
            res.pd_target = self.pd_target.copy()
        if self.force is not None:
            res.force = self.force.copy()
        if self.torque is not None:
            res.torque = self.torque.copy()
        return res

    def to_continuous(self):
        self.pos = np.ascontiguousarray(self.pos)
        self.rot = np.ascontiguousarray(self.rot)
        self.quat = np.ascontiguousarray(self.quat)
        self.linear_vel = np.ascontiguousarray(self.linear_vel)
        self.angular_vel = np.ascontiguousarray(self.angular_vel)

        if self.pd_target is not None:
            self.pd_target = np.ascontiguousarray(self.pd_target)

        if self.force is not None:
            self.force = np.ascontiguousarray(self.force)

        if self.torque is not None:
            self.torque = np.ascontiguousarray(self.torque)

        # if self.contact_info is not None:
        #    pass

        return self

    def cat_to_ndarray(self) -> np.ndarray:
        return np.concatenate([self.pos.reshape(-1), self.rot.reshape(-1), self.quat.reshape(-1), self.linear_vel.reshape(-1), self.angular_vel.reshape(-1)])

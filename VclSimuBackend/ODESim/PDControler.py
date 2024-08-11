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
from scipy.spatial.transform import Rotation
from typing import Optional
from .JointInfoWrapper import JointInfos
from .ODECharacter import ODECharacter
from ..Common.MathHelper import MathHelper


class PDControlerBase:
    def __init__(self, joint_info: JointInfos):
        self.tor_lim = joint_info.torque_limit[:, np.newaxis]
        self.kps = joint_info.kps[:, None]
        self.kds = joint_info.kds[:, None]
        self.world = joint_info.world
        self.joint_info = joint_info
        self.cache_global_torque: Optional[np.ndarray] = None
        self.cache_local_torque: Optional[np.ndarray] = None

    def _add_local_torque(self, parent_qs: np.ndarray, local_torques: np.ndarray) -> np.ndarray:
        """
        param: parent_qs: parent bodies' quaternion in global coordinate
        """
        global_torque: np.ndarray = Rotation(parent_qs, False, False).apply(local_torques)
        self.cache_global_torque = global_torque
        self.cache_local_torque = local_torques
        self.world.add_global_torque(global_torque, self.joint_info.parent_body_c_id, self.joint_info.child_body_c_id)
        return global_torque

    def _add_clipped_torque(self, parent_qs: np.ndarray, local_torques: np.ndarray) -> np.ndarray:
        """
        Clip torque to avoid Numerical explosion.
        Param:
        parent_qs: parent bodies' quaternion in global coordinate
        local_torques: torques added to joints in parent local coordinate
        """
        tor_len = np.linalg.norm(local_torques, axis=-1, keepdims=True)
        tor_len[tor_len < 1e-10] = 1
        ratio = np.clip(tor_len, -self.tor_lim, self.tor_lim)

        new_local_torque = (local_torques / tor_len) * ratio
        return self._add_local_torque(parent_qs, new_local_torque)

    def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# For ode.World.dampedStep
class DampedPDControlerSlow(PDControlerBase):
    def __init__(self, joint_info: JointInfos):
        super(DampedPDControlerSlow, self).__init__(joint_info)

    def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
        """
        Param:
        tar_local_qs: target joints' quaternion in parent local coordinate
        """
        parent_qs, child_qs, local_qs, parent_qs_inv = self.joint_info.get_parent_child_qs()

        delta_local_tar_now = Rotation(tar_local_qs, False, False) * Rotation(local_qs, False, False).inv()
        local_torques: np.ndarray = self.kps * delta_local_tar_now.as_rotvec()

        ret = self._add_clipped_torque(parent_qs, local_torques)

        # test C++ version of pd controller
        pd_ret = self.world.get_pd_control_torque(self.joint_info.joint_c_id, tar_local_qs, self.kps.reshape(-1), self.tor_lim.reshape(-1))
        c_local_torque, c_global_torque = pd_ret
        print("delta local", np.max(np.abs(c_local_torque - self.cache_local_torque)))
        print("delta global", np.max(np.abs(c_global_torque - self.cache_global_torque)))
        print()
        return ret


class DampedPDControler:
    """
    using stable PD control.
    Please refer to [Liu et al. 2013 Simulation and Control of Skeleton-driven Soft Body Characters] for details
    """
    def __init__(self, character: ODECharacter):
        self.character = character
        joint_info = character.joint_info
        self.joint_c_id: np.ndarray = joint_info.joint_c_id
        self.tor_lim = joint_info.torque_limit.flatten()
        self.kps = joint_info.kps.flatten()
        self.world = joint_info.world
        self.joint_info = joint_info
        self.cache_global_torque: Optional[np.ndarray] = None
        self.cache_local_torque: Optional[np.ndarray] = None

    def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
        """
        return: 
        c_global_torque, global torque of each rigid body
        """
        c_local_torque, c_global_torque = self.world.get_pd_control_torque(self.joint_c_id, tar_local_qs, self.kps, self.tor_lim)
        self.cache_global_torque = c_global_torque
        self.cache_local_torque = c_local_torque
        # save the torque.
        self.world.add_global_torque(c_global_torque, self.joint_info.parent_body_c_id, self.joint_info.child_body_c_id)
        return c_global_torque

    def add_torques_by_quat_with_kp(self, tar_local_qs: np.ndarray, kp_ratio: np.ndarray) -> np.ndarray:
        """
        Adjust kp param.
        """
        c_local_torque, c_global_torque = self.world.get_pd_control_torque(
            self.joint_c_id, tar_local_qs, self.kps * kp_ratio, self.tor_lim)
        self.cache_global_torque = c_global_torque
        self.cache_local_torque = c_local_torque
        self.world.add_global_torque(c_global_torque, self.joint_info.parent_body_c_id, self.joint_info.child_body_c_id)
        return c_global_torque


# For ode.World.step
class PDControler(PDControlerBase):
    def __init__(self, joint_info: JointInfos):
        super(PDControler, self).__init__(joint_info)

    def add_torques_by_quat(self, tar_local_qs: np.ndarray) -> np.ndarray:
        parent_qs, child_qs, local_qs, parent_qs_inv = self.joint_info.get_parent_child_qs()

        delta_local_tar_now = MathHelper.flip_quat_by_w(
            Rotation(tar_local_qs, normalize=False, copy=False) *
            (Rotation(local_qs, normalize=False, copy=False).inv()).as_quat())

        local_torques: np.ndarray = self.kps * Rotation(delta_local_tar_now, copy=False).as_rotvec() - \
                                    self.kds * self.joint_info.get_local_angvels(parent_qs_inv)

        return self._add_clipped_torque(parent_qs, local_torques)

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

from typing import List, Optional
import numpy as np
import ModifyODE as ode
from scipy.spatial.transform import Rotation

from .CharacterWrapper import CharacterWrapper, ODECharacter
from .CharacterListWrapper import CharacterListWrapper

class JointExportInfoBase:
    __slots__ = (
        "joint_idx", "body1_idx", "body2_idx", "joint_erp", "body1_r", "body2_r", "inv_mass1",
        "inv_mass2", "inv_inertia1", "inv_inertia2", "rel_quat", "rel_quat_inv", "body_offset",
        "joint_offset"
    )

    def __init__(self):
        self.joint_idx: Optional[np.ndarray] = None
        self.body1_idx: Optional[np.ndarray] = None
        self.body2_idx: Optional[np.ndarray] = None
        self.joint_erp: Optional[np.ndarray] = None
        self.body1_r: Optional[np.ndarray] = None
        self.body2_r: Optional[np.ndarray] = None

        self.inv_mass1: Optional[np.ndarray] = None
        self.inv_mass2: Optional[np.ndarray] = None
        self.inv_inertia1: Optional[np.ndarray] = None
        self.inv_inertia2: Optional[np.ndarray] = None

        self.rel_quat: Optional[np.ndarray] = None
        self.rel_quat_inv: Optional[np.ndarray] = None

        self.body_offset = 0
        self.joint_offset = 0

    def __add__(self, rhs):

        # first combine joint_idx, it should have offset
        j_len = self.joint_offset
        self.joint_idx = np.concatenate([self.joint_idx, rhs.joint_idx + j_len], axis = 0)
        # then we should offset body idx.....
        b_len = rhs.body_offset
        tmp = rhs.body1_idx.copy()
        tmp[tmp != -1] += b_len
        self.body1_idx = np.concatenate([self.body1_idx, tmp], axis = 0)
        tmp = rhs.body2_idx.copy()
        tmp[tmp != -1] += b_len
        self.body2_idx = np.concatenate([self.body2_idx, tmp], axis = 0)

        # some other properties
        self.joint_erp = np.concatenate([self.joint_erp, rhs.joint_erp], axis = 0)
        self.body1_r = np.concatenate([self.body1_r, rhs.body1_r], axis = 0)
        self.body2_r = np.concatenate([self.body2_r, rhs.body2_r], axis = 0)
        self.inv_mass1 = np.concatenate([self.inv_mass1, rhs.inv_mass1], axis = 0)
        self.inv_mass2 = np.concatenate([self.inv_mass2, rhs.inv_mass2], axis = 0)
        self.inv_inertia1 = np.concatenate([self.inv_inertia1, rhs.inv_inertia1], axis = 0)
        self.inv_inertia2 = np.concatenate([self.inv_inertia2, rhs.inv_inertia2], axis = 0)
        self.rel_quat = np.concatenate([self.rel_quat, rhs.rel_quat], axis = 0)
        self.rel_quat_inv = np.concatenate([self.rel_quat_inv, rhs.rel_quat_inv], axis = 0)
        return self

class HingeExportInfo(JointExportInfoBase):
    __slots__ = (
        "joint_idx", "body1_idx", "body2_idx", "joint_erp", "body1_r", "body2_r", "inv_mass1",
        "inv_mass2", "inv_inertia1", "inv_inertia2", "rel_quat", "rel_quat_inv",
        "raw_axis1", "raw_axis2", "limit_lo", "limit_hi")

    def __init__(self):
        super(HingeExportInfo, self).__init__()
        self.raw_axis1: Optional[np.ndarray] = None
        self.raw_axis2: Optional[np.ndarray] = None

        self.limit_lo: Optional[np.ndarray] = None  # hinge lo limit
        self.limit_hi: Optional[np.ndarray] = None  # hinge hi limit

    def __add__(self, other):
        super(HingeExportInfo, self).__add__(other)
        self.raw_axis1 = np.concatenate([self.raw_axis1, other.raw_axis1], axis=0)
        self.raw_axis2 = np.concatenate([self.raw_axis2, other.raw_axis2], axis=0)

        self.limit_lo = np.concatenate([self.limit_lo, other.limit_lo], axis=0)
        self.limit_hi = np.concatenate([self.limit_hi, other.limit_hi], axis=0)
        return self


class BallExportInfo(JointExportInfoBase):
    __slots__ = ("joint_idx", "body1_idx", "body2_idx", "joint_erp", "body1_r", "body2_r", "inv_mass1",
                 "inv_mass2", "inv_inertia1", "inv_inertia2", "rel_quat", "rel_quat_inv")

    def __init__(self):
        super(BallExportInfo, self).__init__()


class CharacterExtractor(CharacterWrapper):

    __slots__ = ("character",)

    def __init__(self, character: ODECharacter):
        super(CharacterExtractor, self).__init__(character)

    def get_info_base(self, info: JointExportInfoBase, joints: List[ode.Joint]):
        info.body1_idx = np.zeros(len(info.joint_idx), dtype=np.long)
        info.body2_idx = np.zeros(len(info.joint_idx), dtype=np.long)

        for idx, joint in enumerate(joints):
            info.body1_idx[idx] = joint.body1.instance_id
            if joint.body2 is not None:
                info.body2_idx[idx] = joint.body2.instance_id
            else:
                info.body2_idx[idx] = -1

        info.body1_r = self.joint_info.get_child_body_relative_pos()[info.joint_idx]
        info.body2_r = self.joint_info.get_parent_body_relative_pos()[info.joint_idx]

        info.inv_mass1 = self.body_info.mass_val[info.body1_idx]
        info.inv_mass2 = self.body_info.mass_val[info.body2_idx]
        inertia_inv = self.body_info.calc_body_init_inertia_inv()
        info.inv_inertia1 = np.ascontiguousarray(inertia_inv[info.body1_idx])
        info.inv_inertia2 = np.ascontiguousarray(inertia_inv[info.body2_idx])

        info.rel_quat = self.joint_info.get_local_q()[info.joint_idx]  # local quat of all joints.
        info.rel_quat_inv = Rotation(info.rel_quat, copy=False).inv().as_quat()

        info.body_offset = self.character.bodies[0].offset_instance_id - self.character.bodies[0].instance_id
        return info

    def get_total_info(self):
        pass

    def get_hinge_info(self) -> HingeExportInfo:
        """
        get hinge information
        """
        info: HingeExportInfo = HingeExportInfo()
        hinges: List[ode.HingeJoint] = self.joint_info.hinge_joints()
        info.joint_idx = np.asarray(self.joint_info.hinge_id(), dtype=np.long)
        info.raw_axis1 = self.joint_info.get_hinge_raw_axis1(hinges)
        info.raw_axis2 = self.joint_info.get_hinge_raw_axis2(hinges)
        info.joint_erp = self.joint_info.get_hinge_erp(hinges)

        info.limit_lo = self.joint_info.get_hinge_lo(hinges)
        info.limit_hi = self.joint_info.get_hinge_hi(hinges)

        self.get_info_base(info, hinges)
        return info

    def get_ball_info(self) -> BallExportInfo:
        """
        get ball information
        """
        info: BallExportInfo = BallExportInfo()
        balls = self.joint_info.ball_joints()
        info.joint_idx = np.asarray(self.joint_info.ball_id(), dtype=np.long)
        info.joint_erp = self.joint_info.get_ball_erp(balls)
        self.get_info_base(info, balls)
        return info


class CharacterListExtractor():

    # __slots__ = ("character_list",)

    def __init__(self, character_list):
        self.character_list = [CharacterExtractor(character) for character in character_list]

    def get_hinge_info(self) -> HingeExportInfo:
        """
        get hinge information
        """
        info = self.character_list[0].get_hinge_info()
        for idx,character in enumerate(self.character_list[1:]):
            info.joint_offset += len(self.character_list[idx-1].joints)
            info = info + character.get_hinge_info()
        return info

    def get_ball_info(self) -> BallExportInfo:
        """
        get ball information
        """
        info = self.character_list[0].get_ball_info()
        for idx,character in enumerate(self.character_list[1:]):
            info.joint_offset += len(self.character_list[idx-1].joints)
            info = info + character.get_ball_info()
        return info

    def get_kp_info(self) -> np.ndarray:
        return np.concatenate([node.joint_info.kps for node in self.character_list if node.joint_info.kps is not None and len(node.joint_info.kps) > 0])

    def get_torque_limit_info(self) -> np.ndarray:
        return np.concatenate([node.joint_info.torque_limit for node in self.character_list if node.joint_info.torque_limit is not None])

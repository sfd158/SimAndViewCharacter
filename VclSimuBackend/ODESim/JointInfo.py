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
from scipy.spatial.transform import Rotation
from typing import Optional, List, Union, Tuple, Dict

from ..Common.Helper import Helper
from ..Common.MathHelper import MathHelper

from MotionUtils import (
    quat_multiply_forward_fast, quat_to_rotvec_single_fast, quat_to_rotvec_fast, quat_from_rotvec_fast, quat_inv_fast,
    decompose_rotation_single_fast, decompose_rotation_single_pair_fast,
    quat_to_vec6d_single_fast, quat_inv_single_fast, quat_apply_single_fast,
    quat_apply_forward_fast, quat_to_vec6d_fast, six_dim_mat_to_quat_fast, quat_normalize_fast
)


def my_concatenate(tup, axis=0) -> np.ndarray:
    a, b = tup
    if np.size(b) == 0 or b is None:
        return a
    if np.size(a) == 0 or a is None:
        return b
    return np.concatenate([a, b], axis=axis)


class JointInfosBase:
    def __init__(self, world: ode.World):
        self.world: ode.World = world  #
        self.joints: List[Union[ode.Joint, ode.BallJointAmotor, ode.BallJoint, ode.HingeJoint]] = []
        self.joint_c_id: Optional[np.ndarray] = None  # address(id) of joints in C code. calc in JointInfoInit.py

        self.hinge_c_id: Optional[np.ndarray] = None  # address(id) of joints in C code. calc in JointInfoInit.py
        self.hinge_lo: Optional[np.ndarray] = None  # lo of hinge angle limit
        self.hinge_hi: Optional[np.ndarray] = None  # hi of hinge angle limit

        self.weights: Optional[np.ndarray] = None  # weight of each joint for computing loss. load from file.
        self.euler_axis_local: Optional[np.ndarray] = None  # local joint euler axis. load from file.

        self.torque_limit: Optional[np.ndarray] = None  # max torque add on the body. load from file.
        self.kps: Optional[np.ndarray] = None  # Kp parameter of each joint. load from file.
        self.kds: Optional[np.ndarray] = None  # Kd parameter of each joint. load from file.

    def __add__(self, other):

        self.joints += other.joints
        self.joint_c_id = my_concatenate([self.joint_c_id, other.joint_c_id], axis=0)

        self.hinge_c_id = my_concatenate([self.hinge_c_id, other.hinge_c_id], axis=0)
        self.hinge_lo = my_concatenate([self.hinge_lo, other.hinge_lo], axis=0)
        self.hinge_hi = my_concatenate([self.hinge_hi, other.hinge_hi], axis=0)
        self.weights = my_concatenate([self.weights, other.weights], axis=0)
        self.euler_axis_local = my_concatenate([self.euler_axis_local, other.euler_axis_local], axis=0)

        self.torque_limit = my_concatenate([self.torque_limit, other.torque_limit], axis=0)
        self.kps = my_concatenate([self.kps, other.kps], axis=0)
        self.kds = my_concatenate([self.kds, other.kds], axis=0)

        return self

    def __len__(self) -> int:
        return len(self.joints)

    def joint_names(self) -> List[str]:
        """
        return: each joints' name
        """
        return [i.name for i in self.joints]

    def ball_id(self) -> List[int]:
        """
        all ball joints' index
        """
        return [idx for idx, joint in enumerate(self.joints) if issubclass(type(joint), ode.BallJointBase)]

    def ball_joints(self) -> List[Union[ode.BallJointAmotor, ode.BallJoint]]:
        return [joint for joint in self.joints if issubclass(type(joint), ode.BallJointBase)]

    def hinge_id(self) -> List[int]:
        """
        All Hinge Joints' index
        """
        return [idx for idx, joint in enumerate(self.joints) if type(joint) == ode.HingeJoint]

    def hinge_joints(self) -> List[ode.HingeJoint]:
        """
        All Hinge Joints
        """
        return [joint for joint in self.joints if type(joint) == ode.HingeJoint]

    def hinge_lo_hi(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.hinge_lo, self.hinge_hi

    def has_hinge(self) -> bool:
        return self.hinge_c_id is not None and self.hinge_c_id.size > 0

    def clear(self):
        for joint in self.joints:
            joint.destroy_immediate()

        self.joints.clear()
        self.joint_c_id: Optional[np.ndarray] = None
        self.hinge_c_id: Optional[np.ndarray] = None
        self.hinge_lo: Optional[np.ndarray] = None
        self.hinge_hi: Optional[np.ndarray] = None

        self.weights: Optional[np.ndarray] = None
        self.euler_axis_local = None

        self.kps: Optional[np.ndarray] = None
        self.kds: Optional[np.ndarray] = None
        self.torque_limit: Optional[np.ndarray] = None

        return self


class JointInfos(JointInfosBase):
    def __init__(self, world: ode.World):
        super(JointInfos, self).__init__(world)
        self.pa_joint_id: List[int] = []  # id of joint's parent joint
        self.sample_win: Optional[np.ndarray] = None  # Sample Window for Samcon. load from file.
        self.sample_mask: Optional[np.ndarray] = None

        self.parent_body_index: Optional[np.ndarray] = None  # TODO: concat it from __add__
        self.child_body_index: Optional[np.ndarray] = None
        self.parent_body_c_id: Optional[np.ndarray] = None  # address(id) of parent body of each joint in C code
        self.child_body_c_id: Optional[np.ndarray] = None  # address(id) of child_body of each joint in C code

        # load from file
        self.has_root: bool = False  # whether character has root joint.
        self.root_idx: Optional[int] = None  # index of root joint. For convenience, it should be None or 0.

    def get_subset_by_name(self, remain_joint_names: List[str]):
        curr_name: List[str] = self.joint_names()
        curr_name_dict = {node: index for index, node in enumerate(curr_name)}
        remain_list: List[int] = [curr_name_dict[name] for name in remain_joint_names]
        remain_list.sort()
        self.get_subset(remain_list)

    def get_subset(self, remain_joint_index: List[int]):
        result = JointInfos()
        result.world = self.world
        result.joints = [self.joints[index] for index in remain_joint_index]
        if self.joint_c_id is not None:
            result.joint_c_id = np.ascontiguousarray(self.joint_c_id[remain_joint_index])
        remain_hinge_index = [index for index in remain_joint_index if isinstance(self.joints[index], ode.HingeJoint)]
        if len(remain_hinge_index) > 0:
            remain_hinge_joints = [self.joints[index] for index in remain_hinge_index]
            if self.hinge_c_id is not None:
                result.hinge_c_id = self.world.jointListToNumpy(remain_hinge_joints)
            if self.hinge_lo is not None:
                result.hinge_lo = self.get_hinge_lo(remain_hinge_joints)
            if self.hinge_hi is not None:
                result.hinge_hi = self.get_hinge_hi(remain_hinge_joints)

        if self.weights is not None:
            result.weights = np.ascontiguousarray(self.weights[remain_joint_index])
        if self.euler_axis_local is not None:
            result.euler_axis_local = np.ascontiguousarray(self.euler_axis_local[remain_joint_index])
        if self.torque_limit is not None:
            result.torque_limit = np.ascontiguousarray(self.torque_limit[remain_joint_index])
        if self.kps is not None:
            result.kps = np.ascontiguousarray(self.kps[remain_joint_index])
        if self.kds is not None:
            result.kds = np.ascontiguousarray(self.kds[remain_joint_index])

        # assume the remaining joints are continuous.
        result.pa_joint_id = []  # TODO: id of joint's parent joint
        if self.sample_win is not None:
            result.sample_win = np.ascontiguousarray(self.sample_win[remain_joint_index])
        if self.sample_mask is not None:
            result.sample_mask = np.ascontiguousarray(self.sample_mask[remain_joint_index])

        if self.parent_body_index is not None:  # TODO
            pass
        if self.child_body_index is not None:
            pass

        if self.parent_body_c_id is not None:
            result.parent_body_c_id = np.ascontiguousarray(self.parent_body_c_id[remain_joint_index])
        if self.child_body_c_id is not None:
            result.child_body_c_id = np.ascontiguousarray(self.child_body_c_id[remain_joint_index])

        result.has_root = self.has_root  # assume root is not modified..
        result.root_idx = self.root_idx
        return result

    def __add__(self, other):
        super(JointInfos, self).__add__(other)
        j_len = len(self.pa_joint_id)
        tmp = [p_id if p_id < 0 else p_id + j_len for p_id in other.pa_joint_id]
        self.pa_joint_id += tmp
        self.sample_win = my_concatenate([self.sample_win, other.sample_win], axis=0)
        self.parent_body_c_id = my_concatenate([self.parent_body_c_id, other.parent_body_c_id], axis=0)
        self.child_body_c_id = my_concatenate([self.child_body_c_id, other.child_body_c_id], axis=0)

        # TODO: Actually you should use this code... but I'm afraid that Teacher Song's test code will not work
        # TODO: if we change the root into List...

        # if type(self.has_root) is List:
        #     self.has_root += other.has_root
        # else:
        #     self.has_root = [self.has_root, other.has_root]
        #
        # if type(self.root_idx) is List:
        #     self.root_idx += other.root_idx
        # else:
        #     self.root_idx = [self.root_idx, other.root_idx]

        # TODO: Currently, we use character 0's root
        return self

    @property
    def root_joint(self) -> Optional[ode.Joint]:
        return self.joints[self.root_idx] if self.has_root else None

    def resize_euler_axis_local(self):
        self.euler_axis_local = np.tile(np.eye(3), len(self)).reshape((-1, 3, 3))

    @staticmethod
    def body_rotvec(body: ode.Body) -> np.ndarray:
        """
        Get Body's Rot Vector in world coordinate
        """
        return Rotation.from_matrix(np.array(body.getRotation()).reshape((3, 3))).as_rotvec()

    def disable_root(self) -> None:
        """
        Disable root joint if exists
        """
        if not self.has_root:
            return
        self.joints[self.root_idx].disable()  # joint->flags |= dJOINT_DISABLED;

    def enable_root(self) -> None:
        """
        enable root joint if exists
        """
        if not self.has_root:
            return
        self.joints[self.root_idx].enable()

    def parent_qs(self) -> np.ndarray:
        """
        Get parent bodies' quaternion in global coordinate
        """
        res: np.ndarray = self.world.getBodyQuatScipy(self.parent_body_c_id).reshape((-1, 4))  # (num joint, 4)
        # if simulation fails, quaternion result will be very strange.
        # for example, rotations will be zero...
        if self.has_root:
            res[self.root_idx] = MathHelper.unit_quat()
        res = quat_normalize_fast(res)
        # res /= np.linalg.norm(res, axis=-1, keepdims=True)
        return res

    def child_qs(self) -> np.ndarray:
        """
        Get Child bodies' quaternion in global coordinate
        """
        res: np.ndarray = self.world.getBodyQuatScipy(self.child_body_c_id).reshape((-1, 4))  # (num joint, 4)
        res /= np.linalg.norm(res, axis=-1, keepdims=True)
        return res

    # return q1s, q2s, local_qs, q1s_inv
    def get_parent_child_qs_old(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Note: implement of scipy is slow..use cython version instead.
        return:
        parent bodies' quaternion in global coordinate,
        child bodies' quaternion in global coordinate,
        joint's quaternion in parent's local coordinate.
        inverse of parent bodies' quaternion in global coordinate
        """
        raise ValueError("This is slow. Please use cython version instead.")
        parent_qs: np.ndarray = self.parent_qs()  # parent bodies' quaternion in global coordinate
        child_qs: np.ndarray = self.child_qs()  # child bodies' quaternion in global coordinate

        parent_qs_inv: Rotation = Rotation(parent_qs, False, False).inv()  # inv parent bodies' quaternion in global
        local_qs: np.ndarray = (parent_qs_inv * Rotation(child_qs, False, False)).as_quat()  # joints' local quaternion

        # Test with C++ version
        # print(self.joint_c_id)
        c_parent_qs, c_child_qs, c_local_qs, c_parent_qs_inv = self.world.get_all_joint_local_angle(self.joint_c_id)
        print(np.max(np.abs(parent_qs - c_parent_qs)))
        print(np.max(np.abs(child_qs - c_child_qs)))
        print(np.max(np.abs(MathHelper.flip_quat_by_w(local_qs) - MathHelper.flip_quat_by_w(c_local_qs))))
        print(np.max(np.abs(MathHelper.flip_quat_by_w(parent_qs_inv.as_quat()) - MathHelper.flip_quat_by_w(c_parent_qs_inv))))
        return parent_qs, child_qs, local_qs, parent_qs_inv.as_quat()

    def get_parent_child_qs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # check when character has real root joint OK
        c_parent_qs, c_child_qs, c_local_qs, c_parent_qs_inv = self.world.get_all_joint_local_angle(self.joint_c_id)
        return c_parent_qs, c_child_qs, c_local_qs, c_parent_qs_inv

    def get_local_q(self) -> np.ndarray:
        """
        joint' quaternion in parent's local coordinate
        """
        _, _, local_qs, _ = self.get_parent_child_qs()
        return local_qs

    def get_local_angvels(self, parent_qs_inv: Optional[np.ndarray] = None) -> np.ndarray:
        """
        param:
        parent_qs_inv: Optional. inverse of parent bodies' quaternion in global coordinate

        return: Joints' angular velocity in parent body's local coordinate, in shape (num joint, 3)
        """
        # if parent_qs_inv is None:
        #     parent_qs_inv = Rotation(self.parent_qs(), normalize=True, copy=False).inv().as_quat()
        global_angvel = self.get_global_angvels()
        return quat_apply_forward_fast(parent_qs_inv, global_angvel)
        # return Rotation(parent_qs_inv, False, False).apply(global_angvel)

    def get_global_angvels(self) -> np.ndarray:
        """
        return: Joints' angular velocity in global coordinate, in shape (num joint, 3)
        """
        ang_parent_body: np.ndarray = self.world.getBodyAngVel(self.parent_body_c_id).reshape((-1, 3))
        ang_child_body: np.ndarray = self.world.getBodyAngVel(self.child_body_c_id).reshape((-1, 3))
        if self.has_root:
            ang_child_body[self.root_idx] = 2 * ang_parent_body[self.root_idx]
        return ang_child_body - ang_parent_body

    # Get Position relative to parent joint in world(global) coordinate
    def get_relative_global_pos(self) -> np.ndarray:
        # Emm..should load root joint when export to bvh...
        # assert self.has_root

        global_pos: np.ndarray = self.world.getBallAndHingeAnchor1(self.joint_c_id).reshape((-1, 3))
        pa_global_pos: np.ndarray = global_pos[self.pa_joint_id]
        if self.has_root:
            pa_global_pos[self.root_idx, :] = self.joints[self.root_idx].getAnchorNumpy()
        else:
            for idx, pa_idx in enumerate(self.pa_joint_id):
                if pa_idx >= 0:
                    continue
                pa_body: ode.Body = self.joints[idx].body2
                pa_global_pos[idx, :] = pa_body.PositionNumpy

        return global_pos - pa_global_pos

    # Get Position relative to parent joint in parent joint's coordinate..
    def get_relative_local_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        # check OK when has real root joint
        parent_qs, child_qs, local_qs, parent_qs_inv = self.get_parent_child_qs()
        global_offset = self.get_relative_global_pos()
        # joint's global rotation should be q2. joint's parent joint's global rotation should be q1..
        offset: np.ndarray = Rotation(parent_qs_inv, copy=False, normalize=False).apply(global_offset)
        return local_qs, offset

    # r' = vector from child body to joint in child body's local frame
    # Rb = rotation matrix of child body
    # xb = global position of child body
    # xj = joint position in world, that is, xj = xb + Rb * r'
    # for joint in in ode v0.12, body0 is child body, and body1 is parent body
    # assume there is only ball joint, hinge joint, and amotor joint in a character
    # this method returns r'
    def get_child_body_relative_pos(self) -> np.ndarray:
        # Note: this method only support dJointID as input.
        # if you takes other as input, the program will crash or fall in dead cycle
        return self.world.getBallAndHingeRawAnchor1(self.joint_c_id).reshape((-1, 3))

    def get_parent_body_relative_pos(self) -> np.ndarray:
        return self.world.getBallAndHingeRawAnchor2(self.joint_c_id).reshape((-1, 3))

    # actually, we need to get hinge joint's index
    def get_hinge_raw_axis1(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
        if hinges is None:
            hinges = self.hinge_joints()

        res = np.zeros((len(hinges), 3))
        for idx, joint in enumerate(hinges):
            res[idx, :] = joint.Axis1RawNumpy
        return np.ascontiguousarray(res)

    def get_hinge_raw_axis2(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
        if hinges is None:
            hinges = self.hinge_joints()

        res = np.zeros((len(hinges), 3))
        for idx, joint in enumerate(hinges):
            res[idx, :] = joint.Axis2RawNumpy
        return np.ascontiguousarray(res)

    def get_hinge_axis1(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
        if hinges is None:
            hinges = self.hinge_joints()

        res = np.zeros((len(hinges), 3))
        for idx, joint in enumerate(hinges):
            res[idx, :] = joint.HingeAxis1
        return np.ascontiguousarray(res)

    def get_hinge_axis2(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
        if hinges is None:
            hinges = self.hinge_joints()

        res = np.zeros((len(hinges), 3))
        for idx, joint in enumerate(hinges):
            res[idx, :] = joint.HingeAxis2
        return np.ascontiguousarray(res)

    def get_hinge_angle(self) -> np.ndarray:
        """
        return angle of each hinge joint
        """
        return self.world.get_all_hinge_angle(self.hinge_c_id)

    def get_global_anchor1(self) -> np.ndarray:
        """
        call dJointGetBallAnchor1 and dJointGetHingeAnchor1
        """
        global_pos: np.ndarray = self.world.getBallAndHingeAnchor1(self.joint_c_id).reshape((-1, 3))
        return global_pos

    def get_global_pos1(self) -> np.ndarray:
        return self.get_global_anchor1()

    def get_global_anchor2(self) -> np.ndarray:
        """
        call dJointGetBallAnchor2 and dJointGetHingeAnchor2
        if simulation is totally correct, result of GetAnchor2 should be equal to GetAnchor1
        """
        return self.world.getBallAndHingeAnchor2(self.joint_c_id).reshape((-1, 3))

    def get_global_pos2(self) -> np.ndarray:
        """
        get global joint anchor
        """
        return self.get_global_anchor2()

    def get_joint_euler_order(self) -> List[str]:
        """
        Only enabled when 
        """
        return [joint.euler_order for joint in self.joints]

    def get_ball_erp(self, balls: Optional[List[Union[ode.BallJoint, ode.BallJointAmotor]]] = None) -> np.ndarray:
        """
        Get erp parameter of all ball joints
        """
        if balls is None:
            balls = self.ball_joints()
        return np.asarray([joint.joint_erp for joint in balls])

    def get_hinge_erp(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
        """
        get erp parameter of all hinge joints
        """
        if hinges is None:
            hinges = self.hinge_joints()
        return np.asarray([joint.joint_erp for joint in hinges])

    def get_hinge_lo(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
        if hinges is None:
            hinges = self.hinge_joints()

        return np.array([joint.AngleLoStop for joint in hinges])

    def get_hinge_hi(self, hinges: Optional[List[ode.HingeJoint]] = None) -> np.ndarray:
        if hinges is None:
            hinges = self.hinge_joints()

        return np.array([joint.AngleHiStop for joint in hinges])

    def get_erp(self) -> np.ndarray:
        """
        Get erp of all joints
        """
        return np.asarray([joint.joint_erp for joint in self.joints])

    def get_cfm(self) -> np.ndarray:
        """
        Get CFM parameter of all Joints
        """
        return np.asarray([joint.joint_cfm for joint in self.joints])

    def clear(self):
        super(JointInfos, self).clear()
        self.sample_win: Optional[np.ndarray] = None
        self.has_root: bool = False
        self.root_idx: Optional[int] = None
        self.parent_body_index: Optional[np.ndarray] = None
        self.child_body_index: Optional[np.ndarray] = None
        self.parent_body_c_id: Optional[np.ndarray] = None
        self.child_body_c_id: Optional[np.ndarray] = None
        self.pa_joint_id.clear()

        return self

    def gen_sample_mask(self, use_joint_names = None) -> np.ndarray:
        result_list = []
        for joint in self.joints:
            if use_joint_names is not None and joint.name not in use_joint_names:
                continue
            buff = np.ones(3, dtype=np.float64)
            if isinstance(joint, ode.HingeJoint):
                buff[:] = 0
                buff[ord(joint.euler_order) - ord('X')] = 1
            result_list.append(buff)
        self.sample_mask = np.array(result_list)
        return self.sample_mask

    def get_mirror_index(self) -> List[int]:
        """
        Modified from Libin Liu's pymotionlib
        TODO: Test
        """
        joint_names = self.joint_names()
        return Helper.mirror_name_list(joint_names)

    def set_joint_weights(self, weight_dict: Dict[str, float]):
        self.weights = np.ones(len(self.joints))
        try:
            joint_name_dict: Dict[str, int] = {name: index for index, name in enumerate(self.joint_names())}
            for key, value in weight_dict.items():
                index: int = joint_name_dict[key]
                self.weights[index] = value
            self.weights = np.ascontiguousarray(self.weights)
        except:
            pass
        return self.weights

    def get_adj_matrix(self) -> np.ndarray:
        """
        get adj matrix of each joints.
        """
        num_joints: int = len(self.joints)
        ret: np.ndarray = np.zeros((num_joints, num_joints), dtype=np.int32)
        for idx, parent in enumerate(self.pa_joint_id):
            if parent == -1:
                continue
            ret[idx, parent] = 1
            ret[parent, idx] = 1
        return ret

    def get_neighbours(self) -> List[List[int]]:
        num_joints: int = len(self.joint_parents_idx)
        result: List[List[int]] = [[] for _ in range(num_joints)]
        for idx, parent_idx in enumerate(self.joint_parents_idx):
            if parent_idx == -1:
                continue
            result[idx].append(parent_idx)
            result[parent_idx].append(idx)
        for idx in range(num_joints):
            result[idx].sort()
        return result

    def enable_joint_feedback(self) -> None:
        for joint_idx, joint in enumerate(self.joints):
            joint.setFeedback(1)

    def disable_joint_feedback(self) -> None:
        for joint_idx, joint in enumerate(self.joints):
            joint.setFeedBack(0)

    def get_feedback_force(self) -> np.ndarray:
        result: np.ndarray = self.world.getJointFeedBackForce(self.joint_c_id)
        return result

    def get_feedback_torque(self) -> np.ndarray:
        result: np.ndarray = self.world.getJointFeedBackTorque(self.joint_c_id)
        return result

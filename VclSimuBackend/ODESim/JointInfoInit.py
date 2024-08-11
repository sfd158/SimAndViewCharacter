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

from typing import Optional, List, Union
import numpy as np
import ModifyODE as ode
from .JointInfoWrapper import JointInfos, JointInfoWrapper


class JointInfoInit(JointInfoWrapper):
    def __init__(self, joint_info: JointInfos):
        super(JointInfoInit, self).__init__(joint_info)

    def calc_joint_c_id(self):
        self.joint_info.joint_c_id = self.world.jointListToNumpy(self.joints)
        hinges: List[ode.HingeJoint] = self.joint_info.hinge_joints()
        if len(hinges) > 0:
            self.joint_info.hinge_c_id = self.world.jointListToNumpy(hinges)

    def init_after_load(self):
        self.calc_joint_c_id()
        self.joint_info.weights = np.ones(len(self.joints))

        self.joint_info.resize_euler_axis_local()
        # for idx, joint in enumerate(self.joints):
        #    self.euler_axis_local[idx, :, :] = joint.euler_axis
        # self.euler_axis_local = np.ascontiguousarray(self.euler_axis_local)

    @staticmethod
    def set_ball_joint_limit(joint: ode.BallJointAmotor, euler_order: str,
                             angle_limits: Union[List, np.ndarray], raw_axis: Optional[np.ndarray] = None):
        raw_axis = np.eye(3) if raw_axis is None else raw_axis
        angle_limits = np.deg2rad(np.asarray(angle_limits))
        assert raw_axis.shape == (3, 3) and len(euler_order) == 3

        euler_order = euler_order.upper()
        if euler_order in ["XZY", "YXZ", "ZYX"]:  # swap angle limit
            idx = ord(euler_order[2]) - ord('X')
            angle_limits[idx] = np.array([-angle_limits[idx][1], -angle_limits[idx][0]])

        joint.setAmotorMode(ode.AMotorEuler)
        joint.setAmotorNumAxes(3)
        # if raw axis are same, the program will crash
        joint.setAmotorAxisNumpy(0, 1, raw_axis[ord(euler_order[0]) - ord("X")])  # Axis 0, body 1
        joint.setAmotorAxisNumpy(2, 2, raw_axis[ord(euler_order[2]) - ord("X")])  # Axis 2, body 2

        joint.setAngleLim1(*angle_limits[0])
        joint.setAngleLim2(*angle_limits[1])
        joint.setAngleLim3(*angle_limits[2])

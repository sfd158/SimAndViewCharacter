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
from scipy.spatial.transform import Rotation as R
from typing import List, Union

from .MotionData import MotionData
from .Utils import align_quaternion, quat_product


class DynamicsData:
    def __init__(
        self,
        motion: MotionData,
        bone_mass: Union[np.ndarray, List[float]],
        bone_inertia: Union[np.ndarray, List[np.ndarray], List[List[float]], List[float]],
        bone_relative_offset: Union[np.ndarray, List[np.ndarray], List[List[float]], List[float]],
        bone_relative_rotation: Union[np.ndarray, List[np.ndarray], List[List[float]], List[float]]
        ):
        """ compuate and cache rigid body dynamics
        """
        bone_mass = np.asarray(bone_mass).reshape(-1, 1)
        bone_inertia = np.asarray(bone_inertia).reshape(-1, 3)
        bone_relative_offset = np.asarray(bone_relative_offset).reshape(-1, 3)
        bone_relative_rotation = np.asarray(bone_relative_rotation).reshape(-1, 4)

        assert (bone_mass.shape[0] == bone_inertia.shape[0])
        assert (bone_mass.shape[0] == bone_relative_offset.shape[0])
        assert (bone_mass.shape[0] == bone_relative_rotation.shape[0])

        assert (motion is None or motion.num_joints == bone_mass.shape[0])

        self.bone_mass = bone_mass
        self.bone_inertia = bone_inertia
        self.bone_relative_offset = bone_relative_offset
        self.bone_relative_rotation = bone_relative_rotation

        # those joint-related values are from the motion
        self.joint_positions = None if motion is None else motion.joint_position
        self.joint_orientations = None if motion is None else motion.joint_orientation
        self.joint_linear_velocities = None if motion is None else motion.compute_linear_velocity(True)
        self.joint_angular_velocities = None if motion is None else motion.compute_angular_velocity(True)

        # the following properties will be computed and cached
        # all of them are in the world coordinates
        self.bone_positions = None
        self.bone_orientations = None
        self.bone_linear_velocities = None
        self.bone_angular_velocities = None

        if motion is not None:
            self.update_bone_info()

        self.center_of_mass = None if motion is None else self.compute_center_of_mass()
        self.linear_momentum = None if motion is None else self.compute_linear_momentum()
        self.angular_momentum = None if motion is None else self.compute_angular_momentum(
            center_of_mass=self.center_of_mass)

    def update_bone_info(self):
        """ update kinematic properties of bones based on joint information
        """

        if self.joint_positions is None or \
                self.joint_orientations is None or \
                self.joint_linear_velocities is None or \
                self.joint_angular_velocities is None:
            raise ValueError('joint information is not complete')

        num_frames = self.joint_positions.shape[0]
        # orientation
        self.bone_orientations = quat_product(self.joint_orientations, self.bone_relative_rotation[None, ...])
        self.bone_orientations /= np.linalg.norm(self.bone_orientations, axis=-1, keepdims=True)
        self.bone_orientations = align_quaternion(self.bone_orientations, True)

        # position
        global_bone_offsets = R(self.joint_orientations.reshape(-1, 4), normalize=False, copy=False).apply(
            np.tile(self.bone_relative_offset, (num_frames, 1))
        ).reshape((num_frames, -1, 3))

        self.bone_positions = self.joint_positions + global_bone_offsets

        # angular velocity
        self.bone_angular_velocities = self.joint_angular_velocities.copy()

        # linear velocity
        self.bone_linear_velocities = self.joint_linear_velocities + \
                                      np.cross(self.joint_angular_velocities, global_bone_offsets)

    def update_joint_info(self):
        """ update kinematic properties of joints based on bone information
        """
        if self.bone_positions is None or \
                self.bone_orientations is None or \
                self.bone_linear_velocities is None or \
                self.bone_angular_velocities is None:
            raise ValueError('bone information is not complete')

        num_frames = self.bone_positions.shape[0]
        # orientation
        self.joint_orientations = quat_product(self.bone_orientations, self.bone_relative_rotation[None, ...])
        self.joint_orientations /= np.linalg.norm(self.joint_orientations, axis=-1, keepdims=True)
        self.joint_orientations = align_quaternion(self.joint_orientations, True)

        # position
        global_bone_offsets = R(self.joint_orientations.reshape(-1, 4), normalize=False, copy=False).apply(
            np.tile(self.bone_relative_offset, (num_frames, 1))
        ).reshape((num_frames, -1, 3))

        self.joint_positions = self.bone_positions - global_bone_offsets

        # angular velocity
        self.joint_angular_velocities = self.bone_angular_velocities.copy()

        # linear velocity
        self.joint_linear_velocities = self.bone_linear_velocities - \
                                       np.cross(self.bone_angular_velocities, global_bone_offsets)

    def compute_center_of_mass(self, bone_list: Union[List[int], None] = None):
        """ compute the center of mass of a given list of bones

            if bone_list is none, the entire skeleton will be considered
        """
        if self.bone_positions is None:
            raise ValueError('bone information is not complete')

        num_frames, num_bones = self.bone_positions.shape[:2]
        bone_list = list(range(num_bones)) if bone_list is None else bone_list
        masses = self.bone_mass[bone_list, :]
        mass_ratio = masses / masses.sum()
        pos = np.sum(self.bone_positions[:, bone_list] * mass_ratio.reshape(1, -1, 1), axis=1)
        return pos

    def compute_linear_momentum(self, bone_list: Union[List[int], None] = None):
        """ compute the total linear momentum of a given list of bones

            if bone_list is none, the entire skeleton will be considered
        """
        if self.bone_linear_velocities is None:
            raise ValueError('bone information is not complete')

        num_frames, num_bones = self.bone_linear_velocities.shape[:2]
        bone_list = list(range(num_bones)) if bone_list is None else bone_list

        momentum = np.sum(self.bone_linear_velocities[:, bone_list] * self.bone_mass[bone_list].reshape(1, -1, 1),
                          axis=1)
        return momentum

    def compute_angular_momentum(self, bone_list: Union[List[int], None] = None,
                                 center_of_mass: Union[np.ndarray, None] = None
                                 ):
        """ compute the total angular momentum of a given list of bones
            around the center of mass of those bones

            if bone_list is none, the entire skeleton will be considered

            if center_of_mass is not provided, it will be computed
        """
        if self.bone_positions is None or \
                self.bone_orientations is None or \
                self.bone_linear_velocities is None or \
                self.bone_angular_velocities is None:
            raise ValueError('bone information is not complete')

        num_frames, num_bones = self.bone_positions.shape[:2]
        bone_list = list(range(num_bones)) if bone_list is None else bone_list

        center_of_mass = np.asarray(center_of_mass).reshape(
            (-1, 3)) if center_of_mass is not None else self.compute_center_of_mass(bone_list)
        assert (center_of_mass.shape[0] == self.bone_angular_velocities.shape[0])

        dcm = R(self.bone_orientations[:, bone_list].reshape(-1, 4), normalize=False, copy=False).as_dcm().reshape(
            num_frames, -1, 3, 3)
        I = dcm * self.bone_inertia[bone_list].reshape(1, -1, 1, 3)
        I = np.einsum('ijkm,ijlm->ijkl', I, dcm)

        angular_momentum = np.einsum('ijkm,ijm->ijk', I, self.bone_angular_velocities[:, bone_list])

        com_to_bone = self.bone_positions[:, bone_list] - center_of_mass.reshape((-1, 1, 3))
        bone_linear_momentum = self.bone_linear_velocities[:, bone_list] * self.bone_mass[bone_list].reshape(1, -1, 1)
        angular_momentum += np.cross(com_to_bone, bone_linear_momentum)

        angular_momentum = angular_momentum.sum(axis=1)

        return angular_momentum

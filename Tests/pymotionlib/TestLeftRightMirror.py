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
import os
import subprocess
from scipy.spatial.transform import Rotation
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib.Utils import flip_quaternion


fdir = os.path.dirname(__file__)
# fname = os.path.join(fdir, "../CharacterData/Human36Reheight/S1/Directions-mocap-100.bvh")
# fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Samcon\0\5-0.bvh.bvh"
fname = r"G:\Samcon-Exps\S11\S11-WalkDog-mocap-100\test-7-0.bvh"
motion = BVHLoader.load(fname)

new_motion = motion.sub_sequence(copy=True)
new_motion = new_motion.flip(np.array([1.0, 0.0, 0.0]))
new_motion_velo = new_motion.compute_linear_velocity()
new_motion_angvel = new_motion.compute_angular_velocity()

new_motion_rot_speed = new_motion.compute_rotational_speed(False)

mirror_index = motion.get_mirror_joint_indices()
mirror_motion_pos = motion.joint_position.copy()
mirror_motion_pos[..., 0] *= -1

motion_velo = motion.compute_linear_velocity()
mirror_motion_velo = motion_velo.copy()
mirror_motion_velo[..., 0] *= -1

motion_angvel = motion.compute_angular_velocity()
mirror_motion_angvel = motion_angvel.copy()
mirror_motion_angvel[..., 1:] *= -1


test_pos = True
if test_pos:
    for index in range(motion.num_joints):
        print(np.max(np.abs(mirror_motion_pos[:, mirror_index[index]] - new_motion.joint_position[:, index])))


test_velo = False
if test_velo:
    print("test velo")
    for index in range(motion.num_joints):
        print(np.max(np.abs(mirror_motion_velo[:, mirror_index[index]] - new_motion_velo[:, index])))

# angular velocity cannot be mirrored directly..
test_angvel = False
if test_angvel:
    print("test angular velo")
    for index in range(motion.num_joints):
        print(np.max(np.abs(mirror_motion_angvel[:, mirror_index[index]] - new_motion_angvel[:, index])))

# test_rot_speed = False
# if test_rot_speed:
#     print("test rotational speed")
#     for index in range(motion.num_joints):
#         print(np.max(np.abs(mirror_motion_rot_speed[:, mirror_index[index]] - new_motion_rot_speed[:, index])))

flip_quat = flip_quaternion(motion.joint_orientation.reshape(-1, 4), np.array([1.0, 0.0, 0.0]), False).reshape(motion.joint_orientation.shape)
flip_quat = flip_quat[:, mirror_index]

test_global_quat = False
if test_global_quat:
    frame = 23
    for index in range(motion.num_frames):
        delta = Rotation(flip_quat[frame, index, :]) * Rotation(new_motion.joint_orientation[frame, index, :]).inv()
        print(index, delta.as_quat())


test_global_angvel = False
if test_global_angvel:
    frame = 23
    for index in range(motion.num_joints):
        delta = mirror_motion_angvel[frame, mirror_index[index], :] - new_motion_angvel[frame, index, :]
        print(index, delta, mirror_motion_angvel[frame, mirror_index[index], :], new_motion_angvel[frame, index, :])

output_fname = "mirrored-test-data.bvh"  # Add by Yulong Zhang, yhy yyds
BVHLoader.save(new_motion, output_fname)
subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", output_fname, fname])
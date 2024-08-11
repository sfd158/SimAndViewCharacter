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
import torch
from torch import nn
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib.PyTorchMotionData import PyTorchMotionData
from VclSimuBackend.pymotionlib import BVHLoader


def main():

    fdir = os.path.dirname(__file__)
    fname = os.path.join(fdir, "../CharacterData/Human3.6/S1/Walking-short-mocap.bvh")
    motion = BVHLoader.load(fname)
    diff_motion = PyTorchMotionData()
    diff_motion.build_from_motion_data(motion)
    index = [2, 5]
    param = nn.Parameter(diff_motion.joint_rotation[:, index, :].clone())
    diff_motion.set_parameter(param, index)
    diff_motion.recompute_joint_global_info()

    # check compute angular velocity
    rot_vel = motion.compute_rotational_speed(True)
    torch_rot_vel = diff_motion.compute_rotational_speed(True)
    print(np.max(np.abs(rot_vel - torch_rot_vel.detach().numpy())))
    # test gradient
    # sum_val = torch.sum(torch_rot_vel)
    # sum_val.backward()

    angvel = motion.compute_angular_velocity(False)
    torch_angvel = diff_motion.compute_angular_velocity(False)
    print(np.max(np.abs(angvel - torch_angvel.detach().numpy())))

    sum_val = torch.sum(torch_angvel)
    sum_val.backward()
    print(param.grad)


if __name__ == "__main__":
    main()

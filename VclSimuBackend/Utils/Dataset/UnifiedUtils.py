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

"""
common joint for ms-coco (17 joints), std-human (19 joints), subset for human3.6m (17 joints)
Note: there is no root joint for coco17 data.
"""

import numpy as np

unified_joints_def = {
    'nose': 0,
    'l-shoulder': 1,
    'l-elbow': 2,
    'l-wrist': 3,
    'l-hip': 4,
    'l-knee': 5,
    'l-ankle': 6,
    'r-shoulder': 7,
    'r-elbow': 8,
    'r-wrist': 9,
    'r-hip': 10,
    'r-knee': 11,
    'r-ankle': 12,
}

unified_bones_def = [
    [0, 1], [0, 7],  # head
    [1, 2], [2, 3],  # left arm
    [7, 8], [8, 9],  # right arm
    [1, 4], [7, 10],  # trunk
    [4, 5], [5, 6],  # left leg
    [10, 11], [11, 12],  # right leg
]

def check_up_vector(pos2d: np.ndarray):
    nose_index: int = unified_joints_def["l-shoulder"]
    ankle_index: int = unified_joints_def["r-ankle"]
    print(pos2d.shape)
    dpos: np.ndarray = pos2d[..., nose_index, :] - pos2d[..., ankle_index, :]
    # dpos = dpos / np.linalg.norm(dpos, axis=-1, keepdims=True)
    mean_up_vector = np.mean(dpos.reshape(-1, 2), axis=0)
    print("mean up vector", mean_up_vector)

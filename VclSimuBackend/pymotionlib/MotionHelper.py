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

# Add by Zhenhua Song
import numpy as np
from typing import Optional, Dict
from .MotionData import MotionData


def calc_children(data: MotionData):
    children = [[] for _ in range(data._num_joints)]
    for i, p in enumerate(data._skeleton_joint_parents[1:]):
        children[p].append(i + 1)
    return children


def calc_name_idx(data: MotionData) -> Dict[str, int]:
    return dict(zip(data.joint_names, range(len(data.joint_names))))


def adjust_root_height(data: MotionData, dh: Optional[float] = None):
    if dh is None:
        min_y_pos = np.min(data._joint_position[:, :, 1], axis=1)
        min_y_pos[min_y_pos > 0] = 0
        dy = np.min(min_y_pos)
    else:
        dy = dh
    data._joint_position[:, :, 1] -= dy
    data._joint_translation[:, 0, 1] -= dy

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
from .MathHelper import RotateType
from MotionUtils import (
    quat_to_rotvec_fast, quat_to_matrix_fast, quat_to_vec6d_fast,
    quat_from_rotvec_fast,
    six_dim_mat_to_quat_fast
)


class RotateConvertFast:
    @staticmethod
    def quat_single_to_other_rotate(x: np.ndarray, rotate_type: RotateType) -> np.ndarray:
        pass

    @staticmethod
    def quat_single_from_other_rotate(x: np.ndarray, rotate_type: RotateType) -> np.ndarray:
        pass

    @staticmethod
    def quat_to_other_rotate(x: np.ndarray, rotate_type: RotateType) -> np.ndarray:
        x: np.ndarray = np.ascontiguousarray(x, dtype=np.float64)
        if rotate_type == RotateType.Matrix or rotate_type == RotateType.SVD9d:
            return quat_to_matrix_fast(x)
        elif rotate_type == RotateType.Quaternion:
            return x
        elif rotate_type == RotateType.AxisAngle:
            return quat_to_rotvec_fast(x)[1]
        elif rotate_type == RotateType.Vec6d:
            return quat_to_vec6d_fast(x)
        else:
            raise NotImplementedError

    @staticmethod
    def quat_from_other_rotate(x: np.ndarray, rotate_type: RotateType) -> np.ndarray:
        q: np.ndarray = np.ascontiguousarray(x, dtype=np.float64)
        if rotate_type == RotateType.Matrix or rotate_type == RotateType.SVD9d:
            raise NotImplementedError
        elif rotate_type == RotateType.Quaternion:
            return q
        elif rotate_type == RotateType.AxisAngle:
            return quat_from_rotvec_fast(q)
        elif rotate_type == RotateType.Vec6d:
            return six_dim_mat_to_quat_fast(q)
        else:
            raise NotImplementedError

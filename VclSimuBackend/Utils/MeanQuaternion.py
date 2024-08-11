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

# Test mean of Rotation
from MotionUtils import simple_mix_quaternion, mix_quat_by_slerp
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, Union



def calc_iterative(quat: np.ndarray, eps: float = 1e-8, total_epoch: int = 100, start_solu: Optional[np.ndarray] = None):
    """
    Rotation averaging
    Richard Hartley, Jochen Trumpf, Yuchao Dai, Hongdong Li
    International journal of computer vision 103 (3), 267-305, 2013
    
    This is more faster than raw gradient descent method implemented by PyTorch.
    I learn it from website:
    https://www.zhihu.com/question/439497100/answer/1686005871
    """
    assert quat.shape[-1] == 4
    quat_in: Rotation = Rotation(quat)
    result: np.ndarray = Rotation(start_solu.copy() if start_solu is not None else quat[0].copy())
    for epoch in range(total_epoch):
        r: np.ndarray = np.mean((result.inv() * quat_in).as_rotvec(), axis=0)
        len_r = np.linalg.norm(r)
        if len_r < eps:
            # print(f"epoch = {epoch}, err = {len_r}, result = {result.as_rotvec()}")
            break
        result = result * Rotation.from_rotvec(r)

    return result


def _eval_mean_quat_loss(mean: Rotation, rot: Rotation):
    dist = np.linalg.norm((mean.inv() * rot).as_rotvec(), axis=-1)
    return np.mean(dist)


def calc_mean_quaternion(input_rot: Union[np.ndarray, Rotation], print_mess: bool = False) -> np.ndarray:
    """
    calc via several algorithm, and choose the result with minimal loss..
    """
    if isinstance(input_rot, np.ndarray):
        rots, quat = Rotation(input_rot), input_rot
    elif isinstance(input_rot, Rotation):
        rots, quat = input_rot, input_rot.as_quat()
    else:
        raise ValueError

    mean_scipy: Rotation = rots.mean()
    mean_iter: Rotation = calc_iterative(quat)
    # mean_simple: Rotation = Rotation(simple_mix_quaternion(quat))
    # mean_slerp: Rotation = Rotation(mix_quat_by_slerp(quat))
    loss = [(_eval_mean_quat_loss(i, rots), i.as_quat(), j)
            for i, j in zip([mean_scipy, mean_iter], ["scipy", "iter"])]
    loss.sort(key=lambda x: x[0])

    if print_mess:
        print(f"Best method is {loss[0][2]}, with loss = {loss[0][0]:.6f}")

    return loss[0][1]

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

from scipy.spatial.transform.rotation import Slerp
from VclSimuBackend.Common.MathHelper import *

def test_slerp():
    q0 = MathHelper.flip_quat_by_w(Rotation.random(100).as_quat())
    q1 = MathHelper.flip_quat_by_w(Rotation.random(100).as_quat())
    print(Rotation(MathHelper.slerp(q0, q1, 0.3)).as_rotvec())
    exit(0)
    slerp = Slerp([0, 1], Rotation(np.concatenate([q0, q1], axis=0)))
    # why sometimes the result is not same...
    print(slerp([0.3]).as_rotvec())


def test_slerp_vec():
    q0 = MathHelper.flip_quat_by_w(Rotation.random(6).as_quat())
    q1 = MathHelper.flip_quat_by_w(Rotation.random(6).as_quat())
    print(q0.shape)
    weight = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.8])
    for idx, val in enumerate(weight):
        slerp = Slerp([0, 1], Rotation(np.concatenate([q0[None, idx], q1[None, idx]], axis=0)))
        print(slerp([val]).as_rotvec())
    print("\n\n")
    res = MathHelper.slerp(q0, q1, weight)
    print(Rotation(res).as_rotvec())


def test_vec6d():
    q: np.ndarray = MathHelper.flip_quat_by_w(Rotation.random(10000).as_quat())
    vec6d: np.ndarray = MathHelper.quat_to_vec6d(q)
    q_new: np.ndarray = MathHelper.flip_quat_by_w(MathHelper.vec6d_to_quat(vec6d))
    assert np.max(np.abs(q - q_new) < 1e-12)
    print("Check OK")


def test_decompose_rotation():
    random_rot = Rotation.random(2333).as_quat()
    my_decompose_y, my_decompose_xz = MathHelper.y_decompose(random_rot)
    libin_decompose = MathHelper.extract_heading_Y_up(random_rot)
    print()


if __name__ == "__main__":
    test_decompose_rotation()

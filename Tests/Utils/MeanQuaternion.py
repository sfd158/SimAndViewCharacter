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

from VclSimuBackend.Utils.MeanQuaternion import *
from VclSimuBackend.DiffODE import DiffQuat


def use_grad_descent(quat, total_epoch: int = 10000):
    """
    This method is slow, and the result is not good...
    """
    assert quat.shape[-1] == 4
    import torch
    result = torch.nn.Parameter(DiffQuat.quat_to_rotvec(quat[None, 0]))
    opt = torch.optim.SGD([result], lr=1e-3, weight_decay=0)
    num: int = quat.shape[0]
    for epoch in range(total_epoch):
        qin = DiffQuat.quat_from_rotvec(result)
        ext_result: torch.Tensor = qin.repeat((num, 1))
        diff = DiffQuat.log_quat_diff_sqr(ext_result, quat)
        opt.zero_grad()
        loss = torch.mean(diff)
        loss.backward()
        if epoch % 100 == 0:
            print(epoch, loss.item())
        
        opt.step()

    print(result)
    return result.detach().numpy()


def main():
    init_vec: np.ndarray = np.array([[0.4, 0.8, 1.0]])
    print(init_vec.flatten())
    noise: np.ndarray = 0.1 * np.random.randn(100, 3)
    rotvec: np.ndarray = init_vec + noise
    rots: Rotation = Rotation.from_rotvec(rotvec)
    mean_q = calc_mean_quaternion(rots)
    print(Rotation(mean_q).as_rotvec())


if __name__ == "__main__":
    main()

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

import smpl_np
from smpl_torch import SMPLModel
import numpy as np
import torch
import os


fdir = os.path.dirname(__file__)
print(fdir)


def compute_diff(a, b):
    """
    Compute the max relative difference between ndarray a and b element-wisely.

    Parameters:
    ----------
    a, b: ndarrays to be compared of same shape.

    Return:
    ------
    The max relative difference.

    """
    return np.max(np.abs(a - b) / np.minimum(a, b))


def pytorch_wrapper(beta, pose, trans):
    device = torch.device('cpu')
    pose = torch.from_numpy(pose).type(torch.float64).to(device)
    beta = torch.from_numpy(beta).type(torch.float64).to(device)
    trans = torch.from_numpy(trans).type(torch.float64).to(device)
    model = SMPLModel(device=device)
    with torch.no_grad():
        result = model(beta, pose, trans)
    return result.cpu().numpy()


def np_wrapper(beta, pose, trans):
    input_fname = os.path.join(fdir, 'model.pkl')
    print(input_fname)
    smpl = smpl_np.SMPLModel(input_fname)
    result = smpl.set_params(pose=pose, beta=beta, trans=trans)
    return result


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.random.seed(9608)
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    beta = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    np_result = np_wrapper(beta, pose, trans)
    torch_result = pytorch_wrapper(beta, pose, trans)

    if np.allclose(np_result, torch_result):
        print('Bingo!')
    else:
        print('Failed')
        print('torch - np: ', compute_diff(torch_result, np_result))

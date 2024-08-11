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
Maybe we should use attention hierarchy here.
Maybe we can use network structure from
Zheng et al. 3D Human Pose Estimation with Spatial and Temporal Transformers, ICCV 2021

"""

import math
import torch
from torch import nn
debug_mode = False


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        qkv_bias: bool,
        atten_drop: float,
        fc_drop: float,
        device=torch.device("cpu")
    ):
        super(MultiHeadAttention, self).__init__()
        if debug_mode:
            print(locals())
        self.hid_dim: int = hid_dim
        self.n_heads: int = n_heads
        self.head_len: int = hid_dim // n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim, qkv_bias)
        self.w_k = nn.Linear(hid_dim, hid_dim, qkv_bias)
        self.w_v = nn.Linear(hid_dim, hid_dim, qkv_bias)
        self.fc = nn.Linear(hid_dim, hid_dim, qkv_bias)
        self.atten_drop = nn.Dropout(atten_drop)
        self.fc_drop = nn.Dropout(fc_drop)

        self.scale: float = math.sqrt(self.head_len)
        self.scale_inv: float = 1.0 / self.scale
        # print(f"scale_inv = {self.scale_inv}")
        self.to(device)

    def forward(self, x: torch.Tensor):
        batch_size, num_vector, x_dim = x.shape
        query: torch.Tensor = self.w_q(x)  # (batch size, num vector, x_dim)
        key: torch.Tensor = self.w_k(x)  # (batch size, num vector, x_dim)
        value: torch.Tensor = self.w_v(x)  # (batch size, num vector, x_dim)

        # (batch size, num head, num vector, head_len)
        query = query.view(batch_size, num_vector, self.n_heads, self.head_len).permute(0, 2, 1, 3).contiguous()
        key = key.view(batch_size, num_vector, self.n_heads, self.head_len).permute(0, 2, 1, 3).contiguous()
        value = value.view(batch_size, num_vector, self.n_heads, self.head_len).permute(0, 2, 1, 3).contiguous()

        x = self.scale_inv * torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch size, num head, num vec, num vec), energy
        x = torch.softmax(x, dim=-1)  # (batch size, num head, num vec, num vec), attention
        x = self.atten_drop(x)

        x = torch.matmul(x, value)  # (batch size, num head, num vec, head len)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch size, num vec, num head, head len)
        x = x.view(batch_size, num_vector, x_dim)

        x = self.fc(x)  # (batch size, num vector, x_dim)
        x = self.fc_drop(x)

        return x

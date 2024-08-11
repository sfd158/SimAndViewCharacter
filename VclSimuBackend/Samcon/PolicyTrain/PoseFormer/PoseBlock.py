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

import torch
from torch import nn
from .MultiHeadAttention import MultiHeadAttention
from .DropPath import DropPath


debug_mode = False


class PoseBlock(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_heads: int,
        qkv_bias: bool,
        mlp_hidden: int,
        linear_drop: float,
        atten_drop: float,
        drop_path: float,
        device=torch.device('cpu')
        ):
        super(PoseBlock, self).__init__()
        if debug_mode:
            print(locals())
        self.norm1 = nn.LayerNorm(x_dim)
        self.attention = MultiHeadAttention(x_dim, num_heads, qkv_bias, atten_drop, linear_drop, device)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(x_dim)
        self.mlp = nn.Sequential(
            nn.Linear(x_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(linear_drop),
            nn.Linear(mlp_hidden, x_dim),
            nn.Dropout(linear_drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

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
use network hierarchy from
3D Human Pose Estimation with Spatial and Temporal Transformers, ICCV 2021
"""
import numpy as np

import torch
from torch import nn

from .PoseBlock import PoseBlock
from ..Common import get_output_dim
from ....Common.MathHelper import RotateType


debug_mode = False

class PoseTransformer(nn.Module):
    def __init__(
        self,
        num_frame: int = 9,
        num_input_joint: int = 17,  # default value for coco dataset.
        num_output_joint: int = 19,  # default value for std-human
        rotate_type: RotateType = RotateType.Vec6d,
        also_output_pd_target: bool = False,
        also_output_local_rot: bool = False,  # I think we should also output root position here..
        also_output_contact_label: bool = False,
        in_channel: int = 2,  # input 2d joints
        embed_ratio: int = 32,  # same as original paper
        transformer_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        qkv_bias: bool = True,
        attention_drop: float = 0.0,
        linear_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        device=torch.device("cpu")
    ):
        super(PoseTransformer, self).__init__()
        assert also_output_pd_target or also_output_local_rot  # there should be 1 part
        if debug_mode:
            print(locals())
        self.rotate_type = rotate_type
        self.out_dim = get_output_dim(num_output_joint, rotate_type, also_output_pd_target, also_output_local_rot, also_output_contact_label)

        self.embed_ratio: int = embed_ratio
        self.embed_dim: int = embed_ratio * num_input_joint

        self.spatial_embed = nn.Linear(in_channel, embed_ratio)
        self.spatial_embed_dummy = nn.Parameter(torch.zeros(1, num_input_joint, embed_ratio))
        self.pos_drop = nn.Dropout(linear_drop)

        self.temporal_embed_dummy = nn.Parameter(torch.zeros(1, num_input_joint * embed_ratio))

        drop_ratio = np.linspace(0, drop_path_rate, transformer_depth)
        self.spatial_blocks = nn.ModuleList([
            PoseBlock(
                embed_ratio, num_heads, qkv_bias, mlp_ratio * embed_ratio,
                linear_drop, attention_drop, drop_ratio[i].item()
            )
            for i in range(transformer_depth)]
        )
        self.spatial_norm = nn.LayerNorm(embed_ratio)

        self.temporal_blocks = nn.ModuleList([
            PoseBlock(
                self.embed_dim, num_heads, qkv_bias, mlp_ratio * self.embed_dim,
                linear_drop, attention_drop, drop_ratio[i].item()
            )
            for i in range(transformer_depth)
        ])

        self.temporal_norm = nn.LayerNorm(self.embed_dim)

        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=(1,))

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.out_dim)
        )

        self.to(device)

    def spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frame, num_input_joint, in_channel = x.shape
        x: torch.Tensor = x.view((batch_size * num_frame * num_input_joint), in_channel)
        x: torch.Tensor = self.spatial_embed(x)  # (batch * frame * joint, embed)
        x: torch.Tensor = x.view((batch_size * num_frame), num_input_joint, self.embed_ratio)

        x: torch.Tensor = x + self.spatial_embed_dummy
        x: torch.Tensor = self.pos_drop(x)

        for block in self.spatial_blocks:
            x: torch.Tensor = block(x)
        x: torch.Tensor = self.spatial_norm(x)  # x.shape == ((batch_size * num_frame), num_joint, in_channel)

        x: torch.Tensor = x.view(batch_size, num_frame, num_input_joint * self.embed_ratio)
        return x

    def temporal_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frame, in_channel = x.shape  # (in_channel = num_joint * embed_ratio)
        x = x + self.temporal_embed_dummy
        x = self.pos_drop(x)

        for block in self.temporal_blocks:
            x = block(x)
        x = self.temporal_norm(x)  # (batch size, num frame, num joint * embed ratio)

        x = self.weighted_mean(x)  # (batch size, 1, num joint * embed ratio)
        x = x.view(batch_size, self.embed_dim)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, num_frame, num_joint, in_channel = x.shape
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        x = self.head(x)

        return x


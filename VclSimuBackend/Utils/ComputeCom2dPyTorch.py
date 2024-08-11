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
import torch
from typing import Optional, Union

from .ComputeCom2d import Joint2dComIgnoreHandToe
from ..ODESim.ODECharacter import ODECharacter


class Joint2dComPyTorch:
    def __init__(self) -> None:
        self.curr_parent_in_use: Optional[torch.Tensor] = None
        self.curr_joint_in_use: Optional[torch.Tensor] = None
        self.mass_in_use: Optional[torch.Tensor] = None
        self.total_mass: Union[torch.Tensor, float, None] = None

    @classmethod
    def build_from_character(cls, character: ODECharacter, dtype=torch.float32):
        result_np = Joint2dComIgnoreHandToe(character)
        result_torch = cls.build_from_numpy(result_np)
        return result_torch

    @classmethod
    def build_from_numpy(cls, other: Joint2dComIgnoreHandToe, dtype=torch.float32, device=torch.device("cpu")):
        result = Joint2dComPyTorch()
        result.curr_parent_in_use = torch.as_tensor(other.curr_parent_in_use, dtype=torch.long, device=device)
        result.curr_joint_in_use = torch.as_tensor(other.curr_joint_in_use, dtype=torch.long, device=device)
        result.mass_in_use = torch.as_tensor(other.mass_in_use, dtype=dtype, device=device)
        result.total_mass = torch.as_tensor(other.total_mass, dtype=dtype, device=device)

        return result

    def calc(self, joint_pos_2d: torch.Tensor) -> torch.Tensor:
        """
        same as numpy version
        """
        parent_pos: torch.Tensor = joint_pos_2d[..., self.curr_parent_in_use, :]
        child_pos: torch.Tensor = joint_pos_2d[..., self.curr_joint_in_use, :]
        body_pos: torch.Tensor = 0.5 * (parent_pos + child_pos)
        mass_shape = (1,) * (body_pos.ndim - 2) + (-1, 1)
        mass_in_use: torch.Tensor = self.mass_in_use.reshape(mass_shape)
        sum_mass: torch.Tensor = torch.sum(mass_in_use * body_pos, dim=-2) / self.total_mass
        return sum_mass

    def to_numpy(self) -> Joint2dComIgnoreHandToe:
        """
        convert to numpy version
        """
        result = Joint2dComIgnoreHandToe()
        result.curr_parent_in_use = self.curr_parent_in_use.detach().cpu().clone().numpy()
        result.curr_joint_in_use = self.curr_joint_in_use.detach().cpu().clone().numpy()
        result.mass_in_use = self.mass_in_use.detach().cpu().clone().numpy().astype(np.float64)
        if isinstance(self.total_mass, float):
            result.total_mass = self.total_mass
        elif isinstance(self.total_mass, torch.Tensor):
            result.total_mass = self.total_mass.item()
        else:
            raise ValueError

        return result

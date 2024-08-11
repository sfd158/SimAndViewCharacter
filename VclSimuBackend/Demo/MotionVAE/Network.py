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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, motion_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(2 * motion_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, curr_state: torch.Tensor, next_state: torch.Tensor):
        hidden = self.net(torch.cat([curr_state, next_state], -1))
        mu, logvar = self.mu.forward(hidden), self.logvar.forward(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAEDecoderV0(nn.Module):
    def __init__(self, motion_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + motion_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, motion_dim)
        )
    
    def forward(self, z: torch.Tensor, curr_state: torch.Tensor):
        return self.net(torch.cat([z, curr_state], -1))


class VAEDecoder(nn.Module):
    def __init__(self, motion_dim: int, hidden_dim: int, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(motion_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim + latent_dim, output_dim)

    def forward(self, z: torch.Tensor, curr_state: torch.Tensor):
        hid = F.elu(self.fc1(torch.cat([z, curr_state], dim=-1)))
        hid = F.elu(self.fc2(torch.cat([z, hid], dim=-1)))
        return self.out.forward(torch.cat([z, hid], dim=-1))


class MotionVAE(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int
    ):
        super().__init__()
        self.encoder = VAEEncoder(motion_dim, hidden_dim, latent_dim)
        self.decoder = GatingMixedDecoder(motion_dim, hidden_dim, latent_dim, output_dim)
        self.decoder = torch.jit.script(self.decoder)
        # TODO: increase beta parameter in training.
        self.beta = nn.Parameter(torch.as_tensor(1e-3), False)

    def forward(self, curr_state: torch.Tensor, next_state: torch.Tensor):
        z, mu, logvar = self.encoder.forward(curr_state, next_state)
        recon = self.decoder.forward(mu, curr_state)
        return recon, z, mu, logvar

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        return self.beta * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), -1))

    def sample(self, z: torch.Tensor, curr_state: torch.Tensor):
        return self.decoder.forward(z, curr_state)


class GatingMixedDecoder(nn.Module):

    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        num_experts: int = 6,
        gate_hsize: int = 64,
    ):
        super().__init__()
        input_size = latent_dim + motion_dim
        inter_size = latent_dim + hidden_dim
        num_layer = 4

        # put in list then initialize and register
        for i in range(num_layer):
            wdim1 = inter_size if i != 0 else input_size
            wdim2 = hidden_dim if i != num_layer - 1 else output_dim
            weight = nn.Parameter(torch.empty(num_experts, wdim1, wdim2))
            bias = nn.Parameter(torch.empty(num_experts, wdim2))

            stdv = 1. / math.sqrt(weight.size(1))
            weight.data.uniform_(-stdv, stdv)
            bias.data.uniform_(-stdv, stdv)

            self.register_parameter(f"w{i}", weight)
            self.register_parameter(f"b{i}", bias)

        # add layer norm
        for i in range(num_layer):
            self.add_module(f"ln{i}", nn.LayerNorm(inter_size if i != 0 else input_size))

        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(inplace=True),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(inplace=True),
            nn.Linear(gate_hsize, num_experts)
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=-1)), dim=-1)  # (batch_size, num_experts)
        # layer 0
        input_x = torch.cat((z, c), dim=-1)  # (batch_size, hid)
        input_x = self.ln0(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b0)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w0)
        layer_out: torch.Tensor = F.elu(mixed_input + mixed_bias, inplace=False)

        # layer 1
        input_x = torch.cat((z, layer_out), dim=-1)  # (batch_size, hid)
        input_x = self.ln1(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b1)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w1)
        layer_out: torch.Tensor = F.elu(layer_out + mixed_input + mixed_bias, inplace=False)

        # layer 2
        input_x = torch.cat((z, layer_out), dim=-1)  # (batch_size, hid)
        # input_x = F.layer_norm(input_x, input_x.shape[1:])
        input_x = self.ln2(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b2)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w2)
        layer_out: torch.Tensor = F.elu(layer_out + mixed_input + mixed_bias, inplace=False)

        # layer 3
        input_x = torch.cat((z, layer_out), dim=-1)  # (batch_size, hid)
        input_x = self.ln3(input_x)
        mixed_bias: torch.Tensor = torch.einsum('be,ek->bk', coefficients, self.b3)  # (batch_size, 512), contract
        mixed_input: torch.Tensor = torch.einsum('be,bj,ejk->bk', coefficients, input_x, self.w3)
        layer_out: torch.Tensor = mixed_input + mixed_bias

        return layer_out

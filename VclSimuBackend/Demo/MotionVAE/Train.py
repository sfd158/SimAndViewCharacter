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

from datetime import datetime
import numpy as np
import os
import subprocess
import torch
from torch.optim import AdamW, RAdam
from torch.nn import functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from types import SimpleNamespace
from VclSimuBackend.Common.DiffQuat import vec6d_to_quat
from VclSimuBackend.Demo.MotionVAE.Data import (
    load_from_file, calc_mean_std, calc_feasible_index, sample_rollout, get_next_input
)
from VclSimuBackend.Demo.MotionVAE.Network import MotionVAE


def main():
    args = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        mocap_file=r"C:\Users\24357\Documents\GitHub\ode-scene-backup\Tests\CharacterData\lafan-mocap-100\walk1_subject5.bvh",
        latent_dim=32,
        num_experts=6,
        rollout=8,
        batch=256,
        lr=1e-3,
        epoch_1=400,
        epoch_2=8000,
        epoch_3=100
    )

    if not os.path.exists(args.mocap_file):
        args.mocap_file = r"H:\GitHub\motion_control\assets\walk1_subject5.bvh"

    # prepare training data.
    data_buffer, nj = load_from_file(args.mocap_file)
    n_data, mean, std = calc_mean_std(data_buffer)
    in_key = ["pos", "vel", "vec6d", "avel"]
    out_key = ["vel", "avel"]
    in_dim = sum([data_buffer[key].shape[-1] for key in in_key]) * nj
    out_dim = sum([data_buffer[key].shape[-1] for key in out_key]) * nj
    mean_in = torch.cat([mean[key] for key in in_key], -1).to(args.device)
    std_in = torch.cat([std[key] for key in in_key], -1).to(args.device)
    mean_out = torch.cat([mean[key] for key in out_key], -1).to(args.device)  # (nj, 9)
    std_out = torch.cat([std[key] for key in out_key], -1).to(args.device)  # (nj, 9)
    b = args.batch

    net = MotionVAE(in_dim, 512, args.latent_dim, out_dim).to(args.device)
    # net = torch.jit.script(net)
    net.train()
    optimizer = RAdam(net.parameters(), lr=args.lr)

    writer_fname = 'runs/{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(writer_fname)
    writer.global_step = 0
    subprocess.Popen(["tensorboard", "--logdir", writer_fname])

    def calc_loss(recon_, out_data_, mu_, logvar_):
        loss = {"recon": F.smooth_l1_loss(recon_, out_data_), "kl": net.kl_loss(mu_, logvar_)}
        loss["total"] = sum(loss.values())
        loss["total"].backward()
        optimizer.step()
        optimizer.zero_grad(True)
        for key, value in loss.items():
            writer.add_scalar(key, value, writer.global_step)
        writer.global_step += 1

    def get_in_data():
        return torch.cat([n_data[key][index] for key in in_key], -1
            ).transpose(0, 1).to(args.device)

    def get_out_data():
        return torch.cat([n_data[key][index] for key in out_key], -1
            ).transpose(0, 1).to(args.device)

    # Training stage 1
    feasible_index = calc_feasible_index(data_buffer["done"], 2)
    for epoch in tqdm(range(args.epoch_1)):
        index = sample_rollout(feasible_index, b, 2)
        in_data = get_in_data()
        out_data = get_out_data()[0]
        recon, z, mu, logvar = net.forward(in_data[0].view(b, -1), in_data[1].view(b, -1))
        calc_loss(recon, out_data.view(b, -1), mu, logvar)
    
    # Training state 2 and 3
    feasible_index = calc_feasible_index(data_buffer["done"], args.rollout + 1)
    for epoch in tqdm(range(args.epoch_2)):
        index = sample_rollout(feasible_index, b, args.rollout + 1)
        in_data = get_in_data()
        out_data = get_out_data()
        n_curr_state = in_data[0]
        curr_state = n_curr_state * std_in + mean_in
        curr_quat = vec6d_to_quat(curr_state[..., 6:12].reshape(b, nj, 3, 2))
        recon_list, mu_list, logvar_list = [None] * args.rollout, [None] * args.rollout, [None] * args.rollout
        for i in range(args.rollout):
            recon_list[i], z, mu_list[i], logvar_list[i] = net.forward(n_curr_state.view(b, -1), in_data[i + 1].view(b, -1))
            flag = True
            if flag:
                curr_state, curr_quat = get_next_input(curr_state, recon_list[i].view(b, nj, 9) * std_out + mean_out, curr_quat)
                n_curr_state = (curr_state - mean_in) / std_in
            else:
                n_curr_state = in_data[i + 1]
                curr_state = n_curr_state * std_in + mean_in

        calc_loss(torch.cat([node.view(1, b, nj, 9) for node in recon_list], 0), out_data[:args.rollout],
                  torch.cat([node[None] for node in mu_list], 0), torch.cat([node[None] for node in logvar_list], 0))

if __name__ == "__main__":
    main()

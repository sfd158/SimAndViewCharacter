"""
Get code from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-

Generate motion sequence through DDPM.
"""
from argparse import Namespace
from datetime import datetime
import numpy as np
import os
import subprocess
import torch
from torch.optim import AdamW
from tqdm import tqdm
from tensorboardX import SummaryWriter
from VclSimuBackend.Demo.MotionVAE.Data import load_and_cat_from_file, calc_feasible_index, sample_rollout
from VclSimuBackend.Demo.Diffusion.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from VclSimuBackend.Demo.Diffusion.Model import SimpleConv1D
from VclSimuBackend.Demo.Diffusion.Scheduler import GradualWarmupScheduler, CosineAnnealingLR


def train():
    args = Namespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        mocap_file=r"C:\Users\24357\Documents\GitHub\ode-scene-backup\Tests\CharacterData\lafan-mocap-100\walk1_subject5.bvh",
        rollout_min=8,
        rollout_max=20,
        batch=256,
        lr=1e-4,
        epoch=100000,
        multiplier=2.,
        beta_1=1e-4,
        beta_T=0.02,
        T=1000
    )

    device, b = args.device, args.batch
    n_data, mean, std, done = load_and_cat_from_file(args.mocap_file, device)
    in_dim = n_data.shape[1] * n_data.shape[2]
    net_model = SimpleConv1D(args.T, in_dim, in_dim, 256).to(device)
    net_model.train()
    optimizer = AdamW(net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.epoch // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(net_model, args.beta_1, args.beta_T, args.T).to(device)
    writer_fname = 'runs/{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(writer_fname)
    subprocess.Popen(["tensorboard", "--logdir", writer_fname])

    # train
    feasible_index = calc_feasible_index(done, args.rollout_max)
    for epoch in tqdm(range(args.epoch)):
        roll_len = np.random.randint(args.rollout_min, args.rollout_max)
        index = sample_rollout(feasible_index, b, roll_len)
        motion = n_data[index].to(device).view(b, roll_len, -1).transpose(1, 2).contiguous()
        optimizer.zero_grad(True)
        loss = trainer(motion).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), 1)
        optimizer.step()
        warmUpScheduler.step()
        writer.add_scalar("loss", loss, epoch)
    writer.close()

    # eval
    net_model.eval()

train()
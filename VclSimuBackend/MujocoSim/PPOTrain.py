
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

import os
from mpi4py import MPI
import psutil
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
mpi_comm = MPI.COMM_WORLD
mpi_world_size: int = mpi_comm.Get_size()
mpi_rank: int = mpi_comm.Get_rank()
is_main: bool = mpi_rank == 0
# p = psutil.Process()
# p.cpu_affinity(range(mpi_rank, mpi_rank + 1))

from argparse import ArgumentParser, Namespace
import numpy as np
import copy
from enum import IntEnum
import json

import os
import random
from tensorboardX import SummaryWriter
import time
import torch
from torch import device, nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer, Adam
from typing import Optional, List, Dict, Any
from VclSimuBackend.Common.MathHelper import MathHelper, RotateType
from VclSimuBackend.MujocoSim.ArgsConfig import parse_args
from VclSimuBackend.MujocoSim.PPO import GaussianDistribution, PPO
from VclSimuBackend.MujocoSim.DeepMimicEnv import DeepMimicEnv


fdir = os.path.dirname(__file__)

torch.set_num_threads(1)

class ScaleModule(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return self.scale * x

class RollInfo:
    max_length = 50
    def __init__(self) -> None:
        self.avg_reward = 0.0
        self.buffer = []
        self.curr_index = 0

    def append(self, data, reward):
        if reward < self.avg_reward:
            return
        info = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "act": data.act.copy(), "qacc_warmstart": data.qacc_warmstart.copy(), "r": reward}
        l = len(self.buffer)
        if l < self.max_length:
            self.buffer.append(info)
            self.avg_reward = (self.avg_reward * l + reward) / (l + 1)
        else:
            self.avg_reward = (self.avg_reward * (l - 1) - self.buffer[self.curr_index]["r"] + reward) / l
            self.buffer[self.curr_index] = info
            self.curr_index = (self.curr_index + 1) % self.max_length

    def select(self):
        index = random.randint(0, len(self.buffer) - 1)
        return self.buffer[index]


class PPOTrain:
    def __init__(self, args: Namespace) -> None:
        args.num_samples = max(100, args.num_samples // mpi_world_size)
        self.args = args
        self.env = DeepMimicEnv(args)
        state_dim = self.env.state_dim
        last_linear = nn.Linear(args.hidden_dim, self.env.action_dim)
        #with torch.no_grad():
        #    last_linear.weight.data *= 0.4  # right: 0.4
        #    last_linear.bias.data *= 0.4
        self.action_net = nn.Sequential(
            nn.Linear(state_dim, args.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ELU(inplace=True),
            last_linear,
            # ScaleModule(0.5)
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, args.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(args.hidden_dim, 1)
        )
        self.policy_net = GaussianDistribution(self.action_net, self.env.action_dim, args.action_noise, "cpu")
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=5e-5)  # right: 1e-4
        self.value_optimizer = Adam(self.value_net.parameters(), lr=5e-3)  # right: 1e-3
        self.ppo = PPO(self.args, self.policy_net, self.value_net, self.policy_optimizer, self.value_optimizer)
        self.epoch: int = 0
        self.max_avg_length: int = 0
        self.state_mean = self.env.state_mean.copy()
        self.state_scale = self.env.state_scale.copy()

        self.avg_length = np.zeros(self.env.num_frames, dtype=np.int32)
        self.start_roll = [RollInfo() for _ in range(self.env.num_frames)]
        for frame in range(self.env.num_frames):
            self.env.reset(frame)
            self.start_roll[frame].append(self.env.data, 0.4)

        if mpi_rank == 0 and False:
            opt = Adam(self.action_net.parameters(), lr=1e-3)
            ref_state = torch.as_tensor(self.env.ref_state, dtype=torch.float32)
            ref_qpos = torch.as_tensor(self.env.ref_qpos[:, 7:], dtype=torch.float32)
            for i in range(5):
                pred = self.action_net(ref_state)
                loss = F.mse_loss(ref_qpos, pred)
                loss.backward()
                print(loss.item())
                opt.step()
                opt.zero_grad(True)

    @torch.no_grad()
    def generate_rollout(
        self,
        use_noise: bool = True,
        max_rollout_length: Optional[int] = None,
        start_frame: Optional[int] = None
    ):
        """
        expert_policy: used for dagger trainning
        use_noise: add noise on action for generate rollout

        Actually, gradient is not required here..
        return: states, state_vectors, action_vectors, action_log_probs, rewards
        """
        if max_rollout_length is None:
            max_rollout_length: int = self.args.max_rollout_length

        states: List[np.ndarray] = []  # un-normalized state
        state_vectors: List[np.ndarray] = []  # normalized state
        action_vectors: List[np.ndarray] = []  # action
        action_log_probs: List[np.ndarray] = []  # log (action probility)
        rewards: List[float] = []  # rewards for each step
        ref_motion_index: List[int] = []  # index of reference motion

        # set starting state for the new rollout.
        if start_frame is None:
            start_frame: int = np.random.randint(0, self.env.num_frames)
        # here we should consider the 6d representation and axis angle of the state
        device = torch.device("cpu")

        state: np.ndarray = self.env.reset(start_frame).reshape(-1)
        # state = self.env.reset_by_state(self.start_roll[start_frame].select(), start_frame)
        terminate: bool = False

        while True:  # here we need to consider whether the character meets last state, or failed..
            normalize_state: np.ndarray = (state.reshape(-1) - self.state_mean) / self.state_scale
            torch_state: torch.Tensor = torch.from_numpy(normalize_state).to(device)

            if use_noise:
                action_vec, action_log_prob = self.policy_net.sample(torch_state)
                action_log_prob: np.ndarray = action_log_prob.cpu().numpy()
            else:
                action_vec = self.policy_net.forward(torch_state)
                action_log_prob: np.ndarray = np.zeros(action_vec.shape, dtype=np.float32)

            action_vec: np.ndarray = action_vec.cpu().numpy()
            # for the 6d representation of gcn network, we should convert to quaternion..
            next_state, reward, terminate, _ = self.env.step(action_vec)
            next_state: np.ndarray = next_state.reshape(-1)

            # here we should store the next state..
            states.append(state[None, :])
            state_vectors.append(normalize_state)
            action_vectors.append(action_vec)
            action_log_probs.append(action_log_prob)
            rewards.append(reward)
            ref_motion_index.append(self.env.curr_time)
            state: np.ndarray = next_state
            if terminate or len(states) >= max_rollout_length:
                normalize_state: np.ndarray = (state - self.state_mean) / self.state_scale
                state_vectors.append(normalize_state)
                break

            # self.start_roll[self.env.curr_time].append(self.env.data, reward)

        end_index: int = self.env.curr_time
        self.avg_length[start_frame] = max(self.avg_length[start_frame], len(state_vectors))
        return end_index, states, state_vectors, action_vectors, action_log_probs, rewards, terminate, ref_motion_index

    @torch.no_grad()
    def get_gather_states(
        self,
        use_noise: bool = True,
        max_rollout_length: Optional[int] = None,
    ) -> List[np.ndarray]:
        if max_rollout_length is None:
            max_rollout_length = self.args.max_rollout_length
        gather_states: List[np.ndarray] = []
        self.action_net.eval()

        num_sample = 0
        while num_sample < self.args.num_samples:
            # after_frame: the frame after generate rollout
            # here we should considere generate rollout by given expert policy
            curr_rollout_length = min(self.args.num_samples - self.ppo.rollouts.num_sample, max_rollout_length)
            next_start_index, states, *rollout = self.generate_rollout(use_noise, curr_rollout_length)
            num_sample += len(states)
            if len(states) <= 1:
                continue
            gather_states.extend(states)
            self.ppo.add_rollout(*rollout)
        return gather_states[:max_rollout_length]

    def to_device(self, node: Dict[str, torch.Tensor], device="cpu"):
        for key, value in node.items():
            if key == "_metadata":
                continue
            value.data = value.data.to(device)
        return node
    
    def get_state_dict(self, worker_sync: bool = True) -> Dict[str, Any]:
        device = torch.device("cpu")
        params = {
            "epoch": self.epoch,
            "max_avg_length": self.max_avg_length,
            "value": self.to_device(self.value_net.state_dict(), device) if self.value_net is not None else None,
            "action": self.to_device(self.action_net.state_dict(), device),
            "value_optimizer": self.value_optimizer.state_dict() if (not worker_sync) and self.value_optimizer is not None else None,
            "policy_optimizer": self.policy_optimizer.state_dict() if not worker_sync else None,
            "args": self.args if not worker_sync else None,
            "state_scale": self.state_scale,
            "state_mean": self.state_mean
        }
        return params

    def load_from_state_dict(self, params: Dict[str, Any], load_optim: bool = True) -> None:
        self.epoch = params.get("epoch", self.epoch)
        self.max_avg_length = params.get("max_avg_length", self.max_avg_length)
        self.value_net.load_state_dict(self.to_device(params["value"]))
        self.action_net.load_state_dict(self.to_device(params["action"]))
        value_opt_param = params.get("value_optimizer")
        if value_opt_param is not None:
            self.value_optimizer.load_state_dict(value_opt_param)
        policy_opt_param = params.get("policy_optimizer")
        if policy_opt_param is not None:
            self.policy_optimizer.load_state_dict(policy_opt_param)
        # all of joints shares the same weight..
        self.state_scale: np.ndarray = params["state_scale"].copy()
        self.state_mean: np.ndarray = params["state_mean"].copy()

    def sync_policy(self) -> None:
        params: Optional[Dict[str, Any]] = self.get_state_dict() if mpi_rank == 0 else None
        params: Dict[str, Any] = mpi_comm.bcast(params, root=0)
        if mpi_rank > 0:
            self.load_from_state_dict(params)

    def train_ppo_multi_mlp(self):
        if mpi_rank == 0:
            print("train ppo multi mlp", flush=True)
        start_time: float = time.time()
        self.best_ckpt_path = os.path.join(fdir, f"policy-result.ckpt-best")
        while self.epoch < self.args.max_epoch:
            self.ppo.clear()
            mpi_comm.barrier()
            self.sync_policy()

            self.get_gather_states()
            lengths: List[int] = self.ppo.rollouts.get_length()
            lengths = mpi_comm.reduce(lengths)

            # collect rollouts
            all_rollouts = mpi_comm.reduce([] if mpi_rank == 0 else [self.ppo.rollouts])
            if mpi_rank == 0:
                for rollout in all_rollouts:
                    self.ppo.rollouts.extend(rollout)

                avg_length = sum(lengths) / len(lengths)
                if self.max_avg_length <= avg_length:
                    self.max_avg_length = avg_length
                    torch.save(self.get_state_dict(False), self.best_ckpt_path)
                if self.epoch % 100 == 0:
                    torch.save(self.get_state_dict(False), os.path.join(fdir, f"policy-result.ckpt-latest"))

                ret = self.ppo.train(self.args.batch_size, True, True)
                self.print_log(start_time, lengths, ret)

            self.epoch += 1

    def print_log(self, start_time: float, lengths: List[int], ret, global_step: Optional[int] = None,):
        if global_step is None:
            global_step = self.epoch
        if ret is None:
            ret = [0, 0, 0, 0]
        print ('epoch: %2d | elapse: %6.1f : ' % (global_step,
            time.time() - start_time
            ),
            '#rollouts: %3d | avg_len %0.3f/%0.3f | min_len %3d max_len %3d | total %3d | v0_mean %0.3f | ret_mean %0.3f | adv_mean %0.3f' % (
                len(lengths),
                sum(lengths) / len(lengths),
                self.max_avg_length,
                min(lengths),
                max(lengths),
                sum(lengths),
                ret[2],
                ret[3],
                ret[4]
            ), flush=True)

        # if self.tb_writer is not None and False:
        #     self.tb_writer.add_scalar(f'rollouts', len(lengths), global_step=global_step)
        #     self.tb_writer.add_scalar(f'avg_length', sum(lengths) / len(lengths), global_step=global_step)
        #     self.tb_writer.add_scalar(f'max_avg_length', self.max_avg_length, global_step=global_step)
        #     self.tb_writer.add_scalar(f'min_length', min(lengths), global_step=global_step)
        #     self.tb_writer.add_scalar(f'max_length', max(lengths), global_step=global_step)
        #     self.tb_writer.add_scalar(f'total', sum(lengths), global_step=global_step)

        #     if ret is not None:
        #         self.tb_writer.add_scalar(f'num_samples', ret[0], global_step=global_step)
        #         self.tb_writer.add_scalar(f'v0_mean', ret[1], global_step=global_step)
        #         self.tb_writer.add_scalar(f'ret_mean', ret[2], global_step=global_step)
        #         self.tb_writer.add_scalar(f'adv_mean', ret[3], global_step=global_step)

    @staticmethod
    def main():
        args = parse_args()
        if mpi_rank == 0:
            print(args, flush=True)
        np.random.seed(mpi_rank)
        random.seed(mpi_rank)
        torch.random.manual_seed(mpi_rank)
        train = PPOTrain(args)
        # if mpi_world_size == 1:
        #    train.load_from_state_dict(torch.load(os.path.join(fdir, "policy-result.ckpt-latest")))
        train.train_ppo_multi_mlp()

if __name__ == "__main__":
    PPOTrain.main()

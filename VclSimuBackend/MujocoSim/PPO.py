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

from argparse import Namespace
import numpy as np
from typing import Optional, List, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Independent, Normal
from torch.nn.utils.clip_grad import clip_grad_norm_
from MotionUtils import compute_gae_fast


class GaussianDistribution(nn.Module):
    """
    Add gaussian noise to the policy
    """
    def __init__(
        self,
        action_net: nn.Module,
        max_num_dim: int,
        scale: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.action_net: nn.Module = action_net

        self.max_num_dim: int = max_num_dim
        self.scale: float = scale
        scale_tensor: torch.Tensor = torch.full((max_num_dim,), scale, device=device)
        self.gauss_dist = Independent(Normal(loc=torch.zeros_like(scale_tensor), scale=scale_tensor), 1)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = self.action_net(input_data)
        return mean

    def sample(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mean = self.forward(input_data).view(-1)
            perturb: torch.Tensor = self.gauss_dist.sample()
            log_prob: torch.Tensor = self.gauss_dist.log_prob(perturb)

            sample: torch.Tensor = mean
            sample += perturb[:mean.numel()]

        return sample, log_prob

    def log_prob(self, input: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean: torch.Tensor = self.forward(input).view(input.shape[0], -1)
        sample: torch.Tensor = action - mean

        return mean, self.gauss_dist.log_prob(sample).reshape((-1, 1))


class Rollouts:

    __slots__ = (
        "s", "a", "r", "v",
        "ret", "logProb", "adv",
        "ref_motion_index", "num_sample",
        "terminate_index"
    )

    def __init__(self):
        self.s: List[np.ndarray] = []
        self.a: List[np.ndarray] = []
        self.r: List[np.ndarray] = []
        self.v: List[np.ndarray] = []
        self.ret: List[np.ndarray] = []
        self.logProb: List[np.ndarray] = []
        self.adv: List[np.ndarray] = [] # advantage

        self.ref_motion_index: List[int] = []  # use for track task
        self.num_sample: int = 0

        self.terminate_index: List[int] = []

    def extend(self, rollout):
        self.s.extend(rollout.s)
        self.a.extend(rollout.a)
        self.r.extend(rollout.r)
        self.v.extend(rollout.v)
        self.ret.extend(rollout.ret)
        self.logProb.extend(rollout.logProb)
        self.adv.extend(rollout.adv)
        self.ref_motion_index.extend(rollout.ref_motion_index)

        # here we should extend the terminate index
        self.terminate_index += [self.num_sample + node for node in rollout.terminate_index]
        self.num_sample += rollout.num_sample

    def clear(self):
        self.s.clear()
        self.a.clear()
        self.r.clear()
        self.v.clear()
        self.ret.clear()
        self.logProb.clear()
        self.adv.clear()
        self.ref_motion_index.clear()
        self.num_sample = 0
        self.terminate_index.clear()

    def get_length(self) -> List[int]:
        return [len(node) for node in self.s]

    def concat_states(self) -> np.ndarray:
        return np.concatenate(self.s, axis=0, dtype=np.float32)

    def concat_actions(self) -> np.ndarray:
        return np.concatenate(self.a, axis=0, dtype=np.float32)

    def concat_expert_actions(self) -> np.ndarray:
        return np.concatenate(self.expert_a, axis=0, dtype=np.float32)

    def concat_r(self) -> np.ndarray:
        return np.concatenate(self.r, axis=0, dtype=np.float32)

    def concat_value(self) -> np.ndarray:
        return np.concatenate(self.v, axis=0, dtype=np.float32)

    def concat_ret(self) -> np.ndarray:
        return np.concatenate(self.ret, axis=0, dtype=np.float32)

    def concat_log_prob(self) -> np.ndarray:
        return np.concatenate(self.logProb, axis=0, dtype=np.float32)

    def concat_adv(self) -> np.ndarray:
        return np.concatenate(self.adv, axis=0, dtype=np.float32)

    def concat_terminate(self) -> np.ndarray:
        return np.array(self.terminate_index)

    def average_reward(self):
        return np.mean(self.concat_r())

    def normalize_adv(self) -> np.ndarray:
        """
        Normalize the advantage by mean and std in training PPO
        """
        self.adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-8)
        return self.adv

    def normalize_ret(self) -> np.ndarray:
        """
        Normalize the discounted return by mean and std.
        """
        self.ret = (self.ret - np.mean(self.ret)) / (np.std(self.ret) + 1e-8)

    def build_dataloader(self, batch_size: int, expert_action: Optional[torch.Tensor] = None, device="cpu"):
        """
        return dataloader for training PPO
        """
        s: torch.Tensor = torch.from_numpy(np.concatenate(self.s, axis=0)).to(device)
        num_samples: int = s.shape[0]
        a: torch.Tensor = torch.from_numpy(np.concatenate(self.a, axis=0)).to(device).view(num_samples, -1)
        ret: torch.Tensor = torch.from_numpy(np.concatenate(self.ret, axis=0)).to(device)
        adv: torch.Tensor = torch.from_numpy(np.concatenate(self.adv, axis=0)).to(device)
        v0: torch.Tensor = torch.from_numpy(np.concatenate(self.v, axis=0)).to(device)
        logProb: torch.Tensor = torch.from_numpy(np.concatenate(self.logProb, axis=0)).to(device)

        dataset: List[torch.Tensor] = [
            s[:num_samples], a[:num_samples],
            ret[:num_samples], adv[:num_samples],
            v0[:num_samples], logProb[:num_samples]
        ]
        if expert_action is not None:
            dataset.append(expert_action[:num_samples])

        dataset = TensorDataset(*dataset)
        dataloader = DataLoader(dataset, batch_size, True)
        return dataloader, s, a, ret, adv, v0, logProb


class PPO:

    def __init__(
        self,
        args: Namespace,
        policy: GaussianDistribution,
        value: nn.Module,
        policy_optimizer: optim.Optimizer,
        value_optimizer: optim.Optimizer,
        floatType=np.float32
    ):
        self.args: Namespace = args
        self.device = torch.device("cpu")
        self.policy: GaussianDistribution = policy
        self.value: nn.Module = value
        self.floatType = floatType

        self.eps_clip: float = 0.2
        self.ratio_clip: float = 3.0
        self.value_clip: float = 0.2
        self.actor_grad_clip: Optional[float] = 3
        self.critic_grad_clip: Optional[float] = 3

        self.gamma: float = 0.95  # same setting as Deepmimic
        self.lamb: float = 0.95  # same setting as Deepmimic
        self.gammalamb: float = self.gamma * self.lamb
        self.value_coef: float = (1.0 - self.gamma) / 0.5

        self.value_optimizer: optim.Optimizer = value_optimizer
        self.policy_optimizer: optim.Optimizer = policy_optimizer

        # for combined action-value net
        self.assume_combined_action_value_net = False

        self.rollouts = Rollouts()

    def compute_gae(self, v: np.ndarray, r: np.ndarray, term: bool):
        """
        generalized advantage estimation (GAE), see https://arxiv.org/abs/1506.02438

        Args:
            v (np.ndarray): value in shape (num frame + 1,)
            r (np.ndarray): reward in shape (num frame,)
            term (bool): the rollout terminates

        Returns:
            _type_: _description_
        """
        adv: np.ndarray = np.zeros_like(r, dtype=np.float32)
        ret: np.ndarray = r.copy()
        if term:
            adv[-1] = 0
        else:
            adv[-1] = r[-1] + self.gamma * v[-1] - v[-2]
            ret[-1] = r[-1] + self.gamma * v[-1]
        for step in range(r.shape[0] - 2, -1, -1):
            delta = r[step] + self.gamma * v[step + 1] - v[step]
            ret[step] = r[step] + self.gamma * ret[step + 1]
            adv[step] = delta + self.gammalamb * adv[step + 1]

        return adv, ret

    def append_rollout(
        self,
        s: np.ndarray,  # state
        a: np.ndarray,  # action
        reward: np.ndarray, # reward
        log_prob: Optional[np.ndarray],
        ret: np.ndarray,  # discounted return
        adv: np.ndarray,  # advantage
        ref_motion_index: Optional[np.ndarray],
        v: Optional[np.ndarray]  # value
    ):
        """
        here we should also consider the terminate index..
        """
        self.rollouts.s.append(s[:-1])
        self.rollouts.a.append(a)

        if reward is not None:
            self.rollouts.r.append(reward)

        if log_prob is not None:
            self.rollouts.logProb.append(log_prob.reshape(-1, 1))

        if ret is not None:
            self.rollouts.ret.append(ret.reshape((-1, 1)))
        if adv is not None:
            self.rollouts.adv.append(adv.reshape((-1, 1)))
        if v is not None:
            self.rollouts.v.append(v.reshape((-1, 1))[:-1])

        if ref_motion_index is not None and len(ref_motion_index) > 0:
            self.rollouts.ref_motion_index.append(ref_motion_index)

        self.rollouts.num_sample += a.shape[0]
        # terminate index
        self.rollouts.terminate_index.append(self.rollouts.num_sample)

    def add_rollout(
        self,
        s: List[np.ndarray],
        a: List[np.ndarray],
        log_prob: List[np.ndarray],
        r: List[np.ndarray],
        term: bool,
        ref_motion_index: List[int],
        v: Optional[np.ndarray] = None,
        need_compute_v: bool = True
    ):
        if len(s) == 0:
            print("zero length rollout", flush=True)
            return

        s: np.ndarray = np.asarray(s, dtype=self.floatType)  # (num frame + 1, *)
        a: np.ndarray = np.asarray(a, dtype=self.floatType)  # (num frame, *)
        log_prob: np.ndarray = np.asarray(log_prob, dtype=self.floatType)
        r: np.ndarray = np.asarray(r, dtype=self.floatType).flatten()  # (num frame,)

        ret = adv = None
        if need_compute_v:
            if v is None: # v is not provided, we need to compute it
                with torch.no_grad():
                    self.value.eval()
                    s_tensor: torch.Tensor = torch.from_numpy(s).to(self.device)  # (num frame + 1, *)
                    v: np.ndarray = self.value(s_tensor).cpu().numpy().flatten()  # (num frame + 1, *)
            else:
                v: np.ndarray = np.asarray(v, dtype=self.floatType).reshape(-1)

            # adv, ret = self.compute_gae(v, r, term)
            adv, ret = compute_gae_fast(v, r, term, self.gamma, self.lamb)  # compute GAE using cython, which is faster..

        # append to the buffer
        self.append_rollout(s, a, r, log_prob, ret, adv, ref_motion_index, v)

    def clear(self):
        if self.rollouts is not None:
            self.rollouts.clear()

    def compute_loss(
        self,
        s_: torch.Tensor,
        a_: torch.Tensor,
        ret_: torch.Tensor,
        adv_: torch.Tensor,
        v0_: torch.Tensor,
        logProb_: torch.Tensor,
        expert_action_: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        cur_log_prob: Optional[torch.Tensor] = None,
        v_: Optional[torch.Tensor] = None,
        train_actor: bool = True,
        train_critic: bool = True
    ):
        """
        Compute ppo training loss here.
        """
        if action is None:
            action, cur_log_prob = self.policy.log_prob(s_, a_)
        if v_ is None:
            v_: torch.Tensor = self.value(s_)

        ratio: torch.Tensor = -logProb_ + cur_log_prob
        ratio: torch.Tensor = ratio.clamp(-self.ratio_clip, self.ratio_clip) # avoid nan in exp
        ratio: torch.Tensor = torch.exp(ratio)
        a_loss: torch.Tensor = -torch.min(ratio * adv_, ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_).mean()
        if not train_actor:
            a_loss = a_loss.detach()

        # try not to move too far from the last estimation
        v_clamp: torch.Tensor = v0_ + (v_ - v0_).clamp(-self.value_clip, self.value_clip)
        v_loss: torch.Tensor = self.value_coef * torch.mean(torch.max((v_-ret_)**2, (v_clamp-ret_)**2))
        if not train_critic:
            v_loss = v_loss.detach()

        loss: torch.Tensor = v_loss + a_loss
        # we can compute dagger loss in ppo training
        if expert_action_ is not None:
            dagger_loss = F.mse_loss(action, expert_action_)
            loss += self.args.dagger_weight * dagger_loss

        return loss

    def train(
        self,
        batch_size: int = 256,
        train_actor: bool = True,
        train_critic: bool = True,
        expert_action: Optional[torch.Tensor] = None,
        do_gradient_descent: bool = True
    ):
        self.policy.train()
        self.value.train()
        dataloader, s, a, ret, adv, v0, logProb = self.rollouts.build_dataloader(batch_size, expert_action, self.device)

        for batch_idx, data_tuple in enumerate(dataloader):
            if expert_action is None:
                s_, a_, ret_, adv_, v0_, logProb_ = data_tuple
                expert_action_ = None
            else:
                s_, a_, ret_, adv_, v0_, logProb_, expert_action_ = data_tuple

            loss = self.compute_loss(s_, a_, ret_, adv_, v0_, logProb_, expert_action_, None, None, None, train_actor, train_critic)

            # for training with multi environment, we need to average the gradient.
            # and we need to return the loss here, for average..
            if do_gradient_descent:
                loss.backward()
                # use gradient clip here.
                if train_critic:
                    if self.critic_grad_clip is not None:
                        clip_grad_norm_(self.value.parameters(), self.critic_grad_clip)
                    self.value_optimizer.step()

                if train_actor:
                    if self.actor_grad_clip is not None:
                        clip_grad_norm_(self.policy.parameters(), self.actor_grad_clip)
                    self.policy_optimizer.step()

                self.policy_optimizer.zero_grad(True)
                self.value_optimizer.zero_grad(True)

        self.rollouts: Rollouts = Rollouts()

        return loss, s.shape[0], v0.mean().cpu().numpy(), ret.mean().cpu().numpy(), adv.mean().cpu().numpy()


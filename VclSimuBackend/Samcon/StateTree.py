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
from typing import Optional, List


class Sample:

    __slots__ = ("parent", "s0", "s1", "a0", "cost", "t", "falldown", "index", "s0_info", "s1_info")
    # instance_cnt = 0

    def __init__(self, parent=None,
                 state0=None,
                 state1=None,
                 action: Optional[np.ndarray] = None,
                 cost: Optional[float] = 0.0,
                 t: int = 0,
                 falldown: Optional[bool] = False,
                 index: Optional[int] = 0,
                 s0_info = None,
                 s1_info = None):
        self.parent = parent
        self.s0 = state0
        self.s1 = state1

        self.a0: Optional[np.ndarray] = action

        self.cost: Optional[float] = cost
        self.t: int = t

        self.falldown: bool = falldown
        self.index: int = index

        self.s0_info = s0_info
        self.s1_info = s1_info

    def __del__(self):
        self.parent = None
        self.s0 = None
        self.s1 = None
        self.s0_info = None
        self.s1_info = None
        self.a0 = None
        # Sample.instance_cnt -= 1

    def copy(self):
        """
        Deep copy
        """
        return Sample(self.parent,
                      self.s0.copy() if self.s0 is not None else None,
                      self.s1.copy() if self.s1 is not None else None,
                      self.a0.copy() if self.a0 is not None else None,
                      self.cost,
                      self.t,
                      self.falldown,
                      self.index,
                      self.s0_info,
                      self.s1_info)

    def shallow_copy(self):
        return Sample(self.parent, self.s0, self.s1, self.a0, self.cost, self.t, self.falldown, self.index, self.s0_info, self.s1_info)

    def set_val(self, a0: Optional[np.ndarray],
                cost: float, s1, t: int,
                falldown: Optional[bool] = False):
        self.a0 = a0

        self.cost = cost
        self.s1 = s1
        # self.s1.to_continuous()
        self.t = t
        self.falldown = falldown

    def create_child(self):
        return Sample(self, self.s1, None, None, None, self.t, False, None, None, None)


class StateTree:

    __slots__ = ("tree",)

    def __init__(self):
        self.tree: List[List[Sample]] = []

    def __len__(self) -> int:
        return len(self.tree)

    def __getitem__(self, item):
        return self.tree[item]

    def insert(self, level: int, samples: Optional[List[Sample]] = None):
        samples = [] if samples is None else samples
        try:
            self.tree[level].extend(samples)
        except IndexError:
            self.tree.extend([[]] * (level + 1 - len(self.tree)))
            self.tree[level].extend(samples)

        return self

    @staticmethod
    def path_iterator(sample: Sample):
        node = sample
        while node:
            rt = node
            node = node.parent
            yield rt

    def path(self, level: int, idx: int) -> List[Sample]:
        node = self.tree[level][idx]
        path: List[Sample] = [x for x in self.path_iterator(node)]
        path.reverse()
        return path

    def reset(self):
        self.tree.clear()
        return self

    def num_level(self) -> int:
        return len(self.tree)

    def level(self, idx: int) -> List[Sample]:
        return self.tree[idx]

    def roll_back(self, nlevel: int, n_min_keep: int = 0):
        n = len(self.tree)
        nkeep = max(n_min_keep, n - nlevel)
        self.tree = self.tree[0:nkeep]

    def total_sample(self) -> int:
        return sum([len(level) for level in self.tree])

    def clear_dead_nodes(self, n: Optional[int]):
        n = len(self.tree) if n is None else n
        for i in reversed(range(n - 1)):
            this_level = self.tree[i]
            if len(this_level) == 1:
                break
            next_level = self.tree[i + 1]
            lived = {id(node.parent): node.parent for node in next_level}
            this_level.clear()
            this_level.extend([k[1] for k in lived.items()])
            # logging.info(f"Clear: in level {i}, {len(this_level)} in tree.")
        # gc.collect()

    def shallow_copy(self):
        res = StateTree()
        res.tree = [[sample.shallow_copy() for sample in level] for level in self.tree]
        return res

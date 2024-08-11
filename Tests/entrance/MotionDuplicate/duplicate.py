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
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from tqdm import tqdm
from typing import Optional
from VclSimuBackend.pymotionlib.BVHLoader import load, MotionData, save
from VclSimuBackend.Common.MathHelper import MathHelper


class SimpleMotionGraph:
    def __init__(self, fname: str):
        self.bvh = load(fname)  # load bvh
        self.bvh = self.bvh.sub_sequence(1)
        self.weights = np.ones(len(self.bvh.joint_names))
        self.noroot_weights = self.weights.copy()
        self.noroot_weights[0] = 0

    def dist(self, idx_a: int, idx_b: int, weight: Optional[np.ndarray] = None) -> float:
        ra: np.ndarray = Rotation(self.bvh.joint_rotation[idx_a], copy=False).as_rotvec()
        rb: np.ndarray = Rotation(self.bvh.joint_rotation[idx_b], copy=False).as_rotvec()
        if weight is None:
            weight = self.weights
        delta = np.dot(weight, np.linalg.norm(ra - rb, axis=-1)).item()
        return delta

    def get_last_pair(self):
        # for "sfu/0005_Jogging001-mocap.bvh", the best result is at:
        # best val = 0.63, pa = 547, pb = 730

        # for "sfu/0005_Walking001-mocap.bvh" the best result is at: 1097 1365/ or 626 758
        best_val = 1000.0
        pa, pb = 0, 0
        for i in range(10, self.bvh.num_frames):
            for j in range(i + 12, self.bvh.num_frames, 12):
                d = self.dist(i, j, self.noroot_weights)
                if d < best_val:
                    best_val = d
                    pa, pb = i, j
                    print(i, best_val, pa, pb, (pb - pa) % 12)
        exit(0)
            # print(f"i = {i}, best val = {best_val}, pa = {pa}, pb = {pb}")

        # for i in range(self.bvh.num_frames-100, -1, -1):
        #    for j in range(i + 100, self.bvh.num_frames):
        #        res = self.dist(i, j, self.noroot_weights)
        #        if res < 0.8:
        #            return i, j, res

    def append_seq(self, res: MotionData, ins: MotionData, dpos0, drot0):
        drot = drot0 * Rotation(res.joint_rotation[-1, 0]) * (Rotation(ins.joint_rotation[0, 0]).inv())
        ins.joint_rotation[:, 0, :] = (drot * Rotation(ins.joint_rotation[:, 0, :])).as_quat()

        delta_pos: np.ndarray = res.joint_position[-1, 0, :] - ins.joint_position[0, 0, :]
        ins.joint_translation[:, 0, :] += delta_pos + Rotation(res.joint_rotation[-1, 0]).inv().apply(dpos0)

        trans_0 = ins.joint_translation[0, 0, :]
        d_trans = ins.joint_translation[:, 0, :] - trans_0
        ins.joint_translation[:, 0, :] = trans_0 + drot.apply(d_trans)

        ins.recompute_joint_global_info()

        res.append(ins)

    def simple_duplicate(self, f_out: str = "test-dup.bvh"):
        start, end = 626, 758
        res = self.bvh.sub_sequence(start - 5, start, copy=True)
        dpos0 = self.bvh.joint_position[start, 0] - self.bvh.joint_position[start - 1, 0]
        drot0 = Rotation(self.bvh.joint_rotation[start, 0]) * Rotation(self.bvh.joint_rotation[start - 1, 0]).inv()
        for i in range(101):
            ins = self.bvh.sub_sequence(start, end, copy=True)
            self.append_seq(res, ins, dpos0, drot0)

        save(res, f_out)

    def mograph(self, f_out: str):
        start, end, dist = self.get_last_pair()
        eps = 5
        # sub = graph.bvh.sub_sequence(i - eps, j + eps)
        # reset root position
        win0 = self.bvh.sub_sequence(start - eps, start)
        win1 = self.bvh.sub_sequence(end - eps, end)
        win = win1.sub_sequence()
        # slerp without root rotation
        win._joint_rotation = MathHelper.slerp(win1.joint_rotation[:, 1:].reshape((-1, 4)),
                                               win0.joint_rotation[:, 1:].reshape((-1, 4)), 0.5).reshape((win1.num_frames, win1.num_joints - 1, 4))


def main():
    fdir = os.path.dirname(__file__)
    # f_in = os.path.join(fdir, "../../CharacterData/WalkF-mocap.bvh")
    f_in = ""
    # f_out = "../../CharacterData/duplicate-0005_Walking001-mocap.bvh" # + os.path.basename(f_in)
    graph = SimpleMotionGraph(f_in)
    graph.get_last_pair()
    # graph.simple_duplicate(f_out)


if __name__ == "__main__":
    main()

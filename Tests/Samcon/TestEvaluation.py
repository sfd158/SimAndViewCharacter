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
from VclSimuBackend.Utils.Evaluation import calc_nsr
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.Samcon.SamconCMA.MainWorkerCMANew import SamconMainWorkerCMA
from VclSimuBackend.Samcon.SamconMainWorkerBase import SamconMainHelper as SamHlp
from VclSimuBackend.Samcon.SamconMainWorkerRaw import WorkerInfo


def test_nsr():
    fdir = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Samcon\0"
    # fdir2 = r"D:\song\desktop\Samcon-Exps\noise-reduction-no-inv-dyn"
    fdir2 = r"D:\song\desktop\Samcon-Exps\noise-reduction-inv-dyn-dup-20"
    sim_motion = os.path.join(fdir2, "test-3.bvh")
    ref_motion = os.path.join(fdir, "motion-input.bvh")
    sim_motion = BVHLoader.load(sim_motion)
    ref_motion = BVHLoader.load(ref_motion)
    nsr = calc_nsr(sim_motion, ref_motion)
    print(f"nsr = {nsr}")



def test_loss():
    fdir = os.path.dirname(__file__)
    fdir2 = r"D:\song\desktop\Samcon-Exps\noise-reduction-inv-dyn-dup-20"
    fdir3 = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Samcon\0"
    # sim_motion = os.path.join(fdir3, "with-invdyn-finetune-samcon/test-traj-opt.bvh")
    sim_motion = os.path.join(fdir2, "test-3.bvh")
    main_worker = SamconMainWorkerCMA(SamHlp(os.path.join(fdir, "../CharacterData/SamconConfig.json"), "0"), WorkerInfo())
    main_worker.n_iter = main_worker.init_n_iter
    main_worker.bvh_loss_eval(sim_motion)


if __name__ == "__main__":
    test_nsr()

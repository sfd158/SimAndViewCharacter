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
loss using Open Dynamics Engine should be close to loss using DiffODE and PyTorch
"""
import os
from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import *


class TestPyTorchLoss(DirectTrajOptBVH):
    def __init__(
        self,
        samhlp: SamHlp,
        worker_info: WorkerInfo, worker: Optional[SamconWorkerFull],
        scene: Optional[ODEScene] = None, sim_character: Optional[ODECharacter] = None):
        super().__init__(samhlp, worker_info, worker, scene, sim_character)

    def test_loss(self):
        for i in range(233):
            self.worker.scene.damped_simulate_once()
            ode_loss, ode_dict = self.worker.loss.loss_debug(i * self.sim_cnt)
            self.worker.diff_world.import_curr_frame()
            # check compute joint pos in ode and diff-ode
            # ode_joint_pos = self.worker.character.joint_info.get_global_anchor1()
            # diff_joint_pos = self.worker.diff_world.curr_frame.compute_joint_pos()
            # print(np.max(np.abs(ode_joint_pos - diff_joint_pos.detach().numpy().squeeze())))

            torch_loss, torch_dict = self.worker.torch_loss.loss_debug(i * self.sim_cnt)
            print(ode_loss - torch_loss)



if __name__ == "__main__":
    info = WorkerInfo()
    samhlp = SamHlp(os.path.join(os.path.dirname(__file__), "../../CharacterData/SamconConfig-duplicate.json"))
    worker = SamconWorkerFull(samhlp, info)
    main_worker = TestPyTorchLoss(samhlp, info, worker, worker.scene)
    main_worker.test_loss()

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
This file is used for visualize samcon training...
"""

import os
import threading
from typing import Optional, Dict, Any
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, UpdateSceneBase, ODEScene, ServerThreadHandle
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
from VclSimuBackend.Samcon.SamconWorkerFull import SamconWorkerFull, WorkerInfo
from VclSimuBackend.Samcon.SamconWorkerBase import SamHlp
from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import DirectTrajOptBVH


file_dir = os.path.dirname(__file__)
conf_dir = os.path.join(file_dir, "../CharacterData/SamconConfig-duplicate.json")
run_mode = "traj-opt"


class UpdateSceneOptim(UpdateSceneBase):

    def __init__(
        self,
        scene: Optional[ODEScene] = None,
        sim_character: Optional[ODECharacter] = None,
        ref_character: Optional[ODECharacter] = None,
        ):
        super().__init__(scene=scene)

        self.sim_character: ODECharacter = sim_character
        self.ref_character: Optional[ODECharacter] = ref_character
        character_list = self.scene.characters
        if ref_character is not None:
            self.scene.characters = [self.sim_character]

        worker_info = WorkerInfo()
        samhlp = SamHlp(conf_dir)
        worker = SamconWorkerFull(samhlp, worker_info, self.scene, sim_character)
        worker.as_sub_thread = True
        self.main_worker = DirectTrajOptBVH(samhlp, worker_info, worker, self.scene, sim_character)
        self.optim_task = threading.Thread(target=self.forward_wrapper)
        if ref_character is not None:
            self.tar_set = SetTargetToCharacter(ref_character, self.main_worker.target.pose)
        else:
            self.tar_set = None

        self.scene.characters = character_list
        ServerThreadHandle.pause_sub_thread()
        self.optim_task.start()

    def forward_wrapper(self):
        try:
            if run_mode == "traj-opt":
                self.main_worker.test_direct_trajopt()
            elif run_mode == "cma-single":
                self.main_worker.run_single_hlp()
        except ValueError as err:
            print(err)
            raise SystemExit()

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        if not self.optim_task.is_alive():
            raise SystemError("Trajectory Optimization finished.")
        if ServerThreadHandle.sub_thread_is_running():
            return True
        ServerThreadHandle.resume_sub_thread()
        ServerThreadHandle.wait_sub_thread_run_end()
        if self.ref_character is not None:
            self.tar_set.set_character_byframe(self.sim_character.curr_frame_index)

        return False


class OptimVisServer(ServerForUnityV2):
    def __init__(self, view_ref_character: bool = False):
        super().__init__(None, None)
        if view_ref_character:
            self.init_instruction_buf = {"DupCharacterNames": [self.gt_str, self.sim_str]}

    def after_load_hierarchy(self):
        self.update_scene = UpdateSceneOptim(self.scene, *self.select_sim_ref())


def main():
    server = OptimVisServer()
    server.run()


if __name__ == "__main__":
    main()

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
import argparse
import os
from typing import Optional, List, Any, Dict

from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2
from VclSimuBackend.Samcon.SamconWorkerFull import SamconWorkerFull
from VclSimuBackend.Samcon.SamconWorkerBase import WorkerInfo, SamconWorkerBase
from VclSimuBackend.Samcon.SamconCMA.MainWorkerCMANew import SamconMainWorkerCMA, SamHlp
from VclSimuBackend.Samcon.SamconUpdateScene import (
    SamconUpdateSceneWithRef,
    UpdateSceneBVHDirectWrapper,
    Sample,
    SamconUpdateScene,
)


# def render1():
#     loader = JsonSceneLoader()
#     scene = loader.load_from_file(conf["filename"]["world"])
#
#     target = SamconTargetPose()
#     target.load(conf, scene.character0, scene)
#
#     update = SamconUpdateScene(scene, scene.character0, best_path, target, conf)
#     server = ServerForUnity(scene, update)
#     server.run()


class ServerForUnityV2Samcon(ServerForUnityV2):
    def __init__(
        self,
        samhlp: SamHlp,
        best_path_: List[Sample],
        render_ref: bool = True
        ):
        super(ServerForUnityV2Samcon, self).__init__()
        self.samhlp = samhlp
        self.best_path = best_path_

        if render_ref:
            self.init_instruction_buf = {"DupCharacterNames": [self.gt_str, self.sim_str]}

        self.load_scene_hook = lambda scene_, json_world_: SamconWorkerBase.load_scene_with_conf(self.samhlp.conf, scene_, json_world_)

    def after_load_hierarchy(self):
        sim_character, ref_character = self.select_sim_ref()
        SamconWorkerBase.handle_contact_conf(self.scene, self.samhlp.conf, sim_character)
        info = WorkerInfo()
        worker = SamconWorkerFull(self.samhlp, info, self.scene, sim_character)
        main_worker = SamconMainWorkerCMA(self.samhlp, info, worker, worker.scene, sim_character)
        self.scene.extract_contact = True
        main_worker.send_target_pose_no_scatter()
        samcon_update = SamconUpdateScene(worker, self.best_path)

        if ref_character is not None:
            bvh_update = UpdateSceneBVHDirectWrapper(self.scene, main_worker.target.pose, ref_character)
        else:
            bvh_update = None

        self.update_scene = SamconUpdateSceneWithRef(self.scene, samcon_update, bvh_update)

    @staticmethod
    def build_from_file(samhlp: SamHlp, render_ref: bool = True):
        best_path: List[Sample] = samhlp.load_best_path_idx()["best_path"]
        server = ServerForUnityV2Samcon(samhlp, best_path, render_ref)
        return server


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_ref", action="store_true")
    args = parser.parse_args()

    fdir = os.path.dirname(__file__)
    fname = os.path.join(fdir, "../CharacterData/SamconConfig-duplicate.json")
    server = ServerForUnityV2Samcon.build_from_file(SamHlp(fname, 0), True)
    server.run()


if __name__ == "__main__":
    test()

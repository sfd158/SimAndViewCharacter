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
Visualize the optimized result in Unity
"""
from argparse import ArgumentParser, Namespace
import os
import pickle
from typing import List, Tuple, Optional, Any, Dict
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter, TargetPose
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.ODEScene import ODEScene, SceneContactInfo
from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, UpdateSceneBase


fdir = os.path.dirname(__file__)


class UpdateSceneForOptimize(UpdateSceneBase):
    def __init__(self, scene: Optional[ODEScene], target: TargetPose, eval_scene_contact: List):
        super().__init__(scene)
        self.target = target
        self.set_tar = SetTargetToCharacter(self.character0, target)
        self.tot_frame = self.target.num_frames
        self.frame = 0
        self.eval_scene_contact = eval_scene_contact

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        """
        # 1. set character pose
        # 2. create contact
        """
        self.set_tar.set_character_byframe(self.frame)
        self.scene.contact_info = self.eval_scene_contact[self.frame]  # here we can visuzlize in Unity..
        if self.scene.contact_info is not None:
            self.scene.contact_info.pos = self.scene.contact_info.pos.reshape((-1, 3))
        print(self.frame, len(self.scene.contact_info))
        self.frame = (self.frame + 1) % self.tot_frame


class ServerForOptim(ServerForUnityV2):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args: Namespace = args
        track_bvh_fname = os.path.join(args.input_dir, "eval-mocap-predict.bvh")
        self.bvh = BVHLoader.load(track_bvh_fname)
        self.bvh = self.bvh.resample(args.simulation_fps)

    def after_load_hierarchy(self):
        """
        The input format is as follows:

        """
        args: Namespace = self.args
        # here we should load the character bvh file.
        self.scene.set_sim_fps(self.args.simulation_fps)
        self.scene.extract_contact = True
        print(f"the simulation fps is {self.scene.sim_fps}")
        target: TargetPose = BVHToTargetBase(self.bvh, self.scene.sim_fps, self.scene.character0).init_target()
        with open(os.path.join(args.input_dir, "network-output.bin"), "rb") as fin:
            optim_result: Dict[str, Any] = pickle.load(fin)
        eval_scene_contact: List[SceneContactInfo] = optim_result["extract_contact_force"]
        if args.reduce_contact_force:
            """
            merge the contact force here..
            1. compute the total contact force
            2. divide each contact force with the length.
            """
            for frame in range(len(eval_scene_contact) - 1):
                scene_contact_info = eval_scene_contact[frame]
                if scene_contact_info is not None:
                    scene_contact_info = scene_contact_info.merge_force_by_body1()
                    scene_contact_info.pos = scene_contact_info.pos.reshape((-1, 3))
                eval_scene_contact[frame] = scene_contact_info
            eval_scene_contact[-1] = eval_scene_contact[-2]
            print(f"After reduce the contact force", flush=True)
        self.update_scene = UpdateSceneForOptimize(self.scene, target, eval_scene_contact)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str,
        default=r"Z:\GitHub\ode-develop\VclSimuBackend\Samcon\optimize_S11-with-simple-plan-400-epoch\SittingDown1-mocap\opt_result"
    )
    parser.add_argument("--reduce_contact_force", action="store_true", default=False)
    parser.add_argument("--simulation_fps", type=int, default=50)
    parser.add_argument("--render_ground_truth", action="store_true", default=False)
    args = parser.parse_args()
    print(f"input_dir = {args.input_dir}")
    server = ServerForOptim(args)
    server.run()


if __name__ == "__main__":
    main()

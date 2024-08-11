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
Visualize in Unity
1. contact position
2. contact force
3. support next phase button of Unity
"""
import os
import threading
from typing import Optional, Dict, Any
from ..SimpleCIO.SimpleTrack import SimpleTrack
from ..ODESim.ODEScene import ODEScene
from ..pymotionlib.MotionData import MotionData
from ..pymotionlib import BVHLoader
from ..Server.v2.ServerForUnityV2 import ServerForUnityV2, ServerThreadHandle, UpdateSceneBase


class SimpleCIOUpdateScene(UpdateSceneBase):
    def __init__(self, scene: Optional[ODEScene] = None):
        super().__init__(scene=scene)

        fdir = os.path.dirname(__file__)
        # bvh_fname = os.path.join(fdir, "../../Tests/CharacterData/Human3.6/S1/Walking-short-mocap.bvh")
        # bvh_fname = os.path.join(fdir, "../../Tests/CharacterData/WalkF-mocap.bvh")
        bvh_fname = "noise.bvh"
        # bvh_fname = os.path.abspath(bvh_fname)
        if not os.path.exists(bvh_fname):
            print(f"{bvh_fname} not exists")
        motion: MotionData = BVHLoader.load(bvh_fname)

        self.tracker = SimpleTrack(scene, scene.character0, motion)
        self.tracker.run_as_sub_thread = True
        ServerThreadHandle.pause_sub_thread()

        self.frame = 0
        self.total_frames: int = motion.num_frames

        self.optim_task = threading.Thread(target=self.tracker.train)
        self.optim_task.start()

    def should_go_next_phase(self):
        if not self.world_signal:
            return False
        for signal in self.world_signal:
            if signal["GoNextPhase"]:
                return True
        return False

    # def update(self, mess_dict: Optional[Dict[str, Any]] = None):
    #     if self.frame < self.total_frames:
    #         pass
    #     else:
    #         ServerThreadHandle.resume_sub_thread()
    #         ServerThreadHandle.wait_sub_thread_run_end()
    #         self.frame = 0

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        if not self.optim_task.is_alive():
            raise SystemError("Trajectory Optimization finished.")
        if ServerThreadHandle.sub_thread_is_running():
            return True

        go_next_phase: bool = self.should_go_next_phase()
        if go_next_phase:  # do a forward step here..
            self.move_next_phase()

        # set current pose to character
        # and set current contact state to character
        self.tracker.extract_current_state(self.frame)
        self.frame = (self.frame + 1) % self.total_frames

        return False

    def move_next_phase(self):
        ServerThreadHandle.resume_sub_thread()
        ServerThreadHandle.wait_sub_thread_run_end()
        self.frame = 0
        print(f"move to next phase")
        if self.tracker.loss_dict:
            print(f"\n=========At epoch {self.tracker.epoch}=============")
            for key, value in self.tracker.loss_dict.items():
                print(f"{key}: {value.item()}")


class SimpleCIOServer(ServerForUnityV2):
    def __init__(self):
        super().__init__()

    def after_load_hierarchy(self):
        self.update_scene = SimpleCIOUpdateScene(self.scene)

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

from argparse import ArgumentParser, Namespace
import os
import time
from typing import Optional
import ModifyODE as ode
import numpy as np

try:
    from ..ODESim.ODEScene import ODEScene
except:
    from VclSimuBackend.ODESim.ODEScene import ODEScene

from ModifyODE import RenderWorld as RenderWorldCython

class RenderWorld(RenderWorldCython):
    def __init__(self, myworld, space=None, geoms=None):
        super().__init__(myworld, space, geoms)
        self.recorder = None
    
    def create_video_recorder(self, export_fname: str):
        import cv2
        if self.recorder is None:
            self.recorder = cv2.VideoWriter(filename=export_fname, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=60, frameSize=self.get_video_size())

    def end_record_video(self):
        self.pause_record_video()
        if self.recorder is not None:
            self.recorder.release()

    def append_to_recorder(self):
        if self.recorder is None:
            return
        import cv2
        video = self.get_record_buffer()
        # maybe we should start another process to call this function..
        for frame_id in range(video.shape[0]):
            self.recorder.write(cv2.flip(video[frame_id], -1))


def main():
    parser = ArgumentParser()
    parser.add_argument("--fname", type=str,
        default=r"f:\GitHub\ControlVAE\smpl-zero-shape.json")
    parser.add_argument("--bvh_fname", type=str, default="")
    args = parser.parse_args()
    # args.fname = r"D:\song\documents\GitHub\ControlVAE\Data\Misc\world-modify.json"
    # args.fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\StdHuman\world.json"
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
    scene = ODEScene()
    if os.path.exists(args.fname):
        print(f"Load scene from {args.fname}", flush=True)
        scene = JsonSceneLoader(scene).load_from_file(args.fname)
        scene.character0.compute_root_inertia()
    else:
        print(f"File {args.fname} not exist. Ignore.")
    character = scene.character0
    render = RenderWorld(scene.world)
    render.draw_background(1)
    # render.draw_localaxis(1)
    # render.draw_hingeaxis(1)
    render.start()
    # render.start_record_video()

    for i in range(20000):
        
        time.sleep(0.01)

    # render.end_record_video()

if __name__ == '__main__':
    main()

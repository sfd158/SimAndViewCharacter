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
import torch

from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.SimpleCIO.SimpleTrack import SimpleTrack
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData


def main():
    fdir = os.path.dirname(__file__)
    bvh_fname = "noise.bvh"
    motion: MotionData = BVHLoader.load(bvh_fname)

    gt_bvh_fname = "../CharacterData/WalkF-mocap-100.bvh"
    gt_motion = BVHLoader.load(os.path.join(fdir, gt_bvh_fname)).remove_end_sites()

    scene_fname: str = os.path.join(fdir, "../CharacterData/stdhuman-400-50-foot0.2.pickle")
    scene: ODEScene = JsonSceneLoader().load_from_pickle_file(scene_fname)
    tracker = SimpleTrack(scene, scene.character0, motion)
    tracker.train()


if __name__ == "__main__":
    main()

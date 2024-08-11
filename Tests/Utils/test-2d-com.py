
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
Test pipeline:
1. load character and bvh file, then project to 2d, then compute 2d com
2. check numpy version and pytorch version
"""
import os
import torch
from typing import Optional
import numpy as np
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.TargetPose import TargetPose
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase

from VclSimuBackend.Utils.ComputeCom2d import Joint2dComIgnoreHandToe
from VclSimuBackend.Utils.ComputeCom2dPyTorch import Joint2dComPyTorch
from VclSimuBackend.Utils.Camera.CameraNumpy import CameraParamNumpy
from VclSimuBackend.Utils.Camera.Human36CameraBuild import CameraParamBuilder


fdir = os.path.dirname(__file__)

def main():
    scene_fname = os.path.join(fdir, "../CharacterData/Samcon-Human.pickle")
    bvh_fname = os.path.join(fdir, "../CharacterData/sfu/0005_Jogging001-mocap-100.bvh")
    motion: MotionData = BVHLoader.load(bvh_fname)

    loader = JsonSceneLoader()
    scene = loader.load_from_pickle_file(scene_fname)
    scene.set_sim_fps(100)

    character = scene.character0
    camera_dict = CameraParamBuilder.build(dtype=np.float64)
    camera: CameraParamNumpy = camera_dict["S1"][0]

    target: TargetPose = BVHToTargetBase(motion, scene.sim_fps, character).init_target()
    # shape == (frame, total joint, 3)
    world_3d_pos: np.ndarray = target.all_joint_global.pos
    camera_3d_pos: np.ndarray = camera.world_to_camera(world_3d_pos, True)
    camera_2d_pos: np.ndarray = camera.project_to_2d_linear(camera_3d_pos)

    joint2d_com = Joint2dComIgnoreHandToe().build(character)
    com2d: np.ndarray = joint2d_com.calc(camera_2d_pos)

    # print(com2d.shape)
    # print(com2d[0])
    # print(np.mean(camera_2d_pos[0], axis=0))

    # test use pytorch
    camera_2d_pos_torch: Optional[torch.Tensor] = torch.as_tensor(camera_2d_pos, dtype=torch.float32)
    joint2d_com_torch = Joint2dComPyTorch.build_from_numpy(joint2d_com, dtype=torch.float32)
    torch_com2d: torch.Tensor = joint2d_com_torch.calc(camera_2d_pos_torch)
    print("delta", np.max(np.abs(torch_com2d.detach().numpy() - com2d)))


if __name__ == "__main__":
    main()

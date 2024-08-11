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

import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog
from typing import Any, Dict, List, Optional


from VclSimuBackend.Server.v2.ServerForUnityV2 import ServerForUnityV2, UpdateSceneBase
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter, TargetPose
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH

from VclSimuBackend.SMPL.stdhuman2SMPL import build_smpl_mirror_index
from VclSimuBackend.pymotionlib.Utils import flip_quaternion
from VclSimuBackend.Common.SmoothOperator import GaussianBase, smooth_operator


class UpdateSceneMultiBVH(UpdateSceneBase):
    def __init__(self, scene: ODEScene, motion_input: List[MotionData]):
        super(UpdateSceneMultiBVH, self).__init__(scene)
        self.motion_input: List[MotionData] = motion_input
        self.max_num_frames: int = max([node.num_frames for node in self.motion_input])
        # min_fps = max([node.fps for node in self.motion_input])
        # self.scene.set_sim_fps(min_fps)
        self.targets, self.tar_sets = [], []
        print(f"num characters = {len(self.scene.characters)}")
        for i in range(len(self.motion_input)):
            character = self.scene.characters[i]
            target = BVHToTargetBase(self.motion_input[i], self.scene.sim_fps, character).init_target()
            tar_set = SetTargetToCharacter(character, target)
            self.targets.append(target)
            self.tar_sets.append(tar_set)

        self.index: int = 0
        self.accum_time = 0.0

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        # for _ in self.scene.step_range():
        for ch_idx, motion in enumerate(self.motion_input):
            if self.index >= self.motion_input[ch_idx].num_frames:
                continue
            self.tar_sets[ch_idx].set_character_byframe(self.index)
        if self.index == 0:
            curr_time = time.time()
            if (self.accum_time != 0):
                print(curr_time - self.accum_time, flush=True)
            self.accum_time = curr_time
        self.index: int = (self.index + 1) % self.max_num_frames


class ServerMultiBVH(ServerForUnityV2):
    def __init__(self, bvh_fname_list: List[str]):
        super(ServerMultiBVH, self).__init__()
        self.init_instruction_buf = {
            "DupCharacterNames": [os.path.basename(node)[:-4] for node in bvh_fname_list]
        }
        # Note: the motion should be left-right mirrored, for rendering in Unity
        self.motion_input = [BVHLoader.load(bvh_fname).resample(120).remove_end_sites() for bvh_fname in bvh_fname_list]
        # here we should remove rotation of toe joint..
        for joint_name in ["R_Foot_and_R_Ankle", "L_Foot_and_L_Ankle"]:
            if joint_name in self.motion_input[0].joint_names:
                self.motion_input[0].joint_rotation[:, self.motion_input[0].joint_names.index(joint_name), :] = np.array([0.0, 0.0, 0.0, 1.0])
        self.motion_input[0].recompute_joint_global_info()
        # here we need to extend the hand and toe..

        if True:
            mirror_index: np.ndarray = build_smpl_mirror_index(self.motion_input[0].joint_names)
            # print(mirror_index, flush=True)
            vector_x: np.ndarray = np.array([1.0, 0.0, 0.0])
            mocap = self.motion_input[0]
            mirror_root_pos: np.ndarray = mocap.joint_translation[:, 0, :].copy()
            mirror_root_pos[..., 0] *= -1

            mirror_joint_quat: np.ndarray = flip_quaternion(mocap.joint_rotation.reshape(-1, 4), vector_x, False).reshape(mocap.joint_rotation.shape)
            mirror_joint_quat: np.ndarray = np.ascontiguousarray(mirror_joint_quat[:, mirror_index])
            mocap._joint_rotation = mirror_joint_quat
            mocap.recompute_joint_global_info()
            self.motion_input = [mocap]

    def modify_SingleLegJump0(self):
        """
        load target pose..
        """
        motion = self.motion_input[0]
        nf = motion.num_frames
        target = BVHToTargetBase(motion, self.scene.sim_fps, self.scene.character0).init_target()
        tar_set = SetTargetToCharacter(self.scene.character0, target)
        dh_list = np.zeros((nf,))
        aabb_list = []
        root_h_list = []
    
        for i in range(nf):
            tar_set.set_character_byframe(i)
            aabb = self.scene.character0.get_aabb()[2]
            root_h = self.scene.character0.root_body.PositionNumpy[1]
            aabb_list.append(aabb)
            root_h_list.append(root_h)
        
        aabb_list = np.array(aabb_list)
        root_h_list = np.array(root_h_list)
        # print(aabb_list)
        # print(root_h_list)
        fk_list = [0, 100, 200, 260, 380, 450, 550, 600, 650, 700]
        x = 10
        dh_list = 0.2 * root_h_list

        dh_list[0:20] = 0
        dh_list[200-x:200+x] = 0
        # dh_list[100-x:100+x] = 0
        # dh_list[260-x:260+x] = 0
        # dh_list[450-x:450+x] = 0
        # dh_list[600-x:600+x] = 0
        # dh_list[690-x:] = 0
        dh_list = smooth_operator(dh_list, GaussianBase(7))
        """
        dh_list[0:100] = 1.2 * root_h_list[0:100]
        dh_list[100:200]
        dh_list[200:260]
        dh_list[260:380]
        dh_list[380:450]
        dh_list[450:550]
        dh_list[550:600]
        dh_list[600:650]
        dh_list[650:700]
        """
        
        motion.joint_translation[:, 0, 1] += dh_list
        motion.recompute_joint_global_info()
        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.plot(aabb_list)
        # plt.subplot(122)
        # plt.plot(root_h_list)
        # plt.show()

    def smooth_arm(self):
        motion = self.motion_input[0]
        nf = motion.num_frames
        for jname in ['R_Elbow_and_R_Shoulder', 'L_Elbow_and_L_Shoulder', 'L_Collar_and_Spine3', 'R_Collar_and_Spine3']:
            jindex = motion.joint_names.index(jname)
            motion.joint_rotation[:, jindex, :] = MathHelper.vec6d_to_quat(smooth_operator(MathHelper.quat_to_vec6d(motion.joint_rotation[:, jindex, :]).reshape(nf, 6), GaussianBase(3)).reshape(nf, 3, 2))
        motion.recompute_joint_global_info()

    def smooth_whole_body(self):
        motion = self.motion_input[0]
        nf = motion.num_frames
        for jindex in range(motion.num_joints):
            motion.joint_rotation[:, jindex, :] = MathHelper.vec6d_to_quat(smooth_operator(MathHelper.quat_to_vec6d(motion.joint_rotation[:, jindex, :]).reshape(nf, 6), GaussianBase(5)).reshape(nf, 3, 2))
        motion.recompute_joint_global_info()

    def after_load_hierarchy(self):
        self.scene.set_sim_fps(120)
        self.scene.set_render_fps(120)
        to_bvh = CharacterTOBVH(self.scene.character0)
        to_bvh.build_hierarchy()

        default_hierarchy: MotionData = to_bvh.motion.sub_sequence()
        nf = default_hierarchy._num_frames = self.motion_input[0].num_frames
        default_hierarchy._joint_position = np.zeros((nf, default_hierarchy.num_joints, 3))
        default_hierarchy._joint_translation = np.zeros_like(default_hierarchy.joint_position)
        default_hierarchy._joint_rotation = np.zeros((nf, default_hierarchy.num_joints, 4))
        default_hierarchy._joint_rotation[..., 3] = 1
        default_hierarchy._joint_orientation = default_hierarchy.joint_rotation.copy()

        d_dict = {node: index for index, node in enumerate(default_hierarchy.joint_names)}
        for joint_index, joint_name in enumerate(self.motion_input[0].joint_names):
            d_index = d_dict[joint_name]
            default_hierarchy.joint_rotation[:, d_index, :] = self.motion_input[0].joint_rotation[:, joint_index, :]
        default_hierarchy.joint_translation[:, 0, :] = self.motion_input[0].joint_translation[:, 0, :]
        default_hierarchy.recompute_joint_global_info()
        self.motion_input = [default_hierarchy]

        # self.modify_SingleLegJump0()
        # self.smooth_arm()
        # self.smooth_whole_body()
        # default_hierarchy.joint_translation[:, 0, [0, 2]] = default_hierarchy.joint_translation[0, 0, [0, 2]]
        
        default_hierarchy.recompute_joint_global_info()
        self.update_scene = UpdateSceneMultiBVH(self.scene, self.motion_input)        


def get_bvh_fname():
    root = tk.Tk()
    root.withdraw()
    init_dir = os.path.join(os.path.dirname(__file__), r"D:\downloads\policy_iter")
    return filedialog.askopenfilename(initialdir=init_dir, filetypes=[("all_file_types", "*.bvh")])


def select_bvh_files():
    file_list: List[str] = []
    while True:
        fname: Optional[str] = get_bvh_fname()
        if fname:
            file_list.append(fname)
        else:
            break
    print(f"select {len(file_list)} bvh file(s). ")
    if len(file_list) == 0:
        exit(0)
    for node in file_list:
        print(node)

    return file_list

def main():
    # file_list = get_bvh_fname()
    # print(file_list)
    # Initialize server
    # file_list = [r"D:\s
    # file_list = [r"F:\GitHub\ControlVAE\Experiments\WildVideo\PencilRoll0\multi-optim.bvh"]
    # file_list = [r"F:\ControlVAEPose\PredContact\OptInput\S11\Photo.54138969\result.bvh"]
    # file_list = [r"H:\desktop\fuck-test\result.bvh"]
    # file_list = [r"F:\ControlVAEPose\DemoVideo\recalc-dir-is-optd\SingleLegJump0\result.bvh"]
    # file_list = [r"F:\ControlVAEPose\DemoVideo\recalc-dir-is-optd\ShoulderRoll\result.bvh"]
    # file_list = [r"F:\ControlVAEPose\DemoVideo\recalc-dir-is-optd\CrabWalk0\result.bvh"]
    # file_list = [r""]
    # file_list = [r"F:\GitHub\ControlVAE/SingleLegJump0-0-seem-ok.bvh"]
    # file_list = [r"F:\GitHub\ControlVAE\ShoulderRoll-100-300-0-seems-ok.bvh"]
    # maybe we can train a policy only supports default smpl character..
    # file_list = [r"F:\GitHub\ode-scene\refine-s11-sittingdown.bvh"]
    # file_list = [r"F:\GitHub\ControlVAE\S11-directions.bvh"]
    file_list = [r"H:\desktop\SingleLegJump0-0.bvh"]
    # file_list = [r"F:\ControlVAEPose\human36\recon-result\s11-directions-right\res-sim.bvh"]
    
    # file_list = [r"F:\GitHub\ControlVAE\SingleLegJump0-0-seem-ok.bvh"]
    # file_list = [r"F:\GitHub\ControlVAE\Experiments\S11-directions\OptimRawData\result.bvh"]
    # file_list = [r"F:\ControlVAEPose\human36\recon-result\s11-directions-mpc-old\res-sim.bvh"]
    # file_list = [r"F:\ControlVAEPose\DemoVideo\recalc-dir-is-optd\SingleLegJump0\result.bvh"]
    # file_list = [r"F:\GitHub\ControlVAE\Experiments\WildVideo\SingleLegJump0\multi-optim.bvh"]
    # file_list = [r"F:\GitHub\ControlVAE\parkour-seem-ok.bvh"]
    # file_list = [r"F:\ControlVAEPose\DemoVideo\raw-bvh-dir\parkour-0\result.bvh"]
    server = ServerMultiBVH(file_list)
    server.run()


if __name__ == "__main__":
    main()

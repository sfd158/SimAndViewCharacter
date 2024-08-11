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
As character will hang in the air for kinematics motion predicted by neural network,
we should modify the height of character..

We can use all of these methods.

Method 1:
if height of body with max contact probability doesn't match,
we should move this body to floor.
Then, smooth the body height by Gaussian filter.

Method 2:
Optimize the body height only, and the loss term is:
2d projection loss, close to initial motion, smooth term.
"""

import numpy as np
import pickle
import os
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from typing import Dict, Optional

from VclSimuBackend.Common.SmoothOperator import GaussianBase, smooth_operator
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter, TargetPose
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.DiffODE import DiffQuat
from VclSimuBackend.Utils.Dataset.StdHuman import stdhuman_with_root_to_unified_index as sub_index
from VclSimuBackend.Utils.MothonSliceSmooth import smooth_motion_data
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData


fdir = os.path.dirname(__file__)

"""
for wild video, The camera may have a top or bottom view. but the angle error is small...
"""

def pre_process_motion_method1(
    input_data: str,
    gaussian_width: Optional[int] = 3,
    output_bvh_fname: str = "pred-motion-method1.bvh",
    add_global_translation: bool = False
):
    in_fname = input_data
    in_dir = os.path.dirname(input_data)
    with open(input_data, "rb") as fin:
        input_data: Dict = pickle.load(fin)
    motion: MotionData = BVHLoader.load(os.path.join(in_dir, input_data["pred_motion"]))

    # target: TargetPose = BVHToTargetBase(motion, motion.fps, character).init_target()
    # tar_set = SetTargetToCharacter(character, target)
    # contact_label: np.ndarray = input_data["pred_contact_label"]
    # num_frame = min(motion.num_frames, contact_label.shape[0])
    # num_body = len(character.bodies)
    # here we should also smooth the contact label

    if gaussian_width is not None and gaussian_width > 0:
        # TODO: do we need to normalize the contact label?
        # That is, by min-max normalize, or sigmoid, to make sure the contact label is bewteen [0, 1] ?
        motion = smooth_motion_data(motion, GaussianBase(2), None, False, True)  # smooth the joint rotation
        input_data["pred_contact_label"] = smooth_operator(input_data["pred_contact_label"], GaussianBase(gaussian_width))
        # we should also smooth the inverse dynamics result. firstly, convert to 6d vector. Then, smooth the result..
        invdyn = input_data["invdyn_target"]
        for joint_index in range(invdyn.shape[1]):
            vec6d = smooth_operator(MathHelper.quat_to_vec6d(invdyn[:, joint_index, :]), GaussianBase(gaussian_width))
            invdyn[:, joint_index, :] = MathHelper.vec6d_to_quat(vec6d, normalize=True)
        input_data["invdyn_target"] = invdyn

    if add_global_translation:
        min_pos = np.min(motion.joint_position[..., 1])
        motion.joint_translation[..., 1] -= min_pos

    # contact_max_value: np.ndarray = np.max(contact_label, axis=-1)
    # delta_y: np.ndarray = np.zeros(num_frame)
    # body_pos: np.ndarray = np.zeros((num_frame, num_body, 3))

    # create contact by contact max place..
    # here we can compute minimal height by bounding box of character..
    # for frame_idx, max_place in enumerate(contact_max_value):
    #     tar_set.set_character_byframe(frame_idx)
    #     bounding_box = character.get_aabb()
    #     min_y = bounding_box[2]
    #     if max_place >= contact_label_eps:
    #         delta_y[frame_idx] = -min_y
    #     body_pos[frame_idx] = character.get_body_pos()

    # we should also smooth the predicted height..
    # if gaussian_width is not None:
    #     new_root_pos = smooth_operator(motion.joint_translation[:, 0, :], GaussianBase(gaussian_width))
    #     delta_y = new_root_pos -  motion.joint_translation[:, 0, :]
    #     motion.joint_translation[:, 0, :] = new_root_pos
    # motion.recompute_joint_global_info()
    # body_pos += delta_y[:, None, :]

    # here we should also make the contact label clean.
    # That is, if contact label for some body is large, but the body height is also large,
    # we should remove this contact label..
    # contact_place = np.where(contact_label >= contact_label_eps)
    # posy_place = body_pos[contact_place[0], contact_place[1], 1]
    # large_posy_index = np.where(posy_place > contact_body_pos_eps)[0]
    # if len(large_posy_index) > 0:
    #     contact_label[contact_place[0][large_posy_index], contact_place[1][large_posy_index]] = 0

    # finally, store the result into a file..
    # input_data["pred_contact_label"] = contact_label

    out_full_fname = os.path.join(in_dir, output_bvh_fname)
    input_data["pred_motion"] = output_bvh_fname
    BVHLoader.save(motion, out_full_fname)
    print(f"save the pre_processed data at {output_bvh_fname}")

    with open(in_fname, "wb") as fout:
        pickle.dump(input_data, fout)


def pre_process_method2(
    pred_motion: MotionData,
    input_2d_numpy: np.ndarray
):
    raise ValueError("This method is useless..")
    local_motion = pred_motion.sub_sequence().to_local_coordinate()
    num_frame = pred_motion.num_frames
    with torch.enable_grad():
        # 1. build global optim parameter.
        rotmat = nn.Parameter(torch.eye(3, dtype=torch.float64))
        offset = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        optim = AdamW([rotmat, offset], lr=1e-4)
        if pred_motion.joint_position is None:
            pred_motion.recompute_joint_global_info()
        local_pos = torch.from_numpy(local_motion.joint_position[:, 1:])
        root_pos = torch.from_numpy(pred_motion.joint_position[:, 0])
        root_mat = torch.from_numpy(Rotation(pred_motion.joint_rotation[:, 0]).as_matrix())
        # root_mat = DiffQuat
        input_2d_torch = torch.from_numpy(input_2d_numpy)
        # 2. here we need also consider root rotation..
        for epoch in range(4000):
            optim.zero_grad(True)
            # compute root rotation
            # rot_ext = rotmat.repeat(num_frame, 1, 1)
            opt_root_rot = rotmat.view(1, 3, 3) @ root_mat
            opt_root_pos = (rotmat.view(3, 3) @ root_pos.T).T + offset.view(1, 3)
            opt_local_pos = (opt_root_rot[:, None] @ local_pos[..., None])[..., 0] + opt_root_pos[:, None]
            opt_pos = torch.cat([opt_root_pos.view(num_frame, 1, 3), opt_local_pos.view(num_frame, -1, 3)], dim=1)
            opt_sub_pos = opt_pos[:, sub_index]
            proj_pos = opt_sub_pos[..., :2] / opt_sub_pos[..., 2:]
            proj_loss = mse_loss(proj_pos, input_2d_torch)
            proj_loss.backward()
            optim.step()
            print(proj_loss.item())
        print(rotmat, offset)
        ret_motion = pred_motion.sub_sequence()
        ret_motion.joint_translation[:, 0, :] = opt_root_pos.detach().numpy()
        ret_motion.joint_rotation[:, 0, :] = Rotation.from_matrix(opt_root_rot.detach().numpy()).as_quat()
        return ret_motion


def test_method1():
    scene = JsonSceneLoader().load_from_pickle_file(os.path.join(fdir, "../../../Tests/CharacterData/Samcon-Human.pickle"))
    scene.set_sim_fps(100)
    pre_process_motion_method1(os.path.join(fdir, "../../../Tests/CharacterData/Samcon/0/network-output.bin"),
        scene.character0)


if __name__ == "__main__":
    test_method1()


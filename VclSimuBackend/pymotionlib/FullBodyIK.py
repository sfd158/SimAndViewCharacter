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
Input: joint position, character hierarchy
Output: bvh file
"""
from argparse import ArgumentParser, Namespace
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D

import numpy as np
import os
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Optional
from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData


fdir = os.path.dirname(__file__)


def build_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_fname", type=str, default=
        r"D:\song\documents\WeChat Files\wxid_cy7nlkpekunr22\FileStorage\File\2022-03\0(1).npy")
        # r"D:\downloads\deal_ted_data\0011.npy")
    parser.add_argument("--invert_y_axis", action="store_true", default=True)
    parser.add_argument("--output_fname", type=str, default="test.bvh")

    args: Namespace = parser.parse_args()
    return args


class FullBodyIK:
    def __init__(self, args: Namespace, joint_names: List[str], joint_pairs: List[Tuple], unit_bvh_offset: np.ndarray, joint_pos: np.ndarray) -> None:
        self.args = args
        self.joint_names = joint_names
        self.joint_pairs = joint_pairs
        self.joint_pos: np.ndarray = joint_pos
        self.num_frames: int = joint_pos.shape[0]
        self.num_joints: int = joint_pos.shape[1]
        self.unit_bvh_offset: np.ndarray = unit_bvh_offset
        self.bone_length: np.ndarray = self.estimate_bone_length()
        self.bvh_offset: np.ndarray = self.bone_length[:, None] * self.unit_bvh_offset
        self.root_pos_np: np.ndarray = np.ascontiguousarray(self.joint_pos[:, 0, :])
        self.motion_hierarchy: MotionData = self.build_motion()

        self.visualize_joints()  # visualize the position directly..
        BVHLoader.save(self.motion_hierarchy, args.output_fname)

    def get_initial_solution(self) -> np.ndarray:
        """
        We can compute joint global rotation here..
        Note: we should normalize the local offset here..
        Here we can output the initial solution..
        """
        quat_list = [[] for _ in range(self.num_joints)]
        for bone_idx, (parent, child) in enumerate(self.joint_pairs):
            global_off: np.ndarray = self.joint_pos[:, child, :] - self.joint_pos[:, parent, :]
            unit_global_off: np.ndarray = global_off / np.linalg.norm(global_off, axis=-1, keepdims=True)
            parent_quat: np.ndarray = MathHelper.quat_between(self.unit_bvh_offset[child], unit_global_off)
            quat_list[parent].append(parent_quat)
        # compute average quaternion..
        for joint_idx in range(self.num_joints):
            if len(quat_list[joint_idx]) == 0:
                quat_list[joint_idx] = MathHelper.unit_quat_arr((self.num_frames, 4))
            elif len(quat_list[joint_idx]) == 1:
                quat_list[joint_idx] = quat_list[joint_idx][0]
            else:  # here we need to compute average quaternions..
                quat_list[joint_idx] = MathHelper.average_quat_by_slerp(quat_list[joint_idx])

        # convert global quaternion to local..
        rots: List[Rotation] = [Rotation(node) for node in quat_list]
        rots_inv: List[Rotation] = [node.inv() for node in rots]
        local_joint_quat: np.ndarray = MathHelper.unit_quat_arr((self.num_frames, self.num_joints, 4))
        local_joint_quat[:, 0, :] = rots[0].as_quat()
        for bone_idx, (parent, child) in enumerate(self.joint_pairs):
            local_joint_quat[:, child, :] = (rots_inv[parent] * rots[child]).as_quat()

        return local_joint_quat

    def build_motion(self):
        motion = MotionData()
        motion._num_frames = self.num_frames
        motion._num_joints = self.num_joints
        motion._fps = 15
        motion._skeleton_joints = self.joint_names
        motion._skeleton_joint_parents = self.build_parent_index()
        motion._skeleton_joint_offsets = self.bvh_offset
        motion._joint_translation = np.zeros((self.num_frames, self.num_joints, 3))
        motion._joint_translation[:, 0, :] = self.root_pos_np[:]
        # motion._joint_rotation = MathHelper.unit_quat_arr((self.num_frames, self.num_joints, 4))
        motion._joint_rotation = self.get_initial_solution()

        return motion

    def visualize_joints(self, pos_3d: Optional[np.ndarray] = None, render_joint_id: bool = True):
        """
        Visualize the input 3d joint position by matplotlib.
        """
        fig: Figure = plt.figure()
        sub3: Axes3D = fig.add_subplot(projection='3d')
        sub3.set_xlabel("x")
        sub3.set_ylabel("y")
        sub3.set_zlabel("z")

        if pos_3d is None:
            pos_3d: np.ndarray = self.joint_pos

        lines: List[Line3D] = [
            sub3.plot(*(pos_3d[0, joint_pair, axis] for axis in range(3)))[0]
            for joint_pair in self.joint_pairs]

        if render_joint_id:
            text_3d: Text3D = [sub3.text(*(pos_3d[0, j_index, axis] for axis in range(3)), str(j_index), color='r')
                for j_index in range(self.num_joints)]
        else:
            text_3d = None

        # here for debug, we can also render joint index.
        def update_human_3d(index: int):
            if text_3d is not None:
                for j_index in range(self.num_joints):
                    text_3d[j_index].set_position_3d(pos_3d[index, j_index])
            for pair_idx, joint_pair in enumerate(self.joint_pairs):
                lines[pair_idx].set_data(pos_3d[index, joint_pair, axis] for axis in range(2))
                lines[pair_idx].set_3d_properties(pos_3d[index, joint_pair, 2])

        anim = FuncAnimation(fig, update_human_3d, pos_3d.shape[0], interval=1, repeat=False)
        plt.show()

    def build_parent_index(self) -> List[int]:
        parent_index = [-1 for _ in range(self.num_joints)]
        for parent, child in self.joint_pairs:
            parent_index[child] = parent
        return parent_index

    def estimate_bone_length(self):
        segment_len: np.ndarray = np.zeros(self.num_joints)
        for bone_idx, (parent, child) in enumerate(self.joint_pairs):
            segment: np.ndarray = self.joint_pos[:, child] - self.joint_pos[:, parent]
            segment_len[child] = np.mean(np.linalg.norm(segment, axis=-1))
        return segment_len


def ortho_projection(tvec):
    mat = np.zeros(9)
    mat[0] = 1 - (tvec[0] * tvec[0])
    mat[1] = -(tvec[0] * tvec[1])
    mat[2] = -(tvec[0] * tvec[2])
    mat[3] = -(tvec[0] * tvec[1])
    mat[4] = 1 - (tvec[1] * tvec[1])
    mat[5] = -(tvec[1] * tvec[2])
    mat[6] = -(tvec[0] * tvec[2])
    mat[7] = -(tvec[1] * tvec[2])
    mat[8] = 1 - (tvec[2] * tvec[2])
    return mat.reshape((3, 3))


def refine_elbow(shoulder_vec, elbow_vec):
    proj_mat = ortho_projection(shoulder_vec)
    projected_vec = proj_mat @ elbow_vec
    mid_vector = (projected_vec + elbow_vec) / 2
    return mid_vector

def refine_nose(nose_vec):
    mid_vector = (np.array([0, 0, 1]) + nose_vec) / 2
    return mid_vector

def refine_spine(spine_vec):
    mid_vector = (np.array([0, -0.1, 1]) + spine_vec) / 2
    return mid_vector


def test_simple_case():
    joint_names = [
        "spine",
        "neck",
        "nose",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist"
    ]
    joint_pairs = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 7), (7, 8), (8, 9)]
    args = build_args()
    joint_pos: np.ndarray = np.load(args.input_fname).astype(np.float64)
    if args.invert_y_axis:
        joint_pos[..., 1] *= -1
    bvh_offset = np.zeros((len(joint_names), 3))
    # note: for shoulder, we need to recompute the offset..
    global_offset = np.zeros_like(joint_pos)
    for parent, child in joint_pairs:
        global_offset[:, child, :] = joint_pos[:, child, :] - joint_pos[:, parent, :]
    lshoulder_xz = np.mean(np.linalg.norm(global_offset[:, 4, [0, 2]], axis=-1), axis=0)
    lshoulder_y = np.mean(global_offset[:, 4, 1])
    rshoulder_xz = np.mean(np.linalg.norm(global_offset[:, 7, [0, 2]], axis=-1), axis=0)
    rshoulder_y = np.mean(global_offset[:, 7, 1])

    shoulder_xz = 0.5 * (lshoulder_xz + rshoulder_xz)
    shoulder_y = 0.5 * (lshoulder_y + rshoulder_y)
    lshoulder_off = np.array([-shoulder_xz, shoulder_y, 0.0])
    lshoulder_off /= np.linalg.norm(lshoulder_off)
    rshoulder_off = np.array([shoulder_xz, shoulder_y, 0.0])
    rshoulder_off /= np.linalg.norm(rshoulder_off)

    for j in range(joint_pos.shape[0]):
        shoulder_vec = joint_pos[:, 7, :] - joint_pos[:, 4, :] # Vector(bones['elbow.R'].head - bones['elbow.L'].head)
        if 'elbow' in bone_name:
                bone = refine_elbow(shoulder_vec, bone)
            elif 'nose' in bone_name.lower():
                bone = refine_nose(bone)
            elif 'neck' in bone_name.lower():
                bone = refine_spine(bone)

    for parent, child in joint_pairs:
        if child == 4:
            bvh_offset[child] = lshoulder_off
        elif child == 7:
            bvh_offset[child] = rshoulder_off
        elif "left" in joint_names[child]:
            bvh_offset[child] = np.array([-1.0, 0.0, 0.0])
        elif "right" in joint_names[child]:
            bvh_offset[child] = np.array([1.0, 0.0, 0.0])
        else:
            bvh_offset[child] = np.array([0.0, 1.0, 0.0])

    full_body_ik = FullBodyIK(args, joint_names, joint_pairs, bvh_offset, joint_pos)


if __name__ == "__main__":
    test_simple_case()

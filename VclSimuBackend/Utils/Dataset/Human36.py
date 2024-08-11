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

import cdflib
import numpy as np
import os
from scipy.spatial.transform import Rotation
from typing import List, Union, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D

from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.Utils.Camera.CameraNumpy import z_up_to_y_up, CameraParamNumpy
from VclSimuBackend.Utils.Camera.Human36CameraBuild import pre_build_camera
from VclSimuBackend.Utils.Dataset.StdHuman import stdhuman_with_root_name_dict


class Human36FullInfo:
    parents: List[int] = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
            16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]
    joints_left: List[int] = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]  # left part of skeleton
    joints_right: List[int] = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]  # right part of skeleton

    keep_joints: List[int] = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    static_joints: List[int] = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]

    children: List[List[int]] = [
        [1, 6, 11], [2], [3], [4], [5], [], [7], [8], [9], [10], [], [12],
        [13, 16, 24], [14], [15], [], [17], [18], [19], [20, 22], [21], [], [23],
        [], [25], [26], [27], [28, 30], [29], [], [31], []
    ]



class Human36Sub17Info:
    parents: List[int] = [-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15]
    joints_left: List[int] = [4, 5, 6, 11, 12, 13]
    joints_right: List[int] =  [1, 2, 3, 14, 15, 16]
    children: List[List[int]] = [
        [1, 4, 7], [2], [3], [], [5], [6], [], [8, 11, 14], [9], [10], [],
        [12], [13], [], [15], [16], []
    ]

    joint_names = [
        "Root",          # 0
        "RightHip",      # 1
        "RightKnee",     # 2
        "RightFoot",     # 3
        "LeftHip",       # 4
        "LeftKnee",      # 5
        "LeftFoot",      # 6
        "LowBack",       # 7
        "Chest",         # 8
        "Neck",          # 9
        "Head",          # 10
        "LeftShoulder",  # 11
        "LeftElbow",     # 12
        "LeftHand",      # 13
        "RightShoulder", # 14
        "RightElbow",    # 15
        "RightHand"      # 16
    ]

    sub_13_index = [
        10,  # 0
        11,  # 1
        12,  # 2
        13,  # 3
        4,   # 4
        5,   # 5
        6,   # 6
        14,  # 7
        15,  # 8
        16,  # 9
        1,   # 10
        2,   # 11
        3,   # 12
    ]
    sub_13_index_np = np.array(sub_13_index, dtype=np.int64)

    head_index = 10
    no_head_index: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]  # head index not contained.
    no_head_index_np: np.ndarray = np.array(no_head_index)

    stdhuman_sub_name_17 = [
        "RootJoint",
        "rHip",
        "rKnee",
        "rAnkle",
        "lHip",
        "lKnee", 
        "lAnkle",
        "pelvis_lowerback",
        "lowerback_torso",
        "torso_head",
        "torso_head_end", # Note: there is no Head joint here, ignore here.
        "lShoulder",
        "lElbow",
        "lWrist",
        "rShoulder",
        "rElbow",
        "rWrist"
    ]

    smpl_sub_name_17 = [
        "RootJoint",               # 0
        "R_Hip_and_Pelvis",        # 1
        "R_Knee_and_R_Hip",        # 2
        "R_Ankle_and_R_Knee",      # 3
        "L_Hip_and_Pelvis",        # 4
        "L_Knee_and_L_Hip",        # 5
        "L_Ankle_and_L_Knee",      # 6
        "Spine1_and_Pelvis",       # 7
        "Spine2_and_Spine1",       # 8
        "Neck_and_Spine3",         # 9
        "Head_and_Neck",           # 10
        "L_Shoulder_and_L_Collar", # 11
        "L_Elbow_and_L_Shoulder",  # 12
        "L_Wrist_and_L_Elbow",     # 13
        "R_Shoulder_and_R_Collar", # 14
        "R_Elbow_and_R_Shoulder",  # 15
        "R_Wrist_and_R_Elbow"      # 16
    ]

    stdhuman_sub_name_16 = [
        "RootJoint",
        "rHip",
        "rKnee",
        "rAnkle",
        "lHip",
        "lKnee", 
        "lAnkle",
        "pelvis_lowerback",
        "lowerback_torso",
        "torso_head",
        # Note: there is no Head joint here, ignore here.
        "lShoulder",
        "lElbow",
        "lWrist",
        "rShoulder",
        "rElbow",
        "rWrist"
    ]  # Note: There is no head joint here..we should consider head from body position..

    # length == len(stdhuman_with_root_subname) == 16
    stdhuman_with_root_map_index: List[int] = [stdhuman_with_root_name_dict[node] for node in stdhuman_sub_name_16]


def std_human_to_human36m(
    joint_pos_with_root: np.ndarray,
    head_pos: np.ndarray) -> np.ndarray:
    """
    convert std human hierarchy to human 3.6m hierarchy..
    Input:
    - joint_pos_with_root (frame, 20, 3)
    - head_pos (frame, 3)

    Output:
    - subset in human3.6m hierarchy (frame, 17, 3)

    Note: for back, neck, and head, if use std-human joint position directly,
    maybe the mpjpe error is large..?

    human3.6 head: head body position in std-human
    human3.6 neck: lower part of capsule for head body in std-human
        (actually, this equals to torso_head joint in std-human)
    human3.6 chest and LowerBack: use std-human joint directly is OK.
    """
    num_frame = joint_pos_with_root.shape[0]
    assert joint_pos_with_root.shape[1:] == (20, 3) and head_pos.shape == (num_frame, 3)
    result: np.ndarray = np.zeros((num_frame, 17, 3))
    result[:, Human36Sub17Info.head_index, :] = head_pos.copy()
    result[:, Human36Sub17Info.no_head_index, :] = joint_pos_with_root[:, Human36Sub17Info.stdhuman_with_root_map_index, :].copy()

    return result


def simple_vis_2d_13(pos_a: np.ndarray, pos_b: np.ndarray):
    fig: Figure = plt.figure()
    sub_plot = fig.add_subplot()
    sub_plot.set_xlim(-1, 1)
    sub_plot.set_ylim(-1, 1)
    text_a = [sub_plot.text(*(pos_a[0, j_index, axis] for axis in range(2)), str(j_index), color='r')
                    for j_index in range(pos_a.shape[1])]
    text_b = [sub_plot.text(*(pos_b[0, j_index, axis] for axis in range(2)), str(j_index), color='g')
                    for j_index in range(pos_b.shape[1])]
    plt.show()

def simple_vis_2d(pos_2d: np.ndarray, other_data: np.ndarray, render_joint_id: bool = True):
    fig: Figure = plt.figure()
    sub_plot = fig.add_subplot()
    sub_plot.set_xlim(-1, 1)
    sub_plot.set_ylim(-1, 1)
    parents = Human36Sub17Info.parents
    other_render = None
    if other_data is not None:
        other_render = sub_plot.plot(other_data[0, :, 0], other_data[0, :, 1], "o")[0]
    lines = [
        sub_plot.plot(*(pos_2d[0, (j_index, pa_index), axis] for axis in range(2)))[0]
        if pa_index != -1 else None
        for j_index, pa_index in enumerate(parents)]

    if render_joint_id:
        text_2d = [sub_plot.text(*(pos_2d[0, j_index, axis] for axis in range(2)), str(j_index), color='r')
                    for j_index in range(len(parents))]
    else:
        text_2d = None

    # here for debug, we can also render joint index.
    def update_human_2d(index: int):
        if text_2d is not None:
            for j_index, text_node in enumerate(text_2d):
                text_node.set_position(pos_2d[index, j_index])
        if other_render is not None:
            other_render.set_data(other_data[index, :, 0], other_data[index, :, 1])
        for j_index, pa_index in enumerate(parents):
            if pa_index == -1:
                continue
            lines[j_index].set_data(pos_2d[index, (j_index, pa_index), axis] for axis in range(2))

    anim = FuncAnimation(fig, update_human_2d, pos_2d.shape[0], interval=1, repeat=False)
    plt.show()


def simple_vis(
    pos_3d: np.ndarray,
    render_joint_id: bool = False,
    render_title: Optional[str] = "",
    export_fname: Optional[str] = None,
    fps: int = 50
):
    fig: Figure = plt.figure()
    sub3: Axes3D = fig.add_subplot(projection='3d')
    sub3.set_xlabel("x")
    sub3.set_ylabel("y")
    sub3.set_zlabel("z")

    sub3.set_xlim3d(-2, 2)
    sub3.set_ylim3d(-2, 2)
    sub3.set_zlim3d(-2, 2)

    if pos_3d.shape[1] == 32:
        parents = Human36FullInfo.parents
    elif pos_3d.shape[1] == 17:
        parents = Human36Sub17Info.parents
    else:
        raise ValueError("num joints doesn't match. Only 32 and 17 are supported.")

    lines: List[Line3D] = [
        sub3.plot(*(pos_3d[0, (j_index, pa_index), axis] for axis in range(3)))[0]
        if pa_index != -1 else None
        for j_index, pa_index in enumerate(parents)]

    if render_joint_id:
        text_3d: Text3D = [sub3.text(*(pos_3d[0, j_index, axis] for axis in range(3)), str(j_index), color='r')
                            for j_index in range(len(parents))]
    else:
        text_3d = None
    
    plt.title(f"{render_title}, frame = 0")

    # here for debug, we can also render joint index.
    def update_human_3d(index: int):
        if text_3d is not None:
            for j_index, text_node in enumerate(text_3d):
                text_node.set_position_3d(pos_3d[index, j_index])
        for j_index, pa_index in enumerate(parents):
            if pa_index == -1:
                continue
            lines[j_index].set_data(pos_3d[index, (j_index, pa_index), axis] for axis in range(2))
            lines[j_index].set_3d_properties(pos_3d[index, (j_index, pa_index), 2])
        plt.title(f"{render_title}, frame = {index}")

    anim = FuncAnimation(fig, update_human_3d, pos_3d.shape[0], interval=1, repeat=False)
    if export_fname is not None and isinstance(export_fname, str):
        anim.save(export_fname, fps=fps)
    else:
        plt.show()
    # plt.close()


def simple_vis_pair_same_scene(
    pos_3d_a: np.ndarray,
    pos_3d_b: np.ndarray,
    render_joint_id: bool = False,
    export_fname: Optional[str] = None,
    fps: int = 50
):
    """
    """
    fig: Figure = plt.figure()
    sub3: Axes3D = fig.add_subplot(projection='3d')
    sub3.set_xlabel("x")
    sub3.set_ylabel("y")
    sub3.set_zlabel("z")

    sub3.set_xlim3d(-2, 2)
    sub3.set_ylim3d(-2, 2)
    sub3.set_zlim3d(-2, 2)

    pos_3d_list = [pos_3d_a, pos_3d_b]
    parents_list = []
    lines_list = []

    for index, pos_3d in enumerate(pos_3d_list):
        if pos_3d.shape[1] == 32:
            parents = Human36FullInfo.parents
        elif pos_3d.shape[1] == 17:
            parents = Human36Sub17Info.parents
        else:
            raise ValueError("num joints doesn't match. Only 32 and 17 are supported.")
        parents_list.append(parents)

        lines: List[Line3D] = [
            sub3.plot(*(pos_3d[0, (j_index, pa_index), axis] for axis in range(3)))[0]
            if pa_index != -1 else None
            for j_index, pa_index in enumerate(parents)]
        lines_list.append(lines)

        if render_joint_id:
            text_3d: Text3D = [sub3.text(*(pos_3d[0, j_index, axis] for axis in range(3)), str(j_index), color='r')
                                for j_index in range(len(parents))]
        else:
            text_3d = None
    
    plt.title(f"frame = 0")

    # here for debug, we can also render joint index.
    def update_human_3d(index: int):
        for pos_3d, parents, lines in zip(pos_3d_list, parents_list, lines_list):
            if text_3d is not None:
                for j_index, text_node in enumerate(text_3d):
                    text_node.set_position_3d(pos_3d[index, j_index])
            for j_index, pa_index in enumerate(parents):
                if pa_index == -1:
                    continue
                lines[j_index].set_data(pos_3d[index, (j_index, pa_index), axis] for axis in range(2))
                lines[j_index].set_3d_properties(pos_3d[index, (j_index, pa_index), 2])
        plt.title(f"frame = {index}")

    anim = FuncAnimation(fig, update_human_3d, pos_3d.shape[0], interval=1, repeat=True)
    if export_fname is not None and isinstance(export_fname, str):
        anim.save(export_fname, fps=fps)
    else:
        plt.show()
    # plt.close()


def simple_vis_pair(
    pos_3d_a: np.ndarray,
    pos_3d_b: np.ndarray,
    export_fname: Optional[str] = None,
    fps: int = 50
):

    fig: Figure = plt.figure()
    sub3: Axes3D = fig.add_subplot(projection="3d")
    sub3.set_xlabel("x")
    sub3.set_ylabel("y")
    sub3.set_zlabel("z")

    sub3.set_xlim3d(-2, 2)
    sub3.set_ylim3d(-2, 2)
    sub3.set_zlim3d(-2, 2)
    pos_3d_list = []

    def init_render():
        for pos_3d in [pos_3d_a, pos_3d_b]:
            if pos_3d is None:
                continue
            if pos_3d.shape[1] == 32:
                parents = Human36FullInfo.parents
            elif pos_3d.shape[1] == 17:
                parents = Human36Sub17Info.parents
            else:
                raise ValueError("num joints doesn't match")

            lines: List[Line3D] = [
                sub3.plot(*(pos_3d[0, (j_index, pa_index), axis] for axis in range(3)))[0]
                if pa_index != -1 else None
                for j_index, pa_index in enumerate(parents)]
            pos_3d_list.append((parents, lines, pos_3d))

    init_render()

    def update_human_3d(index: int):
        for parents, lines, pos_3d in pos_3d_list:
            for j_index, pa_index in enumerate(parents):
                if pa_index == -1:
                    continue
                lines[j_index].set_data(pos_3d[index, (j_index, pa_index), axis] for axis in range(2))
                lines[j_index].set_3d_properties(pos_3d[index, (j_index, pa_index), 2])

    anim = FuncAnimation(fig, update_human_3d, pos_3d_a.shape[0], interval=1, repeat=False)
    if export_fname is None:
        plt.show()
    else:
        anim.save(export_fname, fps=fps)


def load_angles_3d():
    import cdflib
    fdir = r"Z:\Downloads\human3.6m_downloader-master\training\subject\s1\D3_Angles\D3_Angles\S1\MyPoseFeatures\D3_Angles"
    for _fname in os.listdir(fdir):
        fname: str = os.path.join(fdir, _fname)
        print(f"Load from {fname}")
        hf: cdflib.cdfread.CDF = cdflib.CDF(fname)
        pose: np.ndarray = hf["Pose"]  # (1, num frame?, 78)
        break


def load_angles_mono_3d():
    import cdflib
    fdir = r"Z:\Downloads\human3.6m_downloader-master\training\subject\s1\D3_Angles_mono\D3_Angles_mono\S1\MyPoseFeatures\D3_Angles_mono"
    for _fname in os.listdir(fdir):
        fname: str = os.path.join(fdir, _fname)
        print(f"Load from {fname}")
        hf: cdflib.cdfread.CDF = cdflib.CDF(fname)
        pose: np.ndarray = hf["Pose"]  # shape is (1, num frame, 78) ?
        break

    exit(0)


def pos_to_subset17(x: np.ndarray, is_copy: bool = False) -> np.ndarray:
    assert x.shape[1:] == (32, 3)
    result: np.ndarray = x[:, Human36FullInfo.keep_joints, :]
    return result.copy() if is_copy else result


def load_h36m_pos3d_full(fname: str) -> np.ndarray:
    hf = cdflib.CDF(fname)
    pos_3d_global: np.ndarray = hf['Pose'].reshape(-1, 32, 3)  # 3d Position
    return pos_3d_global


def load_h36m_pos3d(fname: str) -> np.ndarray:
    """
    Input: 3D position file
    Output: the global position of each frame.
    """
    pos_3d_global: np.ndarray = load_h36m_pos3d_full(fname)
    pos_3d_global: np.ndarray = pos_3d_global[:, Human36FullInfo.keep_joints, :]
    return pos_3d_global  # in shape (num frame, 17, 3)


def get_h36m_root_rotation(input_fname: str) -> np.ndarray:
    """
    Input: D3_Angles
    output: root quaternion in shape (num_frame, 4)
    """
    assert input_fname.endswith(".cdf")
    cdf_angles = cdflib.CDF(input_fname)
    angles: np.ndarray = cdf_angles.varget("Pose")[0]
    rot_indexes: np.ndarray = np.array([5, 6, 4])
    zr: np.ndarray = angles[:, rot_indexes[2] - 1]
    xr: np.ndarray = angles[:, rot_indexes[0] - 1]
    yr: np.ndarray = angles[:, rot_indexes[1] - 1]
    cat_euler: np.ndarray = np.concatenate([zr[..., None], xr[..., None], yr[..., None]], axis=-1)
    quat: np.ndarray = Rotation.from_euler("ZXY", cat_euler, True).as_quat()
    return quat


def compute_h36m_17_root_quat(pos_3d_global: np.ndarray) -> np.ndarray:
    # root_pos: np.ndarray = pos_3d_global[:, 0, :]
    root_child_pos = pos_3d_global[:, Human36Sub17Info.children[0], :]
    l_to_r: np.ndarray = root_child_pos[:, 0, :] - root_child_pos[:, 1, :]
    unit_l_to_r: np.ndarray = l_to_r / np.linalg.norm(l_to_r, axis=-1, keepdims=True)
    ori_l_to_r = np.zeros_like(l_to_r)
    ori_l_to_r[:, 0] = -1
    root_quat: np.ndarray = MathHelper.quat_between(ori_l_to_r, unit_l_to_r)
    return root_quat


def compute_h36m_root_quat(pos_3d_global: Union[str, np.ndarray]) -> np.ndarray:
    """
    input: human 3.6m cdf filename or human 3.6m global position
    for human3.6 raw data, z axis is the up axis.
    in open dynamics scene, y axis is the up axis.

    as x axis is not changed in process of converting z-up to y-up,
    this part of code can work on both z-up and y-up case.
    """
    if isinstance(pos_3d_global, str):
        pos_3d_global: np.ndarray = load_h36m_pos3d_full(pos_3d_global)

    root_child_pos: np.ndarray = pos_3d_global[:, Human36FullInfo.children[0], :]
    root_pos: np.ndarray = pos_3d_global[:, 0, :]
    # ch0_len = np.mean(np.linalg.norm(root_child_pos[:, 0, :] - root_pos, axis=-1))  # right leg
    # ch1_len = np.mean(np.linalg.norm(root_child_pos[:, 1, :] - root_pos, axis=-1))  # left leg
    l_to_r: np.ndarray = root_child_pos[:, 0, :] - root_child_pos[:, 1, :]
    unit_l_to_r: np.ndarray = l_to_r / np.linalg.norm(l_to_r, axis=-1, keepdims=True)
    ori_l_to_r = np.zeros_like(l_to_r)
    ori_l_to_r[:, 0] = -1
    root_quat: np.ndarray = MathHelper.quat_between(ori_l_to_r, unit_l_to_r)

    # check if root quaternion is correct
    def check_func():
        local_r: np.ndarray = root_child_pos[:, 0, :] - root_pos
        local_l: np.ndarray = root_child_pos[:, 1, :] - root_pos
        root_rot_inv: Rotation = Rotation(root_quat).inv()
        local_r: np.ndarray = root_rot_inv.apply(local_r)
        local_l: np.ndarray = root_rot_inv.apply(local_l)

    return root_quat
    # in human3.6 data set, the upper axis is along z.

    # we can render this 3 points simply
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # for i in range(3):
    #     ax.plot(root_child_pos[:, i, 0], root_child_pos[:, i, 1], root_child_pos[:, i, 2])
    # plt.show()


def get_y_facing_coordinate(pos_3d_global: Union[str, np.ndarray], convert_to_sub17: bool = True):
    """
    convert position to facing coordinate..
    As human3.6m data position is milli-meter, we should scale to meter.
    """
    if isinstance(pos_3d_global, str):
        pos_3d_global: np.ndarray = load_h36m_pos3d_full(pos_3d_global)  # read the global position from file
    pos_3d_global: np.ndarray = 0.001 * pos_3d_global

    # we should convert from z-up to y-up
    pos_3d_global: np.ndarray = z_up_to_y_up(pos_3d_global)
    # simple_vis(pos_3d_global)
    root_quat: np.ndarray = compute_root_quat(pos_3d_global)
    q_y, q_xz = MathHelper.y_decompose(root_quat)

    # convert to facing coordinate
    pos_3d_global -= MathHelper.vec_axis_to_zero(pos_3d_global[:, 0:1, :], 1)  # first subtract xz component
    # then, rotate with inverse of q_y
    qy_inv: Rotation = Rotation(q_y).inv()
    for i in range(1, pos_3d_global.shape[1]):
        pos_3d_global[:, i, :] = qy_inv.apply(pos_3d_global[:, i, :])

    simple_vis(pos_3d_global)
    if convert_to_sub17:
        pos_3d_global: np.ndarray = pos_to_subset17(pos_3d_global, False)

    # for debug, we should visualize here..
    return pos_3d_global


def parse_human36_video_camera(video_path: str) -> CameraParamNumpy:
    dirname, fname = os.path.split(video_path)
    dirname: str = dirname.lower()
    subject = None
    for index in [1, 5, 6, 7, 8, 9, 11]:
        if f"s{index}" in dirname:
            subject = index
            break
    if subject is None:
        raise ValueError(f"not a valid human 3.6 video input file")
    camera_index: str = fname.split(".")[1]
    subject_cams = pre_build_camera[f"S{subject}"]
    for cam_param in subject_cams:
        if str(cam_param.cam_id) == camera_index:
            return cam_param
    raise ValueError(f"Camera {camera_index} not match")


if __name__ == "__main__":
    simple_vis(z_up_to_y_up(1e-3 * load_h36m_pos3d(r"F:\human3.6m_downloader-master\training\D3Pos\S11\WalkDog.cdf")))


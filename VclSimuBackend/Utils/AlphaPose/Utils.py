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
Visualize the result of alpha pose and input video.
for simple, we can render joints by matplotlib
"""

import json
import numpy as np
import os
import shutil
import subprocess
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple


def go_inner_video_dir(start_dir: str):
    for tmp_ in os.listdir(start_dir):
        tmp = os.path.join(start_dir, tmp_)
        if os.path.isdir(tmp):
            return go_inner_video_dir(tmp)
    return start_dir


def call_alpha_pose():
    """
    run alpha pose on human3.6m video dataset.
    Output: in ms-coco format, with 17 joints.
    """

    human36_dir = "/mnt/hdd/human3.6m_downloader-master/training/subject"
    output_dir = "/mnt/hdd/human36m-process-video"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for subject_ in os.listdir(human36_dir):
        subject = os.path.join(human36_dir, subject_, "Videos")
        video_in_dir = go_inner_video_dir(subject)
        out_subject_dir = os.path.join(output_dir, subject_)
        if not os.path.exists(out_subject_dir):
            os.mkdir(out_subject_dir)
        print(f"out_subject_dir = {out_subject_dir}")
        for video_fname_ in os.listdir(video_in_dir):
            if not video_fname_.endswith(".mp4"):
                continue
            video_fname = os.path.join(video_in_dir, video_fname_)
            print(f"##################process {video_fname}")

            subprocess.call(
                [
                    "python",
                    "scripts/demo_inference.py",
                    "--cfg",
                    "configs/coco/hrnet/256x192_w32_lr1e-3.yaml",
                    "--checkpoint",
                    "pretrained_models/hrnet_w32_256x192.pth",
                    "--video",
                    video_fname,
                    "--outdir",
                    out_subject_dir
                ]
            )

            # what's the result file name ..?
            shutil.move(os.path.join(out_subject_dir, "alphapose-results.json"), os.path.join(out_subject_dir, f"{video_fname_}.json"))


def load_alpha_pose(
    input_fname: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    for single human pose estimation in human3.6,
    we need not compute score, as character must exists in camera..

    return:
    joint_2d_pos: np.ndarray with shape == (frame, 17, 2)
    confidence: np.ndarray with shape == (frame, 17)
    """

    num_joints: int = 17  # here use mscoco format.
    with open(input_fname, "r") as fin:
        res_dict: List[Dict] = json.load(fin)
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(res_dict)
    tot_len: int = end_frame - start_frame
    print(f"length of result is {tot_len}", flush=True)
    joint_2d_pos: np.ndarray = np.zeros((tot_len, num_joints, 2), dtype)
    confidence: np.ndarray = np.zeros((tot_len, num_joints), dtype)
    for index in range(start_frame, end_frame):
        result = res_dict[index]
        key_points: np.ndarray = np.array(result["keypoints"], dtype).reshape((num_joints, 3))
        joint_2d_pos[index - start_frame] = key_points[:, :2]
        confidence[index - start_frame] = key_points[:, 2]

    joint_2d_pos: np.ndarray = np.ascontiguousarray(joint_2d_pos, dtype)
    confidence: np.ndarray = np.ascontiguousarray(confidence, dtype)
    return joint_2d_pos, confidence


def simple_visualize(joint_2d_pos: np.ndarray, input_video: Union[str, np.ndarray]):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from VclSimuBackend.Utils.opencv.VideoLoader import load_video

    if isinstance(input_video, str):  # load video by opencv
        input_video: np.ndarray = load_video(input_video)[0:2 * 233]
    num_frames: int = input_video.shape[0]

    # Note: using matplotlib to render is too slow...
    # maybe opencv or other methods is much faster.
    fig = plt.figure()
    in_plot = fig.add_subplot(111)
    # in_plot2 = fig.add_subplot(122)
    # in_plot2.set_xlabel("x")
    # in_plot2.set_ylabel("y")
    # in_plot2.set_xlim(0, input_video.shape[1])
    # in_plot2.set_ylim(0, input_video.shape[2])
    # in_plot2.set_aspect('equal', adjustable='box')
    # in_plot2.invert_yaxis()

    video_render = in_plot.imshow(input_video[0], aspect='equal')

    # render_2d = in_plot2.plot(joint_2d_pos[0, :, 0], joint_2d_pos[0, :, 1], "o")[0]
    render_2d_ref = in_plot.plot(joint_2d_pos[0, :, 0], joint_2d_pos[0, :, 1], "o", color="#FF0000")[0]
    pbar = tqdm()
    def render_func(index: int = 0):
        pbar.update()
        video_render.set_data(input_video[index])

        render_2d_ref.set_xdata(joint_2d_pos[index, :, 0])
        render_2d_ref.set_ydata(joint_2d_pos[index, :, 1])

        # render_2d.set_xdata(joint_2d_pos[index, :, 0])
        # render_2d.set_ydata(joint_2d_pos[index, :, 1])


    anim = FuncAnimation(fig, render_func, num_frames, interval=1, repeat=False)
    anim.save('Walking.54138969-2d-estimate.mp4', writer='ffmpeg', fps=50)
    # plt.show()


def test1():
    # fname = r"Z:\human3.6m_downloader-master\training\processing\alpha-pose-result\S1\Greeting.54138969.json"
    # video_fname = r"Z:\human3.6m_downloader-master\training\subject\s1\Videos\Videos\S1\Videos\Greeting.54138969.mp4"
    fname = r"Z:\human3.6m_downloader-master\training\processing\alpha-pose-result\S1\Walking.54138969.json"
    video_fname = r"Z:\human3.6m_downloader-master\training\subject\s1\Videos\Videos\S1\Videos\Walking.54138969.mp4"
    joint_2d_pos, confidence = load_alpha_pose(fname)
    simple_visualize(joint_2d_pos, video_fname)


if __name__ == "__main__":
    call_alpha_pose()

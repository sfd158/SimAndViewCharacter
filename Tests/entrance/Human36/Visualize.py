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
Visualize human3.6 input images. 2 modes:
1. remove all static joints.
2. keep all static joints.
"""
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import os
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.Utils.Camera.Human36CameraBuild import CameraParamBuilder

def simple_2d_render():
    import cv2
    fdir: str = r"Z:\human3.6m_downloader-master\training\subject\s6"
    pos2d_fname: str = os.path.join(fdir, r"D2_Positions\D2_Positions\S6\MyPoseFeatures\D2_Positions\Directions 1.54138969.cdf")
    video_fname: str = os.path.join(fdir, r"Videos\Videos\S6\Videos\Directions 1.54138969.mp4")
    bvh_fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Human36Reheight\S6\Directions1-mocap-100.bvh"
    motion = BVHLoader.load(bvh_fname)
    camera = CameraParamBuilder.build(np.float64)["S1"][0]
    gt_3d, camera_3d, camera_2d = proj_to_2d(motion.joint_position, camera)
    bvh_2d = camera.image_coordinates(camera_2d)
    remove_static = True

    # load 2d pos
    hf: cdflib.cdfread.CDF = cdflib.CDF(pos2d_fname)
    pos2d: np.ndarray = hf["Pose"].reshape((-1, 32, 2))
    parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
               16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]  # num frames
    # joints_left = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]  # left part of skeleton
    # joints_right = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]  # right part of skeleton

    static_joints = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
    if remove_static:  # TODO
        pass

    # load video
    capture = cv2.VideoCapture(video_fname)
    width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame: int = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps: int = int(capture.get(cv2.CAP_PROP_FPS))
    print(f"Load video file from {video_fname}, width = {width}, height = {height}, "
          f"num_frame = {num_frame}, fps = {output_fps}")
    input_video = np.zeros((num_frame, height, width, 3), dtype=np.uint8)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame in range(num_frame):
        flag, video_frame = capture.read()
        if not flag:
            print("error in loading video")
            break
        input_video[frame] = video_frame

    fig = plt.figure(figsize=(20, 5))
    in_plot = fig.add_subplot(121)
    in_plot2 = fig.add_subplot(122)
    # in_plot.invert_yaxis()

    render_idx = 0
    video_render = in_plot.imshow(input_video[render_idx], aspect='equal')
    video_render2 = in_plot2.imshow(input_video[render_idx], aspect='equal')
    # render 2d joints
    line_2d = []
    for joint_idx, pa_index in enumerate(parents):
        if pa_index == -1:
            continue
        line_2d.append(in_plot.plot(*(pos2d[render_idx, (pa_index, joint_idx), axis] for axis in range(2)))[0])

    bvh_line_2d = []
    for joint_idx, pa_index in enumerate(motion.joint_parents_idx):
        if pa_index == -1:
            continue
        bvh_line_2d.append(in_plot2.plot(*(bvh_2d[render_idx, (pa_index, joint_idx), axis] for axis in range(2)))[0])
    plt.show()


def create_white_background():
    input_fname: str = r"D:\song\desktop\siga-2023\image-0.PNG"
    output_fname: str = r""
    from PIL import Image
    img: np.ndarray = np.array(Image.open(input_fname))
    # img.shape == (height, width, 4)
    shape = img.shape
    img = img.reshape((-1, 4))
    print(img[0, 0])
    r_index_low = np.where(img[:, 0] >= 0)[0]
    r_index_high = np.where(img[:, 0] <= 30)[0]
    r_index = np.intersect1d(r_index_low, r_index_high)

    g_index_low = np.where(img[:, 1] >= 0)[0]
    g_index_high = np.where(img[:, 1] <= 30)[0]
    g_index = np.intersect1d(g_index_low, g_index_high)

    b_index_low = np.where(img[:, 2] >= 0)[0]
    b_index_high = np.where(img[:, 2] <= 30)[0]
    b_index = np.intersect1d(b_index_low, b_index_high)

    rgb_index = np.intersect1d(r_index, np.intersect1d(g_index, b_index))
    img[rgb_index, 0:3] = 255
    img = img.reshape(shape)
    imgplot = plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    create_white_background()
    # simple_2d_render()

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

import cv2
import numpy as np
import os


def load_video(input_video_fname: str, return_info: bool = False):
    if not os.path.isfile(input_video_fname):
        raise AttributeError(f"The input {input_video_fname} is not a valid file name.")

    capture = cv2.VideoCapture(input_video_fname)
    width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame: int = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps: int = int(capture.get(cv2.CAP_PROP_FPS))
    print(
        f"Load video file from {input_video_fname}, width = {width}, height = {height}, "
        f"num_frame = {num_frame}, fps = {output_fps}", flush=True
    )
    # assert width == param.res_w and height == param.res_h
    input_video = np.zeros((num_frame, height, width, 3), dtype=np.uint8)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame in range(0, num_frame):
        flag, video_frame = capture.read()
        if not flag:
            raise ValueError("error in loading video")
        input_video[frame] = video_frame

    if not return_info:
        return input_video
    else:
        info = {
            "width": width,
            "height": height,
            "num_frame": num_frame,
            "output_fps": output_fps
        }
        return input_video, info

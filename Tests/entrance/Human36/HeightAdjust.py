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
Adjust human 3.6 height
"""
import os
from posixpath import join
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData

fdir = os.path.dirname(__file__)
input_dir = os.path.abspath(os.path.join(fdir, "../../CharacterData/Human36Retarget"))
output_dir = os.path.abspath(os.path.join(fdir, "../../CharacterData/Human36Reheight"))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

adjust_value = 0.024
for sub_dir in os.listdir(input_dir):
    out_sub_dir = os.path.join(output_dir, sub_dir)
    os.mkdir(out_sub_dir)
    for sub_bvh in os.listdir(os.path.join(input_dir, sub_dir)):
        if not sub_bvh.endswith(".bvh"):
            continue
        input_fname = os.path.join(input_dir, sub_dir, sub_bvh)
        motion = BVHLoader.load(input_fname)
        print(f"load motion from {input_fname}")
        motion.joint_translation[:, 0, 1] -= adjust_value
        motion.joint_position[:, :, 1] -= adjust_value
        output_fname = os.path.join(out_sub_dir, sub_bvh)
        BVHLoader.save(motion, output_fname)
        print(f"write motion to {output_fname}")

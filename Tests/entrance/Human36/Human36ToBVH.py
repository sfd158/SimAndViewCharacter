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
Modified the code from https://github.com/ALLIESXO/H36M-to-BVH
Creates a file in BVH format for the H36M Dataset (http://vision.imar.ro/human3.6m/description.php).
The BVH file is prefilled with missing Motion content and is completed with a cdf D3 Angle pose file of H36M.
As the standard skeleton structure - the metadata.xml in H36M matlab code has been used as a reference.
"""

import argparse
import cdflib
import subprocess
import numpy as np
import os
from scipy.spatial.transform import Rotation
import shutil

from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib import BVHLoader


def convert(input_dir: str, output_dir: str, skelScale=100):
    input_fnames = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".cdf")]

    for index, pose in enumerate(input_fnames):
        # initialize global variables
        cdf_angles = cdflib.CDF(pose)
        angles = cdf_angles.varget("Pose")[0]

        # List of rotation indexes in the bvh hierarchy order
        rot_order = [[5,6,4], [32,33,31], [35,36,34], [38,39,37], [41,42,40],
        [44,45,43], [47,48,46], [50,51,49], [53,54,52], [56,57,55], [59,60,58],
        [62,63,61], [65,66,64], [68,69,67], [71,72,70], [74,75,73], [77,78,76],
        [20,21,19], [23,24,22], [26,27,25], [29,30,28],
        [8,9,7], [11,12,10], [14,15,13], [17,18,16]]

        frame_time = 0.02
        frames = len(angles)

        src_file = os.path.join(os.path.dirname(__file__), "base_H36M_hierarchy.bvh")
        #copy the base file to destination dir
        src_file = shutil.copy(src_file, output_dir)
        new_dst_file_name = os.path.join(output_dir, (cdf_angles.file.stem + '.bvh').replace(" ",""))
        os.rename(src_file, new_dst_file_name)

        #Append content to new bvh file
        with open(new_dst_file_name, 'a') as file:
            file.write("\nMOTION \n")
            file.write("Frames:\t" + str(frames) + " \n")
            file.write("Frame Time: " + str(frame_time) + " \n")

            for frame in angles:
                xp = frame[0] / skelScale
                yp = frame[1] / skelScale
                zp = frame[2] / skelScale
                file.write(" " + str(xp) + " " + str(yp) + " " + str(zp)+ " ")

                for rot_indexes in rot_order:
                    # channels order
                    zr = frame[rot_indexes[2] - 1]
                    xr = frame[rot_indexes[0] - 1]
                    yr = frame[rot_indexes[1] - 1]
                    file.write(str(zr) + " " + str(xr) + " " + str(yr)+ " ")

                #end of frame
                file.write("\n ")

        print("Created new file:" + new_dst_file_name)
        # convert as y up via pymotionlib
        new_bvh = BVHLoader.load(new_dst_file_name)
        os.remove(new_dst_file_name)
        new_bvh.z_up_to_y_up()

        # set T-Pose as zero pose
        mocap_hierarchy: MotionData = new_bvh.sub_sequence(0, 1, copy=True)
        mocap_hierarchy.joint_rotation[0, :, :] = np.array([[0, 0, 0, 1]])
        mocap_hierarchy.recompute_joint_global_info()
        LeftShoulder = mocap_hierarchy.joint_names.index("LeftShoulder")
        mocap_hierarchy.joint_rotation[0, LeftShoulder, :] = Rotation.from_rotvec(np.array([0, 0, -0.5 * np.pi])).as_quat()
        RightShoulder = mocap_hierarchy.joint_names.index("RightShoulder")
        mocap_hierarchy.joint_rotation[0, RightShoulder, :] = Rotation.from_rotvec(np.array([0, 0, 0.5 * np.pi])).as_quat()
        new_bvh.reconfig_reference_pose(mocap_hierarchy.joint_rotation[0], False, False)

        BVHLoader.save(new_bvh, new_dst_file_name)
        print(f"Convert from z-up to y-up, {new_dst_file_name}")
        # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", new_dst_file_name])



if __name__ == "__main__":
    # src_dir = r"Z:\human3.6m_downloader-master\training\subject\s1\D3_Angles\D3_Angles\S1\MyPoseFeatures\D3_Angles"
    # src_dir = r"Z:\human3.6m_downloader-master\training\subject\s5\D3_Angles\D3_Angles\S5\MyPoseFeatures\D3_Angles"
    # src_dir = r"Z:\human3.6m_downloader-master\training\subject\s6\D3_Angles\D3_Angles\S6\MyPoseFeatures\D3_Angles"
    # src_dir = r"Z:\human3.6m_downloader-master\training\subject\s7\D3_Angles\D3_Angles\S7\MyPoseFeatures\D3_Angles"
    # src_dir = r"Z:\human3.6m_downloader-master\training\subject\s8\D3_Angles\D3_Angles\S8\MyPoseFeatures\D3_Angles"
    # src_dir = r"Z:\human3.6m_downloader-master\training\subject\s9\D3_Angles\D3_Angles\S9\MyPoseFeatures\D3_Angles"
    src_dir = r"Z:\human3.6m_downloader-master\training\subject\s11\D3_Angles\D3_Angles\S11\MyPoseFeatures\D3_Angles"
    dest = os.path.join(os.path.dirname(__file__), "../../CharacterData/Human3.6/S11")
    if not os.path.exists(dest):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=src_dir)
    parser.add_argument("--output_dir", type=str, default=dest)
    parser.add_argument("--scale", type=float, default=100)
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.scale)
    print("Done converting.")

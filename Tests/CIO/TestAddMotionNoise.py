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

import os
import numpy as np
from scipy.spatial.transform import Rotation
from VclSimuBackend.pymotionlib import BVHLoader


def main():
    fdir = os.path.dirname(__file__)
    fname = os.path.join(fdir, "../CharacterData/WalkF-mocap.bvh")
    motion = BVHLoader.load(fname)
    rhip_idx = motion.joint_names.index("rHip")
    rotvec = np.zeros((motion.num_frames, 3))
    rotvec[:, 0] = 0.1 * np.sin(np.linspace(0, 12 * np.pi, motion.num_frames))
    motion.joint_rotation[:, rhip_idx, :] = (Rotation.from_rotvec(rotvec) * Rotation(motion.joint_rotation[:, rhip_idx, :])).as_quat()
    motion.recompute_joint_global_info()
    BVHLoader.save(motion, "noise.bvh")


if __name__ == "__main__":
    main()

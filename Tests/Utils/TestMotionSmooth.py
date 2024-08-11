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
import subprocess
from typing import Any, Dict
from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.Utils.MothonSliceSmooth import MotionSliceSmooth


def main():
    fdir = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(fdir, "../CharacterData/SamconConfig.json")
    conf = Helper.conf_loader(fname)
    dup_conf: Dict[str, Any] = conf["worker_cma"]["duplicate_input"]
    # dup_conf["smooth_mode"] = "NO"
    smoother = MotionSliceSmooth.build_from_conf(conf["filename"]["bvh"], dup_conf)
    mocap, smooth_mocap = smoother.calc({"ref_start": 5, "ref_end": 5})
    # print(mocap.num_frames, smooth_mocap)
    # exit(0)
    # out_f1 = os.path.join(fdir, "mocap.bvh")
    out_f2 = os.path.join(fdir, "smooth-mocap.bvh")
    # BVHLoader.save(mocap, os.path.join(fdir, out_f1))
    BVHLoader.save(smooth_mocap, os.path.join(fdir, out_f2))
    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", out_f2])


if __name__ == "__main__":
    main()

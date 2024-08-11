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
from argparse import ArgumentParser, Namespace
import os
fdir = os.path.dirname(__file__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.path.abspath(os.path.join(fdir, "stdhuman.xml")))
    parser.add_argument("--mocap_fname", type=str, default=os.path.join(fdir, "../../Tests/CharacterData/WalkF-mocap-100.bvh"))
    parser.add_argument("--max_epoch", type=int, default=100000)
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--control_freq", type=int, default=25)
    parser.add_argument("--com_fail_threshold", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_rollout_length", type=int, default=300)
    parser.add_argument("--use_target_pose", action="store_true", default=False)
    parser.add_argument("--w_pose", type=float, default=0.65)
    parser.add_argument("--w_vel", type=float, default=0.1)
    parser.add_argument("--w_end_eff", type=float, default=0.1)
    parser.add_argument("--w_com", type=float, default=0.1)
    parser.add_argument("--a_pose", type=float, default=2)
    parser.add_argument("--a_vel", type=float, default=0.1)
    parser.add_argument("--a_end_eff", type=float, default=40)
    parser.add_argument("--a_com", type=float, default=10)
    parser.add_argument("--action_noise", type=float, default=0.1)

    parser.add_argument("--bvh_start", type=int, default=0)
    parser.add_argument("--bvh_end", type=int, default=120)
    args = parser.parse_args()
    return args
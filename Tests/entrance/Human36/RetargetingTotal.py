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
from retargeting import main


def curr_main():
    fdir = os.path.dirname(__file__)
    data_dir = os.path.join(fdir, "../../CharacterData/Human3.6")
    output_total_dir = os.path.join(fdir, "../../CharacterData/Human36Retarget")
    if not os.path.exists(output_total_dir):
        os.makedirs(output_total_dir)

    for index in range(1, 12):
        print(f"process S{index}")
        curr_dir = os.path.join(data_dir, f"S{index}")
        human36_names = os.listdir(curr_dir)
        if len(human36_names) == 0:
            continue
        output_curr_dir = os.path.join(output_total_dir, f"S{index}")
        if not os.path.exists(output_curr_dir):
            os.makedirs(output_curr_dir)
        for name in human36_names:
            if "mocap-100" in name:
                continue
            input_name = os.path.abspath(os.path.join(curr_dir, name))
            output_name = os.path.abspath(os.path.join(output_curr_dir, name[:-4] + "-mocap-100.bvh"))
            main(input_name, output_name, False)
        break


if __name__ == "__main__":
    curr_main()

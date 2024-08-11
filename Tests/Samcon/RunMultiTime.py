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
import datetime
from multiprocessing import cpu_count
import pickle
import os
import shutil
import subprocess

from main import build_args
from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.Samcon.SamconWorkerBase import SamHlp


def run_multi_time():
    start_time = datetime.datetime.now()
    _, args = build_args()
    samhlp = SamHlp(args.config, args.repeat_index)
    save_dir = samhlp.save_folder_i_dname(args.repeat_index)
    conf = samhlp.load_conf()
    repeat_cnt = len(conf["worker_cma"]["duplicate_input"]["dup_after_cma"]) + 1
    cpu_cnt = cpu_count()
    main_fname = os.path.split(samhlp.main_dump_fname())[-1]
    print(samhlp.main_dump_fname())
    if os.path.isfile(samhlp.main_dump_fname()):
        with open(samhlp.main_dump_fname(), "rb") as fin:
            dump_info = pickle.load(fin)
        i = dump_info["dup_idx"]
        del dump_info
    else:
        i = 1
    print(f"initial i = {i}")
    while i < repeat_cnt + 1:
        subprocess.run(["mpiexec", "-n", str(cpu_cnt), "python", "main.py", "--mode", "cma", "--exit_after_dumps"])
        new_dir = save_dir + f"dup{i}"
        print(f"new dir = {new_dir}")
        os.rename(save_dir, new_dir)
        os.makedirs(save_dir)
        shutil.copyfile(os.path.join(new_dir, main_fname), os.path.join(save_dir, main_fname))
        i += 1

    print("\n\n")
    Helper.print_total_time(start_time)


if __name__ == "__main__":
    run_multi_time()

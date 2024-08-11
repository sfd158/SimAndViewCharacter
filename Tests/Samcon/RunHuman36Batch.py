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
from argparse import ArgumentParser
import os
import json
import subprocess
from VclSimuBackend.Samcon.SamconMainWorkerBase import SamHlp


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="S5")
    args = parser.parse_args()

    fdir = os.path.dirname(__file__)
    conf_folder = os.path.join(fdir, "../CharacterData")
    conf_fname = os.path.join(conf_folder, "SamconConfig-duplicate.json")
    retarget_dir = "Human36Reheight"
    # modify config file..
    samhlp = SamHlp(conf_fname)
    conf = samhlp.conf

    # initial_sigmas: List[float] = conf["worker_cma"]["copy_init_cma_sigma_list"]
    data_dir = os.path.join(conf_folder, retarget_dir, args.name)

    # load data config...
    config_fname = os.path.join(data_dir, "config.json")
    with open(config_fname, "r") as fin:
        store_list = json.load(fin)
    store_dict = {node["file"]: node for node in store_list}

    save_folder_i_dname = samhlp.save_folder_i_dname()
    for name in os.listdir(data_dir):
        if not name.endswith(".bvh"):
            continue
        # check if file is reconstructed by Samcon
        rename_dir = os.path.join(samhlp.root_save_folder_dname(), name[:-4])
        if os.path.exists(os.path.join(rename_dir)):
            continue

        file_config = store_dict[name]
        if not file_config["in_use"]:
            continue
        fname_key = f"{retarget_dir}/{args.name}/{name}"
        conf["filename"]["bvh"] = fname_key

        start, end = file_config["start"], file_config["end"]
        conf["bvh"].update({"start": start, "end": end})
        conf["worker_cma"]["init_cma_sigma_list"] = file_config["sigmas"]
        with open(conf_fname, "w") as fout:
            json.dump(conf, fout)

        # call samcon algorithm
        print(f"===run samcon at {fname_key}, start={start}, end = {end}")
        subprocess.call(
            ["mpiexec", "-n", "36", "python", os.path.join(fdir, "main.py"), "--mode", "cma-dup-start"]
        )

        # rename the result dir..
        os.rename(save_folder_i_dname, rename_dir)


if __name__ == "__main__":
    main()

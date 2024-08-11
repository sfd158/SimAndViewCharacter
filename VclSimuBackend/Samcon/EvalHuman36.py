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
import numpy as np
import json
import os
import subprocess
import shutil
import psutil
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.Utils.Evaluation import mpjpe
fdir: str = os.path.dirname(__file__)
abs_fdir: str = os.path.abspath(fdir)


def run_network():
    # run network for each input bvh file..
    # sitting on the chair should be ignored.
    parser = ArgumentParser()
    parser.add_argument("--input_bvh_dir", type=str, default=
        os.path.join(fdir, "../../Tests/CharacterData/Human36Reheight/S11")
    )
    parser.add_argument("--output_dir", type=str, default=
        os.path.join(fdir, "pred_result_S11")
    )
    parser.add_argument("--pretrain_result", type=str, default=
        os.path.join(fdir, "policy_train.ckpt200")
        # r"G:\Samcon-Exps\learning-unified\Transformer81\policy_train.ckpt200"
    )
    args: Namespace = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main_fname = os.path.join(fdir, "PolicyTrain/Main.py")
    for fname in os.listdir(args.input_bvh_dir):
        # what about the result of sitting..?
        if not fname.endswith(".bvh"):
            continue
        print(f"process bvh file {fname} in neural network")
        tot_fname: str = os.path.join(args.input_bvh_dir, fname)
        output_dir: str = os.path.join(args.output_dir, fname[:-4])
        cmd = [
            "python", main_fname, "--mode", "eval", "--output_data_dir",
            output_dir, "--pretrain_result", args.pretrain_result, "--eval_attr_fname", tot_fname, "--eval_mode", "BVH_MOCAP",
            "--noise2d", "1e-6"
        ]
        print(" ".join(cmd))
        subprocess.call(cmd)
        print("\n\n\n#########################")

    print(f"After process all of bvh files")


def run_optim():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=os.path.join(fdir, "pred_result_S11"), help="input data dir")
    parser.add_argument("--output_dir", type=str, default=os.path.join(fdir, "optimize_S11-with-simple-plan-400-epoch"), help="final optimization result")
    # Note: maybe we can run samcon with smaller sampling window..
    parser.add_argument("--num_cores", type=int, default=36)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"input dir {args.input_dir} doesn't exist. ")
        exit(0)

    main_fname: str = os.path.abspath(os.path.join(fdir, "OptimalGait/OptimizeEachFrameParallel.py"))
    for fname in os.listdir(args.input_dir):
        tot_fname: str = os.path.join(args.input_dir, fname)
        if not os.path.isdir(tot_fname):
            continue
        # judge if the dir is the correct neural network output
        if not os.path.exists(os.path.join(tot_fname, "network-output.bin")):
            continue
        output_dir: str = os.path.join(args.output_dir, fname[:-4])
        print(f"#################\n\n\n\n\nprocess network input: {tot_fname}")
        if os.path.exists(os.path.join(output_dir, "log.txt")):
            print("the output dir already exists. ignore")
            continue

        # find the pre-saved result..
        file_list: str = [node[len("OptimizeEachFrame.ckpt"):] for node in os.listdir(output_dir) if "OptimizeEachFrame.ckpt" in node]
        for k in range(len(file_list)):
            if len(file_list[k]) == 2:
                file_list[k] = "0" + file_list[k]
        print(file_list)
        if len(file_list) > 0:
            file_list.sort()
            print(file_list)
            final_result: str = "OptimizeEachFrame.ckpt" + file_list[-1]
        else:
            final_result = None

        cmd = ["mpiexec", "-n", str(args.num_cores), "python", main_fname, "--process_fname", tot_fname,
            "--output_fdir",
            output_dir
        ]

        if final_result is not None:
            cmd.extend(["--checkpoint_fname", os.path.join(output_dir, final_result)])

        print(" ".join(cmd))
        subprocess.call(cmd)
        # here we should move the result to the output dir

    print(f"After process all of bvh files")


def parse_network_output():
    """
    We can gather the output result of neural network into a same tabular..
    """
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    args: Namespace = parser.parse_args()
    for fname in os.listdir(args.input_dir):
        full_fname: str = os.path.join(args.input_dir, fname)
        if not os.path.isdir(full_fname):
            continue
        log_fname = os.path.join(full_fname, "log.txt")
        with open(log_fname, "r") as fin:
            res = fin.readline()
        print(res)
        res_list = res.split(",")
        facing = res_list[1].split("=")[-1].strip()
        print(facing)
        glo = res.split(",")
        glo = res_list[2].split("=")[-1].strip()
        print(glo)


def run_samcon():
    """
    run samcon on optimized result, to get the total physics plausible result
    """
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="optimize_S9")
    parser.add_argument("--output_dir", type=str, default="Samcon_S9")
    parser.add_argument("--num_cores", type=int, default=36)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"input dir {args.input_dir} not exists. ignore..")
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main_fname: str = os.path.join(fdir, "../../Tests/Samcon/main.py")
    # load the original config file
    config_fname: str = os.path.join(fdir, "../../Tests/CharacterData/SamconConfig-duplicate.json")
    samcon_out_dir = os.path.join(fdir, "../../Tests/CharacterData/Samcon/0")
    with open(config_fname, "r") as fin:
        config = json.load(fin)
    for fname in os.listdir(args.input_dir):
        print(fname)
        # if not os.path.isdir(fname):
        #    continue
        # here we should take the input tracking file..
        # modify the config, and write to origin file..
        config["filename"]["invdyn_target"] = "../../VclSimuBackend/Samcon/" + args.input_dir + f"/{fname}/" + "opt_result/network-output.bin"
        with open(config_fname, "w") as fout:
            json.dump(config, fout)
        cmd = ["mpiexec", "-n", str(args.num_cores), "python", main_fname, "--mode", "cma-dup-start"]
        print(f"##########   run samcon on {fname}")
        print(" ".join(cmd))
        subprocess.call(cmd)
        # move the Samcon result to the output dir
        result_save_dir = os.path.join(args.output_dir, fname)
        shutil.move(samcon_out_dir, result_save_dir)
        print(f"===================move the samcon result to {result_save_dir}===============")


def eval_human36_mpjpe():
    """
    we need to scale the character into the same height.
    """
    parser = ArgumentParser()
    parser.add_argument("--bvh_fname", type=str,
        default=r"Z:\GitHub\ode-develop\VclSimuBackend\Samcon\Samcon_S11--with-simple-plan-400-epoch\Direction1-mocap\test.bvh.bvh")
    parser.add_argument("--gt_fname", type=str, default=r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Human36Reheight\S11\Direction1-mocap-100.bvh")
    args = parser.parse_args()
    # BVHLoader.save(BVHLoader.load(args.gt_fname).resample(50), r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Human36Reheight\S11\SittingDown1-mocap-50.bvh")
    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", args.bvh_fname, r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Human36Reheight\S11\Direction1-mocap-50.bvh"])
    motion1 = BVHLoader.load(args.bvh_fname).remove_end_sites().resample(50)
    motion2 = BVHLoader.load(args.gt_fname).remove_end_sites().resample(50).sub_sequence(42, -41)
    p_mpjpe = mpjpe(motion1.to_local_coordinate().joint_position, motion2.to_local_coordinate().joint_position)
    print(p_mpjpe)


def eval_physcap_loss():
    parser = ArgumentParser()
    parser.add_argument("--name1", type=str, default="")
    parser.add_argument("--name2", type=str, default="")
    args = parser.parse_args()
    motion1 = BVHLoader.load(args.name1)
    motion2 = BVHLoader.load(args.name2)
    # compute linear velocity
    lin_vel1 = np.linalg.norm(motion1.compute_linear_velocity(), axis=-1)
    lin_vel2 = np.linalg.norm(motion1.compute_linear_velocity(), axis=-1)



if __name__ == "__main__":
    # run_network()
    # run_optim()
    # parse_network_output()
    eval_human36_mpjpe()

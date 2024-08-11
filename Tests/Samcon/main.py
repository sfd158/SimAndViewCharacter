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
# for only use 1 thread
if __name__ == "__main__":
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    import importlib
    try:
        mkl = importlib.import_module("mkl")
        mkl.set_num_threads(1)
    except:
        pass

import argparse
import datetime
import logging
from mpi4py import MPI
import numpy as np
import psutil
import random
from threadpoolctl import threadpool_limits

from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.Samcon.SamconWorkerBase import SamHlp, WorkerInfo, LoadTargetPoseMode
from VclSimuBackend.Render.Renderer import RenderWorld  # for debug with Long Ge's framework


def build_args():
    info = WorkerInfo()
    fdir = os.path.dirname(__file__)
    cma_choice: str = ["cma-single", "cma-window", "cma-multi", "cma-dup-start", "traj-opt", "traj-opt-batch"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_conf_path",
                        type=str,
                        default=os.path.abspath(os.path.join(fdir, "../CharacterData")))
    parser.add_argument("--config", type=str,
                        default="SamconConfig-duplicate.json",
                        help="Path of Samcon Config file")
    parser.add_argument("--mode", type=str, default="cma-single", choices=cma_choice)
    parser.add_argument("--repeat_index", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=info.comm_rank)
    parser.add_argument("--load_target_mode", type=str,
        choices=LoadTargetPoseMode._member_names_, default="BVH_MOCAP")

    args = parser.parse_args()
    args.load_target_mode = LoadTargetPoseMode[args.load_target_mode]
    args.config = os.path.join(args.default_conf_path, args.config)

    return info, args


def set_log_conf(info, args):
    samhlp = SamHlp(args.config, args.repeat_index)
    save_dir, log_dir = samhlp.save_folder_i_dname(), samhlp.log_path_dname()

    if info.comm_rank == 0:
        samhlp.create_dir()
    _ = MPI.COMM_WORLD.bcast("sync")  # wait for main worker to create folder.

    logging.basicConfig(filename=os.path.join(log_dir, str(info.comm_rank) + ".log"),
                        filemode='w', level=logging.DEBUG)
    return samhlp


def main(info=None, args=None):
    if info is None:
        info, args = build_args()

    starttime = datetime.datetime.now()
    samhlp = set_log_conf(info, args)
    seed = args.random_seed
    logging.info(f"Working mode: {args.mode}, set random_seed = {seed}")
    np.random.seed(seed)
    random.seed(seed)

    def build_main_worker():
        import torch
        from VclSimuBackend.Samcon.SamconWorkerFull import SamconWorkerFull
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)

        def debug_func():
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()
            main_worker.n_iter = 6
            main_worker.rebuild_func()
            main_worker.run_single()
            main_worker.stop_other_workers()
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.dump_stats("profile.bin")

        if args.mode.startswith("cma"):
            from VclSimuBackend.Samcon.SamconCMA.MainWorkerCMANew import SamconMainWorkerCMA
            worker = SamconWorkerFull(samhlp, info)
            main_worker = SamconMainWorkerCMA(samhlp, info, worker, worker.scene, None, args.load_target_mode)
            if info.comm_size == 1 and False:
                render = RenderWorld(worker.scene.world)  # debug using Long Ge's Framework
                render.set_joint_radius(0.02)
                render.draw_hingeaxis(1)
                render.draw_localaxis(1)
                render.start()

            if args.mode == "cma-single":
                main_worker.run_single_hlp()
            elif args.mode == "cma-window":
                main_worker.fine_tune_with_window_hlp()
            elif args.mode == "cma-dup-start":
                main_worker.duplicate_by_same_start()
            else:
                raise ValueError
        elif args.mode == "traj-opt":
            from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import DirectTrajOptBVH
            worker = SamconWorkerFull(samhlp, info)
            main_worker = DirectTrajOptBVH(samhlp, info, worker, worker.scene)
            main_worker.test_direct_trajopt()
        elif args.mode == "traj-opt-batch":
            from VclSimuBackend.Samcon.SamconCMA.TrajOptBatch import TrajOptBVHBatch
            main_worker = TrajOptBVHBatch(samhlp, info, worker=SamconWorkerFull(args.config, info))
            main_worker.test_direct_trajopt()
        else:
            raise NotImplementedError

    with threadpool_limits(limits=1):
        if info.comm_rank == 0:
            if info.comm_size == 1:
                logging.info("Running in single thread. Not use mpi")
            build_main_worker()
        else:
            from VclSimuBackend.Samcon.SamconWorkerNoTorch import SamconWorkerNoTorch
            worker = SamconWorkerNoTorch(samhlp, info)
            logging.info(f"Worker memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:3f}")
            worker.run()

    if info.comm_rank == 0:
        Helper.print_total_time(starttime)


if __name__ == "__main__":
    main()

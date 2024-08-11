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
import gc
import json
import logging
import numpy as np
import os
import psutil
import random
import time
from typing import Dict, Any, List


class Helper:
    _empty_str_list = ["", "null", "none", "nullptr", "no", "not", "false", "abort"]
    _true_str_list = ["yes", "true", "confirm", "ok", "sure", "ready"]

    def __init__(self):
        pass

    @staticmethod
    def get_curr_time() -> str:
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    @staticmethod
    def print_total_time(starttime):
        endtime = datetime.datetime.now()
        delta_time = endtime - starttime
        # logging.info(f"start time: {starttime.strftime('%Y-%m-%d %H:%M:%S')}")
        # logging.info(f"end time: {endtime.strftime('%Y-%m-%d %H:%M:%S')}")
        # logging.info(f"delta time: {delta_time}")
        # logging.info(f"seconds = {delta_time.seconds}")

        print(f"\n\nstart time: {starttime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"end time: {endtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"delta time: {delta_time}")
        print(f"seconds = {delta_time.seconds}", flush=True)

    @staticmethod
    def is_str_empty(s: str) -> bool:
        return s.lower() in Helper._empty_str_list

    @staticmethod
    def str_is_true(s: str) -> bool:
        return s.lower() in Helper._true_str_list

    @staticmethod
    def pos_3d_viewer(x: np.ndarray):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        assert len(x.shape) == 2 and x.shape[-1] == 3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])
        plt.show()

    @staticmethod
    def conf_loader(fname: str) -> Dict[str, Any]:
        with open(fname, "r") as f:
            conf: Dict[str, Any] = json.load(f)
        filename_conf: Dict[str, str] = conf["filename"]
        for k, v in filename_conf.items():
            if k.startswith("__"):
                continue
            filename_conf[k] = os.path.join(os.path.dirname(fname), v)

        return conf

    @staticmethod
    def show_curr_mem() -> None:
        gcoll = gc.collect()
        mem = psutil.virtual_memory()
        mem_used, mem_free = mem.used / 1024.0 / 1024.0 / 1024.0, mem.free / 1024.0 / 1024.0 / 1024.0

        logging.info(
            f"Total mem use = {mem_used:.2f}GB, mem free = {mem_free:.2f}GB, "
            f"curr mem = {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB, "
            f"gc collect = {gcoll}"
        )

    @staticmethod
    def check_enough_memory(callback) -> None:
        mem = psutil.virtual_memory()
        free_mem: float = mem.available / 1024 / 1024 / 1024
        logging.info(f"available mem = {free_mem:.3f}")
        if free_mem < 10:
            if callback is None:
                callback = lambda : exit(0)

            callback(free_mem)

    @staticmethod
    def set_torch_seed(random_seed: int):
        import torch
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random.seed)
        os.environ['PYTHONHASHSEED'] = str(random_seed)

    @staticmethod
    def load_numpy_random_state(result: Dict[str, Any]) -> None:
        random_state = result.get("random_state")
        if random_state is not None:
            random.setstate(random_state)

        np_rand_state = result.get("np_rand_state")
        if np_rand_state is not None:
            np.random.set_state(np_rand_state)

    @staticmethod
    def save_numpy_random_state() -> Dict[str, Any]:
        result = {
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        }
        return result

    @staticmethod
    def mirror_name_list(name_list: List[str]):  # This is only tested on std-human.
        indices = [i for i in range(len(name_list))]
        def index(name):
            try:
                return name_list.index(name)
            except ValueError:
                return -1

        for i, n in enumerate(name_list):
            # rule 1: left->right
            idx = -1
            if n.find('left') == 0:
                idx = index('right' + n[4:])
            elif n.find('Left') == 0:
                idx = index('Right' + n[4:])
            elif n.find('LEFT') == 0:
                idx = index('RIGHT' + n[4:])
            elif n.find('right') == 0:
                idx = index('left' + n[5:])
            elif n.find('Right') == 0:
                idx = index('Left' + n[5:])
            elif n.find('RIGHT') == 0:
                idx = index('LEFT' + n[5:])
            elif n.find('L') == 0:
                idx = index('R' + n[1:])
            elif n.find('l') == 0:
                idx = index('r' + n[1:])
            elif n.find('R') == 0:
                idx = index('L' + n[1:])
            elif n.find('r') == 0:
                idx = index('l' + n[1:])

            indices[i] = idx if idx >= 0 else i

        return indices

    @staticmethod
    def save_torch_state() -> Dict[str, Any]:
        """
        save random state of library: pytorch, numpy, random
        """
        import torch
        cuda_available = torch.cuda.is_available()
        result = Helper.save_numpy_random_state()
        result.update({
            "torch_rand_state": torch.get_rng_state(),
            "torch_cuda_rand_state": torch.cuda.get_rng_state() if cuda_available else None,
            "torch_cuda_rand_state_all": torch.cuda.get_rng_state_all() if cuda_available else None
        })

        return result

    @staticmethod
    def load_torch_state(saved_model: Dict[str, Any]):
        """
        load random state of library: pytorch, numpy, random
        """
        import torch
        Helper.load_numpy_random_state(saved_model)

        torch_rand_state = saved_model.get("torch_rand_state")
        if torch_rand_state is not None:
            try:
                torch.set_rng_state(torch_rand_state)
            except Exception as e:
                print(torch_rand_state)
                print(type(torch_rand_state))
                return

        if torch.cuda.is_available():
            torch_cuda_rand_state = saved_model.get("torch_cuda_rand_state", None)
            if torch_cuda_rand_state is not None:
                torch.cuda.set_rng_state(torch_cuda_rand_state)
            torch_cuda_rand_state_all = saved_model.get("torch_cuda_rand_state_all", None)
            if torch_cuda_rand_state_all is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rand_state_all)

import copy
import time
import random
import argparse

import torch
from torch.autograd import Variable
from tqdm import tqdm
from typing import List, Optional, Callable, Dict
from VclSimuBackend.Common.Helper import Helper

from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.Samcon.Loss.SamconLoss import SamconLoss

from VclSimuBackend.Samcon.SamconMainWorkerBase import SamHlp
from VclSimuBackend.Samcon.StateTree import Sample
from VclSimuBackend.Samcon.TrajOpt.PyTorchSamconTargetPose import *

from VclSimuBackend.DiffODE.Build import BuildFromODEScene
from VclSimuBackend.DiffODE.DiffODEWorld import DiffODEWorld
from VclSimuBackend.DiffODE.DiffFrameInfo import DiffFrameInfo
from VclSimuBackend.DiffODE.DiffPDControl import DiffDampedPDControler, DiffDampedPDFast
from VclSimuBackend.DiffODE.PyTorchMathHelper import PyTorchMathHelper
from VclSimuBackend.DiffODE.DiffQuat import quat_apply, quat_inv, quat_to_rotvec, quat_from_rotvec, quat_multiply


class SimpleLoss:
    """
    calc loss between curr frame joint local rotation & T pose rotation(That is, all zero..)
    """
    def __init__(self):
        pass


class TPoseTrajOpt:
    """
    Optimize total samcon path.
    Variable: all of samcon control offset
    Loss: sum of samcon loss at each time slice

    """

    def __init__(self, samhlp: SamHlp):
        self.samhlp: SamHlp = samhlp
        self.conf = Helper.conf_loader(samhlp.conf_name)

        self.max_iter: int = self.conf["global_traj_opt"]["max_iter"]
        self.lr: float = self.conf["global_traj_opt"]["lr"]  # learning rate
        rand_seed: int = self.conf["global_traj_opt"]["rand_seed"]
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        # load Samcon results, and convert to torch.Tensor
        self.best_path: List[Sample] = samhlp.load_best_path_idx()
        actions = np.concatenate([sample.a0[None, ...] for sample in self.best_path[1:]], axis=0)
        self.acts = Variable(torch.as_tensor(actions), requires_grad=True)  # l_path-1, num joint, 3
        self.sample_fps: float = self.conf["worker"]["sample_fps"]

        # samcon loss using numpy
        self.samcon_loss: SamconLoss = SamconLoss(self.conf)

        # build ODE Scene
        self.scene: ODEScene = JsonSceneLoader().load_from_file(self.conf["filename"]["world"])
        self.character: ODECharacter = self.scene.character0
        self.builder = BuildFromODEScene(self.scene)
        self.sim_cnt: int = int(self.scene.sim_fps / self.sample_fps)
        self.select_idx = self.sim_cnt * torch.arange(1, len(self.best_path), dtype=torch.long)

        # load samcon target and convert to pytorch
        self.sam_target, self.inv_dyn_target = samhlp.load_target(self.conf, self.scene, self.character)
        self.torch_sam_target = PyTorchSamconTargetPose.build_from_numpy(self.sam_target)
        self.torch_inv_dyn_target = PyTorchTargetPose.build_from_numpy(self.inv_dyn_target)

        # convert ODE Scene to DiffODE Scene..
        self.diff_world: DiffODEWorld = self.builder.build()
        # self.damped_pd = DiffDampedPDControler.build_from_joint_info(self.character.joint_info)
        self.damped_pd = DiffDampedPDFast()

        self.root_idx: int = self.diff_world.curr_frame.body_frame.const_info.root_index

        self.result: Optional[List[Sample]] = None

    @property
    def curr_frame(self) -> DiffFrameInfo:
        return self.diff_world.curr_frame

    @property
    def target_quat(self) -> torch.Tensor:
        return self.torch_inv_dyn_target.locally.quat

    def run(self):
        """
        Global trajectory optimization
        """

        np_loss_buf: List[Tuple[float, Dict[str, float]]] = []

        def calc_loss_numpy(idx: int):
            _np_loss = self.samcon_loss.loss_debug(self.character, self.sam_target, idx * self.sim_cnt)
            np_loss_buf.append(_np_loss)

        def forward_sim(_callback: Optional[Callable[[int], None]] = None) -> List[DiffFrameInfo]:
            """
            Total path simulation
            """
            _start_time = time.time()
            print("Begin forward simulation.")
            _frame_buff: List[DiffFrameInfo] = [self.diff_world.curr_frame]

            for _idx, _path in tqdm(enumerate(self.best_path[1:])):
                for _i in range(1, self.sim_cnt + 1):  # forward simulation
                    # torque of each joint..Here we should use samcon result..
                    _act: torch.Tensor = self.acts[_i - 1]
                    _aquat: torch.Tensor = quat_from_rotvec(_act)
                    _tot_target = quat_multiply(_aquat, self.target_quat[_idx * self.sim_cnt + _i, :, :])
                    self.curr_frame.stable_pd_control_wrapper(self.damped_pd, _tot_target)

                    self.diff_world.step(do_contact=True)

                _frame_buff.append(self.diff_world.curr_frame)
                if _callback is not None:
                    _callback(_idx)

            _end_time = time.time()
            print(f"End forward simulation. Total use time = {_end_time - _start_time}")
            return _frame_buff

        optim = torch.optim.Adam([self.acts], lr=self.lr)  # optimize control signal
        print("Begin Global Trajectory Optimization")
        start_traj = time.time()
        for i in range(self.max_iter):
            # build from state of frame 0
            self.character.load(self.best_path[0].s0)
            frame_buff: List[DiffFrameInfo] = forward_sim(calc_loss_numpy)
            loss = self.calc_samcon_loss(frame_buff)
            # optimization
            print(f"i = {i}, Total path loss = {loss}")
            optim.zero_grad()
            optim.step()

        end_traj = time.time()
        print(f"End Global Trajectory Optimization {end_traj - start_traj}")

        # convert to new path.
        new_path: List[Sample] = copy.deepcopy(self.best_path)

        def convert_callback(n_iter: int):
            state: BodyInfoState = self.diff_world.export_curr_frame()
            new_path[n_iter].s1 = state
            new_path[n_iter + 1].s0 = state
            new_path[n_iter + 1].a0 = np.ascontiguousarray(self.acts[n_iter].detach().numpy(), dtype=np.float64)
            new_path[n_iter + 1].parent = new_path[n_iter]

        with torch.no_grad():
            forward_sim(convert_callback)

        self.result = new_path
        return new_path

    def save(self):
        fname = self.samhlp.best_path_fname() + "traj-optim"
        save_samcon_path(self.result, fname)
        print(f"Save Trajectory Optimization result to {fname}")


def calc():
    fname = "../../../Tests/CharacterData/SamconConfig.json"
    tot_optim = TPoseTrajOpt(SamHlp(fname, 0))
    tot_optim.run()
    tot_optim.save()


if __name__ == "__main__":
    calc()

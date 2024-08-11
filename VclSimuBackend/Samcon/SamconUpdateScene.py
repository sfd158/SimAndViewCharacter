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

import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from typing import Dict, Any, List, Optional, Generator, Tuple

from .StateTree import Sample
from .SamconWorkerNoTorch import SamconWorkerNoTorch
from ..Common.MathHelper import MathHelper
from ..ODESim.BodyInfoState import BodyInfoState
from ..ODESim.ODECharacter import ODECharacter
from ..ODESim.UpdateSceneBase import UpdateSceneBase, ODEScene
from ..ODESim.UpdateScene.UpdateBVHWrapper import UpdateSceneBVHDirectWrapper
from ..ODESim.Saver.CharacterToBVH import CharacterTOBVH
from ..pymotionlib.MotionData import MotionData


class SamconUpdateScene(UpdateSceneBase):

    def __init__(
        self,
        worker: SamconWorkerNoTorch,
        best_path: List[Sample],
        samcon_save_prefix: Optional[str] = "samcon-result",
        debug_mode: bool = True,
        check_sim_err: bool = True,
        ):
        super(SamconUpdateScene, self).__init__(worker.scene)
        self.debug_mode = debug_mode
        self.check_sim_err = check_sim_err

        self.sim_character: ODECharacter = worker.character

        # for export simulation result
        self.to_bvh = CharacterTOBVH(worker.character, worker.scene.sim_fps)
        self.to_bvh.bvh_hierarchy_no_root()
        self.ret_sim_hist: Optional[List[BodyInfoState]] = None
        self.sim_hist: List[BodyInfoState] = []

        self.worker = worker
        self.best_path: List[Sample] = best_path
        self._start_index = self.get_start_index()
        self.frame: int = 0
        self.mask = worker.joint_info.sample_mask

        if samcon_save_prefix is not None:
            self.save_bvh_fname: str = samcon_save_prefix + ".bvh"
            self.save_sim_fname: str = samcon_save_prefix + ".pickle"
        else:
            self.save_bvh_fname = None
            self.save_sim_fname = None

        self.load_initial_state()
        self.uhelp: Optional[Generator[int, Any, None]] = self.update_help()

        self.ret_motion: Optional[MotionData] = None

    def total_num_frames(self) -> int:
        return (len(self.best_path) - self._start_index) * self.sim_cnt + 1

    def get_start_index(self):
        _index: int = 0
        while _index + 1 < len(self.best_path) and self.best_path[_index + 1].a0 is None:
            _index += 1
        return _index

    def load_initial_state(self):
        self.sim_character.load(self.best_path[self._start_index].s1)

    def update_help(self):
        self.ret_motion: Optional[MotionData] = None
        self.load_initial_state()
        if self.to_bvh is not None:
            self.to_bvh.append_no_root_to_buffer()
        sub_path = self.best_path[1:]
        for idx, path in enumerate(sub_path):
            if path.a0 is None:
                continue
            self.scene.str_info = str(
                f"At {idx + 1}, "
                f"com_h = {self.sim_character.body_info.calc_center_of_mass()[1]:.3f}"
            )
            off_rot = Rotation.from_rotvec(self.mask * path.a0).as_quat()
            if off_rot.shape[0] < len(self.character0.joints):
                new_off_rot: np.ndarray = MathHelper.unit_quat_arr((len(self.character0.joints), 4))
                new_off_rot[self.worker.track_joint_index, :] = off_rot  # TODO
                off_rot = np.ascontiguousarray(new_off_rot)
            off_rot = Rotation(off_rot, False, False)
            forward_count, loss_index = self.worker.get_forward_count(idx + 1)
            pd_target, start_index = self.worker.get_pd_target_array(idx + 1)
            if self.debug_mode:
                dbg_state = self.character0.save()
                if self.check_sim_err:
                    dbg_state.check_delta(self.best_path[idx].s1)
                self.sim_character.load(self.best_path[idx].s1)
            for i in range(forward_count):
                local_q: np.ndarray = (off_rot * Rotation(pd_target[i], False, False)).as_quat()
                if self.sim_hist is not None:
                    curr_state: BodyInfoState = self.sim_character.save()
                    curr_state.pd_target = local_q.copy()
                    self.sim_hist.append(curr_state)

                self.worker.damped_pd.add_torques_by_quat(local_q)
                self.scene.simu_func()
                self.frame += 1
                if self.to_bvh is not None:
                    self.to_bvh.append_no_root_to_buffer()
                yield self.frame

        # After whole simulation, save the last state into list
        if self.sim_hist is not None:
            self.sim_hist.append(self.sim_character.save())

        self.reset_uhelp()

    def reset_uhelp(self) -> int:
        self.load_initial_state()
        self.uhelp = self.update_help()
        if self.to_bvh is not None:
            self.to_bvh.motion._fps = self.scene.sim_fps
            self.ret_motion: Optional[MotionData] = self.to_bvh.to_file(self.save_bvh_fname)

        if self.save_sim_fname and self.sim_hist is not None:
            with open(self.save_sim_fname, "wb") as fout:
                pickle.dump(self.sim_hist, fout)

        self.to_bvh: Optional[CharacterTOBVH] = None
        self.ret_sim_hist = self.sim_hist
        self.sim_hist: Optional[List[BodyInfoState]] = None
        self.frame = 0
        return self.frame

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        try:
            # for _ in range(self.scene.step_cnt):
            self.frame = next(self.uhelp)
        except StopIteration:
            self.reset_uhelp()

    @staticmethod
    def visualize_sim_hist(sim_hist: List[BodyInfoState]):
        pass

    @staticmethod
    def eval_best_path(
        worker: SamconWorkerNoTorch,
        best_path: List[Sample],
        samcon_save_prefix: str
        ) -> Tuple[List[BodyInfoState], MotionData]:
        update_scene = SamconUpdateScene(worker, best_path, samcon_save_prefix)
        _ = list(iter(update_scene.uhelp))

        return update_scene.ret_sim_hist, update_scene.ret_motion


class SamconUpdateSceneWithRef(UpdateSceneBase):
    def __init__(
        self,
        scene: ODEScene,
        samcon_update: SamconUpdateScene,
        bvh_update: Optional[UpdateSceneBVHDirectWrapper] = None,
        ):
        super().__init__(scene=scene)
        self.samcon_update = samcon_update
        self.bvh_update = bvh_update
        self.frame = 0

    def set_bvh(self):
        if self.bvh_update is not None:
            self.bvh_update.set_tar.set_character_byframe(self.frame % self.bvh_update.target.num_frames)

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        try:
            for _ in range(self.scene.step_cnt):
                self.frame = next(self.samcon_update.uhelp)
                self.set_bvh()
        except StopIteration:
            self.samcon_update.reset_uhelp()
            self.frame = 0
            self.set_bvh()

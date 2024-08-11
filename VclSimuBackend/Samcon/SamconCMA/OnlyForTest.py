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

from .MainWorkerCMANew import *


class MainWorkerCMATest(SamconMainWorkerCMA):
    def __init__(self, samhlp: SamHlp, worker_info: WorkerInfo, worker: Optional[SamconWorkerFull] = None, scene: Optional[ODEScene] = None, sim_character: Optional[ODECharacter] = None):
        super().__init__(samhlp, worker_info, worker=worker, scene=scene, sim_character=sim_character)

        # TODO: for paper `Guided Learning of Control Graphs for Physics-Based Characters`
        self.linear_policy = []
        self.saved_path = []

    def dup_different_frame_forward(
        self,
        dup_count: int = 10,
        saved_cma: List[CMAUpdate] = None
        ):
        samcon_buffer = []
        for start_index in range(1, dup_count + 1):
            self.rebuild_func(f" start_idx = {start_index}", saved_cma is None)
            self.k_start = start_index
            self.tree.reset()
            for pre_index in range(start_index):
                self.tar_set.set_character_byframe(pre_index * self.sim_cnt)
                state = self.character.save()
                tree_node = Sample(None, state, state, t=pre_index)
                if pre_index > 0:
                    tree_node.parent = self.tree.tree[pre_index - 1][0]
                self.tree.tree.append([tree_node])
            if saved_cma is not None:
                self.cma = copy.deepcopy(saved_cma)

            ret_best_path, pd_target, motion = self.run_single()
            samcon_buffer.append([ret_best_path, pd_target, motion])

        # merge the duplicated action piece.
        prev_cat_action_buf = []
        cat_action_buf = []
        for start_index in range(1, dup_count + 1):
            ret_best_path: List[Sample] = samcon_buffer[start_index - 1][0]["best_path"]
            cat_action: np.ndarray = np.concatenate([sample.a0[None, ...] for sample in ret_best_path[dup_count:]], axis=0)
            cat_action_buf.append(cat_action[None, ...])

            # compute the previous actions
            prev_cat_action: np.ndarray = np.concatenate(
                [sample.a0[None, ...] if sample.a0 is not None else np.zeros((1, self.num_joints, 3))
                    for sample in ret_best_path[1:dup_count]
                ],
                axis=0
            )
            prev_cat_action_buf.append(prev_cat_action[None, ...])

        prev_cat_action_buf = operator.truediv(
            np.sum(np.concatenate(prev_cat_action_buf, axis=0), axis=0),
            np.arange(1, dup_count).reshape(dup_count - 1, 1, 1)
        )
        cat_action_buf: np.ndarray = np.mean(np.concatenate(cat_action_buf, axis=0), axis=0)
        cat_action_buf = np.concatenate([prev_cat_action_buf, cat_action_buf], axis=0)
        self.rebuild_func()
        # copy action piece to new cma window
        for k_start in range(1, self.n_iter + 1):
            self.cma[k_start].x_mean = cat_action_buf[k_start - 1].flatten()

    def duplicate_by_different_frame(
        self,
        dup_count: int = 10,
        full_finetune_win: bool = True
    ):
        self.send_target_pose_no_scatter()
        init_cov_max = np.max(self.joint_info.sample_win).item()
        sigmas = [0.25 / init_cov_max, 0.15]
        self.dup_idx = 0
        self.rebuild_func()
        # self.loads()
        self.set_cma_sigma(sigmas[0])
        saved_cma = copy.deepcopy(self.cma)

        def pre_finetune(_sigma: float):
            self.fine_tune_with_window(min(dup_count // 2, 2), 20, _sigma / 4, 500, dup_count + 1, False)

        while self.dup_idx < len(sigmas) - 1:
            sigma: float = sigmas[self.dup_idx]
            for cma_idx in range(len(saved_cma)):
                saved_cma[cma_idx].sigma = sigma

            self.dup_different_frame_forward(dup_count, saved_cma)

            # fine-tune previous action pieces by sliding window..
            # as the length of previous actions is not long, it's OK
            self.k_start = 1
            sigma = sigmas[self.dup_idx + 1]
            self.set_cma_sigma(sigma)
            with tqdm(None, f"Optim on window, sigma = {self.cma[1].sigma:.4f}"):
                pass

            if full_finetune_win:
                self.fine_tune_with_window(min(dup_count // 2, 2), 20, sigma / 6, 500, do_clear=False)
                ret_best_path, sim_full_list, motion = self.eval_best_path()
            else:
                pre_finetune(sigma)
                # go forward with raw samcon using a small sigma..
                with tqdm(None, f"After prefinetune, k_start = {self.k_start}, sigma = {self.cma[1].sigma:.4f}"):
                    pass
                ret_best_path, sim_full_list, motion = self.run_single()

            self.dumps_path_mode(ret_best_path)
            self.dumps_path_mode(ret_best_path, self.dup_idx)
            BVHLoader.save(motion, os.path.join(self.samhlp.save_folder_i_dname(), f"test-{self.dup_idx}.bvh"))
            self.set_cma_sigma(sigma)
            saved_cma = copy.deepcopy(self.cma)
            self.dup_idx += 1

        self.stop_other_workers()

    def compute_linear_policy(self):
        """
        Implement Linear Policy in paper,
        `Liu. et al. Guided Learning of Control Graphs for Physics-Based Characters, SIGGRAPH 2016`
        """
        raise NotImplementedError
        nb: int = len(self.bodies)
        nj: int = self.num_joints

        def get_policy_input(body_state: BodyInfoState) -> np.ndarray:
            vec6d = np.ascontiguousarray(body_state.rot.reshape((nb, 3, 3))[..., :2])
            return np.concatenate([body_state.pos.reshape(-1), body_state.linear_vel.reshape(-1),
                                   vec6d, body_state.angular_vel.reshape(-1)])

        input_dim: int = get_policy_input(self.character.init_body_state).size
        output_dim: int = 6 * nj  # what's if predict 6d vector, then convert back to rotvec?
        self.linear_policy = [[torch.nn.Linear(input_dim, output_dim), None, None] for _ in range(self.init_n_iter + 1)]

    def finetune_multi_times(self):
        self.dup_idx = 0
        self.send_target_pose_no_scatter()
        self.loads()
        if self.dup_idx == 0:
            self.rebuild_func()
            ret_best_path, pd_target, motion = self.run_single()
            self.dup_idx += 1
            self.dumps_path_mode(ret_best_path)
            self.dumps_path_mode(ret_best_path, str(self.dup_idx))

        params = [None, (1, 20, 0.15),] #  (2, 10, 0.12), (3, 5, 0.08)]
        while self.dup_idx < len(params):
            self.rebuild_func()
            self.loads()
            ret_best_path, pd_target, motion = self.fine_tune_with_window(*params[self.dup_idx])
            self.dup_idx += 1
            self.dumps_path_mode(ret_best_path)
            self.dumps_path_mode(ret_best_path, str(self.dup_idx))

        self.stop_other_workers()

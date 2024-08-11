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

class RunCycleMotion(SamconMainWorkerCMA):
    def __init__(self, samhlp: SamHlp, worker_info: WorkerInfo, worker: Optional[SamconWorkerFull] = None, scene: Optional[ODEScene] = None, sim_character: Optional[ODECharacter] = None):
        super().__init__(samhlp, worker_info, worker=worker, scene=scene, sim_character=sim_character)

        # for samcon repeat noise reduction
    def run_on_dup_motion(self, stop_workers: bool = True, save_bvh=True, exit_after_dumps=False):
        def _exit_func():
            if exit_after_dumps:
                self.stop_other_workers()
                exit(0)

        if self.cma_info.dup_count > 1:
            assert self.n_iter * self.sim_cnt == self.target.num_frames
            self.enable_worker_duplicate()
            logging.info(f"Duplicate for {self.cma_info.dup_count} times")
            self.n_iter: int = self.cma_info.dup_count * self.n_iter
            self.reset_com_list(self.n_iter + 1)

        dup_conf_list: List[Dict[str, Any]] = self.conf["worker_cma"]["duplicate_input"]["dup_after_cma"]
        max_dup_cnt = max([self.cma_info.dup_count] + [dup_conf.get("dup_count", 0) for dup_conf in dup_conf_list])
        self.target.pose.compute_global_root_dup(max_dup_cnt)
        self.send_target_pose_no_scatter()

        self.loads()
        if self.dup_idx == 0:
            if self.cma_info.dup_count:
                self.rebuild_func()
            self.set_loss_list_index(self.dup_idx)
            ret_best_path, pd_target, motion = self.run_single()
            self.dup_idx = 1
            self.dumps_path_mode(ret_best_path)
            self.dumps_path_mode(ret_best_path, str(self.dup_idx))
            if save_bvh:
                BVHLoader.save(motion, os.path.join(self.samhlp.save_folder_i_dname(), f"test-{self.dup_idx}.bvh"))
            _exit_func()
        else:
            ret_best_path, pd_target, motion = self.eval_best_path("test.bvh" if save_bvh else "")

        if self.cma_info.dup_count:
            dup_conf_list: List[Dict[str, Any]] = self.conf["worker_cma"]["duplicate_input"]["dup_after_cma"]
            while self.dup_idx <= len(dup_conf_list):
                self.set_loss_list_index(self.dup_idx)
                retrack_fname: str = os.path.join(self.samhlp.save_folder_i_dname(), f"retrack-{self.dup_idx}.bvh")
                self.merge_dup_result(ret_best_path["best_path"], pd_target, motion, dup_conf_list[:self.dup_idx],
                                      None if self.dup_idx < len(dup_conf_list) else 1,
                                      retrack_out_fname=retrack_fname)
                self.dup_idx += 1
                logging.info(f"re-run samcon at {self.dup_idx}")  # Run Samcon Algorithm again
                with tqdm([], f"Dup {self.dup_idx}, init sigma = {self.cma_info.init_sigma}"):
                    pass
                # TODO: Here we should search sigma adaptively..
                ret_best_path, pd_target, motion = self.run_single()
                self.dumps_path_mode(ret_best_path)
                self.dumps_path_mode(ret_best_path, str(self.dup_idx))
                BVHLoader.save(motion, os.path.join(self.samhlp.save_folder_i_dname(), f"test-{self.dup_idx}.bvh"))
                _exit_func()

        if stop_workers:
            self.stop_other_workers()

        with open(self.samhlp.best_path_fname(), "wb") as fout:
            pickle.dump(ret_best_path, fout)

        return ret_best_path, pd_target, motion

    """
    a good case:
    merge_action: bool = False,
    merge_pd_target: bool = True,
    use_initial_root_pos: bool = False,
    self.cma_info.init_sigma = 0.025 / 0.03.  0.02 doesn't work with dup == 20

    a good case:
    merge_action: bool = True,
    merge_pd_target: bool = False,
    use_initial_root_pos: bool = False,
    self.cma_info.init_sigma = 0.035.    with dup == 20

    """
    def merge_dup_result(self,
                         best_path: List[Sample],
                         pd_target: np.ndarray,
                         motion: MotionData,
                         dup_conf_list: List[Dict[str, Any]],
                         new_dup_count: Optional[int] = 1,
                         merge_action: bool = True,
                         merge_pd_target: bool = False,
                         merge_target_pose: bool = False,
                         use_initial_root_pos: bool = True,
                         retrack_out_fname: Optional[str] = "test-retrack.bvh"
                         ):
        """
        Param:
        best_path:
            Samcon actions

        motion:
            Samcon result

        merge_action:
            if true, use mean of action as initial solution.
            if false, use zero as initial solution

        merge_pd_target:
            if true, use mean of target pose for PD Controller as initial inverse dynamics result.
            if false, use initial inverse dynamics result.

        rot_merge:
            if LOCAL, use mean of joint local quaternion as target pose
            if GLOBAL, use mean of joint global quaternion as target pose (It's wrong)

        TODO: merge sim result as ref motion has bug to fix..
        """
        print(f"Motion num frames: {motion.num_frames}")
        num_frame: int = self.target.num_frames
        # nb: int = len(self.bodies)  # 20 for std-human, num of bodies
        nj: int = self.num_joints  # 19 for std-human, num of joints

        dup_count: int = self.cma_info.dup_count
        for dup_conf in dup_conf_list:
            self.cma_info.change_cma_param(dup_conf)
        if new_dup_count is None:
            new_dup_count: int = self.cma_info.dup_count
        self.n_iter = new_dup_count * self.init_n_iter
        self.rebuild_func()

        def merge_act_func():  # Merge action.
            with tqdm([], "Merge Action") as tmp_info:
                pass

            sample_get = lambda cma_idx_, dup_idx_: best_path[dup_idx_ * self.init_n_iter + cma_idx_]
            for cma_idx in range(1, self.init_n_iter + 1):
                mean_action: np.ndarray = np.mean(np.concatenate([sample_get(cma_idx, dup_idx).a0[None, ...] for dup_idx in range(dup_count)], axis=0), axis=0)
                self.cma[cma_idx].x_mean = mean_action.reshape(-1).copy()
            # duplicate new actions
            if new_dup_count > 1:
                for dup_idx in range(1, new_dup_count):
                    for cma_idx in range(1, self.init_n_iter + 1):
                        self.cma[dup_idx * self.init_n_iter + cma_idx].x_mean = self.cma[cma_idx].x_mean.copy()

        # Merge character pose
        def merge_root_by_init(retrack_: MotionData) -> MotionData:
            retrack_.joint_translation[:, 0, :] = self.target.pose.root.pos.copy()
            retrack_.joint_rotation[:, 0, :] = self.target.pose.root.quat[:, :]

            return retrack_

        def merge_root_by_delta(retrack_: MotionData) -> MotionData:
            # TODO: should merge in facing coordinate
            raise ValueError
            root_quat: Rotation = Rotation(motion.joint_rotation[:, 0, :])  # (dup_count * num_frame, 4)
            root_quat_inv: Rotation = root_quat.inv()  # (dup_count * num_frame, 4)
            root_pos: np.ndarray = motion.joint_translation[:, 0, :]  # (dup_count * num_frame, 3)
            delta_pos: np.ndarray = root_pos[1:, :] - root_pos[:-1, :]  # (dup_count * num_frame - 1, 3)
            delta_pos: np.ndarray = np.concatenate([delta_pos, np.zeros((1, 3))], axis=0)  # (dup_count * num_frame, 3)
            facing_delta_pos: np.ndarray = root_quat_inv.apply(delta_pos)  # (dup_count * num_frame, 3)
            facing_delta_pos: np.ndarray = facing_delta_pos.reshape((dup_count, num_frame, 3))

            delta_root_quat: Rotation = root_quat_inv[:-1] * root_quat[1:]  # (dup_count * num_frame - 1, 4)
            delta_root_quat: np.ndarray = np.concatenate(
                [delta_root_quat.as_quat(), MathHelper.unit_quat_arr((1, 4))], axis=0)  # (dup_count * num_frame, 4)
            delta_root_quat: np.ndarray = delta_root_quat.reshape((dup_count, num_frame, 4))

            # result for root quaternion and position
            root_quat_res: np.ndarray = MathHelper.unit_quat_arr((num_frame, 4))
            root_quat_res[0, :] = motion.joint_rotation[0, 0, :].copy()
            last_root_rot_res: Rotation = Rotation(root_quat_res[0])
            root_pos_res: np.ndarray = np.zeros((num_frame, 3), dtype=np.float64)
            root_pos_res[0, :] = motion.joint_translation[0, 0, :].copy()
            root_pos_res[0, 1] = np.mean(motion.joint_translation[::num_frame, 0, 1])  # mean of y height

            for frame_idx in range(0, num_frame - 1):
                mean_idx_dq: np.ndarray = calc_mean_quaternion(delta_root_quat[:, frame_idx, :])
                mean_idx_dpos: np.ndarray = np.mean(facing_delta_pos[:, frame_idx, :], axis=0)
                root_pos_res[frame_idx + 1, :] = root_pos_res[frame_idx, :] + last_root_rot_res.apply(mean_idx_dpos)
                last_root_rot_res: Rotation = Rotation(mean_idx_dq) * last_root_rot_res
                root_quat_res[frame_idx + 1, :] = last_root_rot_res.as_quat()

            root_quat_res: np.ndarray = MathHelper.flip_quat_by_dot(root_quat_res)
            retrack_.joint_rotation[:, 0, :] = root_quat_res[:, :]
            retrack_.joint_translation[:, 0, :] = root_pos_res[:, :]

            retrack_._joint_rotation = np.ascontiguousarray(retrack_._joint_rotation)
            retrack_._joint_translation = np.ascontiguousarray(retrack_._joint_translation)
            return retrack_

        def merge_retrack():
            raise ValueError
            # TODO: recompute global position here for calc loss...
            # motion.recompute_joint_global_info()
            retrack_: MotionData = motion.get_hierarchy(copy=True)
            retrack_.set_anim_attrs(num_frame, self.sim_fps)

            # Merge Joint Rotation by local quaternion
            logging.info("merge joint rotation locally")
            joint_quat: np.ndarray = motion.joint_rotation.reshape((dup_count, num_frame, motion.num_joints, 4))
            # get mean value of joint local rotation
            end_flag = motion.get_end_flags()
            joint_rots = MathHelper.unit_quat_arr((num_frame, motion.num_joints, 4))
            for frame_idx in tqdm(range(num_frame), "Mean JointQuat"):
                for joint_idx in range(1, motion.num_joints):
                    if end_flag[joint_idx]:
                        continue
                    dup_quats: np.ndarray = np.ascontiguousarray(joint_quat[:, frame_idx, joint_idx, :])
                    mean_quat: np.ndarray = calc_mean_quaternion(dup_quats, print_mess=False)  # (4,)
                    joint_rots[frame_idx, joint_idx, :] = mean_quat

            retrack_._joint_rotation = joint_rots

            # Note: we should not merge root quaternion directly in global coordinate.
            if use_initial_root_pos:
                merge_root_by_init(retrack_)
            else:
                merge_root_by_delta(retrack_)

            retrack_.recompute_joint_global_info()
            # visualize mixed motion. Same to bvh file for visualize.
            if retrack_out_fname:
                BVHLoader.save(retrack_, retrack_out_fname)

            return retrack_

        # recompute target pose
        if merge_action:
            merge_act_func()
        if merge_target_pose:
            raise ValueError
            retrack = merge_retrack()
            self.target: SamconTargetPose = SamconTargetPose.load2(retrack, self.character, int(self.sim_fps))
            self.tar_set = SetTargetToCharacter(self.character, self.target.pose)
            # resend target pose to sub workers. We need not calc inverse dynamics here.
            # because the simulated pose is good enough
            self.send_target_pose_no_scatter()

        self.init_search_tree()
        gc.collect()
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

import copy
import logging
import numpy as np
import os
import pickle
import shutil
from tqdm import tqdm
from typing import Optional, List, Dict, Union, Any, Tuple

from .SamconTargetPose import SamconTargetPose, TargetPose2d
from .StateTree import Sample

from ..Common.Helper import Helper
from ..Common.MathHelper import MathHelper

from ..ODESim.BodyInfoState import BodyInfoState
from ..ODESim.ODEScene import ODEScene
from ..ODESim.ODECharacter import ODECharacter
from ..ODESim.Saver.CharacterToBVH import CharacterTOBVH
from ..ODESim.TargetPose import SetTargetToCharacter, TargetPose

from ..pymotionlib.MotionData import MotionData
from ..pymotionlib import BVHLoader

from ..Utils.InverseDynamics import MotionInvDyn
from ..Utils.MothonSliceSmooth import MotionSliceSmooth
from ..Utils.SoftFootRetarget import generate_ref_motion  # actually, it is not used.
from ..Utils.Camera.CameraNumpy import CameraParamNumpy
from ..Utils.ComputeCom2d import Joint2dComIgnoreHandToe


class SamHlp:
    """
    Save path helper of samcon
    """
    result_save_name: str = "save_path.bin"
    main_dump_name: str = "main-worker.bin"
    log_dir_name: str = "log"
    folder: str = "Samcon"

    def __init__(self, conf_name: str, idx=0):
        self.conf_name = os.path.abspath(conf_name)
        self.conf: Dict[str, Any] = self.load_conf()
        self.idx: str = str(idx)

    def load_conf(self) -> Dict[str, Any]:
        return Helper.conf_loader(self.conf_name)

    def remove_samcon_results(self):
        """
        remove all of samcon results
        """
        folder = self.root_save_folder_dname()
        shutil.rmtree(folder)
        print(f"del all samcon results at {folder}")

    def create_dir(self, idx=None):
        """
        Create directory for save samcon result
        """
        idx = self.idx if idx is None else idx
        save_dir = self.save_folder_i_dname(idx)
        log_dir = self.log_path_dname(idx)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"create save_dir at {save_dir}")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"create log_dir at {log_dir}")

    @staticmethod
    def load_best_path(fname: str) -> List[Sample]:
        """
        load saved best path from file
        """
        with open(fname, "rb") as f_:
            best_path: List[Sample] = pickle.load(f_)
        return best_path

    def load_best_path_idx(self, idx=None):
        idx = self.idx if idx is None else idx
        return self.load_best_path(self.best_path_fname(idx))

    def root_save_folder_dname(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(self.conf_name), self.folder))

    def save_folder_i_dname(self, idx=None) -> str:
        idx = self.idx if idx is None else idx
        return os.path.join(self.root_save_folder_dname(), str(idx))

    def best_path_fname(self, idx=None) -> str:
        idx = self.idx if idx is None else idx
        return os.path.join(self.save_folder_i_dname(idx), self.result_save_name)

    def main_dump_fname(self, idx=None) -> str:
        """
        The main dump file
        """
        idx = self.idx if idx is None else idx
        return os.path.join(self.save_folder_i_dname(idx), self.main_dump_name)

    def log_path_dname(self, idx=None) -> str:
        idx = self.idx if idx is None else idx
        return os.path.join(self.save_folder_i_dname(idx), self.log_dir_name)

    def get_invdyn_bvh_fname(self):
        invdyn_fname = os.path.join(self.save_folder_i_dname(), "invdyn-output.bvh")
        return invdyn_fname

    def load_target(self, scene: ODEScene, character: ODECharacter):
        with tqdm(None, "Load Target Pose"):
            pass

        dup_conf: Optional[Dict[str, Any]] = self.conf["worker_cma"].get("duplicate_input")
        invdyn_fname = self.get_invdyn_bvh_fname()
        if dup_conf and dup_conf["in_use"]:
            res = self.load_dup_target(scene, character, invdyn_fname)
        else:
            res = self.load_single_target(scene, character, invdyn_fname)
        return res

    def _handle_target_motion(self, target_in, character: ODECharacter, to_bvh: CharacterTOBVH):
        if isinstance(target_in, MotionData):
            motion: MotionData = target_in
            return motion

        if isinstance(target_in, str):
            if target_in.endswith(".bvh"):
                motion: MotionData = BVHLoader.load(target_in)
                return motion
            if target_in.endswith(".bin") or target_in.endswith(".pickle"):
                with open(target_in, "rb") as fin:
                    target_in: List[BodyInfoState] = pickle.load(fin)

        if isinstance(target_in, list) and isinstance(target_in[0], BodyInfoState):
            for index, state in enumerate(target_in):
                character.load(state)
                to_bvh.append_no_root_to_buffer()
            motion: MotionData = to_bvh.to_file(None)
            return motion

        raise ValueError("Type of target_in should be str or List")

    def load_inv_dyn_from_pickle(
        self,
        scene: ODEScene,
        character: ODECharacter,
        fname: str,
        debug_print: bool = True,
        compute_samcon_target: bool = True
        ):
        """
        load pre computed target pose and inverse dynamics target from pickle file

        :param
        scene: ODEScene, for getting fps
        character: ODECharacter
        fname: pickle file name

        format of pickle file:
        {
            "camera_param": CameraParamNumpy,
            "input_motion": list of BodyInfoState / BVH Motion / filename as target pose,
            "invdyn_target": np.ndarray in shape (frame, joint, 4), inverse dynamics result locally,
            "confidence": np.ndarray,
            "pred_contact_label": np.ndarray
        }

        Process:
        1. convert the state list to bvh format
        2. compute samcon target pose
        3. convert the inverse dynamics result as bvh format
        4. compute inverse dynamics target pose

        return motion, samcon_target, invdyn_target, confidence, contact_label, camera
        """
        fps: int = int(scene.sim_fps)
        invdyn_bvh_fname: str = self.get_invdyn_bvh_fname()
        fname_dir: str = os.path.dirname(fname)
        with open(fname, "rb") as fin:
            result: Dict[str, Any] = pickle.load(fin)
        info_str = f"Samcon: Load target and inverse dynamics target from {fname}"
        if debug_print:
            with tqdm(None, info_str):
                pass

        to_bvh: CharacterTOBVH = CharacterTOBVH(character, fps)
        to_bvh.bvh_hierarchy_no_root()
        motion: MotionData = self._handle_target_motion(os.path.join(fname_dir, result["pred_motion"]), character, to_bvh)
        invdyn: np.ndarray = result["invdyn_target"]
        camera_param: Optional[CameraParamNumpy] = result.get("camera_param")
        unified_pos_2d: Optional[np.ndarray] = result.get("pos2d")
        confidence: Optional[np.ndarray] = result.get("confidence")
        contact_label: Optional[np.ndarray] = result.get("pred_contact_label")
        motion = motion.resample(fps)
        if fps == 50:
            tmp_frame = 2 * motion.num_frames - 1
            if invdyn is not None and invdyn.shape[0] >= tmp_frame:
                invdyn = np.ascontiguousarray(invdyn[::2])
            if unified_pos_2d is not None and unified_pos_2d.shape[0] >= tmp_frame:
                unified_pos_2d = np.ascontiguousarray(unified_pos_2d[::2])
            if confidence is not None and confidence.shape[0] >= tmp_frame:
                confidence = np.ascontiguousarray(confidence[::2])
            if contact_label is not None and contact_label.shape[0] >= tmp_frame:
                contact_label = np.ascontiguousarray(contact_label[::2])

        # smooth the inverse dynamics by Gaussian / butter-worth filter.
        # it seems if the network output is not good, smooth filter will not work.
        # invdyn_vec6d: np.ndarray = MathHelper.quat_to_vec6d(invdyn)
        # smooth_invdyn_vec6d: np.ndarray = smooth_operator(invdyn_vec6d, GaussianBase(5))
        # invdyn: np.ndarray = MathHelper.vec6d_to_quat(smooth_invdyn_vec6d)

        # 2.
        # TODO: support multiple target pose format..
        if compute_samcon_target:
            samcon_target: SamconTargetPose = SamconTargetPose.load2(motion, character, fps)
            if camera_param is not None:
                joint_pos_3d: np.ndarray = samcon_target.pose.all_joint_global.pos
                camera_pos_3d: np.ndarray = camera_param.world_to_camera(joint_pos_3d)
                camera_pos_2d: np.ndarray = camera_param.project_to_2d_linear(camera_pos_3d)
                samcon_target.pose2d = TargetPose2d()
                samcon_target.pose2d.pos2d = camera_pos_2d
                # here we should also compute center of mass in 2d coordinate..
                com_calc = Joint2dComIgnoreHandToe().build(character)
                samcon_target.pose2d.com_2d = com_calc.calc(camera_pos_2d)
            else:
                raise ValueError("Camera parameter is required.")

            samcon_target.pose2d_unified = TargetPose2d()
            samcon_target.pose2d_unified.pos2d = unified_pos_2d
            samcon_target.pose2d_unified.confidence = confidence
            # maybe we can compute 2d com loss by unified joints.
            # however, I don't think 2d com loss is necessary, I think 2d joint projection loss is enough..

            samcon_target.camera_param = camera_param
        else:
            samcon_target = None

        # 4.
        invdyn_target: TargetPose = TargetPose()
        invdyn_target.locally.quat = invdyn
        invdyn_target.num_frames = invdyn.shape[0]

        invdyn_motion: MotionData = to_bvh.forward_kinematics(motion.joint_position[:, 0, :], motion.joint_rotation[:, 0, :], invdyn)
        invdyn_motion: MotionData = to_bvh.insert_end_site(invdyn_motion)
        # save inverse dynamics result as bvh format
        BVHLoader.save(invdyn_motion, invdyn_bvh_fname)

        return motion, samcon_target, invdyn_target, confidence, contact_label, camera_param

    def load_inv_dyn_from_bvh(self):
        """
        load precomputed inverse dynamics in bvh format
        """
        pass

    def load_single_target(
        self,
        scene: ODEScene,
        character: ODECharacter,
        invdyn_fname: Optional[str] = None
    ):
        """
        Load target
        """
        logging.info("Load single target")
        sim_fps = int(scene.sim_fps)
        motion_raw: Union[str, MotionData] = self.conf["filename"]["bvh"]
        if isinstance(motion_raw, str):
            motion_raw: MotionData = BVHLoader.load(motion_raw)
        elif isinstance(motion_raw, MotionData):
            pass
        else:
            raise ValueError

        joint_name_raw: List[str] = copy.deepcopy(motion_raw.joint_names)
        use_soft_character = self.conf["character"]["use_soft_character"]
        curr_joint_name: List[str] = character.joint_info.joint_names()
        if use_soft_character:
            _, motion = generate_ref_motion(motion_raw, character)
        else:
            motion = motion_raw

        # clip the input motion...
        motion_start, motion_end = self.conf["bvh"]["start"], self.conf["bvh"]["end"]
        motion: MotionData = motion.sub_sequence(motion_start, motion_end)
        target, loader = SamconTargetPose.load2(motion, character, sim_fps, return_loader=True)

        inv_dyn_target: Optional[TargetPose] = None
        if self.conf["inverse_dynamics"]["in_use"]:
            invdyn = MotionInvDyn.builder(scene, character, self.conf, motion, invdyn_fname)
            inv_dyn_target = invdyn.calc(loader.bvh, loader.character_to_bvh)

            # actually, this part is NOT used in current version of character
            if use_soft_character:  # remove joint rotations for soft joints on foot
                untrack_names = list(set(curr_joint_name) - set(joint_name_raw))
                untrack_index = [curr_joint_name.index(node) for node in untrack_names]
                untrack_index.sort()
                inv_dyn_target.locally.quat[:, untrack_index, :] = MathHelper.unit_quat_arr((1, 1, 4))
                inv_dyn_target.locally.quat = np.ascontiguousarray(inv_dyn_target.locally.quat)
            logging.info("After running inverse dynamics")

        target: SamconTargetPose = target.sub_seq(*MotionInvDyn.ref_start_end())
        target.pose.dup_root_pos = target.pose.root.pos
        ret_mocap: MotionData = motion.sub_sequence(*MotionInvDyn.ref_start_end(), copy=True)
        return ret_mocap, target, inv_dyn_target

    def load_dup_target(self,
                        scene: ODEScene,
                        character: ODECharacter,
                        invdyn_fname: Optional[str] = None
                        ) -> Tuple[MotionData, SamconTargetPose, Optional[TargetPose], Optional[np.ndarray]]:
        # TODO: start and end are not checked.
        logging.info("Load smoothed target for duplicate")
        sim_fps: int = int(scene.sim_fps)
        motion: MotionData = BVHLoader.load(self.conf["filename"]["bvh"])
        use_inv_dyn: bool = bool(self.conf["inverse_dynamics"]["in_use"])
        r_start, r_end = MotionInvDyn.ref_start(), abs(MotionInvDyn.ref_end())
        smoother: MotionSliceSmooth = MotionSliceSmooth.build_from_conf(
            motion, self.conf["worker_cma"]["duplicate_input"]
        )
        sub_mocap, smooth_sub_mocap, pos_off_mix, rot_off_mix = smoother.calc(
            {"ref_start": r_start, "ref_end": r_end}
        )
        mocap: MotionData = smooth_sub_mocap if smooth_sub_mocap is not None else sub_mocap
        target, loader = SamconTargetPose.load2(
            mocap.sub_sequence(r_start, -r_end, True), character, sim_fps, return_loader=True
        )

        target.pose.set_dup_off_mix(pos_off_mix, rot_off_mix)
        # for compute duplicated root position and quaternion

        inv_dyn_target: Optional[TargetPose] = None
        if use_inv_dyn:
            invdyn = MotionInvDyn.builder(scene, character, self.conf, mocap, output_bvh_fname=invdyn_fname)
            inv_dyn_target = invdyn.calc(mocap, loader.character_to_bvh)

        ret_mocap: MotionData = mocap.sub_sequence(r_start, -r_end, copy=True)
        return ret_mocap, target, inv_dyn_target

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

from argparse import Namespace
import datetime
from matplotlib import animation
from mpi4py import MPI
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation
import subprocess
import time
import torch
from typing import Optional, List, Union, Dict, Tuple, Any
from tqdm import tqdm

from .Common import visualize_motion
from ...Common.Helper import Helper
from ...Common.MathHelper import MathHelper, RotateType
from ...ODESim.Saver.CharacterToBVH import CharacterTOBVH
from ...ODESim.ODEScene import ODEScene
from ...ODESim.ODECharacter import ODECharacter
from ...ODESim.TargetPose import TargetPose
from ...ODESim.BVHToTarget import BVHToTargetBase
from ...ODESim.BodyInfoState import BodyInfoState
from ...Render.Renderer import RenderWorld
from ...pymotionlib.MotionData import MotionData
from ...pymotionlib import BVHLoader
from ...Utils.Dataset.UnifiedUtils import unified_bones_def
from ...Utils.Dataset.StdHuman import stdhuman_to_unified
from ...Utils.Camera.CameraNumpy import CameraParamNumpy
from ...Utils.Camera.CameraPyTorch import CameraParamTorch
from ...Utils.Camera.Human36CameraBuild import CameraParamBuilder
from ...Utils.MotionProjection import convert_rotation_to_camera
from ...Utils.CharacterStateExtractor import concat_body_info_state, mirror_character_data_no_velo


comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda")


class SamconDataLoader:
    """
    TODO: how can we get the data async..?
    """
    def __init__(self, scene: ODEScene, character: ODECharacter, args: Namespace) -> None:
        self.scene: ODEScene = scene
        self.character: ODECharacter = character
        # self.to_bvh: CharacterTOBVH = CharacterTOBVH(self.character, self.scene.sim_fps)
        # self.to_bvh.bvh_hierarchy_no_root()

        self.args: Namespace = args
        self.param: Namespace = args

        self.device: torch.device = args.device
        self.rotate_type: RotateType = args.rotate_type
        self.batch_size: int = args.batch_size
        self.result_buffer: List[Dict[str, np.ndarray]] = []
        self.camera_numpy: CameraParamNumpy = CameraParamBuilder.build(np.float64)
        self.camera_torch: CameraParamTorch = CameraParamTorch.build_dict_from_numpy(self.camera_numpy, torch.float32)

        self.mirror_body_index: List[int] = self.character.body_info.get_mirror_index()
        self.mirror_joint_index: List[int] = self.character.joint_info.get_mirror_index()

        self.mean_pos2d: Optional[torch.Tensor] = None  # for normalize the input 2d data
        self.std_pos2d: Optional[torch.Tensor] = None  # for normalize the input 2d data

        self.mean_pos3d: Optional[torch.Tensor] = None
        self.std_pos3d: Optional[torch.Tensor] = None

        # self.render = RenderWorld(self.scene)
        # self.render.start()

    def prepare_one_piece_base(
        self,
        body_pos_list: np.ndarray,  # (num frame, body, 3)
        body_quat_list: np.ndarray,  # (num frame, body, 4)
        pd_target_list: np.ndarray,  # (num frame, joint, ?)
        contact_label: Optional[np.ndarray],
        data_label: str,  # only use human3.6 data here..
        position_offset: Optional[np.ndarray] = None,
        rotation_offset: Optional[Rotation] = None,
        sub_slice: Optional[slice] = None,
        debug_output: bool = False
    ):
        """
        """
        assert not self.character.joint_info.has_root  # only tested on std-human..
        assert isinstance(data_label, str)  # for debug..

        if sub_slice is not None:
            body_pos_list = body_pos_list[sub_slice]
            body_quat_list = body_quat_list[sub_slice]
            pd_target_list = pd_target_list[sub_slice]
            if contact_label is not None:
                contact_label = contact_label[sub_slice]

        if position_offset is not None and rotation_offset is not None:  # modify the input data
            # modify position
            init_root_pos: np.ndarray = MathHelper.vec_axis_to_zero(body_pos_list[0, 0, :], 1)
            body_pos_list: np.ndarray = body_pos_list - init_root_pos
            # rotate the position. as the rotation offset is along y axis, y component of position will not changed
            body_pos_list: np.ndarray = rotation_offset.apply(body_pos_list.reshape((-1, 3))).reshape(body_pos_list.shape)
            # assume y component of position offset is 0
            body_pos_list: np.ndarray = body_pos_list + position_offset[None, None, :] + init_root_pos

            # modify quaternion
            body_quat_list: np.ndarray = (rotation_offset * Rotation(body_quat_list.reshape((-1, 4)))).as_quat().reshape(body_quat_list.shape)

            # modify linear velocity
            # body_vel_list: np.ndarray = rotation_offset.apply(body_vel_list.reshape((-1, 3))).reshape(body_vel_list.shape)

            # modify angular velocity
            # body_omega_list: np.ndarray = rotation_offset.apply(body_omega_list.reshape((-1, 3))).reshape(body_omega_list.shape)

            # for test, we should test integrate for position and linear velocity..
            # that is, x_{t+1} = x_{t} + h * v_{t + 1}
            # if False:  # debug flag
            #    delta_x = body_pos_list[1] - (body_pos_list[0] + 0.01 * body_vel_list[1])
            #    print(np.max(np.abs(delta_x)))
        def debug_vis():
            num_frame = body_pos_list.shape[0]
            for frame in range(num_frame):
                self.character.set_body_pos(body_pos_list[frame])
                self.character.body_info.set_body_quat(body_quat_list[frame])
                time.sleep(0.01)
            time.sleep(0.1)
            # input()
        # debug_vis()

        result: Dict = {}
        num_joint: int = len(self.character.joints)

        parent_index: np.ndarray = self.character.joint_info.parent_body_index
        child_index: np.ndarray = self.character.joint_info.child_body_index

        parent_body_quat: np.ndarray = np.ascontiguousarray(body_quat_list[:, parent_index, :])  # (frame - 1, joint, 4)
        child_body_quat: np.ndarray = np.ascontiguousarray(body_quat_list[:, child_index, :])  # (frame - 1, joint, 4)
        joint_local_rotation: Rotation = Rotation(parent_body_quat.reshape((-1, 4))).inv() * Rotation(child_body_quat.reshape((-1, 4)))  # ((frame - 1) * joint, 4)
        joint_local_quat: np.ndarray = joint_local_rotation.as_quat().reshape((-1, num_joint, 4))

        # for debug
        # the pd target of samcon is noisy on time series
        # and I don't think the result is continuous...
        # velocity is more noisy than position
        # and accleration may change suddenly
        if False:
            # pd_target_motion = visualize_motion(self.to_bvh, body_pos_list[:, 0, :], body_quat_list[:, 0, :], pd_target_list)
            sim_motion = visualize_motion(self.to_bvh, body_pos_list[:, 0, :], body_quat_list[:, 0, :], joint_local_rotation.as_quat().reshape((-1, num_joint, 4)))
            # pd_target_fname = "test-pd-target.bvh"
            sim_motion_fname = "test-sim-motion.bvh"
            # BVHLoader.save(pd_target_motion, pd_target_fname)
            BVHLoader.save(sim_motion, sim_motion_fname)
            subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", sim_motion_fname])

        if False:
            pd_err_motion = self.visualize_motion(body_pos_list[:, 0, :], body_quat_list[:, 0, :], pd_err, "test-pd-err.bvh")
            exit(0)

        # result["joint_local"] = MathHelper.quat_to_other_rotate(joint_local_quat, self.rotate_type)
        if pd_target_list is not None:
            result["pd_target"] = MathHelper.quat_to_other_rotate(pd_target_list, self.rotate_type).astype(np.float32)

        if contact_label is not None:
            result["contact_label"] = contact_label.astype(np.float32)

        # get joint global position..
        raw_anchor_1: np.ndarray = self.scene.world.getBallAndHingeRawAnchor1(self.character.joint_info.joint_c_id)  # (joint, 3)
        raw_anchor_1_dup: np.ndarray = np.tile(raw_anchor_1, (body_pos_list.shape[0], 1, 1))  # (frame, joint, 3)
        child_body_pos: np.ndarray = body_pos_list[:, child_index, :]  # (frame, joint, 3)
        joint_pos: np.ndarray = Rotation(child_body_quat.reshape((-1, 4))).apply(raw_anchor_1_dup.reshape((-1, 3)))
        joint_pos: np.ndarray = joint_pos.reshape((body_pos_list.shape[0], num_joint, 3)) + child_body_pos
        # here we should also concat position of root body
        # joint_pos: np.ndarray = np.concatenate([body_pos_list[:, 0:1, :], joint_pos], axis=1)  # also with root joint..
        unified_joint_pos: np.ndarray = stdhuman_to_unified(joint_pos)

        # project joint position to 2d..
        unified_camera_pos_3d, unified_joint_pos_2d, camera_root_pos3d, camera_root_quat = self.project_joint_to_2d(
            body_quat_list[:, 0, :], joint_local_quat, body_pos_list[:, 0, :], unified_joint_pos, data_label)
        del unified_camera_pos_3d

        joint_pos_with_root: np.ndarray = np.concatenate([body_pos_list[:, 0:1, :], joint_pos], axis=1)
        camera_joint_pos3d: np.ndarray = np.concatenate(
            [camera.world_to_camera(joint_pos_with_root)[None, ...] for camera in self.camera_numpy[data_label]], axis=0)

        # for debug, we should check if 2d joint location is inside the camera space...
        camera_root_rot: np.ndarray = MathHelper.quat_to_other_rotate(camera_root_quat, self.rotate_type)

        # if False:
        #    # visualize joint pos 2d..
        #    # check joint pos is same as compute with ode..
        #    for index, body_state in enumerate(input_data):
        #        self.character.load(body_state)
        #        test_joint_anchor = self.joint_info.get_global_anchor1()
        #        assert np.max(np.abs(test_joint_anchor - joint_pos[index])) < 1e-4

        # TODO: here we can also extract contact info by collision detection..
        result.update({
            "data_label": data_label,
            "camera_param": self.camera_numpy[data_label],
            "joint_pos_2d": unified_joint_pos_2d.astype(np.float32),
            "camera_root_pos3d": camera_root_pos3d.astype(np.float32),
            "camera_root_rot": camera_root_rot.astype(np.float32),
            "joint_local_rot": MathHelper.quat_to_other_rotate(joint_local_quat, self.rotate_type).astype(np.float32),
            "root_pos3d": body_pos_list[:, 0, :].astype(np.float32),
            "root_quat": body_quat_list[:, 0, :].astype(np.float32),
            "camera_joint_pos3d": camera_joint_pos3d.astype(np.float32),
        })

        if debug_output and comm_size == 1:
            print(f"After perpare data. total piece = {body_pos_list.shape[0]}")

        return result

    def project_joint_to_2d(
        self,
        root_global_quat: np.ndarray,
        joint_local_quat: np.ndarray,
        root_global_pos: np.ndarray,
        joint_pos: np.ndarray,
        data_label: str = "S1"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :param joint_pos: in shape (*, joints, 3)
        :param data_label:

        :return project 2d joint position: in shape (num camera, *, joints, 2)
        """
        assert root_global_quat.shape[-1] == 4
        assert joint_local_quat.shape[-1] == 4
        assert root_global_pos.shape[-1] == 3
        assert joint_pos.shape[-1] == 3
        camera_list: List[CameraParamNumpy] = self.camera_numpy[data_label]
        joint_2d_pos_list: List[np.ndarray] = []
        camera_root_pos_list: List[np.ndarray] = []
        camera_3d_pos_list: List[np.ndarray] = []
        camera_root_quat_list: List[np.ndarray] = []
        for camera_index, camera_param in enumerate(camera_list):
            camera_root_quat: np.ndarray = convert_rotation_to_camera(root_global_quat, camera_param)
            camera_root_pos_list.append(camera_param.world_to_camera(root_global_pos)[None, ...])
            camera_pos3d: np.ndarray = camera_param.world_to_camera(joint_pos)
            camera_3d_pos_list.append(camera_pos3d[None, ...])
            pos2d: np.ndarray = camera_param.project_to_2d_linear(camera_pos3d)  # in shape (frame, num joint, 2)

            joint_2d_pos_list.append(pos2d[None, ...])
            camera_root_quat_list.append(camera_root_quat[None, ...])

            # for debug..
            # we should visualize here...
            if False:
                gt_motion = self.to_bvh.forward_kinematics(root_global_pos, root_global_quat, joint_local_quat)
                BVHLoader.save(gt_motion, "gt-test.bvh")
                test_root_quat = (Rotation.from_rotvec([0.0, 0.0, np.pi]) * Rotation(camera_root_quat)).as_quat()
                test_root_pos = camera_param.camera_to_world(camera_root_pos_list[camera_index][0])
                # test_root_pos[..., 2] = camera_root_pos_list[camera_index][0][..., 1]
                camera_motion = self.to_bvh.forward_kinematics(test_root_pos, test_root_quat, joint_local_quat)
                BVHLoader.save(camera_motion, "camera-test.bvh")
                print(np.max(camera_root_pos_list[camera_index][0, :, 1]))
                subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", "camera-test.bvh", "gt-test.bvh"])

        return (np.concatenate(camera_3d_pos_list, axis=0),
            np.concatenate(joint_2d_pos_list, axis=0),
            np.concatenate(camera_root_pos_list, axis=0),
            np.concatenate(camera_root_quat_list, axis=0)
        )

    def add_noise_on_2d(
        self,
        joint_pos2d: np.ndarray,
        dtype = np.float32,
        debug_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        return joint_pos2d, confidences
        """
        if debug_output:
            print(f"Add noise on 2d data")

        noise_radius: Optional[float] = self.args.noise2d
        if noise_radius is not None:
            noises_x: np.ndarray = np.random.normal(-noise_radius / 6, noise_radius / 6, size=joint_pos2d.shape[:-1])
            noises_y: np.ndarray = np.random.normal(-noise_radius / 6, noise_radius / 6, size=joint_pos2d.shape[:-1])
            # add noise to joint pos 2d
            joint_pos2d[..., 0] += noises_x
            joint_pos2d[..., 1] += noises_y

            confidences_x: np.ndarray = 1 - np.abs(noises_x) / noise_radius
            confidences_y: np.ndarray = 1 - np.abs(noises_y) / noise_radius
            confidences: np.ndarray = 0.5 * (confidences_x + confidences_y)
            confidences[confidences < 0] = 0
        else:
            confidences: np.ndarray = np.ones(joint_pos2d.shape[:-1])

        return joint_pos2d.astype(dtype), confidences.astype(dtype)

    def prepare_bvh_one_piece_base(self, motion: Union[str, MotionData], data_label = "S1") -> Dict[str, Any]:
        """
        Only for test..
        """
        if isinstance(motion, str):
            motion: MotionData = BVHLoader.load(motion)
        # for test, use 0-th camera in S1 of human 3.6
        target: TargetPose = BVHToTargetBase(motion, self.scene.sim_fps, self.character).init_target()
        joint_pos: np.ndarray = target.globally.pos
        unified_joint_pos: np.ndarray = stdhuman_to_unified(joint_pos)

        _, unified_joint_pos_2d, _, _ = self.project_joint_to_2d(
            target.root.quat,
            target.locally.quat,
            target.root.pos,
            unified_joint_pos,
            data_label
        )

        confidence: Optional[np.ndarray] = None
        if self.args.noise2d is not None:
            unified_joint_pos_2d, confidence = self.add_noise_on_2d(unified_joint_pos_2d)

        result = {
            "data_label": data_label,
            "joint_pos_2d": unified_joint_pos_2d,
            "camera_param": self.camera_numpy[data_label]
        }
        if confidence is not None:
            result["confidences"] = confidence.astype(np.float32)

        return result

    def prepare_one_piece(
        self,
        input_data_full: str,
        data_label: str = "S1",
        use_mirror: bool = True,
        data_aug_count: int = 4,
        pbar: Optional[tqdm] = None,
    ) -> List:
        """
        As motion in lafan dataset may runs outside of camera,
        we need to divide the motion into several pieces,
        and put them in different places.

        we can compute position and rotation offset on lafan dataset.
        for body and joint position, we can subtract and rotate directly
        for velocity and angular velocity, we can rotate directly
        for global body and joint rotation, we can rotate directly
        for control torque, they will not be changed (as they are in local space)
        for contact label, they will not be changed (as they are 0-1 variables)

        As Human 3.6 data will always locate inside camera 2d space,
        we need not no divide them into different places.
        """
        if comm_size == 1:
            pbar_desc = f"prepare file {os.path.split(input_data_full)[1]}"
            if pbar is not None:
                pbar.set_description(pbar_desc)
            else:
                print(pbar_desc, end="")

        with open(input_data_full, "rb") as fin:
            input_data_full: List[BodyInfoState] = pickle.load(fin)
        def _process_data():
            concat_info_ = concat_body_info_state(self.scene, self.character, input_data_full, self.args.also_output_contact_label)
            mirror_concat_info_ = mirror_character_data_no_velo(*concat_info_, self.mirror_body_index, self.mirror_joint_index) if use_mirror else None
            concat_arr_ = [concat_info_] if not use_mirror else [concat_info_, mirror_concat_info_]
            return concat_arr_

        concat_arr = _process_data()

        def debug_vis(_joint_pos_2d: np.ndarray):
            # I think we should also render line, and joint index here..
            import matplotlib.pyplot as plt
            fig = plt.figure()
            subs = [fig.add_subplot(221 + i) for i in range(4)]
            point_plot_list = []
            line_plot_list = [[], [], [], []]
            test_plot_list: List[List] = [[], [], [], []]
            def _set_range(_i: int, _frame: int):
                x_min = _joint_pos_2d[i, _frame, :, 0].min()
                y_min = _joint_pos_2d[i, _frame, :, 1].min()
                width_x = _joint_pos_2d[i, _frame, :, 0].max() - x_min
                width_y = _joint_pos_2d[i, _frame, :, 1].max() - y_min
                width = max(width_x, width_y)
                subs[_i].set_xlim(x_min, x_min + width)
                subs[_i].set_ylim(y_min, y_min + width)
                subs[_i].set_aspect('equal')
            for i in range(4):
                # _set_range(i, 0)
                subs[i].set_aspect('equal')
                point_plot_list.append(subs[i].plot(_joint_pos_2d[i, 0, :, 0], _joint_pos_2d[i, 0, :, 1], 'o')[0])
                for j in range(12):
                    pa, ch = unified_bones_def[j]
                    line_plot_list[i].append(subs[i].plot(_joint_pos_2d[i, 0, [pa, ch], 0], _joint_pos_2d[i, 0, [pa, ch], 1])[0])
                for j in range(13):
                    test_plot_list[i].append(subs[i].text(_joint_pos_2d[i, 0, j, 0], _joint_pos_2d[i, 0, j, 1], str(j)))
                subs[i].set_title(f"camera {i}")

            def _update_func(_frame: int):
                for i in range(4):
                    point_plot_list[i].set_data(_joint_pos_2d[i, _frame, :, 0], _joint_pos_2d[i, _frame, :, 1])
                    for j in range(12):
                        pa, ch = unified_bones_def[j]
                        line_plot_list[i][j].set_data(_joint_pos_2d[i, _frame, [pa, ch], 0], _joint_pos_2d[i, _frame, [pa, ch], 1])
                    # for j in range(13):
                    #    test_plot_list[i][j].set_position(_joint_pos_2d[i, _frame, j, :])
                    # _set_range(i, _frame)
                plt.suptitle(f"{_frame}")

            anim = animation.FuncAnimation(fig, _update_func, _joint_pos_2d.shape[1])
            plt.show()

        def judge_aug_success(_joint_pos_2d: np.ndarray) -> bool:
            # debug_vis(_joint_pos_2d)
            return _joint_pos_2d.min() >= -1 and _joint_pos_2d.max() <= 1

        ret_list = []
        if data_aug_count > 0:  # divide into several pieces..
            input_len = len(input_data_full)
            divide_width: int = 150
            x_bound, z_bound = np.array([-0.5, 0.5]), np.array([-0.5, 0.5])
            start_index: np.ndarray = np.arange(0, input_len, divide_width)
            if input_len - start_index[-1] < divide_width // 5:
                start_index = start_index[:-1]
            end_index: np.ndarray = start_index[1:] + self.args.frame_window
            start_index = start_index[:-1]
            end_index[-1] = input_len

            # generate random start position
            # maybe we can randomly put at scene for several times for data augmentation..
            # here we should also use mirror data
            # maybe here we can also use np.vectorize..?
            for sub_index, (start, end) in enumerate(zip(start_index, end_index)):
                if pbar is not None:
                    pbar.set_description(f"{pbar_desc} at {sub_index}")

                for concat_node in concat_arr:
                    root_pos: np.ndarray = concat_node[0][start, 0, :].copy()
                    sub_slice = slice(start, end)
                    # first, test original position and rotation of bvh mocap data.
                    result = self.prepare_one_piece_base(*concat_node, data_label, None, None, sub_slice)
                    if judge_aug_success(result["joint_pos_2d"]):
                        result["joint_pos_2d"], result["confidences"] = self.add_noise_on_2d(result["joint_pos_2d"])
                        ret_list.append(result)
                    else:
                        if pbar is not None:
                            pbar.set_description(f"{pbar_desc} at {sub_index}, init pos out bound.")

                    for aug_index in range(data_aug_count):
                        # for data augmentation, we should consider that
                        # the character is inside the camera..
                        aug_success = False
                        aug_fail_count = 0
                        while not aug_success and aug_fail_count < 40:
                            rand_angle = np.random.uniform(-np.pi, np.pi)
                            rotation_offset: Rotation = Rotation.from_rotvec(np.array([0.0, rand_angle, 0.0]))
                            new_pos_x = np.random.uniform(x_bound[0], x_bound[1])
                            new_pos_z = np.random.uniform(z_bound[0], z_bound[1])
                            new_pos: np.ndarray = np.array([new_pos_x, root_pos[1], new_pos_z])
                            position_offset: np.ndarray = new_pos - root_pos
                            result = self.prepare_one_piece_base(*concat_node, data_label, position_offset, rotation_offset, sub_slice)
                            # here maybe we can add noise on 2d joints..
                            aug_success = judge_aug_success(result["joint_pos_2d"])
                            if not aug_success:
                                aug_fail_count += 1
                        if not aug_success:
                            if pbar is not None:
                                pbar.set_description(f"{pbar_desc} failed for random")
                            continue
                        # print(f"generate after {aug_fail_count} trials")

                        if self.args.noise2d is not None:
                            result["joint_pos_2d"], result["confidences"] = self.add_noise_on_2d(result["joint_pos_2d"])

                        ret_list.append(result)  # do we need to consider confidence as loss weight ..?
        else:
            ret_list.extend([self.prepare_one_piece_base(*node, data_label) for node in concat_arr])

        if pbar is None and comm_size == 1:
            print(f"total length = {len(input_data_full)}, total frame = {sum([node['joint_pos_2d'].shape[1] for node in ret_list])}", flush=True)
        return ret_list

    def normalize_result_buffer(self):
        print(f"normalize the output position")
        cat_pos3d: np.ndarray = np.concatenate([node["camera_root_pos3d"].reshape((-1, 3)) for node in self.result_buffer], axis=0, dtype=np.float32)
        mean_pos3d: np.ndarray = np.mean(cat_pos3d, axis=0, keepdims=True)
        std_pos3d: np.ndarray = np.max(np.abs(cat_pos3d - mean_pos3d), axis=0, keepdims=True)
        del cat_pos3d
        for node in self.result_buffer:
            node["camera_root_pos3d"] = (node["camera_root_pos3d"] - mean_pos3d) / std_pos3d
        self.mean_pos3d: torch.Tensor = torch.as_tensor(mean_pos3d, dtype=torch.float32, device=self.device)
        self.std_pos3d: torch.Tensor = torch.as_tensor(std_pos3d, dtype=torch.float32, device=self.device)

    def get_data_fname_list(self, input_dir: str) -> List[str]:
        file_name_list: List[str] = []
        for name in os.listdir(input_dir):
            if not name.endswith(".pickle"):
                continue
            if name[:2] not in self.camera_numpy.keys() and self.args.only_use_human36_data:
                print(f"only use human3.6m data. ignore {name}")
                continue
            file_name_list.append(os.path.join(input_dir, name))

        return file_name_list

    def prepare_data(self, input_param: str, save_as_npz: bool = False):
        # load samcon results
        # the dumped best path should contains the json configuration
        starttime = datetime.datetime.now()
        if isinstance(input_param, str):
            input_dir: str = input_param
            precompute_fname: str = os.path.join(input_dir, "precompute_train.binary.npz")
            if os.path.exists(precompute_fname):
                # make sure result is same for training..
                pre_calc_result = np.load(precompute_fname, allow_pickle=True)
                self.result_buffer: List[Dict[str, np.ndarray]] = pre_calc_result["result_buffer"].tolist()
                self.mean_pos3d: Optional[torch.Tensor] = torch.as_tensor(pre_calc_result["mean_pos3d"], dtype=torch.float32, device=self.device)
                self.std_pos3d: Optional[torch.Tensor] = torch.as_tensor(pre_calc_result["std_pos3d"], dtype=torch.float32, device=self.device)

                print(f"load the pre compute result from file {precompute_fname}")
            else:
                self.result_buffer: List[Dict[str, np.ndarray]] = []
                file_name_list: List[str] = self.get_data_fname_list(input_dir)
                def load_func(file_name_ : str, pbar: tqdm):
                    data_label = os.path.split(file_name_)[1][:2]
                    if data_label not in self.camera_numpy.keys():
                        data_label = "S1"
                    return self.prepare_one_piece(file_name_, data_label, self.param.use_mirror_data, pbar=pbar)

                # vec_load_func = np.vectorize(load_func)
                # result_buffer = vec_load_func(file_name_list)
                pbar: tqdm = tqdm(total=len(file_name_list))
                for bvh_findex in range(len(file_name_list)):
                    self.result_buffer.append(load_func(file_name_list[bvh_findex], pbar))
                    pbar.update(1)
                pbar.close()

                self.result_buffer = sum(self.result_buffer, [])
                total_frame = sum(node['joint_pos_2d'].shape[1] for node in self.result_buffer)
                print(f"Total frame = {total_frame}, 4 * frame = {total_frame * 4}")

                # here we should normalize the input data for training..
                # we need not no normalize the output data here..
                # actually, if the input is in [-1, 1], we need not to do normalize here
                if self.args.normalize_input_data and False:
                    print(f"normalize the input data")
                    cat_func = lambda x: x.reshape(x.shape[0] * x.shape[1], -1)
                    cat_pos2d: np.ndarray = np.concatenate([cat_func(node["joint_pos_2d"]) for node in self.result_buffer], axis=0)
                    mean_pos2d: np.ndarray = np.mean(cat_pos2d, axis=0)
                    std_pos2d: np.ndarray = np.std(cat_pos2d, axis=0)
                    self.mean_pos2d: torch.Tensor = torch.as_tensor(mean_pos2d, dtype=torch.float32, device=self.device)
                    self.std_pos2d: torch.Tensor = torch.as_tensor(std_pos2d, dtype=torch.float32, device=self.device)
                    del cat_pos2d

                if self.args.normalize_output_pos and self.args.mode == "train":
                    self.normalize_result_buffer()

                if save_as_npz:
                    print(f"Begin save the result to file..")
                    np.savez_compressed(file=precompute_fname,
                        result_buffer=self.result_buffer, mean_pos3d=self.mean_pos3d.cpu().numpy(),
                        std_pos3d=self.std_pos3d.cpu().numpy())

                    print(f"save the processed training data to {precompute_fname}")

        else:
            raise ValueError
        Helper.print_total_time(starttime)

    def extract_data_xy_base(
        self,
        result_node: Dict[str, Any],
        tmp_buffer_x: Optional[List[np.ndarray]] = None,
        tmp_buffer_y: Optional[List[np.ndarray]] = None,
        tmp_buffer_cam_pos_3d: Optional[List[np.ndarray]] = None,
        tmp_buffer_confidence: Optional[List[np.ndarray]] = None,
        tmp_buffer_camera_f: Optional[List[np.ndarray]] = None,
        tmp_buffer_camera_c: Optional[List[np.ndarray]] = None,
        tmp_buffer_camera_trans: Optional[List[np.ndarray]] = None,
        tmp_buffer_camera_rot: Optional[List[np.ndarray]] = None,
        tmp_buffer_divide_index: Optional[List[np.ndarray]] = None,
        remain_data: Optional[int] = None
    ):
        frame_window: int = self.args.frame_window
        sim_fps: int = int(self.scene.sim_fps)
        assert sim_fps % self.param.fps_2d == 0
        ratio: int = sim_fps // self.param.fps_2d

        def cat_func(arr: np.ndarray):
            result = np.concatenate([arr[i:-ratio*frame_window+i, None] for i in range(0, ratio * frame_window, ratio)], axis=1)
            return result

        joint_pos_2d: np.ndarray = result_node["joint_pos_2d"]  # (num camera, frame, num joint, 2)
        num_camera, num_frame = joint_pos_2d.shape[:2]
        pd_target: Optional[np.ndarray] = result_node.get("pd_target")  # (frame, joint, 3, 2)
        camera_root_rot: Optional[np.ndarray] = result_node.get("camera_root_rot")  # (num camera, frame, 3, 2)
        joint_local_rot: Optional[np.ndarray] = result_node.get("joint_local_rot")  # (frame, num joint, 3, 2)
        camera_root_pos_3d: Optional[np.ndarray] = result_node.get("camera_root_pos3d")  # (num camera, frame, 3)
        cam_list: List[CameraParamNumpy] = result_node["camera_param"]
        for camera_index in range(joint_pos_2d.shape[0]):
            cat_data_x: np.ndarray = joint_pos_2d[camera_index]
            cat_data_y: List[np.ndarray] = []
            if self.args.also_output_pd_target:  # note: here we should extract the middle index of output..
                cat_data_y.append(pd_target.reshape((num_frame, -1)))
            if self.args.also_output_local_rot:
                # note: we should normalize the camera root position at training process..
                cat_data_y.extend([
                    camera_root_rot[camera_index].reshape((num_frame, -1)),
                    joint_local_rot.reshape((num_frame, -1)),
                    camera_root_pos_3d[camera_index].reshape((num_frame, -1))
                ])
            if self.args.also_output_contact_label:
                cat_data_y.append(result_node["contact_label"])
            if len(cat_data_y) > 0:
                cat_data_y: np.ndarray = np.concatenate(cat_data_y, axis=-1)
            else:
                raise ValueError

            cat_data_x: np.ndarray = cat_func(cat_data_x)
            # we need not shuffle here..
            rand_index = slice(None)

            cat_data_x: np.ndarray = np.ascontiguousarray(cat_data_x[rand_index]).reshape(cat_data_x.shape[0], -1)
            cat_frame: int = cat_data_x.shape[0]
            # extract middle index here..
            cat_data_y: np.ndarray = cat_data_y[frame_window:-frame_window]
            cat_data_y: np.ndarray = np.ascontiguousarray(cat_data_y[rand_index])

            # here we should extract 3d joint position in camera space.
            # I think maybe memory is not enough when preparing with large dataset..
            cat_pos_3d: np.ndarray = np.ascontiguousarray(result_node["camera_joint_pos3d"][camera_index, frame_window:-frame_window][rand_index])
            if "confidences" in result_node:
                cat_confidence: np.ndarray = result_node["confidences"][camera_index, frame_window:-frame_window][rand_index]
            else:
                cat_confidence: Optional[np.ndarray] = None

            cam_param = cam_list[camera_index]
            cat_f: np.ndarray = np.ascontiguousarray(np.tile(cam_param.focal_length, (cat_frame, 1)), dtype=np.float32)
            cat_c: np.ndarray = np.ascontiguousarray(np.tile(cam_param.center, (cat_frame, 1)), dtype=np.float32)
            cat_cam_trans: np.ndarray = np.ascontiguousarray(np.tile(cam_param.translation, (cat_frame, 1)), dtype=np.float32)
            cat_cam_rot: np.ndarray = np.ascontiguousarray(np.tile(cam_param.orientation, (cat_frame, 1)), dtype=np.float32)

            if tmp_buffer_x is None:
                return cat_data_x, cat_data_y

            tmp_buffer_x.append(cat_data_x)
            tmp_buffer_y.append(cat_data_y)
            tmp_buffer_cam_pos_3d.append(cat_pos_3d)
            tmp_buffer_confidence.append(cat_confidence)
            tmp_buffer_camera_f.append(cat_f)
            tmp_buffer_camera_c.append(cat_c)
            tmp_buffer_camera_trans.append(cat_cam_trans)
            tmp_buffer_camera_rot.append(cat_cam_rot)

            divide_index: np.ndarray = np.zeros(cat_data_x.shape[0], dtype=np.bool_)
            divide_index[-1] = True
            tmp_buffer_divide_index.append(divide_index)

            assert cat_frame == cat_data_y.shape[0] and cat_frame == cat_pos_3d.shape[0] and cat_frame == cat_confidence.shape[0]
            assert cat_frame == cat_f.shape[0] and cat_frame == cat_c.shape[0]

            remain_data += cat_data_x.shape[0]

        # tmp_buffer_camera_param.extend(result_node["camera_param"])
        return remain_data

    def get_total_data(self, result_node: Dict) -> Tuple[np.ndarray, np.ndarray]:
        return self.extract_data_xy_base(result_node)

    def random_iter(self):
        """
        randomly select the piece and start index..
        we can shuffle data in the same piece..

        The required output:
        x_gt, y_gt, camera_pos3d_gt, camera_2d_confidence, camera_param
        """
        tmp_buffer_x: List[np.ndarray] = []
        tmp_buffer_y: List[np.ndarray] = []
        tmp_buffer_cam_pos_3d: List[np.ndarray] = []
        tmp_buffer_confidence: List[np.ndarray] = []
        tmp_buffer_camera_f: List[np.ndarray] = []
        tmp_buffer_camera_c: List[np.ndarray] = []
        tmp_buffer_camera_trans: List[np.ndarray] = []
        tmp_buffer_camera_rot: List[np.ndarray] = []
        tmp_buffer_divide_index: List[np.ndarray] = []

        remain_data: int = 0

        # in shape (tot frame - window_frame,  window_frame, *)
        result_buf_index: np.ndarray = np.arange(0, len(self.result_buffer), dtype=np.int32)
        np.random.shuffle(result_buf_index)
        for index in result_buf_index:
            result_node = self.result_buffer[index]
            while remain_data >= self.batch_size:
                # get the data from tmp buffer..
                ext_count: int = 0
                ret_x: Optional[np.ndarray] = None
                ret_y: Optional[np.ndarray] = None
                ret_pos3d: Optional[np.ndarray] = None
                ret_confidence: Optional[np.ndarray] = None
                ret_cam_f: Optional[np.ndarray] = None
                ret_cam_c: Optional[np.ndarray] = None
                ret_cam_trans: Optional[np.ndarray] = None
                ret_cam_rot: Optional[np.ndarray] = None
                ret_divide_index: Optional[np.ndarray] = None
                buffer_len = len(tmp_buffer_x)
                for ext_index in range(buffer_len):
                    old_ext_count = ext_count
                    ext_count += len(tmp_buffer_x[ext_index])
                    if ext_count > self.batch_size:
                        tmp_index = self.batch_size - old_ext_count
                        # compute the input and output data
                        ret_x: np.ndarray = np.concatenate(tmp_buffer_x[:ext_index] + [tmp_buffer_x[ext_index][:tmp_index]], axis=0)
                        ret_y: np.ndarray = np.concatenate(tmp_buffer_y[:ext_index] + [tmp_buffer_y[ext_index][:tmp_index]], axis=0)
                        tmp_buffer_x = tmp_buffer_x[ext_index:]
                        tmp_buffer_x[0] = tmp_buffer_x[0][tmp_index:]
                        tmp_buffer_y = tmp_buffer_y[ext_index:]
                        tmp_buffer_y[0] = tmp_buffer_y[0][tmp_index:]

                        # concat the camera 3d position
                        ret_pos3d: np.ndarray = np.concatenate(tmp_buffer_cam_pos_3d[:ext_index] + [tmp_buffer_cam_pos_3d[ext_index][:tmp_index]], axis=0)
                        ret_confidence: np.ndarray = np.concatenate(tmp_buffer_confidence[:ext_index] + [tmp_buffer_confidence[ext_index][:tmp_index]], axis=0)
                        tmp_buffer_cam_pos_3d = tmp_buffer_cam_pos_3d[ext_index:]
                        tmp_buffer_cam_pos_3d[0] = tmp_buffer_cam_pos_3d[0][tmp_index:]
                        tmp_buffer_confidence = tmp_buffer_confidence[ext_index:]
                        tmp_buffer_confidence[0] = tmp_buffer_confidence[0][tmp_index:]

                        # here we should extend the camera parameter into a large tensor..
                        # the camera projection has several parameters..(f, c), f * xx + c
                        ret_cam_f: np.ndarray = np.concatenate(tmp_buffer_camera_f[:ext_index] + [tmp_buffer_camera_f[ext_index][:tmp_index]], axis=0)
                        ret_cam_c: np.ndarray = np.concatenate(tmp_buffer_camera_c[:ext_index] + [tmp_buffer_camera_c[ext_index][:tmp_index]], axis=0)
                        tmp_buffer_camera_f = tmp_buffer_camera_f[ext_index:]
                        tmp_buffer_camera_f[0] = tmp_buffer_camera_f[0][tmp_index:]
                        tmp_buffer_camera_c = tmp_buffer_camera_c[ext_index:]
                        tmp_buffer_camera_c[0] = tmp_buffer_camera_c[0][tmp_index:]

                        ret_cam_trans: np.ndarray = np.concatenate(tmp_buffer_camera_trans[:ext_index] + [tmp_buffer_camera_trans[ext_index][:tmp_index]], axis=0)
                        ret_cam_rot: np.ndarray = np.concatenate(tmp_buffer_camera_rot[:ext_index] + [tmp_buffer_camera_rot[ext_index][:tmp_index]], axis=0)
                        tmp_buffer_camera_trans = tmp_buffer_camera_trans[ext_index:]
                        tmp_buffer_camera_trans[0] = tmp_buffer_camera_trans[0][tmp_index:]
                        tmp_buffer_camera_rot = tmp_buffer_camera_rot[ext_index:]
                        tmp_buffer_camera_rot[0] = tmp_buffer_camera_rot[0][tmp_index:]

                        # compute the index..
                        ret_divide_index: np.ndarray = np.concatenate(tmp_buffer_divide_index[:ext_index] + [tmp_buffer_divide_index[ext_index][:tmp_index]], axis=0)
                        ret_divide_index[-1] = True
                        tmp_buffer_divide_index = tmp_buffer_divide_index[ext_index:]
                        tmp_buffer_divide_index[0] = tmp_buffer_divide_index[0][tmp_index:]

                        break
                    elif ext_count == self.batch_size:
                        ret_x: np.ndarray = np.concatenate(tmp_buffer_x[:ext_index + 1], axis=0)
                        ret_y: np.ndarray = np.concatenate(tmp_buffer_y[:ext_index + 1], axis=0)
                        tmp_buffer_x = tmp_buffer_x[ext_index + 1:]
                        tmp_buffer_y = tmp_buffer_y[ext_index + 1:]

                        # concat the camera 3d position and 2d confidence
                        ret_pos3d: np.ndarray = np.concatenate(tmp_buffer_cam_pos_3d[:ext_index + 1], axis=0)
                        ret_confidence: np.ndarray = np.concatenate(tmp_buffer_confidence[:ext_index + 1], axis=0)
                        tmp_buffer_cam_pos_3d = tmp_buffer_cam_pos_3d[ext_index + 1:]
                        tmp_buffer_confidence = tmp_buffer_confidence[ext_index + 1:]

                        # handle the camera parameter (f, c)..
                        ret_cam_f: np.ndarray = np.concatenate(tmp_buffer_camera_f[:ext_index + 1], axis=0)
                        ret_cam_c: np.ndarray = np.concatenate(tmp_buffer_camera_c[:ext_index + 1], axis=0)
                        tmp_buffer_camera_f = tmp_buffer_camera_f[ext_index + 1:]
                        tmp_buffer_camera_c = tmp_buffer_camera_c[ext_index + 1:]

                        ret_cam_trans: np.ndarray = np.concatenate(tmp_buffer_camera_trans[:ext_index + 1], axis=0)
                        ret_cam_rot: np.ndarray = np.concatenate(tmp_buffer_camera_rot[:ext_index + 1], axis=0)
                        tmp_buffer_camera_trans = tmp_buffer_camera_trans[ext_index + 1:]
                        tmp_buffer_camera_rot = tmp_buffer_camera_rot[ext_index + 1:]

                        # compute the index..
                        ret_divide_index: np.ndarray = np.concatenate(tmp_buffer_divide_index[:ext_index + 1], axis=0)
                        ret_divide_index[-1] = True
                        tmp_buffer_divide_index = tmp_buffer_divide_index[ext_index + 1:]
                        break

                remain_data -= self.batch_size
                if ret_x is not None and ret_y is not None:
                    # here we need not to shuffle the input data batchly.
                    # ret_index = np.arange(0, self.batch_size, dtype=np.int32)
                    # np.ramdom.shuffle(ret_index)
                    # ret_x = np.ascontiguousarray(ret_x[ret_index])
                    # ret_y = np.ascontiguousarray(ret_y[ret_index])
                    ret_x: torch.Tensor = torch.as_tensor(ret_x, dtype=torch.float32, device=self.device)
                    ret_y: torch.Tensor = torch.as_tensor(ret_y, dtype=torch.float32, device=self.device)
                    ret_pos3d: torch.Tensor = torch.as_tensor(ret_pos3d, dtype=torch.float32, device=self.device)
                    ret_confidence: torch.Tensor = torch.as_tensor(ret_confidence, dtype=torch.float32, device=self.device)
                    ret_cam_f: torch.Tensor = torch.as_tensor(ret_cam_f, dtype=torch.float32, device=self.device)
                    ret_cam_c: torch.Tensor = torch.as_tensor(ret_cam_c, dtype=torch.float32, device=self.device)
                    ret_cam_trans: torch.Tensor = torch.as_tensor(ret_cam_trans, dtype=torch.float32, device=self.device)
                    ret_cam_rot: torch.Tensor = torch.as_tensor(ret_cam_rot, dtype=torch.float32, device=self.device)
                    ret_divide_index: torch.Tensor = torch.from_numpy(ret_divide_index)  # as we compute phys loss on CPU, we need not put it on GPU.
                    yield ret_x, ret_y, ret_pos3d, ret_confidence, ret_cam_f, ret_cam_c, ret_cam_trans, ret_cam_rot, ret_divide_index

            remain_data = self.extract_data_xy_base(
                result_node,
                tmp_buffer_x, tmp_buffer_y,
                tmp_buffer_cam_pos_3d, tmp_buffer_confidence,
                tmp_buffer_camera_f, tmp_buffer_camera_c,
                tmp_buffer_camera_trans, tmp_buffer_camera_rot,
                tmp_buffer_divide_index,
                remain_data
            )

    def normalize_pos3d(self, pos: torch.Tensor) -> torch.Tensor:
        if self.mean_pos3d is not None and self.std_pos3d is not None:
            return (pos - self.mean_pos3d) / self.std_pos3d
        else:
            return pos

    def unnormalize_pos3d(self, pos: torch.Tensor) -> torch.Tensor:
        if self.mean_pos3d is not None and self.std_pos3d is not None:
            return pos * self.std_pos3d + self.mean_pos3d
        else:
            return pos

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
import os
import pickle

from scipy.spatial.transform import Rotation
from typing import List, Optional, Tuple, Union

from ..ODESim.Saver.CharacterToBVH import CharacterTOBVH
from ..ODESim.BodyInfoState import BodyInfoState
from ..ODESim.ODEScene import ODEScene
from ..ODESim.ODECharacter import ODECharacter
from ..pymotionlib.Utils import flip_quaternion
from ..pymotionlib import BVHLoader

from ..Server.v2.ServerForUnityV2 import ServerForUnityV2, UpdateSceneBase


fdir = os.path.dirname(__file__)


def character_data_to_body_state(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_velo: np.ndarray,
    body_omega: np.ndarray,
    pd_target_quat: np.ndarray
) -> List[BodyInfoState]:
    num_frame: int = body_pos.shape[0]
    body_rot_mat: np.ndarray = Rotation(body_quat.reshape((-1, 4))).as_matrix().reshape(body_quat.shape[:-1] + (3, 3))
    results = [BodyInfoState().set_value(
        body_pos[i], body_rot_mat[i], body_quat[i], body_velo[i], body_omega[i], pd_target_quat[i]
    ) for i in range(num_frame)]

    return results


def mirror_character_data_no_velo(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    pd_target_quat: np.ndarray,  # in (frame, num joint, 4)
    contact_label: Optional[np.ndarray] = None,
    mirror_body_index: Union[List[int], np.ndarray, None] = None,
    mirror_joint_index: Union[List[int], np.ndarray, None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    for data augmentation (left and right mirror)

    return position, quaternion, pd target
    """
    vector_x = np.array([1.0, 0.0, 0.0])

    mirror_body_pos: np.ndarray = body_pos.copy()
    mirror_body_pos[..., 0] *= -1
    mirror_body_pos: np.ndarray = np.ascontiguousarray(mirror_body_pos[:, mirror_body_index])

    mirror_body_quat: np.ndarray = flip_quaternion(body_quat.reshape(-1, 4), vector_x, False).reshape(body_quat.shape)
    mirror_body_quat: np.ndarray = np.ascontiguousarray(mirror_body_quat[:, mirror_body_index])

    mirror_pd_target_quat: np.ndarray = flip_quaternion(pd_target_quat.reshape(-1, 4), vector_x, False).reshape(pd_target_quat.shape)
    mirror_pd_target_quat: np.ndarray = np.ascontiguousarray(mirror_pd_target_quat[:, mirror_joint_index])

    mirror_contact_label: Optional[np.ndarray] = None
    if contact_label is not None:
        mirror_contact_label: Optional[np.ndarray] = np.ascontiguousarray(contact_label[:, mirror_body_index])

    # TODO: debug using Long Ge's Framework..
    return mirror_body_pos, mirror_body_quat, mirror_pd_target_quat, mirror_contact_label


def mirror_position(pos: np.ndarray, mirror_index: np.ndarray, inplace: bool = False):
    if not inplace:
        mirror_pos: np.ndarray = pos.copy()
    else:
        mirror_pos = pos
    mirror_pos[..., 0] *= -1
    mirror_pos: np.ndarray = pos[..., mirror_index, :]
    return mirror_pos


def mirror_character_data_base(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_velo: np.ndarray,
    body_omega: np.ndarray,
    mirror_body_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vector_x = np.array([1.0, 0.0, 0.0], dtype=body_pos.dtype)

    mirror_body_pos: np.ndarray = body_pos.copy()
    mirror_body_pos[..., 0] *= -1
    mirror_body_pos: np.ndarray = np.ascontiguousarray(mirror_body_pos[..., mirror_body_index, :])

    mirror_body_quat: np.ndarray = flip_quaternion(body_quat.reshape(-1, 4), vector_x, False).reshape(body_quat.shape)
    mirror_body_quat: np.ndarray = np.ascontiguousarray(mirror_body_quat[..., mirror_body_index, :])

    mirror_body_velo: np.ndarray = body_velo.copy()
    mirror_body_velo[..., 0] *= -1
    mirror_body_velo = np.ascontiguousarray(mirror_body_velo[:, mirror_body_index, :])

    mirror_body_angvel: np.ndarray = body_omega.copy()
    mirror_body_angvel[..., 1:] *= -1  # mirror y, z component for angular velocity
    mirror_body_angvel: np.ndarray = np.ascontiguousarray(mirror_body_angvel[..., mirror_body_index, :])

    return mirror_body_pos, mirror_body_quat, mirror_body_velo, mirror_body_angvel # TODO: check dtype..


def mirror_character_data(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_velo: np.ndarray,
    body_omega: np.ndarray,
    pd_target_quat: np.ndarray,  # in (frame, num joint, 4)
    mirror_body_index: Union[List[int], np.ndarray, None] = None,
    mirror_joint_index: Union[List[int], np.ndarray, None] = None
):
    """
    for pd control target pose in local frame,
    we can simple mirror left and right part.

    for root global position:
        (x, y, z) -> (-x, y, z)
    for global angular velocity:
        (x, y, z) -> (x, -y, -z)
    for example: if the character turn left in the initial coordinate, it should turn right in mirrored coordinate

    Test:
    1. after forward kinematics, the global joint position should be mirrored
    2. after a step simulation, x component is mirrored, and y, z component are same.

    Note: after mirror character data, the simulation result will NOT always match!
    Most of the times, the results are same (<1e-12), but sometimes, angular velocity error is large
    I think the reason is that little error on mirror will cause contact change, and the simulation result will not always same.
    but the mirrored result is close to real solution.

    Maybe we should run samcon on mirrored data again..

    return:
    mirror_body_pos, mirror_body_quat, mirror_body_velo, mirror_body_angvel, mirror_pd_target_quat, mirror_contact_flag
    """

    
    mirror_body_pos, mirror_body_quat, mirror_body_velo, mirror_body_angvel = mirror_character_data_base(
        body_pos, body_quat, body_velo, body_omega, mirror_body_index)
    # check angular velocity...
    # forward = False
    # from ..pymotionlib.Utils import quat_product
    # qd = np.diff(mirror_body_quat, axis=0) * 100 # 100 is fps

    # # note that we cannot use R(q).inv() here, because scipy implement inv() as
    # #  (x,y,z,w).inv() = (x,y,z,-w)
    # # which is not the conjugate!
    # q = mirror_body_quat[:-1] if forward else mirror_body_quat[1:]
    # q_conj = q.copy().reshape(-1, 4)
    # q_conj[:, :3] *= -1
    # qw = quat_product(qd.reshape(-1, 4), q_conj)

    # frames = mirror_body_quat.shape[0]
    # w = np.zeros(mirror_body_angvel.shape)
    # frag = 2 * qw[:, :3].reshape(frames - 1, mirror_body_quat.shape[1], 3)
    # if forward:
    #     w[:-1] = frag
    # else:
    #     w[1:] = frag

    # w[-1 if forward else 0] = w[-2 if forward else 1]
    # print(w[20] - mirror_body_angvel[20])
    # exit(0)

    mirror_pd_target_quat: np.ndarray = flip_quaternion(pd_target_quat.reshape(-1, 4), vector_x, False).reshape(pd_target_quat.shape)
    mirror_pd_target_quat: np.ndarray = np.ascontiguousarray(mirror_pd_target_quat[:, mirror_joint_index])

    # mirror_contact_flag: np.ndarray = contact_flag[:, mirror_body_index]

    return mirror_body_pos, mirror_body_quat, mirror_body_velo, mirror_body_angvel, mirror_pd_target_quat


def generate_contact_flag(
    scene: ODEScene,
    character: ODECharacter,
    input_data: Union[str, List[BodyInfoState]],
    print_log: bool = False
) -> np.ndarray:
    """
    use collision detection by open dynamics engine
    """
    # check ok with unity visualize..
    if isinstance(input_data, str):
        with open(input_data, "rb") as fin:
            input_data: List[BodyInfoState] = pickle.load(fin)

    body_contact_flag: np.ndarray = np.zeros((len(input_data), len(character.bodies)))
    for index, body_state in enumerate(input_data):
        character.load(body_state)
        # do collision detection
        body_contact_flag[index, :] = scene.extract_body_contact_label()

    if print_log:
        print(f"After generate contact flag")
    return np.ascontiguousarray(body_contact_flag)


def concat_body_info_state(
    scene: ODEScene,
    character: ODECharacter,
    input_data_full: Union[str, List[BodyInfoState]],
    compute_contact_flag: bool = False,
    output_dtype=np.float64
):
    """
    return: body_pos_list, body_quat_list, pd_target_list, Optional[contact_flag]
    """
    if isinstance(input_data_full, str):
        print(f"load file from {input_data_full}")
        with open(input_data_full, "rb") as fin:
            input_data_full: List[BodyInfoState] = pickle.load(fin)
    input_data: List[BodyInfoState] = input_data_full[:-1]

    # extract body state..
    body_pos_list: np.ndarray = np.concatenate([node.pos.reshape((1, -1, 3)) for node in input_data], axis=0, dtype=output_dtype)
    body_quat_list: np.ndarray = np.concatenate([node.quat.reshape((1, -1, 4)) for node in input_data], axis=0, dtype=output_dtype)  # (frame - 1, body, 4)
    # body_vel_list: np.ndarray = np.concatenate([node.linear_vel.reshape((1, -1, 3)) for node in input_data], axis=0)
    # body_omega_list: np.ndarray = np.concatenate([node.angular_vel.reshape((1, -1, 3)) for node in input_data], axis=0)
    pd_target_list: np.ndarray = np.concatenate([node.pd_target.reshape((1, -1, 4)) for node in input_data], axis=0, dtype=output_dtype)  # (frame - 1, joint, 4)

    contact_flag: Optional[np.ndarray] = None
    if compute_contact_flag:
        contact_flag: np.ndarray = generate_contact_flag(scene, character, input_data).astype(output_dtype)

    return body_pos_list, body_quat_list, pd_target_list, contact_flag

#=====================Test ========================


#  Test Visualize contact body (make sure contact label is correct)
#  Visualize in Unity
class DebugContact(UpdateSceneBase):
    def __init__(self, scene: ODEScene, simu_result: List[BodyInfoState], contact_label: np.ndarray):
        super(DebugContact, self).__init__(scene)
        self.simu_result = simu_result
        self.contact_label = contact_label
        self.frame = 0
        self.visualize_color = self.character0.body_info.visualize_color

    def update(self, mess_dict=None):
        self.character0.load(self.simu_result[self.frame])
        self.visualize_color.clear()
        for body_idx, label in enumerate(self.contact_label[self.frame]):
            if label == 0:
                self.visualize_color.append(None)
            else:
                self.visualize_color.append([1.0, 0.0, 0.0])

        self.frame = (self.frame + 1) % len(self.simu_result)


class DebugContactServer(ServerForUnityV2):
    def __init__(self):
        super(DebugContactServer, self).__init__()
        # self.samhlp = SamHlp(os.path.join(fdir, "../CharacterData/SamconConfig-duplicate.json"))

    def after_load_hierarchy(self):
        simu_fname = os.path.join(fdir, "../CharacterData/Samcon/0/3-0.bvh.pickle")
        with open(simu_fname, "rb") as fin:
            simu_result: List[BodyInfoState] = pickle.load(fin)
        contact_label = generate_contact_flag(self.scene, self.scene.character0, simu_result)
        self.update_scene = DebugContact(self.scene, simu_result, contact_label)


def test_mirror_data(scene: ODEScene, result_fname: str):
    """
    load samcon result, and do forward simulation on original data / mirrored data
    """
    import subprocess
    import time
    from VclSimuBackend.ODESim.PDControler import DampedPDControler
    from VclSimuBackend.Render.Renderer import RenderWorld
    scene.set_sim_fps(100)
    character: ODECharacter = scene.character0
    spd_control: DampedPDControler = DampedPDControler(character)

    # for debug, we can use two characters...?

    mirror_body_index: List[int] = character.body_info.get_mirror_index()
    mirror_joint_index: List[int] = character.joint_info.get_mirror_index()

    with open(result_fname, "rb") as fin:
        samcon_result: List[BodyInfoState] = pickle.load(fin)

    concat_result = concat_body_info_state(scene, character, samcon_result[:-1], True)
    mirror_result = mirror_character_data(*concat_result, mirror_body_index, mirror_joint_index)
    mirror_samcon_result: List[BodyInfoState] = character_data_to_body_state(*mirror_result[:-1])

    # to_bvh = CharacterTOBVH(character, scene.sim_fps)
    # to_bvh.bvh_hierarchy_no_root()
    # mirror_invdyn = to_bvh.forward_kinematics(mirror_result[0][:, 0], mirror_result[1][:, 0], mirror_result[-2])
    # mirror_invdyn = to_bvh.insert_end_site(mirror_invdyn)

    # raw_invdyn = to_bvh.forward_kinematics(concat_result[0][:, 0], concat_result[1][:, 0], concat_result[-2])
    # raw_invdyn = to_bvh.insert_end_site(raw_invdyn)

    # raw_fname = "test-raw.bvh"
    # test_mirror_fname = "test-mirror.bvh"
    # BVHLoader.save(raw_invdyn, raw_fname)
    # BVHLoader.save(mirror_invdyn, test_mirror_fname)
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", raw_fname, test_mirror_fname])
    # exit(0)
    renderObj = RenderWorld(scene.world)
    renderObj.set_joint_radius(0.05)
    renderObj.start()

    num_frame = len(samcon_result)
    # we can visualize the simulation with long ge's framework..
    for frame in range(num_frame - 1):
        # as simulation result on Linux and Windows are NOT same, there will be little error on original data
        # origin_state: BodyInfoState = samcon_result[frame]
        # origin_next_state: BodyInfoState = samcon_result[frame + 1]
        # character.load(origin_state)
        # origin_tau = spd_control.add_torques_by_quat(origin_state.pd_target)
        # scene.damped_simulate_once()
        # origin_sim_state = character.save()
        # d_pos, d_rot, d_quat_ode, d_lin_vel, d_ang_vel = origin_next_state.calc_delta(origin_sim_state)
        # print(d_pos)
        # print(d_rot)
        # print(d_quat_ode)
        # print(d_lin_vel)
        # print(d_ang_vel)
        # print("\n\n\n")

        mirror_state: BodyInfoState = mirror_samcon_result[frame]
        mirror_next_state: BodyInfoState = mirror_samcon_result[frame + 1]
        character.load(mirror_state)
        mirror_tau = spd_control.add_torques_by_quat(mirror_state.pd_target)

        # print(origin_tau - mirror_tau)
        scene.damped_simulate_once()
        mirror_sim_state = character.save()
        d_pos, d_rot, d_quat_ode, d_lin_vel, d_ang_vel = mirror_sim_state.calc_delta(mirror_next_state)
        print(d_pos)
        print(d_rot)
        print(d_quat_ode)

        # # position and rotation are ok
        # # I don't think linear velocity and angular velocity are right..
        print(d_lin_vel)
        print(d_ang_vel)
        print("")

        time.sleep(0.1)

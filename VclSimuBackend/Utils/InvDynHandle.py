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

"""
Input: motion at time t-1, t, t+1
Output: pd target at time t
"""

import ModifyODE as ode
from MotionUtils import DInverseDynamics
from MotionUtils import InvDynForceBatchRes as InvDynForceRes
import numpy as np
import os
import platform
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, Union, Dict, Any
from tqdm import tqdm

from VclSimuBackend.Common.MathHelper import MathHelper
from VclSimuBackend.Common.SmoothOperator import GaussianBase, ButterWorthBase, SmoothMode, smooth_operator
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.CharacterJointInfoRoot import CharacterJointInfoRoot
from VclSimuBackend.ODESim.CharacterWrapper import ODECharacter, BodyInfoState
from VclSimuBackend.ODESim.TargetPose import TargetPose
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.CharacterWrapper import CharacterWrapper
from VclSimuBackend.ODESim.Utils import BVHJointMap

from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData

from MotionUtils import (
    quat_multiply_forward_fast, quat_to_rotvec_single_fast, quat_to_rotvec_fast, quat_from_rotvec_fast, quat_inv_fast,
    decompose_rotation_single_fast, decompose_rotation_single_pair_fast,
    quat_to_vec6d_single_fast, quat_inv_single_fast, quat_apply_single_fast,
    quat_apply_forward_fast, quat_to_vec6d_fast, six_dim_mat_to_quat_fast, decompose_rotation_pair_one2many_fast,
    quat_from_matrix_fast, delta_quat_float32, quat_to_matrix_fast
)


def state_to_BodyInfoState(state: np.ndarray):
    res = BodyInfoState()
    state = state.reshape((-1, 13))
    res.pos = np.ascontiguousarray(state[:, 0:3].flatten(), dtype=np.float64)
    res.quat = np.ascontiguousarray(state[:, 3:7].flatten(), dtype=np.float64)
    res.linear_vel = np.ascontiguousarray(state[:, 7:10].flatten(), dtype=np.float64)
    res.angular_vel = np.ascontiguousarray(state[:, 10:13].flatten(), dtype=np.float64)
    res.rot = quat_to_matrix_fast(np.ascontiguousarray(state[:, 3:7].reshape(-1, 4), dtype=np.float64)).reshape(-1)

    return res

class InvDynHandler(CharacterJointInfoRoot):
    def __init__(self, character: ODECharacter, fps: Optional[float] = None) -> None:
        """
        np.ndarray body_mass, shape=(body_count,)
        np.ndarray body_inertia, shape=(body_count, 3, 3)
        np.ndarray body_position, shape=(body_count, 3)
        np.ndarray body_rotation, shape=(body_count, 3, 3)
        np.ndarray parent_joint_dof, np.int32, shape=(body_count,)
        np.ndarray parent_joint_pos, shape=(body_count, 3)
        list parent_joint_euler_order, list[str], len == body_count
        np.ndarray parent_joint_euler_axis, shape=(body_count)
        np.ndarray parent_body_index, np.int32
        """

        self.character = character
        character.load_init_state()
        self.inv_dyn: DInverseDynamics = DInverseDynamics(
            character.body_info.mass_val,
            character.body_info.calc_body_init_inertia(),
            character.body_info.get_body_pos(),
            character.body_info.get_body_rot(),
            self.get_parent_joint_dof(),
            self.get_parent_joint_pos(),
            self.get_parent_joint_euler_order(),
            self.get_parent_joint_euler_axis(),
            self.get_parent_body_index()
        )

        self.mu: np.ndarray = character.body_info.get_body_contact_mu()
        self.root_pos: Optional[np.ndarray] = None
        self.local_quat: Optional[np.ndarray] = None

        self.body_len = self.get_body_length()

        if fps is None:
            self.fps: float = character.scene.sim_fps
        else:
            self.fps: float = fps
        
        self.scene: ODEScene = character.scene

        self.y_force_clip: float = 3.0
        self.contact_height_eps: float = 0.1
        self.tpose_h = self.character.get_tpose_root_h()

    @property
    def num_frames(self):
        pass

    def calc(self, root_pos: np.ndarray, joint_rot: np.ndarray) -> np.ndarray:
        assert root_pos.shape[0] == joint_rot.shape[0]
        root_pos: np.ndarray = np.ascontiguousarray(root_pos, np.float64)
        joint_rot: np.ndarray = np.ascontiguousarray(joint_rot, np.float64)

    def get_body_length(self) -> np.ndarray:
        """
        return length of each body. np.ndarray with shape (num body,)
        """
        res: np.ndarray = np.zeros(len(self.bodies))
        for idx, body in enumerate(self.bodies):
            geoms = list(body.geom_iter())
            if len(geoms) == 1:
                if isinstance(geoms[0], ode.GeomSphere):
                    res[idx] = geoms[0].geomRadius
                elif isinstance(geoms[0], ode.GeomBox):
                    res[idx] = 0.5 * np.linalg.norm(np.asarray(geoms[0].getLengths()))
                elif isinstance(geoms[0], ode.GeomCapsule):
                    r, length = geoms[0].geomRadiusAndLength
                    res[idx] = np.sqrt(r ** 2 + (0.5 * length) ** 2)
                else:
                    raise NotImplementedError
            else:
                # raise ValueError("This case will not happen")
                aabbs = np.max(np.concatenate([geom.AABBNumpy[None, :] for geom in geoms], axis=0), axis=0)
                res[idx] = np.linalg.norm(aabbs[[1, 3, 5]] - aabbs[[0, 2, 4]]) / 2

        return res

    def calc_q_derivatives(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        calc q, qdot, qdotdot
        """
        batch_func = self.inv_dyn.ConvertToGeneralizeCoordinatesBatch
        general_q: np.ndarray = batch_func(self.root_pos, self.local_quat)
        # in Open Dynamics Engine(ODE),
        # qdotdot_{t} = F(q_{t}, qdot_{t})
        # qdot_{t + 1} = qdot_{t} + dt * qdotdot{t}
        # q_{t + 1} = q_{t} + dt * qdot_{t + 1}
        general_qdot: np.ndarray = self.calc_delta(general_q, False, True)  # shape == (tot_frame, tot_dof)
        general_qdotdot: np.ndarray = self.calc_delta(general_qdot, False, False)  # shape == (tot_frame, tot_dof)
        general_qdot *= self.fps
        general_qdotdot *= self.fps ** 2
        return general_q, general_qdot, general_qdotdot

    def smooth_torque(self, tau: np.ndarray) -> np.ndarray:
        """
        smooth torque compute by inverse dynamics
        """
        assert tau.ndim == 3 and tau.shape[-1] == 3
        res: np.ndarray = np.zeros_like(tau)
        for i in range(tau.shape[1]):
            res[:, i, :] = smooth_operator(tau[:, i, :].copy(), self.smooth_out_type)

        return np.ascontiguousarray(res)
    
    @staticmethod
    def flip_delta_q(frag: np.ndarray) -> np.ndarray:
        frag[:, 3:] %= 2 * np.pi
        frag[:, 3:][frag[:, 3:] >= np.pi] -= 2 * np.pi
        return frag

    @staticmethod
    def calc_delta(x: np.ndarray, forward_: bool = True, flip_to_minus_pi_plus_pi: bool = False) -> np.ndarray:
        xdot: np.ndarray = np.zeros_like(x)
        frag: np.ndarray = np.diff(x, axis=0)
        if flip_to_minus_pi_plus_pi:  # convert to [-pi, pi)
            frag = InvDynHandler.flip_delta_q(frag)

        if forward_:
            xdot[:-1] = frag
        else:
            xdot[1:] = frag

        xdot[-1 if forward_ else 0] = xdot[-2 if forward_ else 1].copy()
        return xdot

    def divide_force(
        self,
        f: np.ndarray,
        tau: np.ndarray,
        com_pos: np.ndarray,
        contact_y_clip_max: Optional[float] = None,  # <= 3mg?
        clip_min_d: float = 1e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide force and torque on CoM to contact points.
        Algorithm in [Simulation and Control of Skeleton-driven Soft Body Characters],
        formula (27) ~ (31)
        TODO: copy to cython when there is no bug for fast..

        return: local force, local torque on each joint in np.ndarray with shape (batch, num joint, 3)
        """
        mu: np.ndarray = self.mu
        r_max: np.ndarray = self.body_len
        character = self.character
        scene = self.scene
        # TODO: maybe we should not use body_pos here for walking..
        # emm, maybe we should use minimal height, 
        # rather than child body position
        # body_pos: np.ndarray = self.target.all_child_body.pos
        
        if contact_y_clip_max is None:
            contact_y_clip_max = np.abs(self.y_force_clip * character.body_info.sum_mass * scene.gravity_numpy[1])
        height_eps = self.contact_height_eps

        # y: np.ndarray = body_pos[:, :, 1]
        # contact_flg = np.min(y, axis=1) < height_eps
        # size_0, size_1 = body_pos.shape[0], body_pos.shape[1]
        # out_force: np.ndarray = np.zeros((size_0, size_1, 3), dtype=np.float64)  # (batch, num joint, 3)
        # out_torque: np.ndarray = np.zeros((size_0, size_1, 3), dtype=np.float64)

        for i in tqdm(range(f.shape[0])):
            self.character.load(state_to_BodyInfoState(self.handle.get_state(i)))
            body_pos: np.ndarray = self.character.get_body_pos()
            self.character.body_info.get_aabb()
            if not contact_flg[i]:  # if there is no contact, set result to zero(ignore root force and torque)
                continue
            yi = y[i]

            contact_idx: np.ndarray = np.asarray(np.argwhere(yi < height_eps).reshape(-1), dtype=np.int32)
            contact_y = np.clip(yi[contact_idx], clip_min_d, None)  # height of each contact point

            # divide force on CoM to each contact joint
            f_ratio = 1.0 / contact_y
            f_ratio /= np.sum(f_ratio)  # shape = (num contact,)

            f_each = f_ratio[:, None] @ f[i, None, :]  # shape = (num contact, 3)

            # divide torque on CoM to each contact joint
            tau_each = f_ratio[:, None] @ tau[i, None, :]  # shape = (num contact, 3)
            com_to_body = com_pos[i, None, :] - body_pos[i, contact_idx, :]  # shape = (num contact, 3)
            tau_each -= com_to_body

            # add contact constraint
            f_each[:, 1] = np.clip(f_each[:, 1], 0, contact_y_clip_max)  # contact force on y axis should >= 0
            # friction force at x, z axies should <= \mu F_y
            f_each_xz_len = np.linalg.norm(f_each[:, [0, 2]], axis=-1, keepdims=True)  # (num contact, 1)
            f_each[:, [0, 2]] /= f_each_xz_len
            f_each[:, [0, 2]] *= np.minimum(f_each_xz_len, mu[contact_idx, None] * f_each[:, 1, None])

            # length or torque should be less than r_max * \sqrt{1 + \mu^2} F_y
            tau_max = r_max[contact_idx] * np.sqrt(1 + mu[contact_idx] ** 2) * f_each[:, 1]
            tau_each_len = np.linalg.norm(tau_each, axis=-1, keepdims=True)  # (num contact, 1)
            tau_each /= tau_each_len
            tau_each *= np.minimum(tau_each_len, tau_max[:, None])

            # set to result
            out_force[i, contact_idx, :] = f_each
            out_torque[i, contact_idx, :] = tau_each

        return out_force, out_torque

    def calc(
        self,
        root_pos: Optional[np.ndarray] = None,
        joint_local_quat: Optional[np.ndarray] = None
    ):
        """
        1. calc force & torque using Inverse Dynamics
        2. move root force to contact point
            Algorithm in [Simulation and Control of Skeleton-driven Soft Body Characters],
            formula (27) ~ (31)
            (1) if there is no contact point, drop root force & torque
            (2) if there is 1 contact point, set contact force = root force,
            contact torque = root torque + (foot CoM - character CoM) \times root force
            (3) if there is 2 contact point, set contact 1 force = y2/(y1 + y2) * root force,
            contact 2 force = y1/(y1 + y2) * root force,
            contact 1 torque = (foot 1 CoM - character CoM) \times contact 1 force
            (4) if there is more than 2 contact point, inverse dynamics may not work.
        3. recompute joint force & torque using Inverse Dynamics
        4. compute reference motion using PD Controller formula or Stable PD formula
            (1) for PD Controller: q_{n}^{PD} = q_{n} + 1/k_p * (\tau_{n} + k_d \\dot{q}_{n})
            (2) for stable PD: q_{n}^{ref} = q_{n} + 1/k_p * (\tau_{n} + k_d \\dot{q}_{n+1})

        return: reference motion used in PD Control. type: np.ndarray with shape (batch, num joint, 3)
        """
        if root_pos is not None and joint_local_quat is not None:
            assert root_pos.shape[0] == joint_local_quat.shape[0]
            self.root_pos = root_pos
            self.local_quat = joint_local_quat

        gravity = self.scene.gravity_numpy

        # convert rotation to generalized coordinate
        qs, dqs, ddqs = self.calc_q_derivatives()  # ddqs[[0, 1]] == 0..

        done = np.zeros(root_pos.shape[0], dtype=np.int32)
        done[-1] = 1
        self.handle = self.character.build_ref_state_handle(root_pos, joint_local_quat, None, done, self.tpose_h, 1.0 / self.fps)

        # calc control force/torque on CoM
        rots = joint_local_quat

        def batch_func(f: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None):
            return self.inv_dyn.ComputeForceTorqueMomentumsBatch(qs, dqs, ddqs, rots, gravity, f, t)

        res: InvDynForceRes = batch_func()
        root_rots = Rotation(rots[:, 0, :])
        com_force: np.ndarray = root_rots.apply(res.f_local[:, 0, :])
        com_tau: np.ndarray = root_rots.apply(res.t_local[:, 0, :])
        com_tau += np.cross(res.qs[:, 0:3] - res.com_pos, com_force)

        # divide force and torque
        sep_force, sep_torque = self.divide_force(com_force, com_tau, res.com_pos)

        # calc local force and torque again.
        # root joint has conduction torque is OK, because contact point is on the foot
        # root joint shouldn't have force.
        new_res: InvDynForceRes = batch_func(sep_force, sep_torque)
        res_q: np.ndarray = self.generate_ref_motion(new_res, self.use_stable_pd)
        
    def calc_by_bvh(self, bvh: Union[str, MotionData]):
        if isinstance(bvh, str):
            bvh = BVHLoader.load(bvh)
        mapper = BVHJointMap(bvh, self.character)
        idx = np.concatenate([np.array([0]), mapper.character_to_bvh])
        quat = np.ascontiguousarray(bvh.joint_rotation[:, idx, :])
        pos = np.ascontiguousarray(bvh.joint_translation[:, 0, :])
        self.calc(pos, quat)


def test_func():
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
    scene_fname = r"D:\song\desktop\curr-yhy\Data\Misc\world.json"
    bvh_fname = r"F:\GitHub\ode-scene\Tests\CharacterData\WalkF-mocap-100.bvh"
    scene = JsonSceneLoader().load_from_file(scene_fname)
    
    inv_dyn = InvDynHandler(scene.character0)
    inv_dyn.calc_by_bvh(bvh_fname)


if __name__ == "__main__":
    test_func()

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

import ModifyODE as ode
from MotionUtils import DInverseDynamics
from MotionUtils import InvDynForceBatchRes as InvDynForceRes
import numpy as np
import os
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, Union, Dict, Any

from ..Common.MathHelper import MathHelper
from ..Common.SmoothOperator import GaussianBase, ButterWorthBase, SmoothMode, smooth_operator
from ..ODESim.BVHToTarget import BVHToTargetBase
from ..ODESim.CharacterJointInfoRoot import CharacterJointInfoRoot
from ..ODESim.CharacterWrapper import ODECharacter
from ..ODESim.TargetPose import TargetPose
from ..ODESim.ODEScene import ODEScene
from ..ODESim.CharacterWrapper import CharacterWrapper
from ..pymotionlib import BVHLoader
from ..pymotionlib.MotionData import MotionData


class InvDynGaussianSmoothAttr:
    __slots__ = ("input", "output")

    def __init__(self, input_width: int = 3, output_width: int = 3):
        self.input: Optional[GaussianBase] = GaussianBase(input_width)
        self.output: Optional[GaussianBase] = GaussianBase(output_width)

    @classmethod
    def build_from_dict(cls, info: Dict[str, int]):
        return cls(info["input_smooth"], info["output_smooth"])


class InvDynButterWorthSmoothAttr:
    __slots__ = ("input", "output")

    def __init__(self):
        self.input: Optional[ButterWorthBase] = None
        self.output: Optional[ButterWorthBase] = None

    @classmethod
    def build_from_dict(cls, info: Dict[str, Any], motion_freq: int):
        result = cls()
        result.input = ButterWorthBase.build_from_dict(info.get("input"), motion_freq)
        result.output = ButterWorthBase.build_from_dict(info.get("output"), motion_freq)
        return result


class InverseDynamicsBuilder(CharacterJointInfoRoot):
    """
    Build CInverseDynamics object by ODECharacter
    """

    def __init__(self, character: ODECharacter):
        super(InverseDynamicsBuilder, self).__init__(character)
        self.inv_dyn: Optional[DInverseDynamics] = None

    def build(self) -> DInverseDynamics:
        # TODO: now, if character doesn't have root joint, joint_id + 1 must == child body id...
        self.inv_dyn = DInverseDynamics(
            self.body_info.mass_val,
            self.body_info.calc_body_init_inertia(),
            self.body_info.get_body_pos(),
            self.body_info.get_body_rot(),
            self.get_parent_joint_dof(),
            self.get_parent_joint_pos(),
            self.get_parent_joint_euler_order(),
            self.get_parent_joint_euler_axis(),
            self.get_parent_body_index()
        )
        return self.inv_dyn


class MotionInvDyn(CharacterWrapper):
    """
    Implementation of Inverse Dynamics algorithm in
    Libin Liu, et al. Simulation and Control of Skeleton-driven Soft Body Characters, SIGGRAPH Asia 2013
    """

    visualize_smooth_torque = False

    def __init__(
        self,
        scene: ODEScene,
        character: ODECharacter,
        bvh_target: Union[str, MotionData, TargetPose, None] = None,
        bvh_start: Optional[int] = None,
        bvh_end: Optional[int] = None,
        use_stable_pd: bool = False,
        y_force_clip: float = 3.0,
        contact_height_eps: float = 0.1,
        smooth_mode: Union[InvDynGaussianSmoothAttr, InvDynButterWorthSmoothAttr, None] = None,
        output_bvh_fname: str = "invdyn-out.bvh"
    ):
        super(MotionInvDyn, self).__init__(character)
        self.scene: ODEScene = scene
        self.sim_fps = int(scene.sim_fps)
        self.sim_dt = scene.sim_dt

        # y <= 3mg, or y <= +inf..
        self.y_force_clip: float = y_force_clip
        self.contact_height_eps: float = contact_height_eps
        self.use_stable_pd: bool = use_stable_pd

        self.loader: Optional[BVHToTargetBase] = None
        self.bvh: Optional[MotionData] = None  # only for save hierarchy

        self.smooth_type = smooth_mode
        if isinstance(output_bvh_fname, str) and not output_bvh_fname.endswith(".bvh"):
            output_bvh_fname += ".bvh"
        self.output_bvh_fname = output_bvh_fname

        if isinstance(bvh_target, str) or isinstance(bvh_target, MotionData):  # load bvh from file
            self.loader = BVHToTargetBase(
                bvh_target, self.sim_fps, character, False, bvh_start, bvh_end, smooth_type=self.smooth_in_type)
            self.bvh: Optional[MotionData] = self.loader.bvh
            if self.smooth_in_type is not None:
                self.target: TargetPose = self.loader.init_smooth_target()
            else:
                print("Not do input smooth in inverse dynamics")
                self.target: TargetPose = self.loader.init_target()

        elif type(bvh_target) == TargetPose:
            self.target: TargetPose = bvh_target.sub_seq(bvh_start, bvh_end, False)  # use from input
        else:
            raise TypeError(f"Type of bvh_target should be str or TargetPose. Input type is {type(bvh_target)}")

        self.mu: np.ndarray = self.body_info.get_body_contact_mu()
        self.body_len: np.ndarray = self.get_body_length()
        self.builder = InverseDynamicsBuilder(self.character)
        self.dinv_dyn: DInverseDynamics = self.builder.build()

        self.tau_result: Optional[np.ndarray] = None

    @property
    def smooth_in_type(self):
        return self.smooth_type.input if self.smooth_type is not None else None

    @property
    def smooth_out_type(self):
        return self.smooth_type.output if self.smooth_type is not None else None

    @staticmethod
    def ref_slice() -> slice:
        return slice(*MotionInvDyn.ref_start_end())

    @staticmethod
    def ref_start() -> int:  # >= 5
        return 5

    @staticmethod
    def ref_end() -> int:  # <= -5
        return -5

    @staticmethod
    def ref_start_end() -> Tuple[int, int]:
        return MotionInvDyn.ref_start(), MotionInvDyn.ref_end()

    @staticmethod
    def ref_range() -> Tuple[int, int]:
        return MotionInvDyn.ref_start_end()

    @staticmethod
    def invdyn_param(
        conf: Dict[str, Any],
        fps: int,
        output_bvh_fname: Optional[str] = None) -> Dict[str, Any]:
        inv_dyn_conf: Dict[str, Any] = conf["inverse_dynamics"]
        result: Dict[str, Any] = {key: inv_dyn_conf[key] for key in ["y_force_clip", "contact_height_eps"]}
        smooth_mode: SmoothMode = SmoothMode[inv_dyn_conf["smooth_mode"]]

        if smooth_mode == SmoothMode.NO:
            result["smooth_mode"] = None
        elif smooth_mode == SmoothMode.GAUSSIAN:
            result["smooth_mode"] = InvDynGaussianSmoothAttr.build_from_dict(inv_dyn_conf["gaussian"])
        elif smooth_mode == SmoothMode.BUTTER_WORTH:
            result["smooth_mode"] = InvDynButterWorthSmoothAttr.build_from_dict(inv_dyn_conf["butter_worth"], fps)
        else:
            raise ValueError("Smooth mode not supported. Only support ")

        if output_bvh_fname is not None:
            result["output_bvh_fname"] = str(output_bvh_fname)
        return result

    @staticmethod
    def builder(scene: ODEScene,
                character: ODECharacter,
                conf: Dict[str, Any],
                bvh: Optional[str] = None,
                output_bvh_fname: Optional[str]=None):
        if bvh is None:
            bvh: str = conf["filename"]["bvh"]
        return MotionInvDyn(
            scene, character, bvh,
            **MotionInvDyn.invdyn_param(conf, scene.sim_fps, output_bvh_fname)
        )

    def get_body_by_name(self, name: str) -> int:
        return [idx for idx, b in enumerate(self.bodies) if b.name.lower() == name][0]

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
                aabbs = np.max(np.concatenate([geom.AABBNumpy[None, :] for geom in geoms], axis=0), axis=0)
                res[idx] = np.linalg.norm(aabbs[[1, 3, 5]] - aabbs[[0, 2, 4]]) / 2

        return res

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
            frag = MotionInvDyn.flip_delta_q(frag)

        if forward_:
            xdot[:-1] = frag
        else:
            xdot[1:] = frag

        xdot[-1 if forward_ else 0] = xdot[-2 if forward_ else 1].copy()
        return xdot

    def calc_q_derivatives(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        calc q, qdot, qdotdot
        """
        batch_func = self.dinv_dyn.ConvertToGeneralizeCoordinatesBatch
        general_q: np.ndarray = batch_func(self.target.root.pos, self.target.all_joint_local.quat)
        # in Open Dynamics Engine(ODE),
        # qdotdot_{t} = F(q_{t}, qdot_{t})
        # qdot_{t + 1} = qdot_{t} + dt * qdotdot{t}
        # q_{t + 1} = q_{t} + dt * qdot_{t + 1}
        general_qdot: np.ndarray = self.calc_delta(general_q, False, True)  # shape == (tot_frame, tot_dof)

        # TODO: forward of compute qdotdot should be True here
        # If we use False, the index should be qs[3:], dqs[3:], ddqs[2:-1], rots[3:] as my implementation..
        general_qdotdot: np.ndarray = self.calc_delta(general_qdot, False, False)  # shape == (tot_frame, tot_dof)
        general_qdot *= self.sim_fps
        general_qdotdot *= self.sim_fps ** 2
        return general_q, general_qdot, general_qdotdot

    def divide_force(
        self,
        f: np.ndarray,
        tau: np.ndarray,
        com_pos: np.ndarray,
        start: Optional[int] = None,
        end: Optional[int] = None,
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
        r_max = self.body_len
        # TODO: maybe we should not use body_pos here for walking..
        body_pos = self.target.all_child_body.pos[
            (0 if start is None else start):
            (self.target.num_frames if end is None else end)
        ]
        if contact_y_clip_max is None:
            contact_y_clip_max = np.abs(self.y_force_clip * self.body_info.sum_mass * self.scene.gravity_numpy[1])
        height_eps = self.contact_height_eps

        y: np.ndarray = body_pos[:, :, 1]
        contact_flg = np.min(y, axis=1) < height_eps
        size_0, size_1 = body_pos.shape[0], body_pos.shape[1]
        out_force: np.ndarray = np.zeros((size_0, size_1, 3), dtype=np.float64)  # (batch, num joint, 3)
        out_torque: np.ndarray = np.zeros((size_0, size_1, 3), dtype=np.float64)

        for i in range(size_0):
            if not contact_flg[i]:  # if there is no contact, set result to zero(ignore root force and torque)
                continue
            yi = y[i]

            contact_idx = np.asarray(np.argwhere(yi < height_eps).reshape(-1), dtype=np.int32)
            # if contact_idx.size == 1:
            #    pass
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
        bvh_hierarchy: Optional[MotionData] = None,
        character_to_bvh: Optional[np.ndarray] = None
        ) -> TargetPose:
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
        mass_tot, gravity = self.character.body_info.sum_mass, self.scene.gravity_numpy
        # target = self.target

        # convert rotation to generalized coordinate
        qs, dqs, ddqs = self.calc_q_derivatives()  # ddqs[[0, 1]] == 0..

        # calc control force/torque on CoM
        rots: np.ndarray = self.target.all_joint_local.quat

        def batch_func(f: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None):
            return self.dinv_dyn.ComputeForceTorqueMomentumsBatch(qs[3:], dqs[3:], ddqs[2:-1], rots[3:], gravity, f, t)

        res: InvDynForceRes = batch_func()
        rots3 = res.joint_rots_unslice
        root_rots3 = Rotation(rots3[:, 0, :], copy=False)
        com_force3 = root_rots3.apply(res.f_local_unslice[:, 0, :])
        com_tau3 = root_rots3.apply(res.t_local_unslice[:, 0, :])
        com_tau3 += np.cross(res.qs_unslice[:, 0:3] - res.com_pos_unslice, com_force3)

        com_force5, com_torque5 = com_force3[2:], com_tau3[2:]

        def test_momentum():
            dbg_idx = 23
            f0 = Rotation(self.dinv_dyn.ConvertToJointRotations(res.qs[dbg_idx])[1][0]).apply(
                res.f_local[dbg_idx, 0, :])
            tmp1 = (f0 + self.body_info.sum_mass * np.array([0.0, -9.8, 0.0])) * self.sim_dt
            print(f0, com_force5[dbg_idx])
            tmp2 = self.body_info.sum_mass * (res.com_linvel[dbg_idx + 1, :] - res.com_linvel[dbg_idx, :])

            tmp = tmp1 - tmp2
            print(tmp1, tmp2, tmp)

        def test_ang_momentum():
            dbg_idx = 233
            tmp1 = com_torque5[dbg_idx] * self.sim_dt
            tmp2 = res.com_ang_momentums[dbg_idx + 1, :] - res.com_ang_momentums[dbg_idx, :]
            tmp = tmp1 - tmp2
            print(tmp1, tmp2, tmp)

        # test_momentum()
        # test_ang_momentum()

        # divide force and torque
        sep_force, sep_torque = self.divide_force(com_force3, com_tau3, res.com_pos_unslice, start=3)

        # calc local force and torque again.
        # root joint has conduction torque is OK, because contact point is on the foot
        # root joint shouldn't have force.
        new_res: InvDynForceRes = batch_func(sep_force, sep_torque)
        res_q: np.ndarray = self.generate_ref_motion(new_res, self.use_stable_pd)
        self.target: TargetPose = self.target.sub_seq(self.ref_start(), None, is_copy=False)
        return self.convert_to_target_pose(res_q, bvh_hierarchy, character_to_bvh)

    def smooth_torque(self, tau: np.ndarray) -> np.ndarray:
        """
        smooth torque compute by inverse dynamics
        """
        if not self.smooth_out_type:
            return tau

        assert tau.ndim == 3 and tau.shape[-1] == 3
        res: np.ndarray = np.zeros_like(tau)
        for i in range(tau.shape[1]):
            res[:, i, :] = smooth_operator(tau[:, i, :].copy(), self.smooth_out_type)

        if self.visualize_smooth_torque and False:
            import matplotlib.pyplot as plt
            for joint_idx in range(tau.shape[1]):
                fig = plt.figure()
                for dim in range(3):
                    sub_plot = fig.add_subplot(311 + dim)
                    sub_plot.plot(tau[:200, joint_idx, dim], label="tau")
                    sub_plot.plot(res[:200, joint_idx, dim], label="smooth_tau")
                    sub_plot.legend()

                    sub_plot.set_title(["x", "y", "z"][dim])
                plt.suptitle(f"joint {joint_idx}")
                plt.show()

        # compute phase decay here..
        # max_decay = 20
        # decay_result = []
        # for decay in range(max_decay):
        #     decay_dist = np.sum((res[:-max_decay] - tau[decay:-max_decay + decay]) ** 2)
        #     decay_result.append(decay_dist)
        # decay = np.argmin(np.array(decay_result))
        return res

    def generate_ref_motion(
        self,
        res: InvDynForceRes,
        use_stable_pd: bool = False,
        use_torque_lim: bool = True
    ) -> np.ndarray:
        """
        Generate reference motion from force & torque by PD Controller or Stable PD Controller.
        Compute reference motion using PD Controller formula or Stable PD formula
            (1) for PD Controller: q_{n}^{PD} = q_{n} + 1/k_p * (\tau_{n} + k_d \\dot{q}_{n})
            (2) for stable PD: q_{n}^{ref} = q_{n} + 1/k_p * (\tau_{n} + k_d \\dot{q}_{n+1})

        Tips: TODO: The last frame is not accurate..perhaps because of change of velocity...
        return: reference motion used in PD Control. type: np.ndarray with shape (batch, num joint, 3)
        """
        has_root = self.character.joint_info.has_root  # character num joint = nj
        # TODO: should we flip quaternion by dot when loading...?
        q_n: np.ndarray = res.joint_rots if has_root else res.joint_rots[:, 1:, :]  # (batch, nj, 4)
        kps_inv: np.ndarray = 1.0 / self.joint_info.kps.reshape((1, -1, 1))  # shape = (1, nj, 3)
        kds: np.ndarray = self.joint_info.kds[None, ...].reshape((1, -1, 1))  # shape = (nj, 3)

        # clip torque
        tau_n: np.ndarray = res.t_local if has_root else res.t_local[:, 1:, :]  # (batch, nj, 3)
        tau_n: np.ndarray = self.smooth_torque(tau_n)
        if use_torque_lim:
            tor_lim: np.ndarray = self.joint_info.torque_limit.reshape((1, -1, 1))  # (1, nj, 1)
            tau_n: np.ndarray = np.clip(tau_n, -tor_lim, tor_lim)
        self.tau_result: np.ndarray = tau_n[:self.ref_end()]

        # get local angular velocity
        dq_n: np.ndarray = self.target.locally.angvel[self.ref_start():]  # (batch, nj, 3)
        if use_stable_pd and False:
            # set velocity of last frame = velocity of the second to last.
            # inverse dynamics is not accurate, so it's OK.
            dq_n: np.ndarray = np.concatenate([dq_n[1:, ...], dq_n[-1, None, ...]], axis=0)  # (batch, nj, 3)

        # calc desired value in PD/Stable PD Controller
        invdyn_offset = kps_inv * (tau_n + kds * dq_n)  # (batch, nj, 3)
        invdyn_offset_rot: Rotation = Rotation.from_rotvec(invdyn_offset.reshape((-1, 3)))
        res_q: np.ndarray = (invdyn_offset_rot * Rotation(q_n.reshape(-1, 4))).as_quat().reshape(q_n.shape)

        # visualize in axis angle format..
        if self.visualize_smooth_torque:
            import matplotlib.pyplot as plt
            qn_rotvec: np.ndarray = Rotation(q_n.reshape(-1, 4)).as_rotvec().reshape(invdyn_offset.shape)
            for joint_idx in range(invdyn_offset.shape[1]):
                fig = plt.figure()
                for dim in range(3):
                    sub_plot = fig.add_subplot(311 + dim)
                    sub_plot.plot(invdyn_offset[:, joint_idx, dim], label="tau offset")
                    sub_plot.plot(qn_rotvec[:, joint_idx, dim], label="raw angle")
                    sub_plot.legend()

                    sub_plot.set_title(["x", "y", "z"][dim])
                plt.suptitle(f"joint {joint_idx}")
                plt.show()
            exit(0)

        return res_q

    def convert_to_target_pose(
        self,
        res_q: np.ndarray,
        bvh_hierarchy: Optional[MotionData] = None,
        character_to_bvh: Optional[np.ndarray] = None
        ) -> TargetPose:
        """
        param:
        res_q:
        bvh_hierarchy:
        Convert inverse dynamics local reference motion to TargetPose type

        return: TargetPose
        """
        assert res_q.shape[-1] == 4
        bvh_hierarchy: Optional[MotionData] = self.bvh if bvh_hierarchy is None else bvh_hierarchy
        if character_to_bvh is None:
            character_to_bvh = self.loader.character_to_bvh

        bvh: MotionData = bvh_hierarchy.get_hierarchy(copy=True)
        bvh._num_frames = res_q.shape[0]
        bvh._joint_position = None
        bvh._joint_translation = np.zeros((bvh.num_frames, bvh.num_joints, 3))
        bvh._joint_rotation = MathHelper.unit_quat_arr((bvh.num_frames, bvh.num_joints, 4))
        bvh._joint_orientation = None

        if not self.character.joint_info.has_root:
            bvh._joint_rotation[:, character_to_bvh, :] = res_q[:, :, :]
            bvh._joint_rotation[:, 0, :] = self.target.root.quat[:, :]
            bvh._joint_translation[:, 0, :] = self.target.root.pos
            bvh.recompute_joint_global_info()
        else:
            raise NotImplementedError

        # run hack on knee joints...
        # lknee_index = bvh.joint_names.index("lKnee")
        # rknee_index = bvh.joint_names.index("rKnee")
        # add_offset_on_knee(bvh._joint_rotation, bvh._joint_rotation, lknee_index, 0.1)
        # add_offset_on_knee(bvh._joint_rotation, bvh._joint_rotation, rknee_index, 0.1)

        # output bvh file..
        if self.output_bvh_fname is not None:
            if not os.path.exists(os.path.dirname(self.output_bvh_fname)):
                os.makedirs(os.path.dirname(self.output_bvh_fname), exist_ok=True)
            BVHLoader.save(bvh, self.output_bvh_fname)
            print(f"invdyn: output to {self.output_bvh_fname}")

        # convert to target..here should NOT smooth
        # notice: we need only compute joint local quaternion for inverse dynamics target pose..
        res = TargetPose()
        res.smoothed = True
        res.num_frames = res_q.shape[0]
        res.locally.quat = res_q
        res.fps = self.scene.sim_fps

        return res.sub_seq(0, self.ref_end(), is_copy=False).to_continuous()

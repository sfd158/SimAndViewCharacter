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

from enum import IntEnum
import logging
import numpy as np
import operator
from typing import Dict, Any, Optional, List, Tuple, Union


class SamconLossMode(IntEnum):
    add = 0
    multiply = 1
    exp_add = 2


class SamconLossBase:
    """
    Calc Samcon loss
    """

    def __init__(self, conf: Dict[str, Any]):
        # Weight should load from file.
        self.loss_mode: Optional[SamconLossMode] = SamconLossMode.add
        self.loss_operator = operator.add
        self.loss_list: List[Dict[str, float]] = conf["loss_list"]
        self.w_hack_contact = 1e2

        self.w_pose: float = 0.0
        self.w_global_root_pos: float = 0.0
        self.w_global_root_quat: float = 0.0
        self.w_root: float = 0.0
        self.w_end_effector: float = 0.0
        self.w_balance: float = 0.0
        self.w_joint_energy: float = 0.0

        self.dv_coef: float = 0.1

        self.com_coef: float = 0.0
        self.angular_momentum_coef: float = 0.0

        self.w_global_joint_pos_coef: float = 0.0
        self.w_facing_joint_pos_coef: float = 0.0

        self.w_global_com_coef: float = 0.0

        # To compute 2d joint loss, we must know the camera param..
        self.w_camera_joint_pos_2d_coef: float = 0.0

        # project the character on 2d camera plane, and compute com in 2d space
        self.w_camera_com_2d_coef: float = 0.0

        self.energy_window: Optional[np.ndarray] = None
        self.weight_normalize: float = None
        # self.un_normalize_loss_dict: Optional[Dict[str, float]] = None
        self.set_loss_list_index(0)

    def only_use_loss_2d(self) -> Dict[str, float]:
        new_dict = self.set_all_weight_to_zero(False)
        new_dict["w_camera_joint_pos_2d_coef"] = self.w_camera_joint_pos_2d_coef
        self.update_loss_weight(new_dict)
        return new_dict

    def set_all_weight_to_zero(self, do_update: bool = True) -> Dict[str, float]:
        export_dict = self.export_loss_dict()
        new_dict = {key: 0.0 for key in export_dict.keys()}
        new_dict["dv_coef"] = self.dv_coef
        if do_update:
            self.update_loss_weight(new_dict)
        return new_dict

    def loss_weight_list(self):
        return [self.w_pose, self.w_global_root_pos, self.w_global_root_quat,
                self.w_root, self.w_end_effector, self.w_balance, self.com_coef, self.angular_momentum_coef]

    def export_loss_dict(self):
        ret = dict(
            w_hack_contact = self.w_hack_contact,
            w_pose = self.w_pose,
            w_global_root_pos = self.w_global_root_pos,
            w_global_root_quat = self.w_global_root_quat,
            w_root = self.w_root,
            w_end_effector = self.w_end_effector,
            w_balance = self.w_balance,
            w_joint_energy = self.w_joint_energy,
            dv_coef = self.dv_coef,
            com_coef = self.com_coef,
            angular_momentum_coef = self.angular_momentum_coef,
            w_global_joint_pos_coef = self.w_global_joint_pos_coef,
            w_facing_joint_pos_coef = self.w_facing_joint_pos_coef,
            w_global_com_coef = self.w_global_com_coef,
            w_camera_joint_pos_2d_coef = self.w_camera_joint_pos_2d_coef,
            w_camera_com_2d_coef = self.w_camera_com_2d_coef
        )
        return ret

    def normalize_loss_weight(self):
        loss_dict = self.export_loss_dict()
        self.weight_normalize = sum([value for key, value in loss_dict.items()])
        for key in loss_dict.keys():
            loss_dict[key] /= self.weight_normalize
        self.update_loss_weight(loss_dict)

    @staticmethod
    def exp_add_loss_func(result, loss_item):
        return result + np.exp(-loss_item)

    def set_loss_mode(self, mode: Union[str, SamconLossMode, None]):
        if mode is None:
            return
        if isinstance(mode, str):
            mode: SamconLossMode = SamconLossMode[mode]

        self.loss_mode = mode
        if mode == SamconLossMode.add:
            self.loss_operator = operator.add
        elif mode == SamconLossMode.multiply:
            self.loss_operator = operator.mul
        elif mode == SamconLossMode.exp_add:
            self.loss_operator = operator.add
        else:
            raise NotImplementedError

    def logging_weight(self):
        loss_dict = self.export_loss_dict()
        log_str = "\n".join([f"{key} = {value:.3f}" for key, value in loss_dict.items()])
        logging.info(log_str)

    def update_loss_weight(self, conf_loss: Dict[str, float]):
        loss_mode = conf_loss.get("loss_mode")
        self.set_loss_mode(loss_mode)

        self.w_hack_contact: float = conf_loss.get("w_hack_contact", self.w_hack_contact)

        self.w_pose: float = conf_loss.get("w_pose", self.w_pose)
        self.w_global_root_pos: float = conf_loss.get("w_global_root_pos", self.w_global_root_pos)
        self.w_global_root_quat: float = conf_loss.get("w_global_root_quat", self.w_global_root_quat)

        self.w_root: float = conf_loss.get("w_root", self.w_root)
        self.w_end_effector: float = conf_loss.get("w_end_effector", self.w_end_effector)
        self.w_balance: float = conf_loss.get("w_balance", self.w_balance)
        self.w_joint_energy: float = conf_loss.get("w_joint_energy", self.w_joint_energy)

        self.dv_coef: float = conf_loss.get("dv_coef", self.dv_coef)

        self.com_coef: float = conf_loss.get("w_com", self.com_coef)
        self.angular_momentum_coef: float = conf_loss.get("w_ang_momentum", self.angular_momentum_coef)

        self.w_global_joint_pos_coef = conf_loss.get("w_global_joint_pos_coef", self.w_global_joint_pos_coef)
        self.w_facing_joint_pos_coef: float = conf_loss.get("w_facing_joint_pos_coef", self.w_facing_joint_pos_coef)

        self.w_global_com_coef: float = conf_loss.get("w_global_com_coef", self.w_global_com_coef)

        self.w_camera_joint_pos_2d_coef: float = conf_loss.get("w_camera_joint_pos_2d_coef", self.w_camera_joint_pos_2d_coef)
        self.w_camera_com_2d_coef: float = conf_loss.get("w_camera_com_2d_coef", self.w_camera_com_2d_coef)

    def set_loss_list_index(self, index: int):
        index = min(index, len(self.loss_list) - 1)
        for i in range(0, index + 1):
            self.update_loss_weight(self.loss_list[i])
        if self.loss_mode == SamconLossMode.exp_add:
            self.normalize_loss_weight()
        self.logging_weight()

    def pose_loss(self, *args):
        raise NotImplementedError

    def global_root_pos_loss(self, *args):
        raise NotImplementedError

    def global_root_quat_loss(self, *args):
        raise NotImplementedError

    def root_loss(self, *args):
        raise NotImplementedError

    def end_effector_loss(self, *args):
        raise NotImplementedError

    def balance_loss(self, *args):
        raise NotImplementedError

    def joint_energy_loss(self, *args):
        raise NotImplementedError

    def hinge_limit_loss(self, *args):
        raise NotImplementedError

    def com_loss(self, *args):
        raise NotImplementedError

    def angular_momentum_loss(self, *args):
        raise NotImplementedError

    def global_joint_pos_loss(self, *args):
        raise NotImplementedError

    def facing_joint_pos_loss(self, *args):
        raise NotImplementedError

    def global_com_loss(self, *args):
        raise NotImplementedError

    def camera_joint_loss_2d(self, *args):
        raise NotImplementedError

    def camera_com_2d_loss(self, *args):
        raise NotImplementedError

    def loss(self, *args):
        # TODO: convert to list
        if self.loss_mode == SamconLossMode.add:
            res = 0.0
        elif self.loss_mode == SamconLossMode.multiply:
            res = 1.0
        elif self.loss_mode == SamconLossMode.exp_add:
            res = 0.0
        else:
            raise NotImplementedError

        if self.w_pose > 0:
            res = self.loss_operator(res, self.w_pose * self.pose_loss(*args))
        if self.w_global_root_pos > 0:
            res = self.loss_operator(res, self.w_global_root_pos * self.global_root_pos_loss(*args))
        if self.w_global_root_quat > 0:
            res = self.loss_operator(res, self.w_global_root_quat * self.global_root_quat_loss(*args))
        if self.w_root > 0:
            res = self.loss_operator(res, self.w_root * self.root_loss(*args))
        if self.w_end_effector > 0:
            res = self.loss_operator(res, self.w_end_effector * self.end_effector_loss(*args))
        if self.w_balance > 0:
            res = self.loss_operator(res, self.w_balance * self.balance_loss(*args))
        if self.w_joint_energy > 0:
            res = self.loss_operator(res, self.w_joint_energy * self.joint_energy_loss(*args))
        if self.com_coef > 0:
            res = self.loss_operator(res, self.com_coef * self.com_loss(*args))
        if self.angular_momentum_coef > 0:
            res = self.loss_operator(res, self.angular_momentum_coef * self.angular_momentum_loss(*args))
        if self.w_global_joint_pos_coef > 0:
            res = self.loss_operator(res, self.w_global_joint_pos_coef * self.global_joint_pos_loss(*args))
        if self.w_facing_joint_pos_coef > 0:
            res = self.loss_operator(res, self.w_facing_joint_pos_coef * self.facing_joint_pos_loss(*args))
        if self.w_global_com_coef > 0:
            res = self.loss_operator(res, self.w_global_com_coef * self.global_com_loss(*args))
        if self.w_camera_joint_pos_2d_coef > 0:
            res = self.loss_operator(res, self.w_camera_joint_pos_2d_coef * self.camera_joint_loss_2d(*args))
        if self.w_camera_com_2d_coef > 0:
            res = self.loss_operator(res, self.w_camera_com_2d_coef * self.camera_com_2d_loss(*args))
        return res

    def loss_debug(self, *args) -> Tuple[float, Dict[str, float]]:
        if self.loss_mode == SamconLossMode.add:
            res = 0.0
        elif self.loss_mode == SamconLossMode.multiply:
            res = 1.0
        elif self.loss_mode == SamconLossMode.exp_add:
            res = 0.0
        else:
            raise NotImplementedError

        res_dict: Dict[str, float] = {}
        if self.w_pose > 0:
            pose_val = self.w_pose * self.pose_loss(*args)
            res_dict["pose"] = pose_val
            res = self.loss_operator(res, pose_val)
        if self.w_global_root_pos > 0:
            root_pos_val = self.w_global_root_pos * self.global_root_pos_loss(*args)
            res_dict["global_root_pos"] = root_pos_val
            res = self.loss_operator(res, root_pos_val)
        if self.w_global_root_quat > 0:
            root_quat_val = self.w_global_root_quat * self.global_root_quat_loss(*args)
            res_dict["global_root_quat"] = root_quat_val
            res = self.loss_operator(res, root_quat_val)
        if self.w_root > 0:
            root_val = self.w_root * self.root_loss(*args)
            res_dict["root"] = root_val
            res = self.loss_operator(res, root_val)
        if self.w_end_effector > 0:
            end_effector_val = self.w_end_effector * self.end_effector_loss(*args)
            res_dict["end_effector"] = end_effector_val
            res = self.loss_operator(res, end_effector_val)
        if self.w_balance > 0:
            w_balance_val = self.w_balance * self.balance_loss(*args)
            res_dict["w_balance"] = w_balance_val
            res = self.loss_operator(res, w_balance_val)
        if self.com_coef > 0:
            com_val = self.com_coef * self.com_loss(*args)
            res_dict["com"] = com_val
            res = self.loss_operator(res, com_val)
        if self.angular_momentum_coef > 0:
            ang_mom_val = self.angular_momentum_coef * self.angular_momentum_loss(*args)
            res_dict["ang_mom"] = ang_mom_val
            res = self.loss_operator(res, ang_mom_val)
        if self.w_global_joint_pos_coef > 0:
            global_joint_3d_val = self.w_global_joint_pos_coef * self.global_joint_pos_loss(*args)
            res_dict["global_joint_pos_3d"] = global_joint_3d_val
            res = self.loss_operator(res, global_joint_3d_val)
        if self.w_facing_joint_pos_coef > 0:
            local_joint_3d_val = self.w_facing_joint_pos_coef * self.facing_joint_pos_loss(*args)
            res_dict["facing_joint_pos_3d"] = local_joint_3d_val
            res = self.loss_operator(res, local_joint_3d_val)
        if self.w_global_com_coef > 0:
            global_com = self.w_global_com_coef * self.global_com_loss(*args)
            res_dict["global_com"] = global_com
            res = self.loss_operator(res, global_com)
        if self.w_camera_joint_pos_2d_coef > 0:
            joint_2d_val = self.w_camera_com_2d_coef * self.camera_joint_loss_2d(*args)
            res_dict["joint_2d_val"] = joint_2d_val
            res = self.loss_operator(res, joint_2d_val)
        if self.w_camera_com_2d_coef > 0:
            camera_com_2d_val = self.w_camera_com_2d_coef * self.camera_com_2d_loss(*args)
            res_dict["camera_com_2d"] = camera_com_2d_val
            res = self.loss_operator(res, camera_com_2d_val)

        return res, res_dict

    def __call__(self, *args):
        return self.loss(*args)

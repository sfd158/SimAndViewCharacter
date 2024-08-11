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

from VclSimuBackend.Common.MathHelper import MathHelper
import copy
import numpy as np
import os
from typing import List, Optional

from ..CharacterWrapper import CharacterWrapper, ODECharacter
from ...pymotionlib.MotionData import MotionData
from ...pymotionlib.ExtEndSite import save_ext_end_site
from ...pymotionlib import BVHLoader


class CharacterTOBVH(CharacterWrapper):
    def __init__(self, character: ODECharacter, sim_fps: int = 120):
        super(CharacterTOBVH, self).__init__(character)
        self.motion = MotionData()
        self.buffer: List[MotionData] = []
        self.end_site_info = None
        self.motion_backup: Optional[MotionData] = None
        self.sim_fps = int(sim_fps)

    @property
    def root_idx(self):
        return self.character.joint_info.root_idx

    def deepcopy(self):
        result = self.__class__(self.character)
        result.motion = copy.deepcopy(self.motion)
        result.end_site_info = copy.deepcopy(self.end_site_info)
        return result

    def build_hierarchy_base(self):
        self.motion._fps = self.sim_fps
        self.motion._num_frames = 0

    def bvh_hierarchy_no_root(self):
        if self.character.joint_info.has_root:
            raise ValueError("There should be no root joint.")

        old_state = self.character.save()
        self.character.load_init_state()
        self.build_hierarchy_base()
        self.motion._num_joints = len(self.joints) + 1

        # When insert virtual root joint at the front, index of other joints add by 1
        self.motion._skeleton_joint_parents = [-1] + (np.array(self.character.joint_info.pa_joint_id) + 1).tolist()
        self.motion._skeleton_joints = ["RootJoint"] + self.character.joint_info.joint_names()

        localqs, offset = self.character.joint_info.get_relative_local_pos()
        self.motion._skeleton_joint_offsets = np.concatenate([np.zeros((1, 3)), offset])

        self.end_site_info = dict(
            [(pa_jidx + 1, self.character.end_joint.jtoj_init_local_pos[idx])
            for idx, pa_jidx in enumerate(self.character.end_joint.pa_joint_id)]
        )

        self.motion_backup = copy.deepcopy(self.motion)
        self.character.load(old_state)

    def build_hierarchy_with_root(self):
        assert self.character.joint_info.has_root
        old_state = self.character.save()
        self.character.load_init_state()
        self.build_hierarchy_base()
        self.motion._num_joints = len(self.joints)
        self.motion._skeleton_joint_parents = copy.deepcopy(self.character.joint_info.pa_joint_id)
        self.motion._skeleton_joints = self.character.joint_info.joint_names()
        localqs, offset = self.character.joint_info.get_relative_local_pos()
        self.motion._skeleton_joint_offsets = offset
        self.end_site_info = dict(
            [(pa_jidx, self.character.end_joint.jtoj_init_local_pos[idx])
            for idx, pa_jidx in enumerate(self.character.end_joint.pa_joint_id)]
        )
        self.motion_backup = copy.deepcopy(self.motion)
        self.character.load(old_state)

    def build_hierarchy(self):
        if self.character.joint_info.has_root:
            self.build_hierarchy_with_root()
        else:
            self.bvh_hierarchy_no_root()

        return self

    def bvh_append_with_root(self):
        assert self.joint_info.has_root
        translation = np.zeros((1, len(self.joints), 3))
        translation[0, 0, :] = self.joint_info.root_joint.getAnchorNumpy()
        localqs: np.ndarray = self.joint_info.get_local_q()
        rotation = localqs[None, ...]

        # Append to the mocap data
        self.motion.append_trans_rotation(translation, rotation)
        return self.motion

    def bvh_append_no_root(self):
        # assume that root body's index is 0.
        assert self.body_info.root_body_id == 0
        root_pos = self.character.root_body.PositionNumpy
        translation = np.concatenate([root_pos[np.newaxis, :], np.zeros((len(self.joints), 3))], axis=0).reshape((1, -1, 3))

        # joint_rotation
        root_rot: np.ndarray = self.character.root_body.getQuaternionScipy()
        localqs: np.ndarray = self.joint_info.get_local_q()
        rotation = np.concatenate([root_rot.reshape((1, 4)), localqs], axis=0)[None, :, :]

        # Append to the mocap data
        self.motion.append_trans_rotation(translation, rotation)
        return self.motion

    def append_with_root_to_buffer(self):
        motion_back = copy.deepcopy(self.motion)
        self.bvh_append_with_root()
        self.buffer.append(self.motion)
        self.motion = motion_back

    def append_no_root_to_buffer(self):
        motion_back = copy.deepcopy(self.motion)
        self.bvh_append_no_root()
        self.buffer.append(self.motion)
        self.motion = motion_back
        return self

    def insert_end_site(self, motion: Optional[MotionData] = None):
        """
        insert end site to self.motion..
        """
        if motion is None:
            motion = self.motion

        end_site_list = [[key, value] for key, value in self.end_site_info.items()]
        end_site_list.sort(key=lambda x: x[0])
        motion._end_sites = []
        parent_res = motion.joint_parents_idx
        jnames = motion.joint_names
        joffs: List[np.ndarray] = [i for i in motion.joint_offsets]
        jtrans: List[np.ndarray] = [motion.joint_translation[:, i, :] for i in range(motion.num_joints)]
        jrots: List[np.ndarray] = [motion.joint_rotation[:, i, :] for i in range(motion.num_joints)]
        trans_zero: np.ndarray = np.zeros_like(jtrans[0])
        rots_zero: np.ndarray = MathHelper.unit_quat_arr(jrots[0].shape)

        for enum_idx, end_node in enumerate(end_site_list):
            end_idx, end_off = end_node
            end_idx = int(end_idx + enum_idx + 1)
            motion.end_sites.append(end_idx)
            after_list = [end_idx - 1]
            jnames.insert(end_idx, jnames[end_idx - 1] + "_end")
            joffs.insert(end_idx, end_off)
            jtrans.insert(end_idx, trans_zero)
            jrots.insert(end_idx, rots_zero)

            for parent in parent_res[end_idx:]:
                if parent < end_idx:
                    after_list.append(parent)
                else:
                    after_list.append(parent + 1)

            parent_res = parent_res[:end_idx] + after_list


        children = [[] for _ in range(len(parent_res))]
        for i, p in enumerate(parent_res[1:]):
            children[p].append(i + 1)
        motion._num_joints = len(parent_res)
        motion._joint_translation = np.concatenate([i[:, None, :] for i in jtrans], axis=1)
        motion._joint_rotation = np.concatenate([i[:, None, :] for i in jrots], axis=1)
        motion._joint_position = None # np.zeros_like(motion._joint_translation)
        motion._joint_orientation = None
        motion._skeleton_joint_offsets = np.concatenate([i[None, ...] for i in joffs], axis=0)
        motion._skeleton_joint_parents = parent_res

        return motion

    def merge_buf(self):
        if self.buffer:
            self.motion._joint_rotation = np.concatenate([motion._joint_rotation for motion in self.buffer], axis=0)
            self.motion._joint_translation = np.concatenate([motion._joint_translation for motion in self.buffer], axis=0)
            self.motion._num_frames = len(self.buffer)
            self.buffer.clear()

        return self.motion

    def ret_merge_buf(self) -> MotionData:
        self.merge_buf()
        if self.end_site_info:
            self.insert_end_site()
        ret_motion: MotionData = self.motion
        self.motion = copy.deepcopy(self.motion_backup)

        return ret_motion

    def forward_kinematics(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_local_quat: np.ndarray
    ) -> MotionData:
        assert root_pos.shape[0] == root_quat.shape[0] == joint_local_quat.shape[0]
        assert root_pos.shape[-1] == 3 and root_quat.shape[-1] == 4 and joint_local_quat.shape[-1] == 4
        # make sure joint order are same..
        num_frame: int = root_pos.shape[0]
        ret_motion = self.motion.get_hierarchy(True)
        assert not ret_motion._end_sites
        ret_motion._num_frames = num_frame
        ret_motion._joint_translation = np.zeros((num_frame, ret_motion.num_joints, 3))
        ret_motion._joint_translation[:, 0, :] = root_pos[:, :]
        ret_motion._joint_rotation = MathHelper.unit_quat_arr((num_frame, ret_motion.num_joints, 4))
        ret_motion._joint_rotation[:, 0, :] = root_quat[:, :]
        ret_motion._joint_rotation[:, 1:, :] = joint_local_quat[:, :, :]
        ret_motion._joint_rotation = np.ascontiguousarray(ret_motion._joint_rotation)

        # recompute global info
        ret_motion.recompute_joint_global_info()

        return ret_motion

    def to_file(self, fname: str = "test.bvh", print_info=False) -> MotionData:
        self.merge_buf()
        if self.end_site_info:
            self.insert_end_site()
        if fname and not os.path.isdir(fname):
            try:
                BVHLoader.save(self.motion, fname)
                if print_info:
                    print(f"Write BVH file to {fname}, with frame = {self.motion.num_frames}, fps = {self.motion.fps}")
            except IOError as arg:
                print(arg)

        ret_motion: MotionData = self.motion
        self.motion = copy.deepcopy(self.motion_backup)

        return ret_motion

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
from mpi4py import MPI
import numpy as np
import os
from tqdm import tqdm

from typing import Callable, List, Optional, Dict, Tuple, Union
from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter, TargetPose
from VclSimuBackend.Samcon.OptimalGait.ContactWithKinematic import ContactLabelExtractor
from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib import BVHLoader


fdir = os.path.abspath(os.path.dirname(__file__))
comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()


class ContactLabelState:

    @staticmethod
    def list_from_hash(hash_value_: int) -> List[int]:
        hash_value: int = hash_value_
        result: List[int] = []
        index: int = 0
        while hash_value > 0:
            if hash_value & 1 == 1:
                result.append(index)
            index += 1
            hash_value //= 2
        return result

    @staticmethod
    def list_to_hash(labels: List[int]) -> int:
        result: int = 0
        for node in labels:
            result += 1 << node
        return result


class ContactPlan:
    """
    Here we can pre-compute contacts for all of the bodies.
    """
    add_contact_height_eps: float = 0.1  # This will be modified by arg parser.
    com_acc_ratio: float = 0.9

    def __init__(self, scene: ODEScene, character: ODECharacter, full_contact: DiffContactInfo, callback: Callable, frame: int) -> None:
        self.callback = callback
        self.frame: int = frame
        self.contact_map: Dict[int, float] = {}
        self.sorted_contact_map: Optional[List[Tuple[int, float]]] = None

        self.character: ODECharacter = character
        self.num_body: int = len(self.character.bodies)
        self.g: float = scene.gravity_numpy[1]
        self.total_body_index: List[int] = list(range(self.num_body))

        # here we can divide the contact into several parts..
        self.full_contact: Optional[DiffContactInfo] = full_contact
        self.divide_contacts: List[DiffContactInfo] = full_contact.divide_by_body0()
        self.body_min_height: Optional[np.ndarray] = np.array([np.min(node.contact_pos[..., 1, 0].numpy()) for node in self.divide_contacts])

        self.body_parents: List[int] = character.body_info.parent.copy()
        self.body_children: List[List[int]] = copy.deepcopy(character.body_info.children)

    def __repr__(self) -> str:
        return f"{self.__name__} at frame {self.frame}, memory = {id(self)}"

    def clear(self):
        self.contact_map.clear()

    @staticmethod
    def build_by_mocap(
        mocap: Union[str, MotionData, TargetPose],
        scene: ODEScene,
        character: ODECharacter,
        forward_callback: Callable,
        contact_extract: Optional[ContactLabelExtractor] = None,
        # start_index: Optional[int] = None,
        # end_index: Optional[int] = None,
        frame_offset: Optional[int] = 0
    ):
        if contact_extract is None:
            contact_extract = ContactLabelExtractor(scene, character)
        if isinstance(mocap, str):
            mocap: MotionData = BVHLoader.load(mocap)
        if isinstance(mocap, MotionData):
            mocap = mocap.resample(scene.sim_fps)
            mocap: TargetPose = BVHToTargetBase(mocap, mocap.fps, character).init_target()
        set_tar: SetTargetToCharacter = SetTargetToCharacter(character, mocap)
        result: List[Optional[ContactPlan]] = [None for _ in range(frame_offset)]
        full_label: np.ndarray = np.ones(len(character.bodies))
        # if start_index is None:
        #    start_index: Optional[int] = 0
        # if end_index is None:
        #    end_index: Optional[int] = mocap.num_frames
        if comm_rank == 0:
            pbar = tqdm(None, "Gen Contact", total=mocap.num_frames)
        else:
            pbar = None
        for frame in range(mocap.num_frames):
            if pbar is not None:
                pbar.update()
            set_tar.set_character_byframe(frame)
            body_pos, body_quat = character.body_info.get_body_pos(), character.body_info.get_body_quat()
            ret_index_arr, ret_pos_arr, ret_label_arr = contact_extract.compute_contact_by_label(body_pos, body_quat, full_label, 0.0, None)
            diff_contact: DiffContactInfo = contact_extract.convert_to_diff_contact_single(ret_index_arr, ret_pos_arr, ret_label_arr)
            contact_plan = ContactPlan(scene, character, diff_contact, forward_callback, frame + frame_offset)
            result.append(contact_plan)

        if pbar is not None:
            pbar.close()
        return result

    def compute_loss_by_contact_label(self, label: List[int]) -> float:
        sub_contact: DiffContactInfo = self.to_subcontact(label)
        return self.callback(self.frame, sub_contact)

    def handle_contact_mess(self, contact_mess: Union[int, List[int]]) -> Tuple[int, List[int]]:
        if isinstance(contact_mess, int):
            contact_hash: int = contact_mess
            contact_label: List[int] = ContactLabelState.list_from_hash(contact_hash)
        elif isinstance(contact_mess, List):
            contact_hash: int = ContactLabelState.list_to_hash(contact_mess)
            contact_label: List[int] = contact_mess
        else:
            raise ValueError
        if contact_hash not in self.contact_map:
            self.contact_map[contact_hash] = self.compute_loss_by_contact_label(contact_label)

        return contact_hash, contact_label

    def add_contact(self, contact_mess: Union[int, List[int]]):
        """
        create contact at minimal position
        """
        # find the body with minimal position..
        contact_hash, contact_label = self.handle_contact_mess(contact_mess)
        if len(contact_label) == self.num_body:
            return self.contact_map[contact_hash], contact_hash
        remain_label: List[int] = list(set(self.total_body_index).difference(contact_label))
        min_h: float = np.min(self.body_min_height[remain_label])
        if min_h >= self.add_contact_height_eps and False:
            return self.contact_map[contact_hash], contact_hash
        else:
            min_index: int = int(np.where(self.body_min_height == min_h)[0][0])
            new_hash: int = contact_hash + (1 << min_index)
            new_label: List[int] = contact_label + [min_index]
            new_label.sort()
            self.contact_map[new_hash] = self.compute_loss_by_contact_label(new_label)
            return self.contact_map[new_hash], new_hash

    def remove_contact(self, contact_mess: Union[int, List[int]], com_acc: float) -> Tuple[float, int]:
        """
        remove one contact
        """
        contact_hash, contact_label = self.handle_contact_mess(contact_mess)
        result_map: Dict[int, float] = {}
        for node_index, node in enumerate(contact_label):  # remove the node..
            node_hash: int = contact_hash - (1 << node)
            if node_hash == 0 and com_acc > self.com_acc_ratio * self.g:  # avoid no collision
                continue
            if node_hash not in self.contact_map:
                # evaluate node value..
                node_label: List[int] = contact_label.copy()
                del node_label[node_index]
                # if len(node_label) == 0:
                #    print("node label is None", flush=True)
                self.contact_map[node_hash] = self.compute_loss_by_contact_label(node_label)
            result_map[node_hash] = self.contact_map[node_hash]
        if len(result_map) > 0:
            min_result: Tuple[float, int] = min(zip(result_map.values(), result_map.keys()))
        else:
            min_result = self.contact_map[contact_hash], contact_hash
        return min_result

    def move_contact(self, contact_mess: Union[int, List[int]]) -> Tuple[float, int]:
        """
        Move one contact to adjacent position..
        """
        # Which contact point should be moved..?
        contact_hash, contact_label = self.handle_contact_mess(contact_mess)
        contact_flags: np.ndarray = np.zeros(self.num_body, dtype=np.int32)
        contact_flags[contact_label] = 1
        result_map: Dict[int, float] = {}
        for node_index, node in enumerate(contact_label):
            parent = self.body_parents[node]
            children = self.body_children[node]
            near_body = children.copy()
            if parent != -1:
                near_body.append(parent)
            curr_h = self.body_min_height[node]
            removed_contact_label: List[int] = contact_label.copy()
            del removed_contact_label[node_index]
            for near in near_body:
                if contact_flags[near] or self.body_min_height[near] > curr_h:
                    continue
                # move the contact to the near position..we can try all of potential movements..
                new_hash: int = contact_hash - (1 << node) + (1 << near)
                if new_hash in self.contact_map:
                    result_map[new_hash] = self.contact_map[new_hash]
                    continue
                new_contact_label = removed_contact_label.copy()
                new_contact_label.append(near)
                new_contact_label.sort()
                result_map[new_hash] = self.contact_map[new_hash] = self.compute_loss_by_contact_label(new_contact_label)
        if len(result_map) > 0:
            min_result: Tuple[float, int] = min(zip(result_map.values(), result_map.keys()))
        else:
            min_result: Tuple[float, int] = self.contact_map[contact_hash], contact_hash
        return min_result

    def all_operation(self, contact_hash: int, com_acc: float):
        self.add_contact(contact_hash)
        self.remove_contact(contact_hash, com_acc)
        self.move_contact(contact_hash)

    def to_subcontact_by_hash(self, contact_hash: int) -> DiffContactInfo:
        """
        build contact by hash value..
        """
        # 1. convert the hash value to list
        label: List[int] = ContactLabelState.list_from_hash(contact_hash)
        # 2. build subcontact from list
        sub_contact: DiffContactInfo = DiffContactInfo.concat([self.divide_contacts[node] for node in label])
        return sub_contact

    def to_subcontact(self, label: List[int]):
        sub_contact: DiffContactInfo = DiffContactInfo.concat([self.divide_contacts[node] for node in label])
        return sub_contact

    def search_solution(
        self,
        com_acc: float,
        contact_mess: Union[int, List[int]],
        other_contact_mess: Union[int, List[int], None] = None,
        max_depth: int = 4,
        best_index: int = 20
    ) -> Tuple[float, int, List[int], DiffContactInfo]:
        """
        Note: we should make sure that contact should be close to previous contact
        """
        contact_hash, contact_label = self.handle_contact_mess(contact_mess)
        if other_contact_mess is not None:
            _, other_contact_list = self.handle_contact_mess(other_contact_mess)
        else:
            other_contact_list: Optional[List[List[int]]] = None
        for i in range(max_depth):
            curr_list = [(value, key) for key, value in self.contact_map.items()]
            curr_list.sort()
            for node in curr_list[:best_index]:
                self.all_operation(node[1], com_acc)
        # get the best solution here..
        # Note: the result should be close to previous solution..
        if other_contact_list is not None:
            if 0 in self.contact_map:
                if len(other_contact_list) > 2 or com_acc > self.com_acc_ratio * self.g:
                    del self.contact_map[0]

        # here we should judge the case: label == []
        self.sorted_contact_map: List[Tuple[int, float]] = [(key, value) for key, value in self.contact_map.items()]
        self.sorted_contact_map.sort(key=lambda x: x[1])
        best_hash, best_value = self.sorted_contact_map[0]

        # here we should also recompute the best DiffContact
        best_label: List[int] = ContactLabelState.list_from_hash(best_hash)
        sub_contact: DiffContactInfo = self.to_subcontact(best_label)
        return best_value, best_hash, best_label, sub_contact

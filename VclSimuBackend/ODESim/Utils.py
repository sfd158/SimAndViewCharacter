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
from typing import List

from ..pymotionlib import MotionHelper
from ..pymotionlib.MotionData import MotionData
from .CharacterWrapper import ODECharacter, CharacterWrapper


class BVHJointMap(CharacterWrapper):
    def __init__(self, bvh: MotionData, character: ODECharacter):
        super(BVHJointMap, self).__init__(character)
        self.bvh = bvh

        bvh_name_idx = MotionHelper.calc_name_idx(self.bvh)

        # build bvh children list
        self.bvh_children = MotionHelper.calc_children(self.bvh)

        # index is joint index in character, and self.character_to_bvh[]
        self.character_to_bvh = np.array([bvh_name_idx[joint.name] for joint in self.joints])
        self.bvh_to_character: List[int] = [2 * self.bvh_joint_cnt for _ in range(self.bvh_joint_cnt)]
        for character_idx, bvh_idx in enumerate(self.character_to_bvh):
            self.bvh_to_character[bvh_idx] = character_idx

        self.end_to_bvh: List[int] = []
        # self.refine_hinge_rotation()
        # assert silce is available
        joint_names = self.joint_names()

        if False:
            assert np.all(np.array(self.bvh.joint_names)[self.character_to_bvh] == np.array(joint_names))
            for index, node in enumerate(self.bvh_to_character):
                if node < 2 * self.bvh_joint_cnt:
                    assert self.bvh.joint_names[index] == joint_names[node]

        # consider there is no end joints
        if bvh.end_sites:
            for pa_character_idx in self.end_joint.pa_joint_id:
                bvh_pa_idx = self.character_to_bvh[pa_character_idx]
                self.end_to_bvh.append(self.bvh_children[bvh_pa_idx][0])

    @property
    def bvh_joint_cnt(self):
        """
        bvh joint count
        """
        return len(self.bvh.joint_names)

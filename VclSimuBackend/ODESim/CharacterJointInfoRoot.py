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
from VclSimuBackend.ODESim.CharacterWrapper import CharacterWrapper, ODECharacter



class CharacterJointInfoRoot(CharacterWrapper):

    def __init__(self, character: ODECharacter):
        super(CharacterJointInfoRoot, self).__init__(character)

    def get_joint_dof(self) -> np.ndarray:  # it is OK
        dofs = np.zeros(len(self.joints), dtype=np.int32)
        for idx, joint in enumerate(self.joints):
            dofs[idx] = joint.joint_dof
        if self.joint_info.has_root:
            return dofs
        else:
            return np.concatenate([np.array([3], dtype=np.int32), dofs])

    def get_parent_joint_dof(self) -> np.ndarray:
        """
        get parent joint dof for each body
        used in Inverse Dynamic
        return: np.ndarray in shape (num body,)
        """
        dofs: np.ndarray = np.zeros(len(self.bodies), dtype=np.int32)
        for body_idx, body in enumerate(self.bodies):
            pa_joint_idx: int = self.child_body_to_joint[body_idx]
            if pa_joint_idx != -1:  # has parent joint
                dofs[body_idx] = self.joints[pa_joint_idx].joint_dof
            else:  # There is no parent joint
                dofs[body_idx] = 3
        return dofs

    def get_parent_joint_pos(self) -> np.ndarray:
        """
        Get global position of parent joint of each body
        return: np.ndarray in shape
        """
        result = np.zeros((len(self.bodies), 3), dtype=np.float64)
        joint_pos = self.joint_info.get_global_pos1()  # shape == (njoints, 3)
        index = np.asarray(self.child_body_to_joint, dtype=np.int32)
        if not self.joint_info.has_root:
            result[0, :] = self.character.root_body.PositionNumpy
            result[1:, :] = joint_pos[index[1:]]
        else:
            result = joint_pos
            result = result[index]
        # resort joints. result[i] is parent joint position of i-th body
        return np.ascontiguousarray(result)

    def get_parent_joint_euler_order(self) -> List[str]:
        """
        used in Inverse Dynamics
        return List[str] with length {num body}
        """
        euler_order = [None for _ in range(len(self.bodies))]
        for body_idx, body in enumerate(self.bodies):
            pa_joint_idx: int = self.child_body_to_joint[body_idx]
            if pa_joint_idx != -1:  # has parent joint
                euler_order[body_idx] = self.joints[pa_joint_idx].euler_order
            else:  # There is no parent joint
                euler_order[body_idx] = "XYZ"

        return euler_order

    def get_parent_joint_euler_axis(self) -> np.ndarray:
        if False:
            euler_axis = self.joint_info.euler_axis_local
            if not self.joint_info.has_root:
                euler_axis = np.concatenate([np.eye(3)[None, ...], euler_axis], axis=0)
            return np.ascontiguousarray(euler_axis)
        else:
            return np.ascontiguousarray(np.tile(np.eye(3)[None], (len(self.bodies), 1, 1)))

    def get_parent_body_index(self) -> np.ndarray:
        result: np.ndarray = np.zeros(len(self.bodies), dtype=np.int32)
        for body_idx, joint_idx in enumerate(self.child_body_to_joint):
            if joint_idx == -1 or self.joints[joint_idx].body2 is None:
                result[body_idx] = -1
            else:
                result[body_idx] = self.joints[joint_idx].body2.instance_id
                # print(self.bodies[body_idx].name, self.bodies[result[body_idx]].name)
        return result


def test_func():
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
    scene_fname = r"D:\song\desktop\curr-yhy\Data\Misc\world.json"
    scene = JsonSceneLoader().load_from_file(scene_fname)
    test = CharacterJointInfoRoot(scene.character0)
    test.get_parent_body_index()


if __name__ == "__main__":
    test_func()

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
build subset dict..
"""
import numpy as np
import os
from typing import List, Dict

try:
    from ...ODESim.Loader.JsonSceneLoader import JsonSceneLoader
    from ...ODESim.ODECharacter import ODECharacter
    from ...ODESim.ODEScene import ODEScene
except ImportError:
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
    from VclSimuBackend.ODESim.ODECharacter import ODECharacter
    from VclSimuBackend.ODESim.ODEScene import ODEScene

fdir = os.path.dirname(__file__)

stdhuman_names = [
    "pelvis_lowerback",  # 0
    "lowerback_torso",   # 1
    "rHip",              # 2
    "lHip",              # 3
    "rKnee",             # 4
    "lKnee",             # 5
    "rAnkle",            # 6
    "lAnkle",            # 7
    "rToeJoint",         # 8
    "lToeJoint",         # 9
    "torso_head",        # 10
    "rTorso_Clavicle",   # 11
    "lTorso_Clavicle",   # 12
    "rShoulder",         # 13
    "lShoulder",         # 14
    "rElbow",            # 15
    "lElbow",            # 16
    "rWrist",            # 17
    "lWrist"             # 18
]

stdhuman_with_root_names: List[str] = ["RootJoint"] + stdhuman_names

def _build_stdhuman_index():
    return {value: index for index, value in enumerate(stdhuman_names)}

stdhuman_name_dict: Dict[str, int] = _build_stdhuman_index()  # length == 19
stdhuman_with_root_name_dict: Dict[str, int] = {value: index for index, value in enumerate(stdhuman_with_root_names)}  # len == 20

stdhuman_to_unified_name = [
    "torso_head",  # 0
    "lShoulder",   # 1
    "lElbow",      # 2
    "lWrist",      # 3
    "lHip",        # 4
    "lKnee",       # 5
    "lAnkle",      # 6
    "rShoulder",   # 7
    "rElbow",      # 8
    "rWrist",      # 9
    "rHip",        # 10
    "rKnee",       # 11
    "rAnkle",      # 12
]


def _build_stdhuman_to_unified_index() -> List[int]:  # we doesn"t consider root joint here..
    result = [stdhuman_name_dict[node] for node in stdhuman_to_unified_name]
    return result

unified_mirror_dict = {
    "torso_head": "torso_head",
    "lShoulder": "rShoulder",
    "lElbow": "rElbow",
    "lWrist": "rWrist",
    "lHip": "rHip",
    "lKnee": "rKnee",
    "lAnkle": "rAnkle",
    "rShoulder": "lShoulder",
    "rElbow": "lElbow",
    "rWrist": "lWrist",
    "rHip": "lHip",
    "rKnee": "lKnee",
    "rAnkle": "lAnkle",
}

stdhuman_to_unified_index: List[int] = _build_stdhuman_to_unified_index()
stdhuman_with_root_to_unified_index: List[int] = (np.array(stdhuman_to_unified_index) + 1).tolist()

def _build_unified_mirror_index():
    unified_dict = {node: index for index, node in enumerate(stdhuman_to_unified_name)}
    result: np.ndarray = np.zeros(len(unified_dict), np.int32)
    for index, node in enumerate(stdhuman_to_unified_name):
        result[index] = unified_dict[unified_mirror_dict[node]]
        # print(index, node, result[index], stdhuman_to_unified_name[result[index]])
    return result

unified_mirror_index: np.ndarray = _build_unified_mirror_index()

def stdhuman_to_unified(joint_pos: np.ndarray) -> np.ndarray:
    """
    Note: the nose position can be computed by this way:
    1. extract head position directyly
    2. use subset of std human.

    assume there is no root joint
    """
    subset: np.ndarray = joint_pos[..., stdhuman_to_unified_index, :]
    return subset


def test():
    fname: str = os.path.join(fdir, "../../../Tests/CharacterData/Samcon-Human.pickle")
    scene: ODEScene = JsonSceneLoader().load_from_pickle_file(fname)
    character: ODECharacter = scene.character0
    print(character.joint_info.joint_names())


if __name__ == "__main__":
    test()

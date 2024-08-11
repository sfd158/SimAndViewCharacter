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

from deprecated.sphinx import deprecated
from typing import Dict, Any
from . import ServerBase
from ...ODESim.ODEScene import ODEScene
from .ODEToUnity import ODEToUnity
from ...ODESim.UpdateSceneBase import UpdateSceneBase


@deprecated(version="2021.04.22", reason="Please Use Server v2")
class ServerForUnity(ServerBase.ServerBase):
    def __init__(self, scene: ODEScene, update_scene: UpdateSceneBase):
        super(ServerForUnity, self).__init__()
        self.scene = scene
        self.update_scene = update_scene
        self.ode2unity = ODEToUnity(self.scene)

    @property
    def world(self):
        return self.scene.world

    def reset(self):
        self.scene.reset()

    @deprecated(version="2021.04.22", reason="Please Use Server v2")
    def calc(self, mess_dict: Dict[str, Any]):
        self.ode2unity.joint_radius = float(mess_dict["JointRadius"])
        if mess_dict["type"] == "GetHierarchyInfo":
            res = self.ode2unity.get_hierarchy_info()
        elif mess_dict["type"] == "GetUpdateInfo":
            self.update_scene.update(mess_dict)
            res = self.ode2unity.get_update_info()
        else:
            raise NotImplementedError("type %s not supported.")

        return res

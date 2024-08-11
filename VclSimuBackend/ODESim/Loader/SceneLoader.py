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

import json
import os
from deprecated.sphinx import deprecated
from typing import Optional, Dict
from ..ODEScene import ODEScene
from ..Loader.XMLCharacterLoader import XMLCharacterLoader


@deprecated(version="2021.04.24", reason="XML Character loader is deprecated. Please use json character Loader")
class SceneLoader:
    def __init__(self, conf_file: str):
        """
        param:
        conf_file:
        """
        with open(conf_file, "r") as f:
            self.conf = json.load(f)
        conf_dir = os.path.dirname(conf_file)
        for k, v in self.f_conf.items():
            self.f_conf[k] = os.path.join(conf_dir, v)

        self.scene: Optional[ODEScene] = None

    @property
    def f_conf(self) -> Dict[str, str]:
        return self.conf["filename"]

    @deprecated(version="2021.04.24", reason="Please use json scene loader")
    def load(self) -> ODEScene:
        """
        return: ODEScene
        """
        c = self.conf.get("scene")
        # Create Scene
        if c is not None:
            self.scene = ODEScene(c["render_fps"], c["sim_fps"], c["gravity"], c["friction"], c["bounce"],
                                  c["self_collision"])
        else:
            self.scene = ODEScene()

        # Load Character
        character_loader = XMLCharacterLoader(self.scene, self.conf)
        character_loader.load(True, False) # self.conf["character"]["load_rootjoint"])
        if self.conf["scene"]["with_floor"]:
            self.scene.create_floor()
        return self.scene

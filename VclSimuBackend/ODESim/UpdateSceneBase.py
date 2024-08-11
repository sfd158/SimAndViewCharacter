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
import time
from typing import Any, Dict, Optional, List
from .ODEScene import ODEScene
from .ODECharacter import ODECharacter


class UpdateSceneBase:

    def __init__(self, scene: Optional[ODEScene] = None):
        self.scene = scene

        # control signal for each character.
        # Now contains horizonal and vertical
        self.world_signal: Optional[List[Dict]] = None

    @property
    def world(self) -> ode.World:
        return self.scene.world

    @property
    def space(self) -> ode.SpaceBase:
        return self.scene.space

    @property
    def character0(self) -> Optional[ODECharacter]:
        return self.scene.character0

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        """
        update scene at each frame.
        @param mess_dict: message recieved from Unity client (e.g. control signal)
        """
        pass


class SimpleUpdateScene(UpdateSceneBase):
    def __init__(self, scene: ODEScene):
        super(SimpleUpdateScene, self).__init__(scene)
        self.sim_cnt: int = 0
        self.tot_time: float = 0.0

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        start_time = time.time()
        for _ in self.scene.step_range():
            self.scene.simulate_once()  # do collision callback in python
            # self.scene.step_fast_collision()  # do collision detection callback in cython
            self.sim_cnt += 1
        end_time = time.time()

        self.tot_time += end_time - start_time


class SimpleDampedUpdateScene(UpdateSceneBase):
    def __init__(self, scene: ODEScene):
        super(SimpleDampedUpdateScene, self).__init__(scene)

    def update(self, mess_dict: Optional[Dict[str, Any]] = None):
        for _ in self.scene.step_range():
            self.scene.damped_simulate_once()

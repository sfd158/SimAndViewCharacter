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
import numpy as np
from typing import Optional, List, Union


class Environment:
    """
    static geometry in environment
    """
    def __init__(self, world: ode.World, space: ode.SpaceBase):
        self.world: ode.World = world
        self.space: ode.SpaceBase = space
        self.floor: Union[ode.GeomPlane, ode.GeomBox] = None
        self.geoms: List[ode.GeomObject] = []
        # Should save position and rotation of non-placeable Geoms (Geom without Body)

    def __len__(self) -> int:
        if self.geoms is not None:
            return len(self.geoms)
        else:
            return 0

    def set_space(self, space: Optional[ode.SpaceBase]) -> ode.SpaceBase:
        if self.geoms is not None:
            for geom in self.geoms:
                geom.space = space
        self.space = space
        return self.space

    def enable(self):
        if self.geoms is None:
            return
        for geom in self.geoms:
            geom.enable()
        return self

    def disable(self):
        if self.geoms is None:
            return
        for geom in self.geoms:
            geom.disable()
        return self

    def get_floor_in_list(self) -> Optional[ode.GeomObject]:
        """
        floor will be GeomBox or GeomPlane type..
        """
        if self.geoms is None:
            return None
        self.floor: Union[ode.GeomPlane, ode.GeomBox] = None
        for geom in self.geoms:
            if isinstance(geom, ode.GeomPlane):
                self.floor = geom
                break
            elif isinstance(geom, ode.GeomBox):
                if np.mean(geom.LengthNumpy) > 50:
                    self.floor = geom
                    break
        return self.floor

    def create_floor(self, friction=0.8) -> ode.GeomPlane:
        self.floor = ode.GeomPlane(self.space, (0, 1, 0), 0)
        self.floor.name = "Floor"
        self.floor.character_id = -1
        self.floor.friction = friction
        self.geoms.append(self.floor)
        # Maybe there is only one GeomPlane in the Environment...
        # Modify ode source code, to get plane's pos...
        return self.floor

    def clear(self):
        self.floor: Union[ode.GeomPlane, ode.GeomBox] = None
        self.geoms.clear()
        return self

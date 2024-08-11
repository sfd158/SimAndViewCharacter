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
import ModifyODE as ode
from ..ODECharacterInit import ODECharacterInit, ODECharacter, CharacterWrapper, JointInfoInit
import trimesh

class MeshCharacterLoader(CharacterWrapper):
    def __init__(self, world: ode.World(), space: ode.SpaceBase):
        super(MeshCharacterLoader, self).__init__()
        self.character = ODECharacter(world, space)
        self.character_init = ODECharacterInit(self.character)
        self.default_friction: float = 0.8
    
    def load_from_obj(self, obj_path, meshname, volume_scale=1, density_scale=1, inverse_xaxis=True):
        self.character.name = meshname

        pymesh = trimesh.load_mesh(obj_path)
        pymesh.density *= density_scale
        pymesh.apply_scale(volume_scale)

        meshbody = ode.Body(self.world)
        meshData = ode.TriMeshData()
        if inverse_xaxis:
            pymesh.vertices[:, 0] = -pymesh.vertices[:, 0]
            trimesh.repair.fix_inversion(pymesh)
        meshData.build(pymesh.vertices-pymesh.center_mass, pymesh.faces)
        meshGeom = ode.GeomTriMesh(meshData, self.space)
        meshGeom.body = meshbody

        meshMass = ode.Mass()
        meshMass.setParameters(pymesh.mass_properties['mass'], 0.0, 0.0, 0.0, 
            pymesh.moment_inertia[0, 0], pymesh.moment_inertia[1, 1], pymesh.moment_inertia[2, 2], 
            pymesh.moment_inertia[0, 1], pymesh.moment_inertia[0, 2], pymesh.moment_inertia[1, 2])
        meshbody.setMass(meshMass)

        self.character_init.append_body(meshbody, meshMass, meshname, -1)
        self.character_init.init_after_load()
        return self.character
        
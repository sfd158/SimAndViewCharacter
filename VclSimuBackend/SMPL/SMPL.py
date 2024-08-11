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

import os
import numpy as np
import pyvista as pv
import tetgen
from typing import Optional
from ..SMPL.config import smpl_param_size, smpl_joint_size, smpl_vertice_size, smpl_param_range, smpl_parent_list
from ..SMPL.config import smpl_render_color, smpl_render_index, smpl_rho, smpl_tetgen_index, smpl_tetgen_radius
from ..SMPL.config import smpl_export_body_size
fdir = os.path.dirname(__file__)

def coor_tuple(coor: np.ndarray) -> tuple:
    return (coor[0], coor[1], coor[2])

class SMPLModel:

    def dump_array(self, fname):
        array = np.loadtxt(os.path.join(fdir, "SMPLParams", str(fname)), skiprows = 1)
        return array

    def __init__(self, smpl_param : np.ndarray, mass_optimize : bool) -> None:
        self.joints : np.ndarray = self.dump_array('joints.txt')
        self.faces : np.ndarray = self.dump_array('faces.txt')
        self.jregressor : np.ndarray = self.dump_array('jregressor.txt')
        self.jshape_blend : np.ndarray = self.dump_array('jshape_blend.txt')
        self.lbs_weight : np.ndarray = self.dump_array('lbs_weights.txt')
        self.shape_blend : np.ndarray = self.dump_array('shape_blend.txt')
        self.vertices : np.ndarray = self.dump_array('vertices.txt')
        self.mass_optimize : bool = mass_optimize
        
        params = smpl_param
        
        self.vertices = self.vertices + np.dot(self.shape_blend, params).reshape((smpl_vertice_size, 3))
        self.joints = self.joints + np.dot(self.jshape_blend, params).reshape((smpl_joint_size, 3))        
        self.argmax_index = np.argmax(np.absolute(self.lbs_weight), axis = 1)
        #print(self.argmax_index.shape)
        self.J : np.ndarray = self.joints
        self.verts : np.ndarray = self.vertices

    def get_subpart_place(self, index: int) -> np.ndarray:
        result : np.ndarray = np.argwhere(self.argmax_index == index)
        result = result.reshape(-1)
        return result

    def visualize_pyvista_impl(self, plot: pv.Plotter, index: Optional[np.ndarray] = None) -> None:
        if index is None:
            #for i in range(0, smpl_joint_size):
            for i in range(0, smpl_joint_size):
                if i in smpl_render_index:
                    index : np.ndarray = np.argwhere(self.argmax_index == i).reshape(-1)
                    points = self.verts[index]
                    plot.add_mesh(points, color = smpl_render_color[i], point_size = 2)
            # Render Part
        else:
            points = self.verts[index]
            plot.add_mesh(points, color = 'r', point_size = 5)

    def _volume(self, a, b, c, d):
        b = b - a
        c = c - a
        d = d - a
        result = np.dot(np.cross(b, c), d)
        if result < 0:
            result = -result
        return result / 6
    
    def make_manifold(self, v, f, argmax_index, verbose = False):
        try:
            import pymeshfix
        except ImportError:
            raise ImportError('pymeshfix not installed.  Please run:\n'
                              'pip install pymeshfix')

        meshfix = pymeshfix.MeshFix(v, f)
        meshfix.repair(verbose)
        result = np.zeros(meshfix.v.shape[0], dtype=np.int32)
        dist = np.zeros(meshfix.v.shape[0], dtype=np.int32)
        for i in range(0, meshfix.v.shape[0]):
            index : int = np.argmin(np.linalg.norm(v - meshfix.v[i], axis = 1))
            dist[i] = np.linalg.norm(v[index] - meshfix.v[i])
            result[i] = argmax_index[index]
        
        print(np.max(dist))
        return meshfix.v, meshfix.f, result
    
    def initialize_mass(self, plot: Optional[pv.Plotter] = None) -> np.ndarray:
        bone_vertice: np.ndarray = np.zeros(((smpl_joint_size - 1) * 2, 3), dtype = np.float64)
        bone_argmax_index: np.ndarray = np.zeros(((smpl_joint_size - 1) * 2), dtype = np.int64)
        for index in range(1, smpl_joint_size):
            parent_idx: int = smpl_parent_list[index]
            point1: np.ndarray = (2 * self.joints[index] + self.joints[parent_idx]) / 3
            point2: np.ndarray = (self.joints[index] + 2 * self.joints[parent_idx]) / 3

            bone_vertice[index * 2 - 2] = point1
            bone_vertice[index * 2 - 1] = point2
            bone_argmax_index[index * 2 - 2] = smpl_parent_list[index]
            bone_argmax_index[index * 2 - 1] = smpl_parent_list[index]
        
        leninfos = len(smpl_tetgen_index)
        infos = zip(smpl_tetgen_radius, smpl_tetgen_index)
        if not self.mass_optimize:
            leninfos = 2
            infos = zip([0.05, 0.05], [4, 5])
        hip_vertice : np.ndarray = np.zeros((leninfos * 12, 3), dtype = np.float64)
        hip_argmax_index : np.ndarray = np.zeros((leninfos * 12), dtype = np.int32)
        for enum, info in enumerate(infos):
            radius, index = info
            parent_idx : int = smpl_parent_list[index]
            xdim : np.ndarray = self.joints[parent_idx] - self.joints[index]
            xdim = xdim / np.linalg.norm(xdim)
            ydim : np.ndarray = np.array([0.0, 0.0, 1.0])
            ydim = ydim - np.dot(xdim, ydim) * xdim
            ydim = ydim / np.linalg.norm(ydim)
            zdim = np.cross(xdim, ydim)
            for j in range(1, 4):
                point = (j * self.joints[index] + (4 - j) * self.joints[parent_idx]) / 4
                loc : int =  enum * 12 + (j - 1) * 4
                hip_vertice[loc + 0] = point + ydim * radius
                hip_vertice[loc + 1] = point - ydim * radius
                hip_vertice[loc + 2] = point + zdim * radius
                hip_vertice[loc + 3] = point - zdim * radius
                hip_argmax_index[loc + 0] = smpl_parent_list[index]
                hip_argmax_index[loc + 1] = smpl_parent_list[index]
                hip_argmax_index[loc + 2] = smpl_parent_list[index]
                hip_argmax_index[loc + 3] = smpl_parent_list[index]
        
        if plot is not None:
            for i in range(0, bone_vertice.shape[0]):
                circ = pv.Sphere(0.01, bone_vertice[i])
                plot.add_mesh(circ, color="red")
            
            for i in range(0, hip_vertice.shape[0]):
                circ = pv.Sphere(0.01, hip_vertice[i])
                plot.add_mesh(circ, color="red")
        
        """
        tet_argmax_index = np.concatenate([self.argmax_index, bone_argmax_index, hip_argmax_index], axis = 0)
        tet_vertices = np.concatenate([self.vertices, bone_vertice, hip_vertice], axis = 0)
        tet_faces = np.array(self.faces, dtype = np.int32)
        """
        tet_vertices, tet_faces, tet_argmax_index = self.make_manifold(self.vertices, np.array(self.faces, dtype = np.int32), self.argmax_index)
        tet_argmax_index = np.concatenate([tet_argmax_index, bone_argmax_index, hip_argmax_index], axis = 0)
        tet_vertices = np.concatenate([tet_vertices, bone_vertice, hip_vertice], axis = 0)

        for index in [22, 23]: # Fingers
            loc: np.ndarray =  np.argwhere(tet_argmax_index == index).reshape(-1)
            tet_argmax_index[loc] = index - 2
        print(tet_vertices.shape[0], "Vertices in tetgen.")
        tet = tetgen.TetGen(tet_vertices, tet_faces)
        node, elem = tet.tetrahedralize(quality = False)

        self.mass = np.zeros(tet_vertices.shape[0], dtype = np.float64)
        self.center_mass = np.zeros((smpl_export_body_size, 3), dtype = np.float64)
        self.joint_mass = np.zeros(smpl_export_body_size, dtype = np.float64)
        for i in range(elem.shape[0]):
            idx0, idx1, idx2, idx3 = elem[i, 0], elem[i, 1], elem[i, 2], elem[i, 3]
            size = self._volume(tet_vertices[idx0], tet_vertices[idx1], tet_vertices[idx2], tet_vertices[idx3])
            for idx in [idx0, idx1, idx2, idx3]:
                self.mass[idx] = self.mass[idx] + size * smpl_rho / 4
        
        for i in range(smpl_export_body_size):
            index : np.ndarray = np.argwhere(tet_argmax_index == i).reshape(-1)
            self.joint_mass[i] = np.sum(self.mass[index])
            self.center_mass[i] = np.dot(tet_vertices[index].transpose(), self.mass[index]) / self.joint_mass[i]
        
        if plot != None:
            for i in range(smpl_export_body_size):
                circ = pv.Sphere(0.02, self.center_mass[i])
                plot.add_mesh(circ, color = 'purple')

        self.inertia : np.ndarray = np.zeros((smpl_joint_size, 3, 3), dtype = np.float64)
        for j in range(tet_vertices.shape[0]):
            i : int = tet_argmax_index[j]
            offset : np.ndarray = tet_vertices[j] - self.center_mass[i]
            p_inertia : np.ndarray = np.outer(offset, offset)
            p_inertia = np.identity(3) * np.dot(offset, offset) - p_inertia
            self.inertia[i] = self.inertia[i] + p_inertia * self.mass[j]
        
        print("Mass of the human: ", np.sum(self.joint_mass), "Kg")
        # print(self.joint_mass)
        return self.joint_mass, self.center_mass, self.inertia      
        #It seems that there are duplicated points

    def visualize_joint_pyvista_impl(self, plot: pv.Plotter, joint: np.ndarray) -> None:
        for i in range(1, smpl_joint_size):
            line = pv.Line(coor_tuple(joint[i]), coor_tuple(joint[smpl_parent_list[i]]))
            plot.add_mesh(line, color = 'k', line_width = 6)

        for i in range(0, smpl_joint_size):
            sphere = pv.Sphere(radius = 0.02, center = coor_tuple(joint[i]))
            plot.add_mesh(sphere, show_edges = False, color = 'blue')
        
        

# 4 pnt, 6 edge, 4 face
# 8 pnt, 12edge, 6 face
# 6 pnt, 12edge, 8 face
# 12 pnt, 50edge, 20 face,
# 13776 face, 9184 edge, 6890 points
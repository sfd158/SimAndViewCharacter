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
import os
import pickle
import sys
from typing import Dict, Optional, Any, Union, List

from VclSimuBackend.SMPL.config import smpl_hierarchy, smpl_name_list

fdir = os.path.dirname(__file__)
_default_model_path = os.path.join(fdir, "model.pkl")


class SMPLModel:
    def __init__(self, model_path: str = _default_model_path) -> None:
        # load color file
        self.color_conf = np.load(os.path.join(fdir, "tableau_color.npy"))
        self.vert_color: Optional[np.ndarray] = None

        with open(model_path, 'rb') as f:
            params: Dict[str, Any] = pickle.load(f)
            self.J_regressor = params['J_regressor']  # (24, 6890)
            self.weights: np.ndarray = params['weights']  # (6890, 24)
            self.posedirs: np.ndarray = params['posedirs']  # (6890, 3, 207)
            self.v_template: np.ndarray = params['v_template']  # (6890, 3)
            self.shapedirs: np.ndarray = params['shapedirs']  # (6890, 3, 10)
            self.faces: np.ndarray = params['f']  # (13776, 3)
            self.kintree_table: np.ndarray = params['kintree_table']  # (2, 24)

            id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
            self.parent: Dict[int, int] = {
                i: id_to_col[self.kintree_table[0, i]]
                for i in range(1, self.kintree_table.shape[1])
            }

            self.pose_shape = (24, 3)
            self.beta_shape = (10,)
            self.trans_shape = (3,)

            self.pose: Optional[np.ndarray] = np.zeros(self.pose_shape)
            self.beta: Optional[np.ndarray] = np.zeros(self.beta_shape)
            self.trans: Optional[np.ndarray] = np.zeros(self.trans_shape)

            self.verts: Optional[np.ndarray] = None
            self.J: Optional[np.ndarray] = None
            self.R: Optional[np.ndarray] = None

            self.update()
            self.get_vertex_color()

    def set_params(self, pose=None, beta=None, trans: Optional[np.ndarray]=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Parameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack([self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape(3, 1))])
                )
            )
        G = G - self.pack(np.matmul(G, np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])))
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape(-1, 4, 1)).reshape(-1, 4)[:, :3]
        self.verts = v + self.trans.reshape([1, 3])

    def rodrigues(self, r: np.ndarray) -> np.ndarray:
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x: np.ndarray) -> np.ndarray:
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x: np.ndarray) -> np.ndarray:
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def get_subpart_place(self, subpart: Union[int, str], eps: float = 0.3) -> np.ndarray:
        """
        export the character part as point cloud
        """
        if isinstance(subpart, str):
            subpart: int = smpl_hierarchy[subpart]
        sub_weights: np.ndarray = self.weights[:, subpart]
        sub_place: np.ndarray = np.where(sub_weights > eps)[0]
        return sub_place

    def get_subpart_verts(self, subpart: Union[int, str], eps: float = 0.3) -> np.ndarray:
        return self.verts[self.get_subpart_place(subpart, eps)]

    def get_vertex_color(self):
        len_color: int = self.color_conf.shape[0]
        color = self.color_conf
        if len_color < self.pose_shape[0]:  # enhaunce reference color
            count = self.pose_shape[0] - len_color
            rand_index = np.random.choice(len_color, count, replace=False)
            color = np.concatenate([color, color[rand_index]], axis=0)

        # compute weight for visualize each joint
        self.vert_color: np.ndarray = self.weights @ color
        return self.vert_color

    def visualize_pyvista_impl(self, plotter_, subset: Union[List[int], np.ndarray, None] = None, render_joint_hierarchy: bool = True):
        """
        performance of pyvista is much better than matplotlib
        """
        import pyvista as pv
        p: pv.Plotter = plotter_
        if subset is None:
            subset = slice(None, None)
        points: np.ndarray = self.verts[subset]
        point_cloud = pv.PolyData(points)

        p.add_mesh(point_cloud, scalars=self.vert_color[subset], rgb=True)

        # Render x, y, z axis.
        for index, axis in enumerate(np.eye(3)):
            line = pv.Line(np.zeros(3), axis)
            p.add_point_labels(axis, "xyz"[index], font_size=30)
            p.add_mesh(line, color="rgb"[index], line_width=20)
        if render_joint_hierarchy:
            self.visualize_joint_pyvista_impl(p)

    def visualize_joint_pyvista_impl(self, plotter_, joint_pos: Optional[np.ndarray] = None):
        """
        visualize all of all of joints
        """
        import pyvista as pv
        p: pv.Plotter = plotter_
        if joint_pos is None:
            joint_pos: np.ndarray = self.J
        point_cloud = pv.PolyData(joint_pos)
        p.add_mesh(point_cloud, color='maroon', point_size=15.0, render_points_as_spheres=True)
        p.add_point_labels(joint_pos, np.arange(joint_pos.shape[0], dtype=np.int32), font_size=20)
        # Note: we should also visualize the joint tree hierarchy
        for node, parent in self.parent.items():
            line = pv.Line(joint_pos[node], joint_pos[parent])
            p.add_mesh(line, color='c', line_width=10)

    def visualize_pyvista(self, subset: Union[List[int], np.ndarray, None] = None):
        import pyvista as pv
        plotter = pv.Plotter()
        self.visualize_pyvista_impl(plotter, subset)
        plotter.show()

    def simple_vis(self, ax, subset: Union[List[int], np.ndarray, None]):
        ax.scatter(self.verts[subset, 0], self.verts[subset, 1], self.verts[subset, 2], c=self.vert_color[subset])

    def visualize_matplotlib(self, subset: Union[List[int], np.ndarray, None] = None):
        """
        visualize vertex using matplotlib
        render all 3d positions
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        if subset is None:
            subset = slice(None, None)
        self.simple_vis(ax, subset)
        plt.show()

    def visualize_subset_matplotlib(self, sub_part: Union[str, int, None] = 0):
        sub_place = self.get_subpart_place(sub_part)
        self.visualize_matplotlib(sub_place)

    def visualize_all_subset(self):
        for i in range(24):
            self.visualize_subset_matplotlib(i)

    def visualize_joints(self):
        """
        show joint hierarchy..
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_aspect("auto")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        for node, parent in self.parent.items():
            node_pos = self.J[node]
            parent_pos = self.J[parent]
            ax.plot([node_pos[0], parent_pos[0]], [node_pos[1], parent_pos[1]], [node_pos[2], parent_pos[2]])
        plt.show()

    def save_to_obj(self, path: str):
        with open(path, 'w') as fp:
            for index, v in enumerate(self.verts):
                fp.write('v %f %f %f' % (v[0], v[1], v[2]))
                out_col = self.vert_color[index]
                fp.write(f" {out_col[0]} {out_col[1]} {out_col[2]}\n")
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def test():
    smpl = SMPLModel(os.path.join(fdir, './model.pkl'))
    np.random.seed(9608)
    # pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
    # beta = (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06
    # trans = np.zeros(smpl.trans_shape)
    # smpl.set_params(beta=beta, pose=pose, trans=trans)
    # smpl.save_to_obj(os.path.join(fdir, './smpl_np.obj'))
    # smpl.visualize_all_subset()
    smpl.visualize_pyvista()


def test_pyvista():
    import pyvista as pv
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]])
    # faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]])
    # lines = np.hstack([[2, 0, 1], [2, 1, 2]])
    lines = np.array([0, 3, 1])
    mesh = pv.PolyData(vertices, lines=lines)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.show()
    print("show")


if __name__ == "__main__":
    test()

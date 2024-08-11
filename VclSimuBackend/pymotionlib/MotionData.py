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
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Union, Optional

from .Utils import flip_quaternion, flip_vector, align_quaternion, quat_product
from ..Common.MathHelper import MathHelper


comm_rank = 0
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
except:
    pass

class MotionData:

    __slots__ = (
        "_skeleton_joints",
        "_skeleton_joint_parents",
        "_skeleton_joint_offsets",
        "_end_sites",
        "_num_joints",
        "_num_frames",
        "_fps",
        "_joint_rotation",
        "_joint_translation",
        "_joint_position",
        "_joint_orientation"
    )

    def __init__(self) -> None:
        # skeleton
        self._skeleton_joints: Optional[List[str]] = None  # name of each joint
        self._skeleton_joint_parents: Optional[List[int]] = None
        self._skeleton_joint_offsets: Optional[np.ndarray] = None  # Joint OFFSET in BVH file
        self._end_sites: Optional[List[int]] = None
        self._num_joints = 0

        # animation
        self._num_frames = 0
        self._fps = 0

        self._joint_rotation: Optional[np.ndarray] = None  # joint local rotation
        self._joint_translation: Optional[np.ndarray] = None  # [:, 0, :] is root position. other is zero.
        # num frame, num joint, 3

        # pre-computed global information
        self._joint_position: Optional[np.ndarray] = None
        self._joint_orientation: Optional[np.ndarray] = None

    @property
    def joint_rotation(self) -> Optional[np.ndarray]:
        return self._joint_rotation

    @property
    def joint_translation(self) -> Optional[np.ndarray]:
        return self._joint_translation

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def joint_position(self) -> Optional[np.ndarray]:
        return self._joint_position

    @property
    def joint_orientation(self) -> Optional[np.ndarray]:
        return self._joint_orientation

    @property
    def joint_parents_idx(self):
        return self._skeleton_joint_parents

    @property
    def joint_names(self) -> Optional[List[str]]:
        return self._skeleton_joints

    @property
    def end_sites(self) -> Optional[List[int]]:
        return self._end_sites

    # Add by Zhenhua Song
    def get_end_flags(self) -> np.ndarray:
        flags = [0] * self._num_joints
        if self._end_sites:
            for end_idx in self._end_sites:
                flags[end_idx] = 1
        return np.array(flags)

    @property
    def joint_offsets(self):
        return self._skeleton_joint_offsets

    @property
    def fps(self):
        return self._fps

    # Add by Zhenhua Song
    def remove_end_sites(self, copy: bool = True):
        ret = self.sub_sequence(copy=copy)
        if not ret._end_sites:
            return ret

        # modify attr index
        joint_idx: np.ndarray = np.arange(0, ret._num_joints, dtype=np.uint64)
        joint_idx: np.ndarray = np.delete(joint_idx, np.array(ret._end_sites, dtype=np.uint64))
        if ret._joint_translation is not None:
            ret._joint_translation = ret._joint_translation[:, joint_idx, :]
        if ret._joint_rotation is not None:
            ret._joint_rotation = ret._joint_rotation[:, joint_idx, :]
        if ret._joint_position is not None:
            ret._joint_position = ret._joint_position[:, joint_idx, :]
        if ret._joint_orientation is not None:
            ret._joint_orientation = ret._joint_orientation[:, joint_idx, :]

        # modify parent index
        for i in range(len(ret._end_sites)):
            end_idx = ret._end_sites[i] - i
            before = ret._skeleton_joint_parents[:end_idx]
            after = ret._skeleton_joint_parents[end_idx + 1:]
            for j in range(len(after)):
                if after[j] > end_idx:
                    after[j] -= 1
            ret._skeleton_joint_parents = before + after

        # modify other attributes
        ret._skeleton_joints = np.array(ret._skeleton_joints)[joint_idx].tolist()
        ret._skeleton_joint_offsets = np.array(ret._skeleton_joint_offsets)[joint_idx]
        ret._num_joints -= len(ret._end_sites)
        ret._end_sites.clear()

        return ret.to_contiguous()

    # Add by Zhenhua Song
    def set_anim_attrs(self, num_frames, fps):
        self._num_frames = num_frames
        self._fps = fps
        self._joint_translation = np.zeros((num_frames, self._num_joints, 3))

    # Add by Zhenhua Song
    def to_contiguous(self):
        if self._joint_rotation is not None:
            self._joint_rotation = np.ascontiguousarray(self._joint_rotation)
        if self._joint_translation is not None:
            self._joint_translation = np.ascontiguousarray(self._joint_translation)
        if self._joint_orientation is not None:
            self._joint_orientation = np.ascontiguousarray(self._joint_orientation)
        if self._joint_position is not None:
            self._joint_position = np.ascontiguousarray(self._joint_position)

        return self

    def align_joint_rotation_representation(self):
        """ make sure that the quaternions are aligned
        """
        if self._joint_rotation is not None:
            align_quaternion(self._joint_rotation, True)

        return self

    def reset_global_info(self):
        self._joint_position: Optional[np.ndarray] = None
        self._joint_orientation: Optional[np.ndarray] = None

    def compute_joint_global_info(self, joint_translation: np.ndarray, joint_rotation: np.ndarray,
                                  joint_position: np.ndarray = None, joint_orientation: np.ndarray = None):
        """ compute global information based on given local information
        """

        joint_translation = np.asarray(joint_translation).reshape((-1, self._num_joints, 3))
        joint_rotation = np.asarray(joint_rotation).reshape((-1, self._num_joints, 4))

        num_frames, num_joints = joint_rotation.shape[:2]
        if joint_position is None:
            joint_position = np.zeros((num_frames, num_joints, 3))
        else:
            joint_position.fill(0)
            joint_position = joint_position.reshape((num_frames, num_joints, 3))

        if joint_orientation is None:
            joint_orientation = np.zeros((num_frames, num_joints, 4))
        else:
            joint_orientation.fill(0)
            joint_orientation = joint_orientation.reshape((num_frames, num_joints, 4))

        for i, pi in enumerate(self._skeleton_joint_parents):
            joint_position[:, i, :] = joint_translation[:, i, :] + self._skeleton_joint_offsets[i, :]

            joint_orientation[:, i, :] = joint_rotation[:, i, :]

            if pi < 0:
                assert (i == 0)
                continue

            parent_orient = R(joint_orientation[:, pi, :], normalize=False, copy=False)
            joint_position[:, i, :] = parent_orient.apply(joint_position[:, i, :]) + joint_position[:, pi, :]
            joint_orientation[:, i, :] = (
                        parent_orient * R(joint_orientation[:, i, :], normalize=False, copy=False)).as_quat()
            joint_orientation[:, i, :] /= np.linalg.norm(joint_orientation[:, i, :], axis=-1, keepdims=True)

        return joint_position, joint_orientation

    def recompute_joint_global_info(self):
        #########
        # now pre-compute joint global positions and orientations
        self._joint_position, self._joint_orientation = self.compute_joint_global_info(
            self._joint_translation, self._joint_rotation, self._joint_position, self._joint_orientation)

        align_quaternion(self._joint_orientation, True)

        return self

    def compute_joint_local_info(self, joint_position: np.ndarray, joint_orientation: np.ndarray,
                                 joint_translation: np.ndarray = None, joint_rotation: np.ndarray = None):
        """ compute local information based on given global information
        """

        joint_position = np.asarray(joint_position).reshape((-1, self._num_joints, 3))
        joint_orientation = np.asarray(joint_orientation).reshape((-1, self._num_joints, 4))

        num_frames, num_joints = joint_position.shape[:2]
        if joint_translation is None:
            joint_translation = np.zeros((num_frames, num_joints, 3))
        else:
            joint_translation.fill(0)
            joint_translation = joint_translation.reshape((num_frames, num_joints, 3))

        if joint_rotation is None:
            joint_rotation = np.zeros((num_frames, num_joints, 4))
        else:
            joint_rotation.fill(0)
            joint_rotation = joint_rotation.reshape((num_frames, num_joints, 4))

        joint_translation[:, 0] = joint_position[:, 0]
        joint_translation[:, 1:] = joint_position[:, 1:] - joint_position[:, self._skeleton_joint_parents[1:]]
        joint_translation[:, 1:] = R(joint_orientation[:, self._skeleton_joint_parents[1:]].ravel().reshape(-1, 4),
                                     normalize=False, copy=False).apply(
            joint_translation[:, 1:].reshape(-1, 3), inverse=True
        ).reshape((num_frames, num_joints - 1, 3))

        joint_translation[:, 1:] -= self._skeleton_joint_offsets[1:]

        joint_rotation[:, 0] = joint_orientation[:, 0]
        joint_rotation[:, 1:] = quat_product(
            joint_orientation[:, self._skeleton_joint_parents[1:]],
            joint_orientation[:, 1:],
            inv_p=True)

        return joint_translation, joint_rotation

    def resample(self, new_fps: int):
        if new_fps == self.fps:
            return self

        if self.num_frames == 1:
            self._fps = new_fps
            return self

        from scipy.spatial.transform import Rotation, Slerp
        from scipy.interpolate import interp1d

        length = (self.num_frames - 1) / self.fps
        new_num_frames = int(np.floor(length * new_fps)) + 1

        if comm_rank == 0:
            print('fps: %d -> %d' % (self.fps, new_fps))
            print('num frames: %d -> %d' % (self.num_frames, new_num_frames), flush=True)

        ticks = np.arange(0, self.num_frames, dtype=np.float64) / self.fps
        new_ticks = np.arange(0, new_num_frames, dtype=np.float64) / new_fps

        # deal with root position with linear interp
        joint_trans_interp = interp1d(ticks, self._joint_translation, kind='linear', axis=0, copy=False,
                                      bounds_error=True, assume_sorted=True)
        self._joint_translation = joint_trans_interp(new_ticks)

        # handle joint rotations using slerp
        cur_joint_rots = self._joint_rotation
        num_joints = self.num_joints

        self._joint_rotation = np.zeros((new_num_frames, num_joints, 4))
        for i in range(num_joints):
            rotations = Rotation.from_quat(cur_joint_rots[:, i])
            self._joint_rotation[:, i] = Slerp(ticks, rotations)(new_ticks).as_quat()

        self._num_frames = new_num_frames
        self._fps = new_fps

        self._joint_position = None
        self._joint_orientation = None

        align_quaternion(self._joint_rotation, True)
        self.recompute_joint_global_info()

        # print("After bvh resample")
        return self

    def sub_sequence(self, start: Optional[int] = None,
                     end: Optional[int] = None,
                     skip: Optional[int] = None, copy: bool = True):
        sub = MotionData()

        sub._skeleton_joints = self._skeleton_joints
        sub._skeleton_joint_parents = self._skeleton_joint_parents
        sub._skeleton_joint_offsets = self._skeleton_joint_offsets
        sub._num_joints = self._num_joints

        sub._end_sites = self._end_sites
        sub._fps = self._fps

        key = slice(start, end, skip)

        sub._joint_rotation = self._joint_rotation[key] if self._joint_rotation is not None else None
        sub._joint_translation = self._joint_translation[key] if self._joint_translation is not None else None

        sub._joint_position = self._joint_position[key] if self._joint_position is not None else None
        sub._joint_orientation = self._joint_orientation[key] if self._joint_orientation is not None else None

        sub._num_frames = sub._joint_rotation.shape[0] if self._joint_rotation is not None else 0

        if copy:
            import copy
            sub._skeleton_joints = copy.deepcopy(sub._skeleton_joints)
            sub._skeleton_joint_parents = copy.deepcopy(sub._skeleton_joint_parents)
            sub._skeleton_joint_offsets = copy.deepcopy(sub._skeleton_joint_offsets)
            sub._num_joints = sub._num_joints

            sub._end_sites = copy.deepcopy(sub._end_sites)

            sub._joint_rotation = sub._joint_rotation.copy() if sub._joint_rotation is not None else None
            sub._joint_translation = sub._joint_translation.copy() if sub._joint_translation is not None else None

            sub._joint_position = sub._joint_position.copy() if sub._joint_position is not None else None
            sub._joint_orientation = sub._joint_orientation.copy() if sub._joint_orientation is not None else None

        return sub

    # Add by Zhenhua Song
    def get_t_pose(self):
        if self._num_frames >= 1:
            ret = self.sub_sequence(0, 1)
        else:
            ret = self.sub_sequence(0, 1)
            ret._num_frames = 1
            ret._joint_translation = np.zeros((1, self.num_joints, 3))
            ret._joint_rotation = np.zeros((1, self.num_joints, 4))
            ret._joint_position = np.zeros((1, self.num_joints, 3))
            ret._joint_orientation = np.zeros((1, self.num_joints, 4))

        ret._joint_translation.fill(0)
        # TODO: Here we should make sure the character is on the ground.
        ret._joint_rotation.fill(0)
        ret._joint_rotation[..., 3] = 1
        ret.recompute_joint_global_info()

        min_height: float = np.min(ret.joint_position[0, :, 1], axis=0) - 2e-2
        ret.joint_translation[:, :, 1] -= min_height
        ret.joint_position[:, :, 1] -= min_height

        return ret

    def get_hierarchy(self, copy: bool = False):
        """
        Get bvh hierarchy
        """
        return self.sub_sequence(self._num_frames, self._num_frames, copy=copy)

    def append_trans_rotation(self, trans: np.ndarray, rotation: np.ndarray):
        self._num_frames += trans.shape[0]
        if self._joint_rotation is not None:
            self._joint_rotation = np.concatenate([self._joint_rotation, rotation], axis=0)
        else:
            self._joint_rotation = rotation.copy()

        if self._joint_translation is not None:
            self._joint_translation = np.concatenate([self._joint_translation, trans], axis=0)
        else:
            self._joint_translation = trans.copy()

    def append(self, other_):
        import operator
        other: MotionData = other_
        assert self.fps == other.fps
        assert operator.eq(self.joint_names, other.joint_names)
        assert operator.eq(self.joint_parents_idx, other.joint_parents_idx)
        assert operator.eq(self.end_sites, other.end_sites)
        self._num_frames += other._num_frames

        if self.joint_rotation is not None:
            self._joint_rotation = np.concatenate([self._joint_rotation, other._joint_rotation], axis=0)
        else:
            self._joint_rotation = other._joint_rotation.copy()

        if self._joint_translation is not None:
            self._joint_translation = np.concatenate([self._joint_translation, other._joint_translation], axis=0)
        else:
            self._joint_translation = other._joint_translation.copy()

        if self._joint_position is not None:
            self._joint_position = np.concatenate([self._joint_position, other._joint_position], axis=0)
        else:
            self._joint_position = other._joint_position.copy()

        if self._joint_orientation is not None:
            self._joint_orientation = np.concatenate([self._joint_orientation, other._joint_orientation], axis=0)
        else:
            self._joint_translation = other._joint_translation.copy()

        return self

    def scale(self, factor: float):
        self._skeleton_joint_offsets *= factor

        if self._joint_translation is not None:
            self._joint_translation *= factor

        if self._joint_position is not None:
            self._joint_position *= factor

        return self

    def compute_linear_velocity(self, forward: bool = False):
        """ compute linear velocities of every joint using finite difference

            the velocities are in the world coordinates

            return: an array of size (num_frame, num_joint, 3),
                for forward/backward difference, the last/first frame is the
                frame next to it
        """
        if self._joint_position is None:
            self.recompute_joint_global_info()
        # Modified by Zhenhua Song
        # v = np.empty_like(self._joint_position)
        # frag = v[:-1] if forward else v[1:]
        # frag[:] = np.diff(self._joint_position, axis=0) * self._fps

        v: np.ndarray = np.zeros_like(self._joint_position)
        frag: np.ndarray = np.diff(self._joint_position, axis=0) * self._fps
        if forward:
            v[:-1] = frag  # v[t] = x[t + 1] - x[t]. e.g. v[0] = x[1] - x[0], v[1] = x[2] - x[1]
        else:
            v[1:] = frag  # v[t] = x[t] - x[t - 1]. e.g. v[1] = x[1] - x[0]

        v[-1 if forward else 0] = v[-2 if forward else 1]
        return v

    def compute_angular_velocity(self, forward: bool = False):
        """ compute angular velocities of every joint using finite difference

            the velocities are in the world coordinates

            forward: if True, we compute w_n = 2 (q_n+1 - q_n) * q_n.inv() ,
                   otherwise, we compute w_n = 2 (q_n - q_n-1) * q_n-1.inv()

            return: an array of size (num_frame, num_joint, 3),
                for forward/backward difference, the last/first frame is the
                frame next to it
        """
        if self._joint_orientation is None:
            self.recompute_joint_global_info()

        qd = np.diff(self._joint_orientation, axis=0) * self._fps

        # note that we cannot use R(q).inv() here, because scipy implement inv() as
        #  (x,y,z,w).inv() = (x,y,z,-w)
        # which is not the conjugate!
        
        # modified by heyuan Yao 
        q = self._joint_orientation[:-1] #if forward else self._joint_orientation[1:]
        q_conj = q.copy().reshape(-1, 4)
        q_conj[:, :3] *= -1
        try:
            qw = quat_product(qd.reshape(-1, 4), q_conj)
        except:
            pass
        # Modify By Zhenhua Song
        # w = np.empty((self._num_frames, self._num_joints, 3))
        # frag = w[:-1] if forward else w[1:]
        # frag[:] = qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
        # frag[:] *= 2
        w = np.zeros((self._num_frames, self._num_joints, 3))
        frag = 2 * qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
        if forward:
            w[:-1] = frag
        else:
            w[1:] = frag

        w[-1 if forward else 0] = w[-2 if forward else 1]
        return w

    def compute_translational_speed(self, forward: bool):
        """ compute the `local` translational velocities of every joint using finite difference

            note that different from `compute_linear_velocity`, this is the relative
            speed of joints wrt. their parents, and the values are represented in the
            parents' local coordinates

            return: an array of size (num_frame, num_joint, 3),
                for forward/backward difference, the last/first frame is the
                frame next to it
        """
        # Modify by Zhenhua Song
        # v = np.empty_like(self._joint_translation)
        # frag = v[:-1] if forward else v[1:]
        # frag[:] = np.diff(self._joint_translation, axis=0) * self._fps
        v = np.zeros_like(self._joint_translation)
        frag = np.diff(self._joint_translation, axis=0) * self._fps
        if forward:
            v[:-1] = frag
        else:
            v[1:] = frag
        v[-1 if forward else 0] = v[-2 if forward else 1]
        return v

    def compute_rotational_speed(self, forward: bool):
        """ compute the `local` rotational speed of every joint using finite difference

            note that different from `compute_angular_velocity`, this is the relative
            speed of joints wrt. their parents, and the values are represented in the
            parents' local coordinates

            forward: if True, we compute w_n = 2 (q_n+1 - q_n) * q_n.inv() ,
                   otherwise, we compute w_n = 2 (q_n - q_n-1) * q_n.inv()

            return: an array of size (num_frame, num_joint, 3),
                for forward/backward difference, the last/first frame is the
                frame next to it
        """
        qd = np.diff(self._joint_rotation, axis=0) * self._fps

        # note that we cannot use R(q).inv() here, because scipy implement inv() as
        #  (x,y,z,w).inv() = (x,y,z,-w)
        # which is not the conjugate!
        # modified by heyuan Yao 
        q = self._joint_rotation[:-1]
        q_conj = q.copy().reshape(-1, 4)
        q_conj[:, :3] *= -1
        qw = quat_product(qd.reshape(-1, 4), q_conj)

        w = np.zeros((self._num_frames, self._num_joints, 3))
        # frag = w[:-1] if forward else w[1:]
        # frag[:] = qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
        # frag[:] *= 2

        # Modify By Zhenhua Song
        frag = 2 * qw[:, :3].reshape(self._num_frames - 1, self._num_joints, 3)
        if forward:
            w[:-1] = frag
        else:
            w[1:] = frag

        w[-1 if forward else 0] = w[-2 if forward else 1].copy()
        return w

    def reconfig_reference_pose(self,
                                rotations: Union[List[np.ndarray], np.ndarray, Dict[str, np.ndarray]],
                                treat_as_global_orientations: bool,
                                treat_as_reverse_rotation: bool
                                ):
        """ reconfigurate the reference pose (T pose) of this bvh object
        Parameters
        -------
        rotations: rotations on the current T pose

        treat_as_global_orientations: if true, the input rotations will be treat as
            target orientations of the bones

        treat_as_reverse_rotation: if true, the input rotations are considered as those
            rotating the target pose to the current pose
        """

        if isinstance(rotations, list):
            rotations += [np.array([0, 0, 0, 1.0]) for i in range(self.num_joints - len(rotations))]
            rotations = np.array(rotations)
        elif isinstance(rotations, np.ndarray):
            rotations = np.concatenate((rotations, np.tile([0, 0, 0, 1.0], (self.num_joints - len(rotations), 1))), axis=0)
        elif isinstance(rotations, dict):
            rotations_ = np.array([[0, 0, 0, 1.0] for i in range(self.num_joints)])
            for jnt, rot in rotations.items():
                idx = self.joint_names.index(jnt)
                rotations_[idx, :] = rot
            rotations = rotations_
        else:
            raise ValueError('unsupported type: rotations (' + type(rotations) + ')')

        rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)

        if not treat_as_global_orientations:
            for i, p in enumerate(self.joint_parents_idx[1:]):
                idx = i + 1
                r = R(rotations[p])
                rotations[idx, :] = (r * R(rotations[idx])).as_quat()
                rotations[idx, :] /= np.sqrt(np.sum(rotations[idx, :] ** 2))

        if treat_as_reverse_rotation:
            rotations[:, -1] = -rotations[:, -1]

        rotations = R(rotations)

        for i, p in enumerate(self.joint_parents_idx):
            new_rot = R.from_quat(self.joint_rotation[:, i]) * rotations[i].inv()
            if p >= 0:
                new_rot = rotations[p] * new_rot
                self._skeleton_joint_offsets[i] = rotations[p].apply(self._skeleton_joint_offsets[i])

                self._joint_translation[:, i] = rotations[p].apply(self._joint_translation[:, i])

            self._joint_rotation[:, i] = new_rot.as_quat()

        self._joint_rotation /= np.linalg.norm(self._joint_rotation, axis=-1, keepdims=True)

        align_quaternion(self._joint_rotation, True)
        self.recompute_joint_global_info()

        return self

    def get_mirror_joint_indices(self):
        indices = list(range(self._num_joints))

        index_dict = {node: index for index, node in enumerate(self._skeleton_joints)}
        def index(name) -> int:
            if name in index_dict:
                return index_dict[name]
            else:
                return -1

        for i, n in enumerate(self._skeleton_joints):
            # rule 1: left->right
            idx = -1
            if n.find('left') == 0:
                idx = index('right' + n[4:])
            elif n.find('Left') == 0:
                idx = index('Right' + n[4:])
            elif n.find('LEFT') == 0:
                idx = index('RIGHT' + n[4:])
            elif n.find('right') == 0:
                idx = index('left' + n[5:])
            elif n.find('Right') == 0:
                idx = index('Left' + n[5:])
            elif n.find('RIGHT') == 0:
                idx = index('LEFT' + n[5:])
            elif n.find('L') == 0:
                idx = index('R' + n[1:])
            elif n.find('l') == 0:
                idx = index('r' + n[1:])
            elif n.find('R') == 0:
                idx = index('L' + n[1:])
            elif n.find('r') == 0:
                idx = index('l' + n[1:])

            indices[i] = idx if idx >= 0 else i

        return indices

    def symmetrize_skeleton(self, plane_of_symmetry_normal: Union[List[float], np.ndarray],
                            mirror_joint_indices: Union[None, List[int]]):
        """ fix skeleton joint offsets to make the skeleton symmetric

        Parameters
        ----------
        plane_of_symmetry_normal : the normal of the plan of symmetry of the skeleton
            note that the

        mirror_joint_indices: should be the index of the mirror joint of a joint
                    if not provided, get_mirror_joint_indices() will be called to get a best estimation

        """
        if mirror_joint_indices is None:
            mirror_joint_indices = self.get_mirror_joint_indices()

        mirror_offsets = flip_vector(self._skeleton_joint_offsets, plane_of_symmetry_normal, inplace=False)
        self._skeleton_joint_offsets += mirror_offsets[mirror_joint_indices]
        self._skeleton_joint_offsets /= 2

        self.recompute_joint_global_info()

        return self

    def flip(self, plane_of_symmetry_normal: Union[List[float], np.ndarray],
             mirror_joint_indices: Union[None, List[int]] = None):
        """ flip the animation wrt the plane of symmetry while assuming the plane passes the origin point

        Note that if the character is not symmetric or if a wrong normal vector is given, the result will not look good

        Parameters
        ----------
        plane_of_symmetry_normal : the normal of the plan of symmetry of the skeleton
            note that the

        mirror_joint_indices: should be the index of the mirror joint of a joint
                    if not provided, get_mirror_joint_indices() will be called to get a best estimation


        Returns
        -------
        None
        """
        flip_quaternion(self._joint_rotation.reshape(-1, 4), plane_of_symmetry_normal, inplace=True)
        flip_vector(self._joint_translation.reshape(-1, 3), plane_of_symmetry_normal, inplace=True)

        if mirror_joint_indices is None:
            mirror_joint_indices = self.get_mirror_joint_indices()

        self._joint_rotation[:] = self._joint_rotation[:, mirror_joint_indices]
        self._joint_translation[:] = self._joint_translation[:, mirror_joint_indices]

        align_quaternion(self._joint_rotation, True)
        self.recompute_joint_global_info()

        return self

    def get_reference_pose(self):
        pos = self.joint_offsets.copy()
        for i, p in enumerate(self.joint_parents_idx[1:]):
            pos[i + 1] += pos[p]

        return pos

    def retarget(self, joint_map: Dict[str, Union[str, List[str]]]):
        """ create a new skeleton based on the joint map and retarget the motion to it

        the hierarchy of current skeleton will be maintained.

        """

        joint_map_inv = [None] * self._num_joints
        try:
            for k, v in joint_map.items():
                if isinstance(v, str):
                    joint_map_inv[self._skeleton_joints.index(v)] = k
                else:
                    for v_ in v:
                        joint_map_inv[self._skeleton_joints.index(v_)] = k

        except ValueError:
            print('cannot find joint', v)
            raise

        if joint_map_inv[0] is None:
            print('root joint is not specified')
            raise ValueError('root joint is not specified')

        ref_pose = self.get_reference_pose()

        data = MotionData()
        data._skeleton_joints = [joint_map_inv[0]]
        data._skeleton_joint_parents = [-1]
        data._skeleton_joint_offsets = [ref_pose[0]]

        for i, n in enumerate(joint_map_inv[1:]):
            if n is None:
                continue

            if n in data._skeleton_joints:
                continue

            idx = i + 1

            data._skeleton_joints.append(n)
            p = self._skeleton_joint_parents[idx]
            while p >= 0:
                if joint_map_inv[p] is not None:
                    break
                p = self._skeleton_joint_parents[p]

            if p < 0:
                print('cannot find the parent joint for', n)
                raise ValueError('cannot find the parent joint for ' + str(n))

            while (self._skeleton_joint_parents[p] >= 0 and
                   joint_map_inv[self._skeleton_joint_parents[p]] == joint_map_inv[p]):
                p = self._skeleton_joint_parents[p]

            data._skeleton_joint_parents.append(data._skeleton_joints.index(joint_map_inv[p]))
            data._skeleton_joint_offsets.append(ref_pose[idx] - ref_pose[p])

        data._num_joints = len(joint_map)
        data._skeleton_joint_offsets = np.asarray(data._skeleton_joint_offsets)

        # now retarget the motion by copying the data
        data._num_frames = self._num_frames
        data._fps = self._fps

        data._joint_rotation = np.zeros((data._num_frames, data._num_joints, 4))
        data._joint_rotation.reshape(-1, 4)[:, -1] = 1
        data._joint_translation = np.zeros((data._num_frames, data._num_joints, 3))

        for i, n in enumerate(joint_map_inv):
            if n is None:
                continue

            idx = data._skeleton_joints.index(n)
            data._joint_rotation[:, idx] = (R.from_quat(data._joint_rotation[:, idx]) *
                                            R.from_quat(self._joint_rotation[:, i])).as_quat()
            data._joint_rotation[:, idx] /= np.linalg.norm(data._joint_rotation[:, idx], axis=-1, keepdims=True)

            data._joint_translation[:, idx] += self._joint_translation[:, i]

        align_quaternion(data._joint_rotation, True)
        data.recompute_joint_global_info()

        return data

    def remore_reference_nodes(self, new_root):
        """ create a new skeleton with the root joint as specified

            some software may export motions with 'reference node', this function will remove those node and bake the
            corresponding transformations into the new root

            note that we only allows a single root joint, so that the siblings of the new_root will be removed
        """
        try:
            new_root_idx = self.joint_names.index(new_root)
        except ValueError:
            raise ValueError('cannot find joint ' + new_root)

        data = MotionData()

        keep_joints = np.zeros(self.num_joints, dtype=bool)
        keep_joints[new_root_idx] = True
        for i in range(new_root_idx + 1, self.num_joints):
            keep_joints[i] = keep_joints[self.joint_parents_idx[i]]

        new_joint_indices = np.cumsum(keep_joints.astype(int)) - 1

        # skeleton
        data._skeleton_joints = list(np.asarray(self.joint_names)[keep_joints])
        data._skeleton_joint_parents = list(new_joint_indices[np.asarray(self.joint_parents_idx)[keep_joints]])
        data._skeleton_joint_offsets = np.asarray(self.joint_offsets)[keep_joints]
        data._end_sites = None if self.end_sites is None else list(
            new_joint_indices[list(n for n in self._end_sites if keep_joints[n])])
        data._num_joints = len(data._skeleton_joints)

        # animation
        data._num_frames = self._num_frames
        data._fps = self._fps

        data._joint_rotation = self._joint_rotation[:, keep_joints, :]
        data._joint_translation = self._joint_translation[:, keep_joints, :]

        data._joint_rotation[:, 0, :] = self.joint_orientation[:, new_root_idx, :]
        data._joint_translation[:, 0, :] = self.joint_position[:, new_root_idx, :]

        data.recompute_joint_global_info()

        return data

    # Add by Zhenhua Song
    def z_up_to_y_up(self):
        # 1. change root quaternion
        delta_rot: R = R.from_rotvec(np.array([-0.5 * np.pi, 0.0, 0.0]))
        self._joint_rotation[:, 0, :] = (delta_rot * R(self._joint_rotation[:, 0, :])).as_quat()

        # 2. change root position
        root_pos_y: np.ndarray = self._joint_translation[:, 0, 1].copy()
        root_pos_z: np.ndarray = self._joint_translation[:, 0, 2].copy()
        self._joint_translation[:, 0, 1] = root_pos_z
        self._joint_translation[:, 0, 2] = -root_pos_y

        # 3. we should NOT change local offset..
        self.recompute_joint_global_info()

    def re_root(self, new_root):
        """ change the root to another joint

            the joints will be reordered to ensure that a joint always behind its parent
        """
        raise NotImplementedError

    # Add by Zhenhua Song
    def to_torch(self):
        from .PyTorchMotionData import PyTorchMotionData
        result: PyTorchMotionData = PyTorchMotionData()
        result.build_from_motion_data(self)
        return result

    # Add by Zhenhua Song
    def remove_root_pos(self):
        """
        Note: this method is in place
        """
        self._joint_translation[:, :, :] = 0
        self._joint_position[:, :, :] -= self._joint_position[:, :, 0:1]

    # Add by Zhenhua Song
    def to_facing_coordinate(self):
        """
        Note: this method is in place
        """
        self._joint_translation[:, 0, :] = 0
        self._joint_rotation[:, 0, :] = MathHelper.y_decompose(self._joint_rotation[:, 0, :])[1]
        # assert np.all(np.abs(self._joint_rotation[:, 0, 1] < 1e-10))
        self._joint_orientation = None
        self._joint_position = None
        self.recompute_joint_global_info()
        return self

    # Add by Zhenhua Song
    def to_local_coordinate(self):
        """
        Note: this method is in place operation
        """
        self._joint_translation[:, 0, :] = 0
        self._joint_rotation[:, 0, :] = MathHelper.unit_quat_arr((self._num_frames, 4))
        self._joint_orientation = None
        self._joint_position = None
        self.recompute_joint_global_info()
        return self

    # Add by Zhenhua Song
    def get_adj_matrix(self) -> np.ndarray:
        num_joints = len(self.joint_parents_idx)
        result: np.ndarray = np.zeros(num_joints, dtype=np.int32)
        for idx, parent_idx in enumerate(self.joint_parents_idx):
            if parent_idx == -1:
                result[idx, parent_idx] = 1
                result[parent_idx, idx] = 1
        return result

    # Add by Zhenhua Song
    def get_neighbours(self) -> List[List[int]]:
        num_joints: int = len(self.joint_parents_idx)
        result: List[List[int]] = [[] for _ in range(num_joints)]
        for idx, parent_idx in enumerate(self.joint_parents_idx):
            if parent_idx == -1:
                continue
            result[idx].append(parent_idx)
            result[parent_idx].append(idx)
        for idx in range(num_joints):
            result[idx].sort()
        return result
    
    # Add by Yulong Zhang
    def concat(self, other, smooth_frame_num: int):
        """
        connect 2 motion data smoothly
        make interpolation between end of "self" and beginning of "other"
        return connected motion data
        """
        import copy
        from scipy.spatial.transform import Rotation, Slerp
        from scipy.interpolate import splprep, splev
        from ..Common.SmoothOperator import smooth_operator, GaussianBase
        result = MotionData()
        result._skeleton_joints = copy.deepcopy(self._skeleton_joints)
        result._skeleton_joint_parents = copy.deepcopy(self._skeleton_joint_parents)
        result._skeleton_joint_offsets = copy.deepcopy(self._skeleton_joint_offsets)
        result._num_joints = self._num_joints
        result._end_sites = copy.deepcopy(self._end_sites)
        result._fps = self._fps
        result._num_frames = self._num_frames + other._num_frames

        result._joint_rotation = np.zeros((result._num_frames, result._num_joints, 4))
        result._joint_translation = np.zeros((result._num_frames, result._num_joints, 3))
        
        result._joint_translation[:self._num_frames, :, :] = self._joint_translation
        result._joint_translation[self._num_frames:, :, :] = other._joint_translation

        smoothed_translation = smooth_operator(result._joint_translation[self._num_frames - smooth_frame_num: self._num_frames + smooth_frame_num, 0], GaussianBase(smooth_frame_num))
        result._joint_translation[self._num_frames - smooth_frame_num: self._num_frames + smooth_frame_num, 0] = smoothed_translation
        
        result._joint_rotation[:self._num_frames, :, :] = self._joint_rotation
        result._joint_rotation[self._num_frames:, :, :] = other._joint_rotation

        for j_idx in range(self.num_joints):
            vec6d: np.ndarray = MathHelper.quat_to_vec6d(result._joint_rotation[self._num_frames - smooth_frame_num: self._num_frames + smooth_frame_num, j_idx, :]).reshape((-1, 6))
            vec6d_new = smooth_operator(vec6d, GaussianBase(smooth_frame_num))
            result._joint_rotation[self._num_frames - smooth_frame_num: self._num_frames + smooth_frame_num, j_idx, :] = MathHelper.flip_quat_by_dot(MathHelper.vec6d_to_quat(vec6d_new.reshape((-1, 3, 2))))
        
        result.recompute_joint_global_info()
        return result

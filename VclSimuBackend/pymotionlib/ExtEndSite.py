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

# Add by Zhenhua Song
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict, Union, Optional, IO
from .MotionData import MotionData


def load_no_end_site(f: IO, ignore_root_offset=True, max_frames=None):
    channels = []
    joints = []
    joint_parents = []
    joint_offsets = []
    end_sites = []
    fps = 0

    parent_stack = [None]
    buf = f.readlines()
    buf_idx = 0
    while buf_idx < len(buf):
        line = buf[buf_idx].lstrip()
        buf_idx += 1
        if 'ROOT' in line or 'JOINT' in line:
            joints.append(line.split()[-1])
            joint_parents.append(parent_stack[-1])
            channels.append(None)
            joint_offsets.append([0, 0, 0])

        elif 'End Site' in line:
            while "}" not in buf[buf_idx]:
                buf_idx += 1
            buf_idx += 1

        elif '{' in line:
            parent_stack.append(joints[-1])

        elif '}' in line:
            parent_stack.pop()

        elif 'OFFSET' in line:
            joint_offsets[-1] = [float(x) for x in line.split()[-3:]]

        elif 'CHANNELS' in line:
            trans_order = []
            trans_channels = []
            rot_order = []
            rot_channels = []
            for i, token in enumerate(line.split()):
                if 'position' in token:
                    trans_order.append(token[0])
                    trans_channels.append(i - 2)

                if 'rotation' in token:
                    rot_order.append(token[0])
                    rot_channels.append(i - 2)

            channels[-1] = [(''.join(trans_order), trans_channels), (''.join(rot_order), rot_channels)]

        elif 'Frame Time:' in line:
            _frame_time = float(line.split()[-1])
            print('frame time: ', _frame_time)
            fps = round(1. / _frame_time)
            break

    values = []
    while buf_idx < len(buf):
        line = buf[buf_idx]
        buf_idx += 1
        tokens = line.split()
        if len(tokens) == 0:
            break
        values.append([float(x) for x in tokens])
        if max_frames is not None and len(values) >= max_frames:
            break

    values = np.array(values)

    assert (parent_stack[0] is None)
    data = MotionData()
    data._fps = fps

    data._skeleton_joints = joints
    data._skeleton_joint_parents = [joints.index(n) if n is not None else -1 for n in joint_parents]
    data._skeleton_joint_offsets = np.array(joint_offsets)
    data._end_sites = end_sites

    if ignore_root_offset:
        data._skeleton_joint_offsets[0].fill(0)

    data._num_frames = values.shape[0]
    data._num_joints = len(data._skeleton_joints)

    data._joint_translation = np.zeros((data._num_frames, data._num_joints, 3))
    data._joint_rotation = np.zeros((data._num_frames, data._num_joints, 4))
    data._joint_rotation[:, :, -1] = 1

    value_idx = 0
    for i, ch in enumerate(channels):
        if ch is None:
            continue

        joint_num_channels = len(ch[0][1]) + len(ch[1][1])
        joint_values = values[:, value_idx:value_idx + joint_num_channels]
        value_idx += joint_num_channels

        if not ch[0][0] == '':
            data._joint_translation[:, i] = joint_values[:, ch[0][1]]
            if not ch[0] == 'XYZ':
                data._joint_translation[:, i] = data._joint_translation[:, i][:, [ord(c) - ord('X') for c in ch[0][0]]]

        if not ch[1][0] == '':
            rot = Rotation.from_euler(ch[1][0], joint_values[:, ch[1][1]], degrees=True)
            data._joint_rotation[:, i] = rot.as_quat()

    print('loaded %d frames @ %d fps' % (data._num_frames, data._fps))

    data._joint_position = None
    data._joint_orientation = None
    data.align_joint_rotation_representation()
    data.recompute_joint_global_info()

    return data


# assume data doesn't have end site
# if a joint has no child, there must be a end site
# the default offset is (0, 0, 0)
def save_ext_end_site(data: MotionData, f: IO, fmt: str = '%10.6f',
                      euler_order: str = 'XYZ',
                      ext_end_site: Optional[Dict[int, np.ndarray]] = None):
    if data.end_sites:
        raise ValueError("Assume data dose not have end site.")

    if not euler_order in ['XYZ', 'XZY', 'YZX', 'YXZ', 'ZYX', 'ZXY']:
        raise ValueError('euler_order ' + euler_order + ' is not supported!')

    # save header
    children = [[] for _ in range(data._num_joints)]
    for i, p in enumerate(data._skeleton_joint_parents[1:]):
        children[p].append(i + 1)

    tab = ' ' * 4
    f.write('HIERARCHY\nROOT ' + data._skeleton_joints[0] + '\n{\n')
    f.write(tab + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[0]) + '\n')
    f.write(tab + 'CHANNELS 6 Xposition Yposition Zposition ' + ' '.join(c + 'rotation' for c in euler_order) + '\n')

    q = [(i, 1) for i in children[0][::-1]]
    last_level = 1
    output_order = [0]
    while len(q) > 0:
        idx, level = q.pop()
        output_order.append(idx)

        while last_level > level:
            f.write(tab * (last_level - 1) + '}\n')
            last_level -= 1

        indent = tab * level
        f.write(indent + 'JOINT ' + data._skeleton_joints[idx] + '\n')
        f.write(indent + '{\n')
        level += 1
        indent += tab
        f.write(indent + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[idx]) + '\n')
        f.write(indent + 'CHANNELS 3 ' + ' '.join(c + 'rotation' for c in euler_order) + '\n')

        if len(children[idx]) == 0:
            if ext_end_site is None:
                offset = np.array([0, 0, 0])  # Add default end site: offset (0, 0, 0)
            else:
                offset = ext_end_site[idx]
            f.write(tab * level + "End Site\n")
            f.write(tab * level + "{\n")
            f.write(tab * (level + 1) + "OFFSET " + ' '.join(fmt % s for s in offset) + '\n')
            f.write(tab * level + "}\n")

        q.extend([(i, level) for i in children[idx][::-1]])
        last_level = level

    while last_level > 0:
        f.write(tab * (last_level - 1) + '}\n')
        last_level -= 1

    # Write frames
    f.write('MOTION\n')
    f.write('Frames: %d\n' % data.num_frames)
    f.write('Frame Time: ' + (fmt % (1 / data.fps)) + '\n')

    # prepare channels
    value_idx = 0
    num_channels = 6 + 3 * (data._num_joints - 1)
    values = np.zeros((data.num_frames, num_channels))
    for i in output_order:
        if data._end_sites is not None and i in data._end_sites:
            continue

        if i == 0:
            values[:, value_idx:value_idx + 3] = data._joint_translation[:, i]
            value_idx += 3

        rot = Rotation.from_quat(data._joint_rotation[:, i])
        values[:, value_idx:value_idx + 3] = rot.as_euler(euler_order, degrees=True)
        value_idx += 3

    f.write('\n'.join([' '.join(fmt % x for x in line) for line in values]))

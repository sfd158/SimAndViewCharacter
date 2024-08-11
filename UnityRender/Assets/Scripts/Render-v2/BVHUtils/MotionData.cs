/*
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
*/
using System.Collections.Generic;
using UnityEngine;
using static RenderV2.Helper;
using static RenderV2.MathHelper;

namespace RenderV2
{
    // save motion data.
    public class MotionData
    {
        public List<string> joints; // store joint names.
        public List<Vector3> joint_offsets; // offset from joint to parent
        public List<int> end_sites; // index of end site joint.
        public List<string> joint_parents; // parent index.
        public List<int> joint_parents_idx; // index of parent joint

        public List<Vector3> joint_translation; // global translation of root joint
        public List<List<Quaternion>> joint_rotation; // local joint rotation

        public List<List<Vector3>> joint_position; // global joint position
        public List<List<Quaternion>> joint_orientation; // global joint orientation

        public int nFrames = 0;
        public int nJoints = 0;
        public float fps = 1;

        public MotionData() { }

        public MotionData Init()
        {
            joints = new List<string>(); // store joint names.
            joint_offsets = new List<Vector3>(); // offset from joint to parent
            end_sites = new List<int>(); // index of end site joint.
            joint_parents = new List<string>(); // parent index.
            joint_parents_idx = new List<int>(); // index of parent joint

            joint_translation = new List<Vector3>(); // global translation of root joint
            joint_rotation = new List<List<Quaternion>>();

            joint_position = new List<List<Vector3>>();
            joint_orientation = new List<List<Quaternion>>();
            return this;
        }

        public int GetNJoints() => nJoints;

        public int GetNFrames() => nFrames;

        public MotionData sub_sequence(int start, int end, bool is_copy = false)
        {
            MotionData res = new MotionData();
            res.fps = fps;
            res.nJoints = nJoints;
            res.nFrames = end - start;
            if (!is_copy)
            {
                res.joints = joints;
                res.joint_offsets = joint_offsets;
                res.end_sites = end_sites;
                res.joint_parents = joint_parents;
                res.joint_parents_idx = joint_parents_idx;
                res.joint_translation = joint_translation.GetRange(start, end - start);
                res.joint_rotation = joint_rotation.GetRange(start, end - start);
                res.joint_position = joint_position.GetRange(start, end - start);
                res.joint_orientation = joint_orientation.GetRange(start, end - start);
            }
            else
            {
                res.joints = DeepCopyList(joints);
                res.joint_offsets = DeepCopyList(joint_offsets);
                res.end_sites = DeepCopyList(end_sites);
                res.joint_parents = DeepCopyList(joint_parents);
                res.joint_parents_idx = DeepCopyList(joint_parents_idx);
                res.joint_translation = DeepCopyList(joint_translation, start, end);
                res.joint_rotation = DeepCopyList(joint_rotation, start, end);
                res.joint_position = DeepCopyList(joint_position, start, end);
                res.joint_orientation = DeepCopyList(joint_orientation, start, end);
            }
            return res;
        }

        public static MotionData Concatenate(MotionData data1, MotionData data2)
        {
            var result = new MotionData().Init();
            result.fps = data1.fps;
            result.nJoints = data1.nJoints;
            result.nFrames = data1.nFrames + data2.nFrames;

            result.joints.AddRange(data1.joints);
            result.joint_offsets.AddRange(data1.joint_offsets);
            result.end_sites.AddRange(data1.end_sites);
            result.joint_parents.AddRange(data1.joint_parents);
            result.joint_parents_idx.AddRange(data1.joint_parents_idx);

            result.joint_translation.AddRange(data1.joint_translation);
            result.joint_translation.AddRange(data2.joint_translation);

            result.joint_rotation.AddRange(data1.joint_rotation);
            result.joint_rotation.AddRange(data2.joint_rotation);

            result.joint_position.AddRange(data1.joint_position);
            result.joint_position.AddRange(data2.joint_position);

            result.joint_orientation.AddRange(data1.joint_orientation);
            result.joint_orientation.AddRange(data2.joint_orientation);
            
            return result;
        }

        public MotionData resample(int new_fps)
        {
            if (new_fps == fps) return this; // do nothing
            var res = sub_sequence(0, 0, true);
            res.joint_translation = Resample(joint_translation, fps, new_fps);
            res.joint_rotation = Resample(joint_rotation, fps, new_fps);
            res.nFrames = res.joint_translation.Count;
            res.fps = new_fps;
            res.forward_kinematics_all();
            return res;
        }

        public void Scale(float scale)
        {
            for (int i = 0; i < nJoints; i++) joint_offsets[i] *= scale;
            for (int i = 0; i < nFrames; i++)
            {
                joint_translation[i] *= scale;
                for (int j = 0; j < nJoints; j++) joint_position[i][j] *= scale;
            }
        }

        public Vector3[,] CalculateVelocity(bool forward)
        {
            Vector3[,] v = new Vector3[nFrames, nJoints];

            if (forward)
            {
                for (int i = 0; i < nFrames - 1; i++)
                    for (int j = 0; j < nJoints; j++)
                        v[i, j] = (joint_position[i + 1][j] - joint_position[i][j]) * fps;
                for (int j = 0; j < nJoints; j++) v[v.Length - 1, j] = v[v.Length - 2, j];
            }
            else
            {
                for (int i = 1; i < nFrames; i++)
                    for (int j = 0; j < nJoints; j++)
                        v[i, j] = (joint_position[i][j] - joint_position[i - 1][j]) * fps;
                for (int j = 0; j < nJoints; j++) v[0, j] = v[1, j];
            }
            return v;
        }

        public void forward_kinematics(int time_step)
        {
            Vector3 pos;
            Quaternion orient;
            var positions = new List<Vector3>();
            var orientations = new List<Quaternion>();
            for (int i = 0; i < nJoints; i++)
            {
                int pi = joint_parents_idx[i];
                if (pi < 0)
                {
                    pos = joint_translation[time_step] + joint_offsets[i];
                    orient = joint_rotation[time_step][i];
                }
                else
                {
                    pos = orientations[pi] * joint_offsets[i] + positions[pi];
                    orient = orientations[pi] * joint_rotation[time_step][i];
                }
                positions.Add(pos);
                orientations.Add(orient);
            }
            joint_position[time_step] = positions;
            joint_orientation[time_step] = orientations;
        }

        public void forward_kinematics_all()
        {
            if (joint_position.Count == 0)
            {
                for (int i = 0; i < nFrames; i++)
                {
                    joint_position.Add(null);
                }
            }
            if (joint_orientation.Count == 0)
            {
                for (int i = 0; i < nFrames; i++)
                {
                    joint_orientation.Add(null);
                }
            }
            
            for (int i = 0; i < nFrames; i++)
            {
                forward_kinematics(i);
            }
        }
    }
}
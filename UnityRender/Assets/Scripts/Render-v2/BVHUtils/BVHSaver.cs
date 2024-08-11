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
using System.Linq;
using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using System.Text;

namespace RenderV2
{
    // save mocap data to .bvh file
    public static class SaveBVH
    {
        public static string PrintVec3(Vector3 v, string fmt)
        {
            return v.x.ToString(fmt) + ' ' + v.y.ToString(fmt) + ' ' + v.z.ToString(fmt);
        }

        // export motion data to .bvh file
        public static void SaveTo(
            MotionData data,
            string filePath,
            string fmt = "F6",
            string eulerOrder = "XYZ"
        )
        {
            if (!new[] { "XYZ", "XZY", "YZX", "YXZ", "ZYX", "ZXY" }.Contains(eulerOrder))
            {
                throw new ArgumentException("euler_order " + eulerOrder + " is not supported!");
            }
            StreamWriter writer = new StreamWriter(filePath);
            int nj = data.GetNJoints();
            List<int>[] children = new List<int>[nj];
            //for i, p in enumerate(data._skeleton_joint_parents[1:]):
            //    children[p].append(i + 1)
            for (int i = 0; i < nj; i++) children[i] = new List<int>();
            for (int i = 1; i < nj; i++)
            {
                int parent = data.joint_parents_idx[i];
                children[parent].Add(i);
            }

            var tab = new string(' ', 4);
            writer.WriteLine("HIERARCHY");
            writer.WriteLine("ROOT " + data.joints[0]);
            writer.WriteLine("{");
            writer.WriteLine(tab + "OFFSET " + PrintVec3(data.joint_offsets[0], fmt));
            writer.WriteLine(tab + "CHANNELS 6 Xposition Yposition Zposition " + string.Join(" ", eulerOrder.Select(c => c + "rotation")));

            var q = new Stack<(int, int)>(children[0].Select(child => (child, 1)).Reverse());
            int lastLevel = 1;
            List<int> outputOrder = new List<int> { 0 };

            while (q.Count > 0)
            {
                var (idx, level) = q.Pop();
                outputOrder.Add(idx);

                while (lastLevel > level)
                {
                    writer.WriteLine(string.Concat(Enumerable.Repeat(tab, --lastLevel)) + "}");
                }

                string indent = string.Concat(Enumerable.Repeat(tab, level));
                bool endSite = data.end_sites != null && data.end_sites.Contains(idx);

                if (endSite)
                {
                    writer.WriteLine(indent + "End Site");
                }
                else
                {
                    writer.WriteLine(indent + "JOINT " + data.joints[idx]);
                }

                writer.WriteLine(indent + "{");
                level++;
                indent += tab;
                writer.WriteLine(indent + "OFFSET " + PrintVec3(data.joint_offsets[idx], fmt));

                if (!endSite)
                {
                    writer.WriteLine(indent + "CHANNELS 3 " + string.Join(" ", eulerOrder.Select(c => c + "rotation")));
                    // q.extend([(i, level) for i in children[idx][::-1]])
                    foreach (var child in children[idx]) q.Push((child, level));
                }
                lastLevel = level;
            }

            while (lastLevel > 0)
            {
                writer.WriteLine(string.Concat(Enumerable.Repeat(tab, --lastLevel)) + "}");
            }

            writer.WriteLine("MOTION");
            writer.WriteLine("Frames: " + data.GetNFrames());
            writer.WriteLine("Frame Time: " + (1.0f / data.fps).ToString(fmt));

            for(int f = 0; f < data.nFrames; f++)
            {
                foreach(int j in outputOrder)
                {
                    if (data.end_sites.Contains(j)) continue;
                    if (j == 0)
                    {
                        writer.Write(PrintVec3(data.joint_position[f][j], fmt) + ' ');
                    }
                    var euler = MathHelper.QuaternionToEuler(data.joint_rotation[f][j], eulerOrder, true); ;
                    writer.Write(PrintVec3(euler, fmt) + ' ');
                }
                writer.WriteLine();
            }
            
            writer.Close();
        }
    }
}
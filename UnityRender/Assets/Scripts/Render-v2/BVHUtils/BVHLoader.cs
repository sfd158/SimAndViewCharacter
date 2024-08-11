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
using System.IO;
using System.Collections;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// using in load bvh file
    /// </summary>
    public class channelObject
    {
        public string trans_order = "";
        public List<int> trans_channel = new List<int>();
        public string rot_order = "";
        public List<int> rot_channel = new List<int>();
    }

    /// <summary>
    /// load .bvh mocap file
    /// </summary>
    public class BVHLoader
    {
        private List<channelObject> channels = new List<channelObject>(); // rotation channel of bvh file

        private List<GameObject> gameObjects = new List<GameObject>();
        private int frame_num = 0; // total frame of bvh

        private List<string> parent_stack = new List<string> { null };

        /// <summary>
        /// load .bvh mocap file
        /// </summary>
        /// <param name="bvh_fname">.bvh file name</param>
        /// <returns>MotionData data</returns>
        public MotionData load_bvh(string bvh_fname)
        {
            MotionData data = new MotionData().Init();
            StreamReader bvh_file = new StreamReader(new FileStream(bvh_fname, FileMode.Open));
            while (!bvh_file.EndOfStream)
            {
                string line = bvh_file.ReadLine().Trim();
                string str = new Regex("[\\s]+").Replace(line, " ");
                string[] split_line = str.Split(' ');
                // bvh character hierarchy
                if (line.Contains("HIERARCHY")) continue;
                if (line.Contains("ROOT") || line.Contains("JOINT"))
                {
                    data.joints.Add(split_line[split_line.Length - 1]);
                    data.joint_parents.Add(parent_stack[parent_stack.Count - 1]);
                    channels.Add(null);
                    data.joint_offsets.Add(new Vector3(0, 0, 0));
                }
                else if (line.Contains("End Site"))
                {
                    data.end_sites.Add(data.joints.Count);
                    data.joints.Add(parent_stack[parent_stack.Count - 1] + "_end");
                    data.joint_parents.Add(parent_stack[parent_stack.Count - 1]);
                    channels.Add(null);
                    data.joint_offsets.Add(new Vector3(0, 0, 0));
                }
                else if (line.Contains("{"))
                {
                    parent_stack.Add(data.joints[data.joints.Count - 1]);
                }
                else if (line.Contains("}"))
                {
                    parent_stack.RemoveAt(parent_stack.Count - 1);
                }
                else if (line.Contains("OFFSET"))
                {
                    Vector3 tmp = new Vector3(
                        float.Parse(split_line[split_line.Length - 3]),
                        float.Parse(split_line[split_line.Length - 2]),
                        float.Parse(split_line[split_line.Length - 1])
                    );
                    data.joint_offsets[data.joint_offsets.Count - 1] = tmp;
                }
                else if (line.Contains("CHANNELS"))
                {
                    channelObject channel = new channelObject();
                    int num = split_line.Length;
                    for (int i = 0; i < num; i++)
                    {
                        if (split_line[i].Contains("position"))
                        {
                            channel.trans_order += split_line[i][0];
                            channel.trans_channel.Add(i - 2);
                        }
                        if (split_line[i].Contains("rotation"))
                        {
                            channel.rot_order += split_line[i][0];
                            channel.rot_channel.Add(i - 2);
                        }
                    }
                    channels[channels.Count - 1] = channel;
                }
                else if (line.Contains("Frame Time"))
                {
                    data.fps = 1.0f / float.Parse(line.Split(':')[1]);
                    break;
                }
                else if (line.Contains("Frames:"))
                {
                    frame_num = int.Parse(split_line[split_line.Length - 1]);
                }
            }

            while (!bvh_file.EndOfStream)
            {
                string line = bvh_file.ReadLine().Trim();
                string str = new Regex("[\\s]+").Replace(line, " ");
                string[] split_line = str.Split(' ');

                int value_idx = 0;
                List<Quaternion> rotations = new List<Quaternion>();
                for (int i = 0; i < channels.Count; i++)
                {
                    Quaternion quat = Quaternion.identity;
                    if (channels[i] is null)
                    {
                        rotations.Add(quat);
                        continue;
                    }
                    int value_num = channels[i].rot_channel.Count + channels[i].trans_channel.Count;
                    float[] joint_value = new float[value_num];
                    float[] tmp = new float[3];
                    try
                    {
                        for (int j = 0; j < value_num; j++)
                            joint_value[j] = float.Parse(split_line[value_idx + j]);
                    }
                    catch
                    {
                        Debug.Log(value_num);
                    }
                    value_idx += value_num;
                    if (channels[i].trans_channel.Count == 3)
                    {
                        for (int k = 0; k < 3; k++)
                            tmp[channels[i].trans_order[k] - 'X'] = joint_value[channels[i].trans_channel[k]];
                        Vector3 trans = new Vector3(tmp[0], tmp[1], tmp[2]);
                        data.joint_translation.Add(trans);
                    }
                    for (int k = 2; k >= 0; k--)
                    {
                        if (channels[i].rot_order[k] - 'X' == 0)
                            quat = Quaternion.Euler(joint_value[channels[i].rot_channel[k]], 0.0F, 0.0F) * quat;
                        else if (channels[i].rot_order[k] - 'X' == 1)
                            quat = Quaternion.Euler(0.0F, joint_value[channels[i].rot_channel[k]], 0.0F) * quat;
                        else if (channels[i].rot_order[k] - 'X' == 2)
                            quat = Quaternion.Euler(0.0F, 0.0F, joint_value[channels[i].rot_channel[k]]) * quat;
                    }
                    rotations.Add(quat);
                }
                data.joint_rotation.Add(rotations);
            }

            int nJoint = data.joints.Count;
            data.nJoints = nJoint;
            data.nFrames = frame_num;
            
            for (int i = 0; i < nJoint; i++)
            {
                if (data.joint_parents[i] != null)
                    data.joint_parents_idx.Add(data.joints.IndexOf(data.joint_parents[i]));
                else
                    data.joint_parents_idx.Add(-1);
            }
            data.forward_kinematics_all();

            return data;
        }

        

        /* public void set_by_frame(int time_step)
        {
            for (int i = 0; i < joints.Count; i++)
            {
                if (end_sites.Contains(i)) continue;
                gameObjects[i].transform.position = joint_position[time_step][i];
                gameObjects[i].transform.rotation = joint_orientation[time_step][i];
            }
        } */
    }
}

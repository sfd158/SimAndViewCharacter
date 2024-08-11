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

using System.Collections;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace RenderV2
{
    [CustomEditor(typeof(PDController))]
    public class InspectorPDController : Editor
    {
        PDController pdController;
        DCharacter dCharacter;
        string[] KpBuffer;
        string[] KdBuffer;
        string[] LimitBuffer;
        string[] WeightBuffer;

        List<string> jointNames;
        List<DJoint> JointList;
        int JointCount;

        string DefaultKp;
        string DefaultKd;
        string DefaultTorLim;
        string DefaultWeight;

        private void OnEnable()
        {
            pdController = target as PDController;
            dCharacter = pdController.transform.parent.GetComponent<DCharacter>();
            dCharacter.CalcAttrs();
            JointCount = dCharacter.JointCount;
            pdController.ResetParams(JointCount);

            KpBuffer = new string[JointCount];
            KdBuffer = new string[JointCount];
            LimitBuffer = new string[JointCount];
            WeightBuffer = new string[JointCount];

            JointList = dCharacter.JointList;
            jointNames = new List<string>();
            int maxLen = 0;
            for(int jointIdx=0; jointIdx<JointCount; jointIdx++)
            {
                string jName = dCharacter.JointList[jointIdx].gameObject.name;
                if (jName.Length > maxLen)
                {
                    maxLen = jName.Length;
                }
                else
                {
                    jName += new string('*', maxLen - jName.Length);
                }
                jointNames.Add(jName);
            }
            LoadJointBuffer();

            DefaultKp = pdController.DefaultKp.ToString();
            DefaultKd = dCharacter.DefaultJointDamping.ToString();
            DefaultTorLim = pdController.DefaultTorqueLimit.ToString();
            DefaultWeight = "1";
        }

        private void LoadJointBuffer()
        {
            for(int jointIdx=0; jointIdx<JointCount; jointIdx++)
            {
                DJoint joint = dCharacter.JointList[jointIdx];
                KpBuffer[jointIdx] = pdController.Kps[jointIdx].ToString();
                KdBuffer[jointIdx] = joint.Damping.ToString();
                LimitBuffer[jointIdx] = pdController.TorqueLimits[jointIdx].ToString();
                WeightBuffer[jointIdx] = joint.weight.ToString();
            }
        }

        private void SaveJointBuffer()
        {
            for(int jointIdx=0; jointIdx<JointCount; jointIdx++)
            {
                DJoint joint = dCharacter.JointList[jointIdx];
                if (float.TryParse(KpBuffer[jointIdx], out var kp))
                {
                    pdController.Kps[jointIdx] = kp;
                }
                else
                {
                    Debug.Log("Parse kp[" + jointIdx + "] err");
                }
                if (float.TryParse(KdBuffer[jointIdx], out var kd))
                {
                    joint.Damping = kd;
                }
                else
                {
                    Debug.Log("Parse Damping[" + jointIdx + "] err");
                }
                if (float.TryParse(LimitBuffer[jointIdx], out var torlim))
                {
                    pdController.TorqueLimits[jointIdx] = torlim;
                }
                else
                {
                    Debug.Log("Parse TorqueLimits[" + jointIdx + "] err");
                }
                if (float.TryParse(WeightBuffer[jointIdx], out var weight))
                {
                    joint.weight = weight;
                }
                else
                {
                    Debug.Log("Parse Joint Weight[" + jointIdx + "] err");
                }
            }
        }

        public override void OnInspectorGUI()
        {
            for(int jointIdx=0; jointIdx<JointCount; jointIdx++)
            {
                DJoint joint = dCharacter.JointList[jointIdx];
                GUILayout.BeginHorizontal();
                GUILayout.Label(jointNames[jointIdx]);
                GUILayout.Label("Kp");
                KpBuffer[jointIdx] = GUILayout.TextArea(KpBuffer[jointIdx]);
                GUILayout.Label("Kd");
                KdBuffer[jointIdx] = GUILayout.TextArea(KdBuffer[jointIdx]);
                GUILayout.Label("TorLim");
                LimitBuffer[jointIdx] = GUILayout.TextArea(LimitBuffer[jointIdx]);
                GUILayout.Label("Weight");
                WeightBuffer[jointIdx] = GUILayout.TextArea(WeightBuffer[jointIdx]);
                GUILayout.EndHorizontal();
            }

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Load Value"))
            {
                LoadJointBuffer();
            }

            if (GUILayout.Button("Save Value"))
            {
                SaveJointBuffer();
            }

            if (GUILayout.Button("Print Value"))
            {
                for(int jointIdx=0; jointIdx<JointCount; jointIdx++)
                {
                    DJoint joint = dCharacter.JointList[jointIdx];
                    Debug.Log(jointNames[jointIdx] + " Kp = " +
                                pdController.Kps[jointIdx] + " Kd = " + joint.Damping +
                                " torlim = " + pdController.TorqueLimits[jointIdx] +
                                " weight = " + joint.weight);
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            DefaultKp = GUILayout.TextArea(DefaultKp);
            if (GUILayout.Button("Set Default Kp"))
            {
                for(int i=0; i<JointCount; i++)
                {
                    KpBuffer[i] = DefaultKp;
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            DefaultKd = GUILayout.TextArea(DefaultKd);
            if (GUILayout.Button("Set Default Kd"))
            {
                for(int i=0; i<JointCount; i++)
                {
                    KdBuffer[i] = DefaultKd;
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            DefaultTorLim = GUILayout.TextArea(DefaultTorLim);
            if (GUILayout.Button("Set Default TorLim"))
            {
                for(int i=0; i<JointCount; i++)
                {
                    LimitBuffer[i] = DefaultTorLim;
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            DefaultWeight = GUILayout.TextArea(DefaultWeight);
            if (GUILayout.Button("Set Default Weight"))
            {
                for(int i=0; i<JointCount; i++)
                {
                    WeightBuffer[i] = DefaultWeight;
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Dump joint param"))
            {
                string fname = EditorUtility.SaveFilePanel("Save Joint Param", ".", "JointParam", "txt");
                if (fname == null || fname.Length == 0)
                {
                    return;
                }
                if (System.IO.File.Exists(fname))
                {
                    System.IO.File.Delete(fname);
                }
                StreamWriter streamWriter = new StreamWriter(new FileStream(fname, FileMode.CreateNew));
                for(int jointIdx=0; jointIdx<JointCount; jointIdx++)
                {
                    DJoint joint = dCharacter.JointList[jointIdx];
                    streamWriter.Write(joint.gameObject.name);
                    streamWriter.Write(" ");
                    streamWriter.Write(pdController.Kps[jointIdx]);
                    streamWriter.Write(" ");
                    streamWriter.Write(joint.Damping);
                    streamWriter.Write(" ");
                    streamWriter.Write(pdController.TorqueLimits[jointIdx]);
                    streamWriter.Write(" ");
                    streamWriter.Write(joint.weight);
                    streamWriter.WriteLine();
                }
                streamWriter.Close();
                Debug.Log("Write to " + fname);
            }
            if (GUILayout.Button("Load joint param"))
            {
                string fname = EditorUtility.OpenFilePanel("Load Joint Param", ".", "txt");
                if (fname == null || fname.Length == 0 || !File.Exists(fname))
                {
                    return;
                }
                StreamReader reader = new StreamReader(new FileStream(fname, FileMode.Open));
                int jointIdx = 0;
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    if (line.Length == 0)
                    {
                        continue;
                    }
                    string[] attrs = line.Split(' ');
                    KpBuffer[jointIdx] = attrs[1];
                    KdBuffer[jointIdx] = attrs[2];
                    LimitBuffer[jointIdx] = attrs[3];
                    WeightBuffer[jointIdx] = attrs[4];
                    jointIdx++;
                }
                reader.Close();
                Debug.Log("Load from " + fname);
            }
            GUILayout.EndHorizontal();
        }
    }
}


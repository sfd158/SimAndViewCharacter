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

using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

namespace RenderV2
{
    /// <summary>
    /// Inspector for DCharacter Component
    /// </summary>
    [CustomEditor(typeof(DCharacter))]
    public class InspectorDCharacter : Editor
    {
        DCharacter dCharacter;

        private void OnEnable()
        {
            dCharacter = target as DCharacter;
        }

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("SetGeomMu"))
            {
                dCharacter.SetDefaultMu();
            }
            if (GUILayout.Button("SetJointDamping"))
            {
                dCharacter.SetDefaultDamping();
            }
            if (GUILayout.Button("Recompute"))
            {
                dCharacter.ReCompute();
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Show Joint Pos"))
            {
                // dCharacter.ReCompute();
                dCharacter.CalcAttrs();
                dCharacter.IsRenderJointPosition = true;
                dCharacter.IsRenderBodyPosition = true;
            }
            if (GUILayout.Button("Close Show Joint Pos"))
            {
                dCharacter.IsRenderJointPosition = false;
                dCharacter.IsRenderBodyPosition = false;
            }
            if (GUILayout.Button("Remove Euler Components"))
            {
                var eulers = dCharacter.gameObject.GetComponentsInChildren<DEulerAxis>();
                foreach(var euler in eulers)
                {
                    DestroyImmediate(euler.gameObject);
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("unset all angle limit"))
            {
                var balls = dCharacter.gameObject.GetComponentsInChildren<DBallJoint>();
                foreach(var joint in balls)
                {
                    for(int i=0; i<3; i++)
                    {
                        joint.AngleLoLimit[i] = -180;
                        joint.AngleHiLimit[i] = 180;
                    }
                }

                var hinges = dCharacter.gameObject.GetComponentsInChildren<DHingeJoint>();
                foreach (var joint in hinges)
                {
                    joint.AngleLimit = new Vector2(-180, 180);
                }
            }
            if (GUILayout.Button("Add Joint View"))
            {
                dCharacter.AddJointView();
            }
            if (GUILayout.Button("Remove Joint View"))
            {
                dCharacter.RemoveJointView();
            }
            // if (GUILayout.Button("Modify Character Shape"))
            // {
            //     this.dCharacter.ReCompute();
            // }
            if (GUILayout.Button("View Capsule"))
            {
                // View Capsule as 2 half ball and a cylinder
            }
            GUILayout.EndHorizontal();
        }
    }
}

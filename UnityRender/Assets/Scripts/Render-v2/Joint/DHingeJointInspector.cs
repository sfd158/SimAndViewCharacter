﻿/*
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
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace RenderV2
{
    [CustomEditor(typeof(DHingeJoint))]
    public class DHingeJointInspector : Editor
    {
        DHingeJoint dHingeJoint;
        bool enableEditAngle = true;
        private void OnEnable()
        {
            dHingeJoint = (DHingeJoint)target;
        }

        /* private void TestAddRotate()
        {
            Quaternion localRot = dHingeJoint.transform.localRotation;
            Vector3 axis = dHingeJoint.GetHingeLocalAxis();
            Quaternion deltaRot = Quaternion.AngleAxis(5, axis);
            dHingeJoint.transform.localRotation = deltaRot * localRot;            
        }

        private void TestAddRotate2()
        {
            dHingeJoint.HingeAngleDegree += 5;
        } */

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("ReCompute"))
            {
                dHingeJoint.ReComputeButton();
            }
            GUILayout.EndHorizontal();

            if (Application.isPlaying)
            {
                enableEditAngle = false;
            }
            else
            {
                enableEditAngle = GUILayout.Toggle(enableEditAngle, "Enable Edit Angle"); ;
            }

            float degree = GUILayout.HorizontalScrollbar(dHingeJoint.HingeAngleDegree, 10, dHingeJoint.AngleLimit[0], dHingeJoint.AngleLimit[1]);
            if (enableEditAngle)
            {
                dHingeJoint.HingeAngleDegree = degree;
            }

        }
    }
}


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
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace RenderV2
{
    /// <summary>
    /// Inspector for Ball Joint
    /// </summary>
    [CustomEditor(typeof(DBallJoint))]
    public class DBallJointInspector: Editor
    {
        DBallJoint dBallJoint;
        bool enableEditAngle = false;
        // bool showAngles = true;

        private void OnEnable()
        {
            dBallJoint = (DBallJoint)target;
        }

        /* void TestAddRotate(int axis)
        {
            Quaternion localRot = dBallJoint.transform.localRotation;
            Vector3 rotVec = Vector3.zero;
            rotVec[axis] = 1;
            rotVec = dBallJoint.EulerAxisLocalRot * rotVec;
            Quaternion deltaRot = Quaternion.AngleAxis(5, rotVec);
            dBallJoint.transform.localRotation = deltaRot * localRot;
        } */

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("ReCompute"))
            {
                dBallJoint.ReComputeButton();
            }
            GUILayout.EndHorizontal();

            Eigen.EulerResultf result = dBallJoint.GetEulerResultByTransform();
            if (Application.isPlaying)
            {
                enableEditAngle = false;
            }
            else
            {
                enableEditAngle = GUILayout.Toggle(enableEditAngle, "Enable Edit Angle"); ;
            }

            GUILayout.BeginVertical();
            GUILayout.Label("local x");
            float angleXRes = GUILayout.HorizontalScrollbar(result.XDeg(), 10, dBallJoint.AngleLoLimit[0], dBallJoint.AngleHiLimit[0]);
            if (enableEditAngle)
            {
                dBallJoint.AngleXDegree = angleXRes;
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical();
            GUILayout.Label("local y");
            float angleYRes = GUILayout.HorizontalScrollbar(result.YDeg(), 10, dBallJoint.AngleLoLimit[1], dBallJoint.AngleHiLimit[1]);
            if (enableEditAngle)
            {
                dBallJoint.AngleYDegree = angleYRes;
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical();
            GUILayout.Label("local z");
            float angleZRes = GUILayout.HorizontalScrollbar(result.ZDeg(), 10, dBallJoint.AngleLoLimit[2], dBallJoint.AngleHiLimit[2]);
            if (enableEditAngle)
            {
                dBallJoint.AngleZDegree = angleZRes;
            }
            GUILayout.EndVertical();

        }
    }
}
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

namespace RenderV2
{
    public class DHingeJoint : DJoint
    {
        public string EulerOrder = "X"; // euler order should be X, Y, Z

        public Vector2 AngleLimit = new Vector2(-180, 180); // low and high angle limit

        /// <summary>
        /// local hinge angle in rad
        /// </summary>
        public float HingeAngle
        {
            get
            {
                Vector3 axis = GetHingeLocalAxis();
                return Utils.GetHingeAngleFromRelativeQuat(transform.localRotation, axis);
            }
            set
            {
                Vector3 axis = GetHingeLocalAxis();
                transform.localRotation = Quaternion.AngleAxis(value * Mathf.Rad2Deg, axis);
            }
        }

        /// <summary>
        /// local hinge angle in deg
        /// </summary>
        public float HingeAngleDegree
        {
            get
            {
                return HingeAngle * Mathf.Rad2Deg;
            }
            set
            {
                HingeAngle = value * Mathf.Deg2Rad;
            }
        }

        public override string ToString()
        {
            return base.ToString();
        }

        public override string JointType()
        {
            return "HingeJoint";
        }

        public override DJointType EnumJointType()
        {
            return DJointType.HingeJointType;
        }

        public Vector3 GetHingeRawAxis()
        {
            Vector3 res = Vector3.zero;
            res[EulerOrder[0] - 'X'] = 1;
            return res;
        }

        public Vector3 GetHingeLocalAxis()
        {
            Vector3 res = GetHingeRawAxis();
            return EulerAxisLocalRot * res;
        }

        public Vector3 GetGlobalAxis()
        {
            return transform.rotation * GetHingeLocalAxis();
        }

        public override DJointExportInfo ExportInfo()
        {
            DJointExportInfo jointInfo = ExportInfoBase();
            jointInfo.AngleLoLimit = new float[1] { AngleLimit[0] };
            jointInfo.AngleHiLimit = new float[1] { AngleLimit[1] };
            jointInfo.EulerOrder = EulerOrder;
            return jointInfo;
        }

        public static DHingeJoint AddHingeJoint(
            GameObject jointObject,
            int JointID,
            string JointName,
            float Damping,
            Vector3 Position,
            float AngleLoLimit,
            float AngleHiLimit,
            string EulerOrder,
            GameObject character
        )
        {
            if (jointObject == null)
            {
                jointObject = new GameObject();
            }
            DHingeJoint hingeJoint = jointObject.AddComponent<DHingeJoint>();
            hingeJoint.EulerOrder = EulerOrder;
            hingeJoint.AngleLimit = new Vector2(AngleLoLimit, AngleHiLimit);
            hingeJoint.InitialPosition = Position;

            hingeJoint.SetParam(JointName, JointID, Damping, character);
            return hingeJoint;
        }

        public static DHingeJoint AddHingeJoint(int JointID, float Damping, Vector3 Position, string EulerOrder, GameObject character)
        {
            return AddHingeJoint(null, JointID, "hinge" + JointID, Damping, Position, -180, 180, EulerOrder, character);
        }

        public static DHingeJoint AddHingeJoint(DJointExportInfo info, GameObject character)
        {
            return AddHingeJoint(null, info.JointID, info.Name, info.Damping, Utils.ArrToVector3(info.Position), info.AngleLoLimit[0], info.AngleHiLimit[0], info.EulerOrder, character);
        }
    }
}


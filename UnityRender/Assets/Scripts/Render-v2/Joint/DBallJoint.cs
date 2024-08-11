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
    /// <summary>
    /// Ball Joint
    /// </summary>
    public class DBallJoint : DJoint
    {
        [Tooltip("Euler Order")]
        public string EulerOrder = "XYZ";

        [Tooltip("Angle Low Limit")]
        public float[] AngleLoLimit = new float[3] { -180, -180, -180};

        [Tooltip("Angle High Limit")]
        public float[] AngleHiLimit = new float[3] { 180, 180, 180};

        private Eigen.EulerResultf eulerResult;
        private bool eulerResultInitialized = false;

        public Eigen.EulerResultf GetEulerResultByTransform()
        {
            Quaternion q = transform.localRotation;
            Eigen.Matrix3f mat = new Eigen.Matrix3f(q.x, q.y, q.z, q.w);
            eulerResult = mat.ExtractEuler(EulerOrder);
            eulerResultInitialized = true;
            return eulerResult;
        }

        void SetTransfromByEulerResult()
        {
            Eigen.Matrix3f mat = Eigen.Matrix3f.MakeEuler(eulerResult, EulerOrder);
            Eigen.QuaternionResultf q = mat.ToQuaternion();
            transform.localRotation = new Quaternion(q.x, q.y, q.z, q.w);
        }

        /// <summary>
        /// Euler angle x
        /// </summary>
        /// <value></value>
        public float AngleX
        {
            get
            {
                if (!eulerResultInitialized)
                {
                    GetEulerResultByTransform();
                }
                return eulerResult.xAngle;
            }
            set
            {
                eulerResult.xAngle = value;
                SetTransfromByEulerResult();
            }
        }

        /// <summary>
        /// euler angle y
        /// </summary>
        /// <value></value>
        public float AngleY
        {
            get
            {
                if (!eulerResultInitialized)
                {
                    GetEulerResultByTransform();
                }
                return eulerResult.yAngle;
            }
            set
            {
                eulerResult.yAngle = value;
                SetTransfromByEulerResult();
            }
        }

        /// <summary>
        /// euler angle z
        /// </summary>
        /// <value></value>
        public float AngleZ
        {
            get
            {
                if (!eulerResultInitialized)
                {
                    GetEulerResultByTransform();
                }
                return eulerResult.zAngle;
            }
            set
            {
                eulerResult.zAngle = value;
                SetTransfromByEulerResult();
            }
        }

        public float AngleXDegree
        {
            get
            {
                return AngleX * Mathf.Rad2Deg;
            }
            set
            {
                AngleX = value * Mathf.Deg2Rad;
            }
        }

        public float AngleYDegree
        {
            get
            {
                return AngleY * Mathf.Rad2Deg;
            }
            set
            {
                AngleY = value * Mathf.Deg2Rad;
            }
        }

        public float AngleZDegree
        {
            get
            {
                return AngleZ * Mathf.Rad2Deg;
            }
            set
            {
                AngleZ = value * Mathf.Deg2Rad;
            }
        }

        private void FixedUpdate()
        {
            GetEulerResultByTransform();
        }

        public override string JointType()
        {
            return "BallJoint";
        }

        public override DJointType EnumJointType()
        {
            return DJointType.BallJointType;
        }
        public override DJointExportInfo ExportInfo()
        {
            DJointExportInfo jointInfo = ExportInfoBase();
            jointInfo.AngleLoLimit = AngleLoLimit;
            jointInfo.AngleHiLimit = AngleHiLimit;
            jointInfo.EulerOrder = EulerOrder;
            return jointInfo;
        }

        public static DBallJoint AddBallJoint(
            GameObject jointObject,
            int JointID,
            string JointName,
            float Damping,
            Vector3 Position,
            float[] AngleLoLimit,
            float[] AngleHiLimit,
            string eulerOrder,
            GameObject character
        )
        {
            if (jointObject == null)
            {
                jointObject = new GameObject();
            }

            DBallJoint ballJoint = jointObject.AddComponent<DBallJoint>();

            ballJoint.EulerOrder = eulerOrder;
            ballJoint.AngleLoLimit = new float[3];
            ballJoint.AngleHiLimit = new float[3];

            if (AngleLoLimit.Length != 3 || AngleHiLimit.Length != 3)
            {
                throw new System.ArgumentException("AngleLoLimit and AngleHiLimit's Length must be 3");
            }
            ballJoint.AngleLoLimit = AngleLoLimit;
            ballJoint.AngleHiLimit = AngleHiLimit;
            ballJoint.InitialPosition = Position;

            ballJoint.SetParam(JointName, JointID, Damping, character);

            return ballJoint;
        }

        public static DBallJoint AddBallJoint(int JointID, float Damping, Vector3 Position, GameObject character)
        {
            return AddBallJoint(null, JointID, "ball" + JointID, Damping, Position, new float[]{-180, -180, -180}, new float[]{180, 180, 180}, "XYZ", character);
        }
        public static DBallJoint AddBallJoint(DJointExportInfo info, GameObject character)
        {
            return AddBallJoint(null, info.JointID, info.Name, info.Damping, Utils.ArrToVector3(info.Position), info.AngleLoLimit, info.AngleHiLimit, info.EulerOrder, character);
        }
    }
}

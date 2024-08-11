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

using Newtonsoft.Json.Linq;
using System;
using System.Runtime.InteropServices;
using UnityEngine;
using static RenderV2.Eigen.EigenBindingf;

namespace RenderV2
{
    namespace Eigen
    {
        /// <summary>
        /// Eigen::Matrix3f in C++
        /// </summary>
        public class Matrix3f
        {
            /// <summary>
            /// Matirx Pointer
            /// </summary>
            public IntPtr Ptr;

            private bool Deleted;

            public Matrix3f()
            {
                Ptr = Matrix3fCreate();
                Deleted = false;
            }

            public Matrix3f(IntPtr T)
            {
                Ptr = T;
                Deleted = false;
            }

            /// <summary>
            /// Build From Quaternion
            /// </summary>
            /// <param name="qx"></param>
            /// <param name="qy"></param>
            /// <param name="qz"></param>
            /// <param name="qw"></param>
            public Matrix3f(float qx, float qy, float qz, float qw)
            {
                Ptr = Matrix3fFromQuaternion(qx, qy, qz, qw);
                Deleted = false;
            }

            public Matrix3f(Quaternion quat)
            {
                Ptr = Matrix3fFromQuaternion(quat.x, quat.y, quat.z, quat.w);
                Deleted = false;
            }

            public Matrix3f(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
            {
                Ptr = Matrix3fCreateValue(a00, a01, a02, a10, a11, a12, a20, a21, a22);
                Deleted = false;
            }

            ~Matrix3f()
            {
                Delete();
            }

            public void Delete()
            {
                if (!Deleted)
                {
                    Matrix3fDelete(Ptr);
                    Deleted = true;
                }
            }

            public static Matrix3f matmul(Matrix3f a, Matrix3f b)
            {
                var res = new Matrix3f();
                Matrix3fProduct(a.Ptr, b.Ptr, res.Ptr);
                return res;
            }

            public static Matrix3f operator * (Matrix3f a, Matrix3f b)
            {
                var res = new Matrix3f();
                Matrix3fProduct(a.Ptr, b.Ptr, res.Ptr);
                return res;
            }

            public static Vector3 operator * (Matrix3f a, Vector3 b)
            {
                float[] res = new float[3];
                Mat3fProductVec3fPointer(a.Ptr, b.x, b.y, b.z, res);
                return new Vector3(res[0], res[1], res[2]);
            }

            public float GetValue(int row, int col)
            {
                //if (row >= GetRows() || col >= GetCols())
                //{
                //    Debug.LogWarning("Getting out of bounds at [" + row + ", " + col + "].");
                //    return 0f;
                //}
                return Matrix3fGetValue(Ptr, row, col);
            }

            public QuaternionResultf ToQuaternion()
            {
                return Matrix3fToQuaternion(Ptr);
            }

            public Quaternion ToQuatUnity()
            {
                var res = Matrix3fToQuaternion(Ptr);
                return new Quaternion(res.x, res.y, res.z, res.w);
            }

            public override string ToString()
            {
                return string.Format("{0}, {1}, {2},\n {3}, {4}, {5},\n {6}, {7}, {8}\n",
                    GetValue(0, 0), GetValue(0, 1), GetValue(0, 2),
                    GetValue(1, 0), GetValue(1, 1), GetValue(1, 2),
                    GetValue(2, 0), GetValue(2, 1), GetValue(2, 2));
            }

            public static Matrix3f MakeEuler(EulerResultf res, string EulerOrder)
            {
                return MakeEuler(res.xAngle, res.yAngle, res.zAngle, EulerOrder);
            }

            public static Matrix3f MakeEuler(float xAngle, float yAngle, float zAngle, string EulerOrder)
            {
                switch (EulerOrder)
                {
                    case "XYZ":
                        return MakeEulerXYZ(xAngle, yAngle, zAngle);
                    case "XZY":
                        return MakeEulerXZY(xAngle, yAngle, zAngle);
                    case "YXZ":
                        return MakeEulerYXZ(xAngle, yAngle, zAngle);
                    case "YZX":
                        return MakeEulerYZX(xAngle, yAngle, zAngle);
                    case "ZXY":
                        return MakeEulerZXY(xAngle, yAngle, zAngle);
                    case "ZYX":
                        return MakeEulerZYX(xAngle, yAngle, zAngle);
                    default:
                        throw new NotImplementedException();
                }
            }

            public static Matrix3f MakeEulerXYZ(float xAngle, float yAngle, float zAngle)
            {
                IntPtr res = Matrix3fMakeEulerXYZ(xAngle, yAngle, zAngle);
                return new Matrix3f(res);
            }

            public static Matrix3f MakeEulerXZY(float xAngle, float yAngle, float zAngle)
            {
                IntPtr res = Matrix3fMakeEulerXZY(xAngle, yAngle, zAngle);
                return new Matrix3f(res);
            }

            public static Matrix3f MakeEulerYXZ(float xAngle, float yAngle, float zAngle)
            {
                IntPtr res = Matrix3fMakeEulerYXZ(xAngle, yAngle, zAngle);
                return new Matrix3f(res);
            }

            public static Matrix3f MakeEulerYZX(float xAngle, float yAngle, float zAngle)
            {
                IntPtr res = Matrix3fMakeEulerYZX(xAngle, yAngle, zAngle);
                return new Matrix3f(res);
            }

            public static Matrix3f MakeEulerZXY(float xAngle, float yAngle, float zAngle)
            {
                IntPtr res = Matrix3fMakeEulerZXY(xAngle, yAngle, zAngle);
                return new Matrix3f(res);
            }

            public static Matrix3f MakeEulerZYX(float xAngle, float yAngle, float zAngle)
            {
                IntPtr res = Matrix3fMakeEulerZYX(xAngle, yAngle, zAngle);
                return new Matrix3f(res);
            }

            /// <summary>
            /// Extract 3x3 rotation matrix to Euler Angle with EulerOrder
            /// </summary>
            /// <param name="EulerOrder"></param>
            /// <returns>Euler Angle Result</returns>
            public EulerResultf ExtractEuler(string EulerOrder)
            {
                // Note: DO NOT take Visual Studio's potential fix IDE0066: Use 'switch' expression
                // Unity 2019.4.18 dosen't support this potential fix.
                switch (EulerOrder)
                {
                    case "XYZ":
                        return ExtractEulerXYZ();
                    case "XZY":
                        return ExtractEulerXZY();
                    case "YXZ":
                        return ExtractEulerYXZ();
                    case "YZX":
                        return ExtractEulerYZX();
                    case "ZXY":
                        return ExtractEulerZXY();
                    case "ZYX":
                        return ExtractEulerZYX();
                    default:
                        throw new NotImplementedException();
                };
            }

            /// <summary>
            /// Extract 3x3 rotation matrix to Euler Angle with EulerOrder XYZ
            /// </summary>
            /// <returns>Euler Angle Result</returns>
            public EulerResultf ExtractEulerXYZ()
            {
                return Matrix3fExtractEulerXYZ(Ptr);
            }

            public EulerResultf ExtractEulerXZY()
            {
                return Matrix3fExtractEulerXZY(Ptr);
            }

            public EulerResultf ExtractEulerYXZ()
            {
                return Matrix3fExtractEulerYXZ(Ptr);
            }

            public EulerResultf ExtractEulerYZX()
            {
                return Matrix3fExtractEulerYZX(Ptr);
            }

            public EulerResultf ExtractEulerZXY()
            {
                return Matrix3fExtractEulerZXY(Ptr);
            }

            public EulerResultf ExtractEulerZYX()
            {
                return Matrix3fExtractEulerZYX(Ptr);
            }
        }
    }
    
}

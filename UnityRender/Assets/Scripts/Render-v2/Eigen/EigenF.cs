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

using System;
using UnityEngine;
using System.Runtime.InteropServices;

namespace RenderV2.Eigen
{
    /// <summary>
    /// Euler Angle Result. It has same structure in C code.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct EulerResultf
    {
        public float xAngle;
        public float yAngle;
        public float zAngle;
        public int unique;

        public override string ToString()
        {
            return string.Format("xAngle={0}, yAngle={1}, zAngle={2}", xAngle, yAngle, zAngle);
        }

        /// <summary>
        /// result in degree for debug
        /// </summary>
        /// <returns></returns>
        public readonly string DegString()
        {
            return string.Format("xAngle={0}, yAngle={1}, zAngle={2}", XDeg(), YDeg(), ZDeg());
        }

        public readonly float XDeg() => xAngle * Mathf.Rad2Deg;
        public readonly float YDeg() => yAngle * Mathf.Rad2Deg;
        public readonly float ZDeg() => zAngle * Mathf.Rad2Deg;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct QuaternionResultf
    {
        public float x;
        public float y;
        public float z;
        public float w;
        public override readonly string ToString()
        {
            return string.Format("x={0}, y={1}, z={2}, w={3}", x, y, z, w);
        }
    }

    /// <summary>
    /// if unity shows DllNotFoundException, please compile extension library by Assets/Plugins/src/CMakeLists.txt
    /// in Windows system, EigenForUnity.dll is required.
    /// </summary>
    public static class EigenBindingf
    {
        // ===========MatrixXf==================================
        //Default
        [DllImport("EigenForUnity")]
        public static extern IntPtr MatrixXfCreate(int rows, int cols);

        [DllImport("EigenForUnity")]
        public static extern IntPtr MatrixXfCreateGaussian(int rows, int cols);

        [DllImport("EigenForUnity")]
        public static extern IntPtr MatrixXfDelete(IntPtr T);

        //Setters and Getters
        [DllImport("EigenForUnity")]
        public static extern int MatrixXfGetRows(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern int MatrixXfGetCols(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXfSetZero(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXfSetSize(IntPtr T, int rows, int cols);

        //Arithmetics
        [DllImport("EigenForUnity")]
        public static extern void MatrixXfAdd(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXfSubtract(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXfProduct(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXfScale(IntPtr lhs, float value, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXfSetValue(IntPtr T, int row, int col, float value);

        [DllImport("EigenForUnity")]
        public static extern float MatrixXfGetValue(IntPtr T, int row, int col);
        // ===========End MatrixXf==============================

        // ===========Matrix3f==================================
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fCreate();

        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fCreateValue(
            float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22
        );

        /// <summary>
        /// Delete Matrix
        /// </summary>
        /// <param name="T"></param>
        [DllImport("EigenForUnity")]
        public static extern void Matrix3fDelete(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern void Matrix3fSetValue(IntPtr T, int row, int col, float value);

        [DllImport("EigenForUnity")]
        public static extern float Matrix3fGetValue(IntPtr T, int row, int col);

        [DllImport("EigenForUnity")]
        public static extern void Matrix3fProduct(IntPtr lhs, IntPtr rhs, IntPtr OUT); // return lhs @ rhs

        [DllImport("EigenForUnity")]
        public static extern void Mat3fProductVec3f(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void Mat3fProductVec3fPointer(IntPtr lhs, float x, float y, float z, [Out] float[] OUT);

        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fFromQuaternion(float x, float y, float z, float w);

        [DllImport("EigenForUnity")]
        public static extern QuaternionResultf Matrix3fToQuaternion(IntPtr T);

        /// <summary>
        /// Convert Euler angle to rotation matrix by XYZ Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3f Pointer (Pointer need to be deleted)</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fMakeEulerXYZ(float xAngle, float yAngle, float zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by XZY Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3f Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fMakeEulerXZY(float xAngle, float yAngle, float zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by YXZ Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3f Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fMakeEulerYXZ(float xAngle, float yAngle, float zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by YZX Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3f Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fMakeEulerYZX(float xAngle, float yAngle, float zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by ZXY Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3f Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fMakeEulerZXY(float xAngle, float yAngle, float zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by ZYX Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3f Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3fMakeEulerZYX(float xAngle, float yAngle, float zAngle);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with XYZ order
        /// </summary>
        /// <param name="T">Pointer of Matrix3f</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultf Matrix3fExtractEulerXYZ(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with XZY order
        /// </summary>
        /// <param name="T">Pointer of Matrix3f</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultf Matrix3fExtractEulerXZY(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with YXZ order
        /// </summary>
        /// <param name="T">Pointer of Matrix3f</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultf Matrix3fExtractEulerYXZ(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with YZX order
        /// </summary>
        /// <param name="T">Pointer of Matrix3f</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultf Matrix3fExtractEulerYZX(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with ZXY order
        /// </summary>
        /// <param name="T">Pointer of Matrix3f</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultf Matrix3fExtractEulerZXY(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with ZYX order
        /// </summary>
        /// <param name="T">Pointer of Matrix3f</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultf Matrix3fExtractEulerZYX(IntPtr T);
    }
}


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
    [System.Runtime.InteropServices.StructLayout(LayoutKind.Sequential)]
    public struct EulerResultd
    {
        public double xAngle;
        public double yAngle;
        public double zAngle;
        public int unique;

        public override string ToString()
        {
            return string.Format("xAngle={0}, yAngle={1}, zAngle={2}", xAngle, yAngle, zAngle);
        }

        /// <summary>
        /// result in degree for debug
        /// </summary>
        /// <returns></returns>
        public string DegString()
        {
            return string.Format("xAngle={0}, yAngle={1}, zAngle={2}", XDeg(), YDeg(), ZDeg());
        }

        public double XDeg() => xAngle * Mathf.Rad2Deg;
        public double YDeg() => yAngle * Mathf.Rad2Deg;
        public double ZDeg() => zAngle * Mathf.Rad2Deg;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct QuaternionResultd
    {
        public double x;
        public double y;
        public double z;
        public double w;
        public override string ToString()
        {
            return string.Format("x={0}, y={1}, z={2}, w={3}", x, y, z, w);
        }
    }

    /// <summary>
    /// if unity shows DllNotFoundException, please compile extension library by Assets/Plugins/src/CMakeLists.txt
    /// in Windows system, EigenForUnity.dll is required.
    /// </summary>
    public static class EigenBindingd
    {
        // ===========MatrixXd==================================
        //Default
        [DllImport("EigenForUnity")]
        public static extern IntPtr MatrixXdCreate(int rows, int cols);

        [DllImport("EigenForUnity")]
        public static extern IntPtr MatrixXdDelete(IntPtr T);

        //Setters and Getters
        [DllImport("EigenForUnity")]
        public static extern int MatrixXdGetRows(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern int MatrixXdGetCols(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXdSetZero(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXdSetSize(IntPtr T, int rows, int cols);

        //Arithmetics
        [DllImport("EigenForUnity")]
        public static extern void MatrixXdAdd(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXdSubtract(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXdProduct(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXdScale(IntPtr lhs, double value, IntPtr OUT);

        [DllImport("EigenForUnity")]
        public static extern void MatrixXdSetValue(IntPtr T, int row, int col, double value);

        [DllImport("EigenForUnity")]
        public static extern double MatrixXdGetValue(IntPtr T, int row, int col);
        // ===========End MatrixXd==============================

        // ===========Matrix3d==================================
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dCreate();

        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dCreateValue(
            double a00, double a01, double a02,
            double a10, double a11, double a12,
            double a20, double a21, double a22
        );

        /// <summary>
        /// Delete Matrix
        /// </summary>
        /// <param name="T"></param>
        [DllImport("EigenForUnity")]
        public static extern void Matrix3dDelete(IntPtr T);

        [DllImport("EigenForUnity")]
        public static extern void Matrix3dSetValue(IntPtr T, int row, int col, double value);

        [DllImport("EigenForUnity")]
        public static extern double Matrix3dGetValue(IntPtr T, int row, int col);

        [DllImport("EigenForUnity")]
        public static extern void Matrix3dProduct(IntPtr lhs, IntPtr rhs, IntPtr OUT); // return lhs @ rhs


        [DllImport("EigenForUnity")]
        public static extern void Mat3fProductVec3f(IntPtr lhs, IntPtr rhs, IntPtr OUT);


        [DllImport("EigenForUnity")]
        public static extern void Mat3fProductVec3fPointer(IntPtr lhs, double x, double y, double z, [Out] double[] OUT);


        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dFromQuaternion(double x, double y, double z, double w);

        [DllImport("EigenForUnity")]
        public static extern QuaternionResultd Matrix3dToQuaternion(IntPtr T);

        /// <summary>
        /// Convert Euler angle to rotation matrix by XYZ Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3d Pointer (Pointer need to be deleted)</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dMakeEulerXYZ(double xAngle, double yAngle, double zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by XZY Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3d Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dMakeEulerXZY(double xAngle, double yAngle, double zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by YXZ Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3d Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dMakeEulerYXZ(double xAngle, double yAngle, double zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by YZX Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3d Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dMakeEulerYZX(double xAngle, double yAngle, double zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by ZXY Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3d Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dMakeEulerZXY(double xAngle, double yAngle, double zAngle);

        /// <summary>
        /// Convert Euler angle to rotation matrix by ZYX Euler order
        /// </summary>
        /// <param name="xAngle"></param>
        /// <param name="yAngle"></param>
        /// <param name="zAngle"></param>
        /// <returns>Eigen Matrix3d Pointer</returns>
        [DllImport("EigenForUnity")]
        public static extern IntPtr Matrix3dMakeEulerZYX(double xAngle, double yAngle, double zAngle);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with XYZ order
        /// </summary>
        /// <param name="T">Pointer of Matrix3d</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultd Matrix3dExtractEulerXYZ(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with XZY order
        /// </summary>
        /// <param name="T">Pointer of Matrix3d</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultd Matrix3dExtractEulerXZY(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with YXZ order
        /// </summary>
        /// <param name="T">Pointer of Matrix3d</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultd Matrix3dExtractEulerYXZ(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with YZX order
        /// </summary>
        /// <param name="T">Pointer of Matrix3d</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultd Matrix3dExtractEulerYZX(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with ZXY order
        /// </summary>
        /// <param name="T">Pointer of Matrix3d</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultd Matrix3dExtractEulerZXY(IntPtr T);

        /// <summary>
        /// Extract 3x3 rotation matrix to euler angle with ZYX order
        /// </summary>
        /// <param name="T">Pointer of Matrix3d</param>
        /// <returns>Euler Angle Result</returns>
        [DllImport("EigenForUnity")]
        public static extern EulerResultd Matrix3dExtractEulerZYX(IntPtr T);
    }
}


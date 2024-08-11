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
using System.Runtime.InteropServices;
using UnityEngine;
using static RenderV2.Eigen.EigenBindingd;

namespace RenderV2
{
    namespace Eigen
    {
        /// <summary>
        /// Wrapper of Eigen::MatrixXd
        /// TODO: Add functions when needed
        /// </summary>
        public class Tensord
        {
            public IntPtr Ptr;

            public Tensord() { Ptr = IntPtr.Zero; }

            public Tensord(int rows, int cols)
            {
                Ptr = MatrixXdCreate(rows, cols);
            }

            //public static Tensor CreateNormal(int rows, int cols)
            //{
            //    return new Tensor() { Ptr = MatrixXdCreateGaussian(rows, cols) };
            //}

            ~Tensord()
            {
                Delete();
            }

            public void Delete()
            {
                if (Ptr != IntPtr.Zero)
                {
                    MatrixXdDelete(Ptr);
                    Ptr = IntPtr.Zero;
                }
            }

            public IntPtr GetPtr() => Ptr;

            public int rows => Ptr == IntPtr.Zero ? 0 : MatrixXdGetRows(Ptr);

            public int cols => Ptr == IntPtr.Zero ? 0 : MatrixXdGetCols(Ptr);

            public int GetRows()
            {
                return MatrixXdGetRows(Ptr);
            }

            public int GetCols()
            {
                return MatrixXdGetCols(Ptr);
            }

            public void SetZero()
            {
                MatrixXdSetZero(Ptr);
            }

            public void SetSize(int rows, int cols)
            {
                MatrixXdSetSize(Ptr, rows, cols);
            }

            public void CheckSize(int row, int col)
            {
                if (row >= GetRows() || col >= GetCols())
                {
                    Debug.LogWarning("Setting out of bounds at [" + row + ", " + col + "].");
                }
            }

            public void SetValue(int row, int col, double value)
            {
                CheckSize(row, col);
                MatrixXdSetValue(Ptr, row, col, value);
            }

            public double GetValue(int row, int col)
            {
                CheckSize(row, col);
                return MatrixXdGetValue(Ptr, row, col);
            }

            public static Tensord operator +(Tensord lhs, Tensord rhs)
            {
                return Add(lhs, rhs);
            }

            public static Tensord Add(Tensord lhs, Tensord rhs, Tensord OUT = null)
            {
                if (lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols())
                {
                    Debug.LogWarning("Incompatible tensor dimensions.");
                }
                else
                {
                    OUT ??= new Tensord(lhs.GetRows(), lhs.GetCols());
                    MatrixXdAdd(lhs.Ptr, rhs.Ptr, OUT.Ptr);
                }
                return OUT;
            }

            public static Tensord operator - (Tensord lhs, Tensord rhs)
            {
                return Subtract(lhs, rhs);
            }

            public static Tensord Subtract(Tensord lhs, Tensord rhs, Tensord OUT = null)
            {
                if (lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols())
                {
                    Debug.LogWarning("Incompatible tensor dimensions.");
                }
                else
                {
                    OUT ??= new Tensord(lhs.GetRows(), lhs.GetCols());
                    MatrixXdSubtract(lhs.Ptr, rhs.Ptr, OUT.Ptr);
                }
                return OUT;
            }

            public static Tensord Product(Tensord lhs, Tensord rhs, Tensord OUT = null)
            {
                if (lhs.GetCols() != rhs.GetRows())
                {
                    Debug.LogWarning("Incompatible tensor dimensions.");
                }
                else
                {
                    OUT ??= new Tensord(lhs.GetRows(), rhs.GetCols());
                    MatrixXdProduct(lhs.Ptr, rhs.Ptr, OUT.Ptr);
                }
                return OUT;
            }

            public static Tensord Scale(Tensord lhs, float value, Tensord OUT)
            {
                OUT ??= new Tensord(lhs.GetRows(), lhs.GetCols());
                MatrixXdScale(lhs.Ptr, value, OUT.Ptr);
                return OUT;
            }

            public double[,] data
            {
                get
                {
                    if (Ptr == IntPtr.Zero) return null;
                    int r = rows, c = cols;
                    var ret = new double[r, c];
                    for (int i = 0; i < r; i++)
                        for (int j = 0; j < c; j++)
                            ret[i, j] = GetValue(i, j);
                    return ret;
                }
                set
                {
                    if (Ptr == IntPtr.Zero) return;
                    int r = rows, c = cols;
                    for (int i = 0; i < r; i++)
                        for (int j = 0; j < c; j++)
                            SetValue(i, j, value[i, j]);
                }
            }

            public Tensor ToFloat32()
            {
                var ret = new Tensor();
                ret.Ptr = MatrixXd_to_f(Ptr);
                return ret;
            }

            [DllImport("EigenForUnity")]
            public static extern IntPtr MatrixXd_to_f(IntPtr ptr);
        }
    }
}
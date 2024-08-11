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
using static RenderV2.Eigen.EigenBindingf;

namespace RenderV2
{
    namespace Eigen
    {
        /// <summary>
        /// Wrapper of Eigen::MatrixXf
        /// TODO: Add functions when needed
        /// </summary>
        public class Tensor
        {
            public IntPtr Ptr;

            public Tensor() { }

            public Tensor(int rows, int cols)
            {
                Ptr = MatrixXfCreate(rows, cols);
            }

            public static Tensor CreateNormal(int rows, int cols)
            {
                return new Tensor() { Ptr = MatrixXfCreateGaussian(rows, cols) };
            }

            ~Tensor()
            {
                Delete();
            }

            public void Delete()
            {
                if (Ptr != IntPtr.Zero)
                {
                    MatrixXfDelete(Ptr);
                    Ptr = IntPtr.Zero;
                }
            }

            public IntPtr GetPtr() => Ptr;

            public int rows => Ptr == IntPtr.Zero ? 0 : MatrixXfGetRows(Ptr);

            public int cols => Ptr == IntPtr.Zero ? 0 : MatrixXfGetCols(Ptr);

            public int GetRows()
            {
                return MatrixXfGetRows(Ptr);
            }

            public int GetCols()
            {
                return MatrixXfGetCols(Ptr);
            }

            public void SetZero()
            {
                MatrixXfSetZero(Ptr);
            }

            public void SetSize(int rows, int cols)
            {
                MatrixXfSetSize(Ptr, rows, cols);
            }

            public void CheckSize(int row, int col)
            {
                if (row >= GetRows() || col >= GetCols())
                {
                    Debug.LogWarning("Setting out of bounds at [" + row + ", " + col + "].");
                }
            }

            public void SetValue(int row, int col, float value)
            {
                CheckSize(row, col);
                MatrixXfSetValue(Ptr, row, col, value);
            }

            public void SetValue(int row, int col, double value)
            {
                CheckSize(row, col);
                MatrixXfSetValue(Ptr, row, col, (float)value);
            }

            public float GetValue(int row, int col)
            {
                CheckSize(row, col);
                return MatrixXfGetValue(Ptr, row, col);
            }

            public static Tensor operator + (Tensor lhs, Tensor rhs)
            {
                return Add(lhs, rhs);
            }

            public static Tensor Add(Tensor lhs, Tensor rhs, Tensor OUT = null)
            {
                if (lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols())
                {
                    Debug.LogWarning("Incompatible tensor dimensions.");
                }
                else
                {
                    OUT ??= new Tensor(lhs.GetRows(), lhs.GetCols());
                    MatrixXfAdd(lhs.Ptr, rhs.Ptr, OUT.Ptr);
                }
                return OUT;
            }

            public static Tensor operator - (Tensor lhs, Tensor rhs)
            {
                return Subtract(lhs, rhs);
            }

            public static Tensor Subtract(Tensor lhs, Tensor rhs, Tensor OUT = null)
            {
                if (lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols())
                {
                    Debug.LogWarning("Incompatible tensor dimensions.");
                }
                else
                {
                    OUT ??= new Tensor(lhs.GetRows(), lhs.GetCols());
                    MatrixXfSubtract(lhs.Ptr, rhs.Ptr, OUT.Ptr);
                }
                return OUT;
            }

            public static Tensor Product(Tensor lhs, Tensor rhs, Tensor OUT)
            {
                if (lhs.GetCols() != rhs.GetRows())
                {
                    Debug.LogWarning("Incompatible tensor dimensions.");
                }
                else
                {
                    OUT ??= new Tensor(lhs.GetRows(), rhs.GetCols());
                    MatrixXfProduct(lhs.Ptr, rhs.Ptr, OUT.Ptr);
                }
                return OUT;
            }

            public static Tensor Scale(Tensor lhs, float value, Tensor OUT)
            {
                OUT ??= new Tensor(lhs.GetRows(), lhs.GetCols());
                MatrixXfScale(lhs.Ptr, value, OUT.Ptr);
                return OUT;
            }

            public float[,] data
            {
                get
                {
                    if (Ptr == IntPtr.Zero) return null;
                    int r = rows, c = cols;
                    var ret = new float[r, c];
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

            public Tensord ToFloat64()
            {
                var ret = new Tensord();
                ret.Ptr = MatrixXf_to_d(Ptr);
                return ret;
            }

            [DllImport("EigenForUnity")]
            public static extern IntPtr MatrixXf_to_d(IntPtr ptr);
        }
    }
}
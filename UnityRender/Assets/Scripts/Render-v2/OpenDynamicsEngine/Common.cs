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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

using UnityEngine;


namespace RenderV2.ODE
{
    public enum JointType
    {
        dJointTypeNone = 0,
        dJointTypeBall,
        dJointTypeHinge,
        dJointTypeSlider,
        dJointTypeContact,
        dJointTypeUniversal,
        dJointTypeHinge2,
        dJointTypeFixed,
        dJointTypeNull,
        dJointTypeAMotor,
        dJointTypeLMotor,
        dJointTypePlane2D,
        dJointTypePR,
        dJointTypePU,
        dJointTypePiston,
        dJointTypeContact2,
        dJointTypeContactMaxForce,
        dJointTypeEmptyBall
    }

    public enum CommonODE { dMaxUserClasses = 4}
    public enum GeomType
    {
        dSphereClass = 0,
        dBoxClass,
        dCapsuleClass,
        dCylinderClass,
        dPlaneClass,
        dRayClass,
        dConvexClass,
        dGeomTransformClass,
        dTriMeshClass,
        dHeightfieldClass,

        dFirstSpaceClass,
        dSimpleSpaceClass = dFirstSpaceClass,
        dHashSpaceClass,
        dSweepAndPruneSpaceClass,
        dQuadTreeSpaceClass,
        dLastSpaceClass = dQuadTreeSpaceClass,

        dFirstUserClass,
        dLastUserClass = dFirstUserClass + CommonODE.dMaxUserClasses - 1,
        dGeomNumClasses
    }

    public enum ParamType
    {
        dParamLoStop = 0,
        dParamHiStop,
        dParamVel,
        dParamFMax,
        dParamFudgeFactor,
        dParamBounce,
        dParamCFM,
        dParamStopERP,
        dParamStopCFM,

        // parameters for suspension
        dParamSuspensionERP,
        dParamSuspensionCFM,
        dParamERP,
        dParamVelMax,

        dParamLoStop1 = 0x000,
        dParamHiStop1,
        dParamVel1,
        dParamFMax1,
        dParamFudgeFactor1,
        dParamBounce1,
        dParamCFM1,
        dParamStopERP1,
        dParamStopCFM1,
        
        //parameters for suspension
        dParamSuspensionERP1,
        dParamSuspensionCFM1,
        dParamERP1,
        dParamVelMax1,

        dParamLoStop2 = 0x100,
        dParamHiStop2,
        dParamVel2,
        dParamFMax2,
        dParamFudgeFactor2,
        dParamBounce2,
        dParamCFM2,
        dParamStopERP2,
        dParamStopCFM2,
        // parameters for suspension
        dParamSuspensionERP2,
        dParamSuspensionCFM2,
        dParamERP2,
        dParamVelMax2,

        dParamLoStop3 = 0x200,
        dParamHiStop3,
        dParamVel3,
        dParamFMax3,
        dParamFudgeFactor3,
        dParamBounce3,
        dParamCFM3,
        dParamStopERP3,
        dParamStopCFM3,
        // parameters for suspension
        dParamSuspensionERP3,
        dParamSuspensionCFM3,
        dParamERP3,
        dParamVelMax3
    }

    public enum ContactType
    {
        dContactMu2 = 0x001,
        dContactFDir1 = 0x002,
        dContactBounce = 0x004,
        dContactSoftERP = 0x008,
        dContactSoftCFM = 0x010,
        dContactMotion1 = 0x020,
        dContactMotion2 = 0x040,
        dContactMotionN = 0x080,
        dContactSlip1 = 0x100,
        dContactSlip2 = 0x200,

        dContactApprox0 = 0x0000,
        dContactApprox1_1 = 0x1000,
        dContactApprox1_2 = 0x2000,
        dContactApprox1 = 0x3000
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Int32Wrapper
    {
        public int value;
        public readonly int GetValue() => value;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Int64Wrapper
    {
        public long value;
        public readonly long GetValue() => value;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct FloatWrapper
    {
        public float value;
        public readonly float GetValue() => value;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DoubleWrapper
    {
        public double value;
        public readonly double GetValue() => value;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct dVector3
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public double[] values;

        /* public dVector3()
        {
            values = new double[4] { 0, 0, 0, 0 };
        } */

        public dVector3(double x, double y, double z)
        {
            values = new double[4] { x, y, z, 0.0 };
        }

        public dVector3(Vector3 v)
        {
            values = new double[4] { v.x, v.y, v.z, 0.0 };
        }

        public dVector3(double[] v)
        {
            values = new double[4] { v[0], v[1], v[2], 0.0 };
        }

        public double x { readonly get => values[0]; set => values[0] = value; }
        public double y { readonly get => values[1]; set => values[1] = value; }
        public double z { readonly get => values[2]; set => values[2] = value; }

        public void setZero()
        {
            if (values != null)
            {
                for (int i = 0; i < 4; i++) values[i] = 0.0;
            }
            else
            {
                values = new double[4];
            }
        }

        public readonly void setValue(double[] v)
        {
            for (int i = 0; i < 3; i++) values[i] = v[i];
        }

        public override readonly string ToString()
        {
            if (values != null) return "(" + values[0] + ", " + values[1] + ", " + values[2] + ")";
            else return "(null)";
        }

        public readonly Vector3 ToVec3() => new Vector3(
            Convert.ToSingle(values[0]), Convert.ToSingle(values[1]), Convert.ToSingle(values[2]));

        public readonly double[] ToArray()
        {
            var res = new double[3];
            for(int i = 0; i < 3; i++) res[i] = values[i];
            return res;
        }

        public readonly double this[int index]
        {
            get => values[index];
            set => values[index] = value;
        }

        public readonly bool IsClose3(dVector3 other, double eps = 1e-14)
        {
            for(int i = 0; i < 3; i++) if (Math.Abs(values[i] - other.values[i]) > eps) return false;
            return true;
        }

        public readonly bool IsClose4(dVector3 other, double eps = 1e-14)
        {
            for (int i = 0; i < 4; i++) if (Math.Abs(values[i] - other.values[i]) > eps) return false;
            return true;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct dMatrix3
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 12)]
        public double[] values;

        static readonly int[] ids = new int[] { 0, 1, 2, 4, 5, 6, 8, 9, 10 };

        public dMatrix3(double[] values)
        {
            if (values.Length != 12)
                throw new ArgumentException("Array must have 12 elements");
            this.values = values;
        }

        public readonly double this[int index]
        {
            get => values[index];
            set => values[index] = value;
        }

        public readonly int Length => 12;

        public readonly double[] GetValues() => values;

        public readonly void setZero()
        {
            if (values != null)
                for (int i = 0; i < 12; i++) values[i] = 0.0;
        }

        public override readonly string ToString()
        {
            if (values != null)
            {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                    {
                        sb.Append(values[i * 4 + j]);
                        sb.Append(", ");
                    }

                return sb.ToString();
            }
            else
            {
                return "null";
            }
        }

        public static dMatrix3 BuildFromDense(double[,] denseMat3)
        {
            dMatrix3 odeMat3 = new dMatrix3();
            odeMat3[0] = denseMat3[0, 0];
            odeMat3[1] = denseMat3[0, 1];
            odeMat3[2] = denseMat3[0, 2];
            odeMat3[3] = 0;
            odeMat3[4] = denseMat3[1, 0];
            odeMat3[5] = denseMat3[1, 1];
            odeMat3[6] = denseMat3[1, 2];
            odeMat3[7] = 0;
            odeMat3[8] = denseMat3[2, 0];
            odeMat3[9] = denseMat3[2, 1];
            odeMat3[10] = denseMat3[2, 2];
            odeMat3[11] = 0;
            return odeMat3;
        }

        public static dMatrix3 BuildFromDense(double[] denseMat3)
        {
            return (new dMatrix3()).SetValueDense(denseMat3);
        }

        public dMatrix3 SetValueDense(double[] denseMat3)
        {
            for (int i = 0; i < ids.Length; i++)
            {
                values[ids[i]] = denseMat3[i];
            }
            values[3] = values[7] = values[11] = 0;
            return this;
        }

        public dMatrix3 SetValueDense(float[] denseMat3)
        {
            for (int i = 0; i < ids.Length; i++)
            {
                values[ids[i]] = denseMat3[i];
            }
            values[3] = values[7] = values[11] = 0;
            return this;
        }

        public static dMatrix3 BuildFromODEPtr(IntPtr ptr)
        {
            return new dMatrix3(CommonFunc.ArrayFromDoublePtr(ptr, 12));
        }
        
        public void Add_(double[,] m)
        {
            values[0] += m[0, 0];
            values[1] += m[0, 1];
            values[2] += m[0, 2];
            values[4] += m[1, 0];
            values[5] += m[1, 1];
            values[6] += m[1, 2];
            values[8] += m[2, 0];
            values[9] += m[2, 1];
            values[10] += m[2, 2];
        }

        public readonly double[,] ToDenseMat2D()
        {
            var m = new double[3, 3];
            m[0, 0] = values[0];
            m[0, 1] = values[1];
            m[0, 2] = values[2];
            m[1, 0] = values[4];
            m[1, 1] = values[5];
            m[1, 2] = values[6];
            m[2, 0] = values[8];
            m[2, 1] = values[9];
            m[2, 2] = values[10];
            return m;
        }

        public readonly double[] ToDenseMat()
        {
            double[] odeMat3 = values;
            double[] denseMat3 = new double[9];
            for (int i = 0; i < 9; i++)
                denseMat3[i] = odeMat3[ids[i]];
            return denseMat3;
        }

        public readonly bool IsClose(dMatrix3 other, double eps = 1e-14)
        {
            for(int i = 0; i < 9; i++) if (Math.Abs(values[ids[i]] - other.values[ids[i]]) > eps) return false;
            return true;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector2Wrapper
    {
        public double x, y;
        public readonly Vector2 ToVec2() => new Vector3(Convert.ToSingle(x), Convert.ToSingle(y));

        public static Vector2Wrapper FromVec3(Vector2 v) => new Vector2Wrapper { x = v.x, y = v.y };

        public override readonly string ToString() => "(" + x + ", " + y + ")";
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3Wrapper
    {
        public double x, y, z;
        public readonly Vector3 ToVec3() => new Vector3(Convert.ToSingle(x), Convert.ToSingle(y), Convert.ToSingle(z));

        public static Vector3Wrapper FromVec3(Vector3 v) => new Vector3Wrapper { x = v.x, y = v.y, z = v.z };

        public override readonly string ToString() => "(" + x + ", " + y + ", " + z + ")";
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector4Wrapper
    {
        public double a, b, c, d;
        public readonly Vector4 ToVec4()
        {
            return new Vector4(Convert.ToSingle(a), Convert.ToSingle(b), Convert.ToSingle(c), Convert.ToSingle(d));
        }
        public static Vector4Wrapper FromVec4(Vector4 v)
        {
            return new Vector4Wrapper { a = v.x, b = v.y, c = v.z, d = v.w };
        }
        public override readonly string ToString() => "(" + a + ", " + b + ", " + c + ", " + d + ")";
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct Quat4Wrapper
    {
        public double x, y, z, w;
        public readonly Quaternion ToQuat4() => new Quaternion(Convert.ToSingle(x), Convert.ToSingle(y), Convert.ToSingle(z), Convert.ToSingle(w));

        public static Quat4Wrapper FromQuat4(Quaternion q) => new Quat4Wrapper { x = q.x, y = q.y, z = q.z, w = q.w };
        public static Quat4Wrapper FromArray(double[] q) => new Quat4Wrapper { x = q[0], y = q[1], z = q[2], w = q[3] };
        public override readonly string ToString() => "(" + x + ", " + y + ", " + z + ", " + w + ")";
    };

    // [StructLayout(LayoutKind.Sequential)]
    public class GeomData
    {
        public double friction;
        public double bounce;

        public static readonly int MaxIgnore = 64;
        // [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
        public ulong[] ignore;

        public int ignore_count;

        public GeomData()
        {
            friction = 0.8f;
            bounce = 0;
            ignore = new ulong[MaxIgnore];
            ignore_count = 0;
        }

        public void SetFriction(double friction_) => friction = friction_;
        public double GetFriction() => friction;
        public void SetBounce(double bounce_) => bounce = bounce_;
        public double GetBounce() => bounce;

        public void AddIgnore(ulong uid)
        {
            if (ignore is null)
            {
                ignore = new ulong[MaxIgnore];
            }
            ignore[ignore_count++] = uid;
        }

        public bool IgnoreContains(ulong uid)
        {
            if (ignore is null) return false;
            for(int i = 0; i < ignore_count; i++)
            {
                if (ignore[i] == uid) return true;
            }
            return false;
        }
    };

    public class GeomMemoryPool<T> where T: new()
    {
        private readonly T[] _pool;
        private readonly Stack<int> _availableIndices;
        private readonly int _maxSize;

        public GeomMemoryPool(int maxSize)
        {
            _maxSize = maxSize;
            _pool = new T[maxSize];
            _availableIndices = new Stack<int>(maxSize);

            // Initialize the stack with all indices
            for (int i = maxSize - 1; i >= 0; i--)
            {
                _availableIndices.Push(i);
                _pool[i] = new T();
            }
        }

        public T[] GetPool() => _pool;

        public int Alloc(out T item)
        {
            if (_availableIndices.Count > 0)
            {
                int index = _availableIndices.Pop();
                item = _pool[index];
                return index;
            }

            // If no available slot is found, throw an exception
            throw new InvalidOperationException("Memory pool exhausted.");
        }

        public void Dealloc(int index)
        {
            if (index < 0 || index >= _maxSize)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Index out of range.");
            }

            if (!_availableIndices.Contains(index))
            {
                _pool[index] = default;
                _availableIndices.Push(index);
            }
            else
            {
                throw new InvalidOperationException("Memory at this index is already deallocated.");
            }
        }

        public T Get(int index)
        {
            if (index < 0 || index >= _maxSize)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Index out of range.");
            }

            return _pool[index];
        }

        public void Set(int index, T item)
        {
            if (index < 0 || index >= _maxSize)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Index out of range.");
            }

            _pool[index] = item;
        }

        public T this[int index]
        {
            get => _pool[index];
            set => _pool[index] = value;
        }
    }


    public static class CommonFunc
    {
        static bool _InitCalled = false;
        public static readonly double dInfinity = getInfinity();
        public static bool InitCalled { get => _InitCalled; }
        public static GeomMemoryPool<GeomData> GeomPool;

        public static void InitODE(int MaxGeomNum = 128)
        {
            dInitODE();
            dRandSetSeed(0);
            GeomPool = new GeomMemoryPool<GeomData>(MaxGeomNum);
            _InitCalled = true;
        }

        public static void CloseODE()
        {
            dCloseODE();
            _InitCalled = false;
        }

        public static float[] CopyArr(float[] arr)
        {
            var res = new float[arr.Length];
            for(int i = 0; i < arr.Length; i++) res[i] = arr[i];
            return res;
        }

        public static double[,] EyeArrayD(int length)
        {
            double[,] res = new double[length, length];
            for (int i = 0; i < length; i++)
                for (int j = 0; j < length; j++)
                    res[i, j] = 0;
            for (int i = 0; i < length; i++)
                res[i, i] = 1;
            return res;
        }

        public static double[,] EyeArray3D()
        {
            return new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        }

        public static double[][] EyeArray3DImpl2()
        {
            var res = new double[3][];
            res[0] = new double[3] { 1, 0, 0 };
            res[1] = new double[3] {0, 1, 0 };
            res[2] = new double[3] {0, 0, 1};
            return res;
        }

        public static double[] ArrayFromDoublePtr(IntPtr ptr, int length)
        {
            double[] res = new double[length];
            Marshal.Copy(ptr, res, 0, length);
            return res;
        }

        public static double[] ArrayFromDoublePtr3(IntPtr ptr)
        {   
            double[] res = new double[3];
            Marshal.Copy(ptr, res, 0, 3);
            return res;
        }

        public static Vector3 Vec3FromDoubleArr(double[] arr)
        {
            return new Vector3(Convert.ToSingle(arr[0]), Convert.ToSingle(arr[1]), Convert.ToSingle(arr[2]));
        }

        public static Vector3 Vec3FromDoublePtr(IntPtr ptr)
        {
            double[] res = new double[3];
            Marshal.Copy(ptr, res, 0, 3);
            return CommonFunc.Vec3FromDoubleArr(res);
        }

        public static Quaternion QuatFromODEPtr(IntPtr ptr)
        {
            double[] q = new double[4];
            Marshal.Copy(ptr, q, 0, 4);
            return QuatFromODEArr(q);
        }

        public static Quaternion QuatFromODEArr(double[] ptr)
        {
            return new Quaternion(Convert.ToSingle(ptr[1]),
                Convert.ToSingle(ptr[2]),
                Convert.ToSingle(ptr[3]),
                Convert.ToSingle(ptr[0]));
        }

        public static double[] QuatArrFromODEArr(double[] ptr, bool inplace = true) // TODO: Check
        {
            double x = ptr[1], y = ptr[2], z = ptr[3], w = ptr[0];
            if (!inplace) ptr = new double[4];
            ptr[0] = x; ptr[1] = y; ptr[2] = z; ptr[3] = w;
            return ptr;
        }

        public static double[] QuatArrFromODEPtr(IntPtr ptr)
        {
            double[] q = new double[4];
            Marshal.Copy(ptr, q, 0, 4);
            double x = q[1], y = q[2], z = q[3], w = q[0];
            q[0] = x; q[1] = y; q[2] = z; q[3] = w;
            return q;
        }

        public static double[] QuaternionToODEArray(Quaternion q)
        {
            double[] ptr = new double[4];
            ptr[0] = q.w;
            ptr[1] = q.x;
            ptr[2] = q.y;
            ptr[3] = q.z;
            return ptr;
        }

        public static float[,] QuatToMatrixf(Quaternion q)
        {
            float[,] result = new float[3, 3];
            float x = q.x, y = q.y, z = q.z, w = q.w;

            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;
            float w2 = w * w;

            float xy = x * y;
            float zw = z * w;
            float xz = x * z;
            float yw = y * w;
            float yz = y * z;
            float xw = x * w;

            result[0, 0] = x2 - y2 - z2 + w2;
            result[0, 1] = 2 * (xy - zw);
            result[0, 2] = 2 * (xz + yw);

            result[1, 0] = 2 * (xy + zw);
            result[1, 1] = -x2 + y2 - z2 + w2;
            result[1, 2] = 2 * (yz - xw);

            result[2, 0] = 2 * (xz - yw);
            result[2, 1] = 2 * (yz + xw);
            result[2, 2] = -x2 - y2 + z2 + w2;

            return result;
        }

        public static double[,] QuatToMatrixd(Quaternion q)
        {
            double[,] result = new double[3, 3];
            double x = q.x, y = q.y, z = q.z, w = q.w;

            double x2 = x * x;
            double y2 = y * y;
            double z2 = z * z;
            double w2 = w * w;

            double xy = x * y;
            double zw = z * w;
            double xz = x * z;
            double yw = y * w;
            double yz = y * z;
            double xw = x * w;

            result[0, 0] = x2 - y2 - z2 + w2;
            result[0, 1] = 2 * (xy - zw);
            result[0, 2] = 2 * (xz + yw);

            result[1, 0] = 2 * (xy + zw);
            result[1, 1] = -x2 + y2 - z2 + w2;
            result[1, 2] = 2 * (yz - xw);

            result[2, 0] = 2 * (xz - yw);
            result[2, 1] = 2 * (yz + xw);
            result[2, 2] = -x2 - y2 + z2 + w2;

            return result;
        }

        /// <summary>
        /// Deallocate some extra memory used by ODE that can not be deallocated
        /// using the normal destroy functions.
        /// </summary>
        [DllImport("ModifyODE")]
        public static extern void dCloseODE();

        /// <summary>
        /// Initialize some ODE internals. This will be called for you when you
        /// call InitODE(), but you should call this again if you CloseODE().'''
        /// </summary>
        [DllImport("ModifyODE")]
        public static extern void dInitODE();

        [DllImport("ModifyODE")]
        public static extern void dInitODE2();

        [DllImport("ModifyODE")]
        public static extern uint dRandGetSeed();

        [DllImport("ModifyODE")]
        public static extern void dRandSetSeed(uint seed);

        [DllImport("ModifyODE")]
        public static extern double getInfinity();

        //[DllImport("ModifyODE")]
        //public static extern void DenseMat3ToODEMat3(out dMatrix3 odeMat3, UIntPtr denseMat3In, int offset);
    }
}
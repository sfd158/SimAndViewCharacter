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
using static RenderV2.ODE.CommonFunc;

namespace RenderV2.ODE
{
    public class Inertia
    {
        private double[,] I;  // 3x3 inertia matrix
        private double mass;

        public Inertia()
        {
            I = new double[3, 3];
            mass = 0.0f;
        }

        public Inertia(dMass dmass)
        {
            mass = dmass.mass;
            I = dmass.I.ToDenseMat2D();
        }

        // Translates the inertia tensor by (tx, ty, tz)
        public double[,] TransInertia(double tx, double ty, double tz)
        {
            double[,] t = new double[3, 3];

            t[0, 0] = mass * (ty * ty + tz * tz);
            t[0, 1] = -mass * tx * ty;
            t[0, 2] = -mass * tx * tz;
            t[1, 0] = t[0, 1];
            t[1, 1] = mass * (tx * tx + tz * tz);
            t[1, 2] = -mass * ty * tz;
            t[2, 0] = t[0, 2];
            t[2, 1] = t[1, 2];
            t[2, 2] = mass * (tx * tx + ty * ty);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    I[i, j] += t[i, j];

            return I;
        }

        // Overload of TransInertia that accepts a Vector3 for translation
        public double[,] TransInertia(Vector3 v)
        {
            return TransInertia(v.x, v.y, v.z);
        }

        public double[,] RotInertia(Quaternion q)
        {
            var mat = QuatToMatrixd(q);
            return RotInertia(mat);
        }

        // Rotates the inertia tensor using a 3x3 rotation matrix
        public double[,] RotInertia(double[,] R)
        {
            double[,] bI = new double[3, 3] {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            double[,] newI = new double[3, 3] {{0, 0, 0 }, {0, 0, 0}, {0, 0, 0}};

            // Compute R * I
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        bI[i, j] += R[i, k] * I[k, j];

            // Compute (R * I) * R^T
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        newI[i, j] += bI[i, k] * R[j, k];

            I = newI;
            return I;
        }

        public void Add(Inertia o)
        {
            mass += o.mass;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    I[i, j] += o.I[i, j];
        }

        public void AddToMass(dMass dmass)
        {
            dmass.mass += mass;
            dmass.I.Add_(I);
        }

        public dMass ToMass()
        {
            var ret = new dMass();
            ret.mass = mass;
            ret.I = dMatrix3.BuildFromDense(I);
            // TODO: check c.
            return ret;
        }
    }
    /// <summary>
    /// mass of rigid body. same structure as ODE.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct dMass
    {
        public double mass;
        public dVector3 c; // align with c code.
        public dMatrix3 I; // Inertia matrix. align with c code.

        public static dMass BuildZero()
        {
            return new dMass { mass = 0, c = new dVector3(0, 0, 0), I = new dMatrix3(new double[12]) };
        }

        public dMass(double mass, dVector3 c, dMatrix3 I)
        {
            this.mass = mass;
            this.c = c;
            this.I = I;
        }

        public void setZero()
        {
            mass = 0.0;
            c.setZero();
            I.setZero();
        }

        public override readonly string ToString()
        {
            return "mass: " + mass + " " + c.ToString() + " " + I.ToString();
        }

        public static bool operator == (dMass lhs, dMass rhs)
        {
            const double eps = 1e-14;
            if (Math.Abs(lhs.mass - rhs.mass) > eps) return false;
            if (!lhs.c.IsClose3(rhs.c, eps)) return false;
            if (!lhs.I.IsClose(rhs.I, eps)) return false;
            return true;
        }

        public static bool operator != (dMass lhs, dMass rhs)
        {
            const double eps = 1e-14;
            if (Math.Abs(lhs.mass - rhs.mass) > eps) return true;
            if (!lhs.c.IsClose3(rhs.c, eps)) return true;
            if (!lhs.I.IsClose(rhs.I, eps)) return true;
            return false;
        }

        public override readonly bool Equals(object other)
        {
            if (other == null) return false;
            if (GetType() != other.GetType()) return false;
            return this == (dMass)other;
        }

        public override int GetHashCode() => base.GetHashCode();
    }

    /// <summary>
    /// mass of rigid body.
    /// </summary>
    public class ODEMass
    {
        public dMass dmass;
        public ODEMass()
        {
            dmass = dMass.BuildZero();
        }

        public double MassValue
        {
            get => dmass.mass;
            set => dmass.mass = value;
        }
        public override string ToString() => dmass.ToString();

        public override bool Equals(object other)
        {
            if (other == null) return false;
            if (GetType() != other.GetType()) return false;
            return dmass == ((ODEMass)other).dmass;
        }

        public override int GetHashCode() => base.GetHashCode();

        #region FuncWrapper
        public ODEMass SetZero()
        {
            dMassSetZero(ref dmass);
            return this;
        }

        public ODEMass SetParameters(double mass, double cgx, double cgy, double cgz, double I11, double I22, double I33, double I12, double I13, double I23)
        {
            dMassSetParameters(ref dmass, mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23);
            return this;
        }

        public ODEMass SetSphere(double density, double radius)
        {
            dMassSetSphere(ref dmass, density, radius);
            return this;
        }

        public ODEMass SetSphereTotal(double totalMass, double radius)
        {
            dMassSetSphereTotal(ref dmass, totalMass, radius);
            return this;
        }

        public ODEMass SetCapsule(double density, int direction, double radius, double length)
        {
            dMassSetCapsule(ref dmass, density, direction, radius, length);
            return this;
        }

        public ODEMass SetCapsuleTotal(double totalMass, int direction, double radius, double length)
        {
            dMassSetCapsuleTotal(ref dmass, totalMass, direction, radius, length);
            return this;
        }

        public ODEMass SetCylinder(double density, int direction, double radius, double height)
        {
            dMassSetCylinder(ref dmass, density, direction, radius, height);
            return this;
        }

        public ODEMass SetCylinderTotal(double totalMass, int direction, double radius, double height)
        {
            dMassSetCylinderTotal(ref dmass, totalMass, direction, radius, height);
            return this;
        }

        public ODEMass SetBox(double density, double lx, double ly, double lz)
        {
            dMassSetBox(ref dmass, density, lx, ly, lz);
            return this;
        }

        public ODEMass SetBox(double density, Vector3 length)
        {
            dMassSetBox(ref dmass, density, length.x, length.y, length.z);
            return this;
        }

        public ODEMass SetBoxTotal(double totalMass, double lx, double ly, double lz)
        {
            dMassSetBoxTotal(ref dmass, totalMass, lx, ly, lz);
            return this;
        }

        public ODEMass Adjust(double newMass)
        {
            dMassAdjust(ref dmass, newMass);
            return this;
        }

        public ODEMass Add(ODEMass b)
        {
            dMassAdd(ref dmass, ref b.dmass);
            return this;
        }
        #endregion

        #region ImportWrapper
        [DllImport("ModifyODE")]
        private static extern void dMassSetZero(ref dMass mass);

        [DllImport("ModifyODE")]
        private static extern void dMassSetParameters(ref dMass mass, double massValue, double cgx, double cgy, double cgz, double I11, double I22, double I33, double I12, double I13, double I23);

        [DllImport("ModifyODE")]
        private static extern void dMassSetSphere(ref dMass mass, double density, double radius);

        [DllImport("ModifyODE")]
        private static extern void dMassSetSphereTotal(ref dMass mass, double totalMass, double radius);

        [DllImport("ModifyODE")]
        private static extern void dMassSetCapsule(ref dMass mass, double density, int direction, double radius, double length);

        [DllImport("ModifyODE")]
        private static extern void dMassSetCapsuleTotal(ref dMass mass, double totalMass, int direction, double radius, double length);

        [DllImport("ModifyODE")]
        private static extern void dMassSetCylinder(ref dMass mass, double density, int direction, double radius, double height);

        [DllImport("ModifyODE")]
        private static extern void dMassSetCylinderTotal(ref dMass mass, double totalMass, int direction, double radius, double height);

        [DllImport("ModifyODE")]
        private static extern void dMassSetBox(ref dMass mass, double density, double lx, double ly, double lz);

        [DllImport("ModifyODE")]
        private static extern void dMassSetBoxTotal(ref dMass mass, double totalMass, double lx, double ly, double lz);

        [DllImport("ModifyODE")]
        private static extern void dMassAdjust(ref dMass mass, double newMass);

        [DllImport("ModifyODE")]
        private static extern void dMassAdd(ref dMass a, ref dMass b);
        #endregion
    }
} 
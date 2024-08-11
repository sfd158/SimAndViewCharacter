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

using System.Runtime.InteropServices;
using System;
using UnityEngine;


namespace RenderV2.ODE
{
    /// <summary>
    ///  hinge joint
    /// </summary>
    public class ODEHingeJoint : ODEJoint
    {
        // Constructor
        public ODEHingeJoint(ODEWorld world, JointGroup jointgroup = null)
        {
            if (world is null) throw new ArgumentNullException(nameof(world));

            UIntPtr gid = UIntPtr.Zero;
            if (!(jointgroup is null))
            {
                gid = jointgroup.gid;
            }
            jid = dJointCreateHinge(world.wid, gid);

            _world = world;
            jointgroup?.AddJoint(this);
        }

        public Vector3 GetAnchorVec3()
        {
            double[] p = new double[3];
            dJointGetHingeAnchor(jid, p);
            return CommonFunc.Vec3FromDoubleArr(p);
        }

        /// <summary>
        /// Set the hinge anchor which must be given in world coordinates.
        /// </summary>
        /// <param name="pos"></param>
        /// <exception cref="ArgumentException"></exception>
        public void SetAnchor(double[] pos)
        {
            if (pos.Length != 3) throw new ArgumentException("Position array must have exactly 3 elements.");
            dJointSetHingeAnchor(jid, pos[0], pos[1], pos[2]);
        }

        /// <summary>
        /// Get the joint anchor point, in world coordinates. This returns
        /// the point on body 1. If the joint is perfectly satisfied, this
        /// will be the same as the point on body 2.
        /// </summary>
        /// <returns></returns>
        public new double[] GetAnchor()
        {
            double[] p = new double[3];
            dJointGetHingeAnchor(jid, p);
            return p;
        }

        /// <summary>
        /// Get the joint anchor point, in world coordinates. This returns
        /// the point on body 2. If the joint is perfectly satisfied, this
        // will be the same as the point on body 1.
        /// </summary>
        /// <returns></returns>
        public new double[] GetAnchor2()
        {
            double[] p = new double[3];
            dJointGetHingeAnchor2(jid, p);
            return p;
        }

        // Get the first joint anchor point as raw data
        public double[] GetAnchor1Raw()
        {
            if (jid == UIntPtr.Zero) throw new ArgumentNullException();
            IntPtr res = dJointGetHingeAnchor1Raw(jid);
            return CommonFunc.ArrayFromDoublePtr3(res);
        }

        // Get the second joint anchor point as raw data
        public double[] GetAnchor2Raw()
        {
            IntPtr res = dJointGetHingeAnchor2Raw(jid);
            return CommonFunc.ArrayFromDoublePtr3(res);
        }

        /// <summary>
        /// Set the hinge axis
        /// </summary>
        /// <param name="axis"></param>
        /// <exception cref="ArgumentException"></exception>
        public void SetAxis(double[] axis)
        {
            if (axis.Length != 3) throw new ArgumentException("Axis array must have exactly 3 elements.");
            dJointSetHingeAxis(jid, axis[0], axis[1], axis[2]);
        }

        public void SetAxis(Vector3 axis)
        {
            dJointSetHingeAxis(jid, axis.x, axis.y, axis.z);
        }

        /// <summary>
        /// Get the hinge axis
        /// </summary>
        /// <returns></returns>
        public double[] GetAxis()
        {
            double[] a = new double[3];
            dJointGetHingeAxis(jid, a);
            return a;
        }
        
        public Vector3 GetAxisVec3()
        {
            double[] a = new double[3];
            dJointGetHingeAxis(jid, a);
            return CommonFunc.Vec3FromDoubleArr(a);
        }

        public double[] GetHingeAxis1()
        {
            double[] a = new double[3];
            dJointGetHingeAxis1(jid, a);
            return a;
        }

        public double[] GetHingeAxis1Raw()
        {
            double[] a = new double[3];
            dJointGetHingeAxis1Raw(jid, a);
            return a;
        }

        public double[] GetHingeAxis2()
        {
            double[] a = new double[3];
            dJointGetHingeAxis2(jid, a);
            return a;
        }

        public double[] GetHingeAxis2Raw()
        {
            double[] a = new double[3];
            dJointGetHingeAxis2Raw(jid, a);
            return a;
        }

        public Quaternion HingeQRel()
        {
            double[] q = new double[4];
            dJointGetHingeQRel(jid, q);
            return CommonFunc.QuatFromODEArr(q);
        }

        public void SetHingeAxisOffset(double x, double y, double z, double dangle)
        {
            dJointSetHingeAxisOffset(jid, x, y, z, dangle);
        }

        /// <summary>
        /// Get the hinge angle. The angle is measured between the two
        /// bodies, or between the body and the static environment.The
        /// angle will be between -pi..pi.
        /// When the hinge anchor or axis is set, the current position of
        /// the attached bodies is examined and that position will be the
        /// zero angle.
        /// </summary>
        public double HingeAngle { get => dJointGetHingeAngle(jid); }

        /// <summary>
        /// Get the time derivative of the angle.
        /// </summary>
        public double HingeAngleRate
        {
            get => dJointGetHingeAngleRate(jid);
        }

        public double JointERP
        {
            get => dJointGetHingeParam(jid, (int)ParamType.dParamERP);
            set => dJointSetHingeParam(jid, (int)ParamType.dParamERP, value);
        }

        public double JointCFM
        {
            get => dJointGetHingeParam(jid, (int)ParamType.dParamCFM);
            set => dJointSetHingeParam(jid, (int)ParamType.dParamCFM, value);
        }

        public void AddHingeTorque(double value) => dJointAddHingeTorque(jid, value);

        public uint HingeFlags() => dJointGetHingeFlags(jid);

        public void SetAngleLimit(double lo, double hi)
        {
            if (lo > hi) Debug.Log("Warning: lo > hi in SetAngleLimit");
            dJointSetHingeParam(jid, (int)ParamType.dParamLoStop, lo);
            dJointSetHingeParam(jid, (int)ParamType.dParamHiStop, hi);
        }
 
        // cpp declarations for the ODE functions
        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern UIntPtr dJointCreateHinge(UIntPtr world, UIntPtr jointGroup);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetHingeAnchor(UIntPtr joint, double x, double y, double z);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointGetHingeAnchor(UIntPtr joint, [Out] double[] result);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointGetHingeAnchor2(UIntPtr joint, [Out] double[] result);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr dJointGetHingeAnchor1Raw(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr dJointGetHingeAnchor2Raw(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetHingeAxis(UIntPtr joint, double x, double y, double z);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointGetHingeAxis(UIntPtr joint, [Out] double[] result);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern double dJointGetHingeAngle(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern double dJointGetHingeAngleRate(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern double dJointGetHingeParam(UIntPtr joint, int parameter);

        [DllImport("ModifyODE")]
        static extern void dJointSetHingeParam(UIntPtr joint, int parameter, double value);

        [DllImport("ModifyODE")]
        static extern void dJointAddHingeTorque(UIntPtr joint, double torque);

        [DllImport("ModifyODE")]
        static extern uint dJointGetHingeFlags(UIntPtr joint);

        [DllImport("ModifyODE")]
        static extern void dJointGetHingeAxis1(UIntPtr joint, [Out] double[] res);

        [DllImport("ModifyODE")]
        static extern void dJointGetHingeAxis1Raw(UIntPtr joint, [Out] double[] res);

        [DllImport("ModifyODE")]
        static extern void dJointGetHingeAxis2(UIntPtr joint, [Out] double[] res);

        [DllImport("ModifyODE")]
        static extern void dJointGetHingeAxis2Raw(UIntPtr joint, [Out] double[] res);

        [DllImport("ModifyODE")]
        static extern void dJointGetHingeQRel(UIntPtr j, [Out] double[] q);

        [DllImport("ModifyODE")]
        static extern void dJointSetHingeAxisOffset(UIntPtr j, double x, double y, double z, double dangle);
    }
}
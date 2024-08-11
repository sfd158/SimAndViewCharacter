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
    public enum MotorType
    {
        dAMotorUser = 0,
        dAMotorEuler = 1
    }

    /// <summary>
    /// ball joint
    /// </summary>
    public class BallJointBase: ODEJoint
    {
        public BallJointBase()
        {

        }

        public double JointERP
        {
            get => dJointGetBallParam(jid, (int)ParamType.dParamERP);
            set => dJointSetBallParam(jid, (int)ParamType.dParamERP, value);
        }

        // Property for joint_cfm
        public double JointCFM
        {
            get => dJointGetBallParam(jid, (int) ParamType.dParamCFM);
            set => dJointSetBallParam(jid, (int)ParamType.dParamCFM, value);
        }

        [DllImport("ModifyODE")]
        protected static extern UIntPtr dJointCreateBall(UIntPtr world, UIntPtr group);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        protected static extern double dJointGetBallParam(UIntPtr joint, int parameter);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        protected static extern double dJointSetBallParam(UIntPtr joint, int parameter, double value);
    }

    public class BallJointAmotor : BallJointBase
    {
        private UIntPtr amotor_jid;  // ball joint and amotor joint are both attached to bodies

        public BallJointAmotor(ODEWorld world, JointGroup jointgroup = null)
        {
            if (world == null) throw new ArgumentNullException(nameof(world));

            var gid = UIntPtr.Zero;
            if (jointgroup != null)
            {
                gid = jointgroup.gid;
            }
            jid = dJointCreateBall(world.WorldPtr(), gid);
            amotor_jid = dJointCreateAMotor(world.WorldPtr(), gid);

            _world = world;
            if (jointgroup != null)
            {
                jointgroup.AddJoint(this);
            }
        }

        private void _destroy_amotor()
        {
            if (amotor_jid != UIntPtr.Zero)
            {
                dJointDestroy(amotor_jid);
                amotor_jid = UIntPtr.Zero;
            }
        }

        public new void Destroy()
        {
            _destroy_amotor();
            base.Destroy();
        }

        public void attach_ext(ODEBody body1, ODEBody body2)
        {
            base.Attach(body1, body2);
            dJointAttach(amotor_jid, body1 == null ? UIntPtr.Zero: body1.bid, body2 == null ? UIntPtr.Zero : body2.bid);
        }

        public UIntPtr get_amotor_jid() => amotor_jid;

        /// <summary>
        /// Set the joint anchor point which must be specified in world
        /// coordinates.
        /// </summary>
        /// <param name="pos"></param>
        /// <exception cref="ArgumentException"></exception>
        public void setAnchor(double[] pos)
        {
            if (pos.Length != 3) throw new ArgumentException("Position array must have exactly 3 elements.");
            dJointSetBallAnchor(jid, pos[0], pos[1], pos[2]);
        }

        public Vector3 Anchor => getAnchorVec3();

        public Vector3 getAnchorVec3()
        {
            double[] p = new double[3];
            dJointGetBallAnchor(jid, p);
            return CommonFunc.Vec3FromDoubleArr(p);
        }

        /// <summary>
        /// Get the joint anchor point, in world coordinates.  This
        /// returns the point on body 1.  If the joint is perfectly
        /// satisfied, this will be the same as the point on body 2.
        /// </summary>
        /// <returns></returns>
        public double[] getAnchor()
        {
            double[] p = new double[3];
            dJointGetBallAnchor(jid, p);
            return p;
        }

        /// <summary>
        /// Get the joint anchor point, in world coordinates.  This
        /// returns the point on body 2. If the joint is perfectly
        /// satisfied, this will be the same as the point on body 1.
        /// </summary>
        public Vector3 Anchor2 => getAnchor2Vec3();

        public Vector3 getAnchor2Vec3()
        {
            double[] p = new double[3];
            dJointGetBallAnchor2(jid, p);
            return CommonFunc.Vec3FromDoubleArr(p);
        }

        public double[] getAnchor2()
        {
            double[] p = new double[3];
            dJointGetBallAnchor2(jid, p);
            return p;
        }

        public void setAnchor2(double[] pos)
        {
            dJointSetBallAnchor2(jid, pos[0], pos[1], pos[2]);
        }

        public void setAnchor2(Vector3 pos)
        {
            dJointSetBallAnchor2(jid, pos[0], pos[1], pos[2]);
        }

        public double[] getAnchor1Raw()
        {
            IntPtr res = dJointGetBallAnchor1Raw(jid);
            return CommonFunc.ArrayFromDoublePtr3(res);
        }

        public double[] getAnchor2Raw()
        {
            IntPtr res = dJointGetBallAnchor2Raw(jid);
            double[] p = new double[3];
            Marshal.Copy(res, p, 0, 3);
            return p;
        }

        public void setAmotorMode(int mode)
        {
            dJointSetAMotorMode(amotor_jid, mode);
        }

        public int getAmotorMode()
        {
            return dJointGetAMotorMode(amotor_jid);
        }

        public int AMotorMode
        {
            get => dJointGetAMotorMode(amotor_jid);
        }

        public void setAmotorNumAxes(int num)
        {
            dJointSetAMotorNumAxes(amotor_jid, num);
        }

        public int AMotorNumAxes
        {
            get => dJointGetAMotorNumAxes(amotor_jid);
        }

        public void setAmotorAxis(int anum, int rel, double[] axis)
        {
            if (axis.Length != 3) throw new ArgumentException("Axis array must have exactly 3 elements.");
            dJointSetAMotorAxis(amotor_jid, anum, rel, axis[0], axis[1], axis[2]);
        }

        public double[] getAmotorAxis(int anum)
        {
            double[] a = new double[3];
            dJointGetAMotorAxis(amotor_jid, anum, a);
            return a;
        }

        public double[,] getAmotorAllAxis()
        {
            double[,] allAxis = new double[3, 3];
            for (int i = 0; i < 3; i++)
            {
                double[] axis = getAmotorAxis(i);
                allAxis[i, 0] = axis[0];
                allAxis[i, 1] = axis[1];
                allAxis[i, 2] = axis[2];
            }
            return allAxis;
        }

        public int getAmotorAxisRel(int anum)
        {
            return dJointGetAMotorAxisRel(amotor_jid, anum);
        }

        public int[] AllAxisRel
        {
            get => new int[] { getAmotorAxisRel(0), getAmotorAxisRel(1), getAmotorAxisRel(2) };
        }

        public void setAmotorAngle(int anum, double angle)
        {
            dJointSetAMotorAngle(amotor_jid, anum, angle);
        }

        public double getAmotorAngle(int anum)
        {
            return dJointGetAMotorAngle(amotor_jid, anum);
        }

        public double GetAMotorAngleRate(int anum)
        {
            return dJointGetAMotorAngleRate(amotor_jid, anum);
        }
        
        public void addAmotorTorques(double torque0, double torque1, double torque2)
        {
            dJointAddAMotorTorques(amotor_jid, torque0, torque1, torque2);
        }

        public void setAmotorParam(int param, double value)
        {
            dJointSetAMotorParam(amotor_jid, param, value);
        }

        public double getAmotorParam(int param)
        {
            return dJointGetAMotorParam(amotor_jid, param);
        }

        public void setAngleLim1(double lo, double hi)
        {
            dJointSetAMotorParam(amotor_jid, (int)ParamType.dParamLoStop, lo);
            dJointSetAMotorParam(amotor_jid, (int)ParamType.dParamHiStop, hi);
        }

        public void setAngleLim2(double lo, double hi)
        {
            dJointSetAMotorParam(amotor_jid, (int)ParamType.dParamLoStop2, lo);
            dJointSetAMotorParam(amotor_jid, (int)ParamType.dParamHiStop2, hi);
        }

        public void setAngleLim3(double lo, double hi)
        {
            dJointSetAMotorParam(amotor_jid, (int)ParamType.dParamLoStop3, lo);
            dJointSetAMotorParam(amotor_jid, (int)ParamType.dParamHiStop3, hi);
        }

        public (double lo, double hi) getAngleLimit1()
        {
            return (dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamLoStop),
                dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamHiStop));
        }

        public (double lo, double hi) getAngleLimit2()
        {
            return (dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamLoStop2),
                dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamHiStop2));
        }

        public (double lo, double hi) getAngleLimit3()
        {
            return (dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamLoStop3),
                dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamHiStop3));
        }

        public (double lo1, double hi1, double lo2, double hi2, double lo3, double hi3) AngleLimit
        {
            get => (getAngleLimit1().lo, getAngleLimit1().hi,
                getAngleLimit2().lo, getAngleLimit2().hi,
                getAngleLimit3().lo, getAngleLimit3().hi);
        }

        public (double angle0, double angle1, double angle2) Angles
        {
            get => (dJointGetAMotorAngle(amotor_jid, 0),
                dJointGetAMotorAngle(amotor_jid, 1),
                dJointGetAMotorAngle(amotor_jid, 2));
        }

        public double ball_erp
        {
            get => dJointGetBallParam(jid, (int)ParamType.dParamERP);
        }

        public double ball_cfm
        {
            get => dJointGetBallParam(jid, (int)ParamType.dParamCFM);
        }

        public double amotor_erp
        { 
            get => dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamERP);
        }

        public double amotor_cfm
        {
            get => dJointGetAMotorParam(amotor_jid, (int)ParamType.dParamCFM);
        }

        public int get_joint_dof()
        {
            return 3;
        }

        #region ImportWrapper
        // declarations for the ODE functions

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern UIntPtr dJointCreateAMotor(UIntPtr world, UIntPtr jointGroup);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointDestroy(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointAttach(UIntPtr joint, UIntPtr body1, UIntPtr body2);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetBallAnchor(UIntPtr joint, double x, double y, double z);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointGetBallAnchor(UIntPtr joint, [Out] double[] result);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointGetBallAnchor2(UIntPtr joint, [Out] double[] result);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetBallAnchor2(UIntPtr joint, double x, double y, double z);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr dJointGetBallAnchor1Raw(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr dJointGetBallAnchor2Raw(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetAMotorMode(UIntPtr joint, int mode);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern int dJointGetAMotorMode(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetAMotorNumAxes(UIntPtr joint, int num);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern int dJointGetAMotorNumAxes(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetAMotorAxis(UIntPtr joint, int anum, int rel, double x, double y, double z);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointGetAMotorAxis(UIntPtr joint, int anum, [Out] double[] result);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern int dJointGetAMotorAxisRel(UIntPtr joint, int anum);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetAMotorAngle(UIntPtr joint, int anum, double angle);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern double dJointGetAMotorAngle(UIntPtr joint, int anum);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern double dJointGetAMotorAngleRate(UIntPtr joint, int anum);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointAddAMotorTorques(UIntPtr joint, double torque0, double torque1, double torque2);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetAMotorParam(UIntPtr joint, int parameter, double value);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern double dJointGetAMotorParam(UIntPtr joint, int parameter);
        
        #endregion
    }
}
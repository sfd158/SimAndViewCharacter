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
using static RenderV2.ODE.BodyDataClass;

namespace RenderV2.ODE
{
    using dJointFeedback = ODEHeader.dJointFeedback;

    /// <summary>
    /// joint class
    /// </summary>
    public class ODEJoint
    {
        protected UIntPtr jid = default; // joint pointer
        protected ODEWorld _world = null;
        protected IntPtr feedback; // get torque and force on the joint

        protected ODEBody _body1 = null;
        protected ODEBody _body2 = null;
        protected string _name = default;
        protected string _eulerOrder = default;
        protected int _instanceId = 0;

        public void EnableImplicitDamping() => dJointEnableImplicitDamping(jid);

        public void DisableImplicitDamping() => dJointDisableImplicitDamping(jid);

        // Immediate destroy
        public void Destroy()
        {
            if (jid != UIntPtr.Zero)
            {
                SetFeedback(false);
                Detach();
                dJointDestroy(jid);
                jid = UIntPtr.Zero;
            }
        }

        /// <summary>
        /// Enable the joint. Disabled joints are completely ignored during the
        /// simulation.Disabled joints don't lose the already computed information
        /// like anchors and axes.
        /// </summary>
        public void Enable() => dJointEnable(jid);

        /// <summary>
        /// Disable the joint. Disabled joints are completely ignored during the
        /// simulation.Disabled joints don't lose the already computed information
        /// like anchors and axes.
        /// </summary>
        public void Disable() => dJointDisable(jid);

        public bool IsEnabled() => dJointIsEnabled(jid);

        public UIntPtr GetPtr() => jid;

        /// <summary>
        /// Detach the joint
        /// </summary>
        public void Detach()
        {
            if (_body1 != null && _body1.BidIsNotNull())
            {
                _body1 = null;
            }

            if (_body2 != null && _body2.BidIsNotNull())
            {
                _body2 = null;
            }

            dJointAttach(jid, UIntPtr.Zero, UIntPtr.Zero);
        }

        /// <summary>
        /// Attach the joint to some new bodies. A body can be attached
        /// to the environment by passing null as second body.
        /// </summary>
        /// <param name="body1"></param>
        /// <param name="body2"></param>
        public void Attach(ODEBody body1, ODEBody body2)
        {
            Detach();
            UIntPtr id1 = body1 != null ? body1.BodyPtr() : UIntPtr.Zero;
            UIntPtr id2 = body2 != null ? body2.BodyPtr() : UIntPtr.Zero;
            if (id1 != UIntPtr.Zero && id2 != UIntPtr.Zero && id1 == id2)
            {
                Debug.LogWarning("Warning: body1.bid == body2.bid in joint attach");
            }

            _body1 = body1;
            _body2 = body2;

            dJointAttach(jid, id1, id2);
        }

        // Properties and other methods
        public ODEWorld World => _world;

        public ODEWorld world => _world;
        
        public int InstanceId
        {
            get => _instanceId;
            set => _instanceId = value;
        }

        public string Name
        {
            get => _name;
            set => _name = value;
        }

        public string EulerOrder
        {
            get => _eulerOrder;
            set => _eulerOrder = value;
        }

        public ODEBody GetBody(int index)
        {
            if (index == 0) return _body1;
            if (index == 1) return _body2;
            throw new IndexOutOfRangeException();
        }

        public UIntPtr DJointGetBody(int index)
        {
            return dJointGetBody(jid, index);
        }

        /// <summary>
        ///  Create a feedback buffer. If flag is True then a buffer is
        /// allocated and the forces/torques applied by the joint can
        /// be read using the getFeedback() method.If flag is False the
        /// buffer is released.
        /// </summary>
        /// <param name="flag"></param>
        public void SetFeedback(bool flag = true)
        {
            if (flag)
            {
                if (feedback != IntPtr.Zero) return;
                feedback = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(dJointFeedback)));
                dJointSetFeedback(jid, feedback);
            }
            else
            {
                if (feedback != IntPtr.Zero)
                {
                    dJointSetFeedback(jid, IntPtr.Zero);
                    Marshal.FreeHGlobal(feedback);
                    feedback = IntPtr.Zero;
                }
            }
        }

        public Vector3 GetFeedbackForce() // TODO: check
        {
            IntPtr fb = dJointGetFeedback(jid);
            if (fb == IntPtr.Zero) throw new InvalidOperationException("Feedback is not enabled.");
            dJointFeedback feedbackData = Marshal.PtrToStructure<dJointFeedback>(fb);
            return feedbackData.f1.ToVec3();
        }

        public Vector3 GetFeedbackTorque()
        {
            IntPtr fb = dJointGetFeedback(jid);
            if (fb == IntPtr.Zero) throw new InvalidOperationException("Feedback is not enabled.");
            dJointFeedback feedbackData = Marshal.PtrToStructure<dJointFeedback>(fb);
            return feedbackData.t1.ToVec3();
        }

        /// <summary>
        /// Get the forces/torques applied by the joint. If feedback is
        /// activated(i.e.setFeedback(True) was called) then this method
        /// returns a tuple (force1, torque1, force2, torque2) with the
        /// forces and torques applied to body 1 and body 2.  The
        /// forces/torques are given as 3-tuples.
        /// If feedback is deactivated then the method always returns None.
        /// </summary>
        /// <returns></returns>
        public (Vector3, Vector3, Vector3, Vector3) GetFeedback()
        {
            IntPtr fb = dJointGetFeedback(jid);
            if (fb == IntPtr.Zero) return (Vector3.zero, Vector3.zero, Vector3.zero, Vector3.zero);
            dJointFeedback feedbackData = Marshal.PtrToStructure<dJointFeedback>(fb);
            return (feedbackData.f1.ToVec3(), feedbackData.f2.ToVec3(), feedbackData.t1.ToVec3(), feedbackData.t2.ToVec3());
        }

        public Vector3 GetKd()
        {
            IntPtr kdPtr = dJointGetKd(jid);
            return CommonFunc.Vec3FromDoublePtr(kdPtr);
        }

        public void SetKd(double kdx, double kdy, double kdz)
        {
            dJointSetKd(jid, kdx, kdy, kdz);
        }

        public void SetSameKd(double kd)
        {
            dJointSetKd(jid, kd, kd, kd);
        }

        public void SetKdArray(double[] kd)
        {
            dJointSetKdArr(jid, kd);
        }

        public int GetJointType() => dJointGetType(jid);

        public JointType GetJointEnumType() => (JointType) dJointGetType(jid);

        public int GetJointDof()
        {
            return 0;
        }

        // The following methods require more specific implementations based on the actual usage in different joint types
        public void SetAnchor(Vector3 anchor)
        {
            throw new NotImplementedException();
        }

        public Vector3 GetAnchor()
        {
            throw new NotImplementedException();
        }

        public Vector3 GetAnchor2()
        {
            throw new NotImplementedException();
        }

        public Quaternion GetGlobalQuat()
        {
            var body0 = dJointGetBody(jid, 0);
            var body1 = dJointGetBody(jid, 1);
            if (body1 == UIntPtr.Zero)
            {
                return CommonFunc.QuatFromODEPtr(dBodyGetQuaternion(body0));
            }
            if (dBodyGetData(body0).ToInt32() > dBodyGetData(body1).ToInt32())
            {
                body1 = body0;
            }
            return CommonFunc.QuatFromODEPtr(dBodyGetQuaternion(body1));
        }

        public Quaternion GetLocalQuat()
        {
            var body0 = dJointGetBody(jid, 0);
            var body1 = dJointGetBody(jid, 1);
            if (body1 == UIntPtr.Zero)
            {
                return CommonFunc.QuatFromODEPtr(dBodyGetQuaternion(body0));
            }
            if (dBodyGetData(body0).ToInt32() > dBodyGetData(body1).ToInt32())
            {
                (body1, body0) = (body0, body1);
            }
            var q0 = CommonFunc.QuatFromODEPtr(dBodyGetQuaternion(body0));
            var q1 = CommonFunc.QuatFromODEPtr(dBodyGetQuaternion(body1));
            return Quaternion.Inverse(q0) * q1;
        }

        #region WrapperImport
        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointEnableImplicitDamping(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointDisableImplicitDamping(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointDestroy(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointEnable(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointDisable(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool dJointIsEnabled(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointAttach(UIntPtr joint, UIntPtr body1, UIntPtr body2);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern UIntPtr dJointGetBody(UIntPtr joint, int index);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetFeedback(UIntPtr joint, IntPtr feedback);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr dJointGetFeedback(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetData(UIntPtr joint, UIntPtr data);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern UIntPtr dJointGetData(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr dJointGetKd(UIntPtr joint);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetKd(UIntPtr joint, double kdx, double kdy, double kdz);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern void dJointSetKdArr(UIntPtr joint, [Out] double[] kd);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        private static extern int dJointGetType(UIntPtr joint);

        [DllImport("ModifyODE")]
        static extern void dJointSetData(UIntPtr joint, IntPtr data);

        #endregion
    }

    public class ODEFixedJoint: ODEJoint
    {
        public ODEFixedJoint(ODEWorld world, JointGroup group = null)
        {
            jid = dJointCreateFixed(world.WorldPtr(), group == null ? UIntPtr.Zero : group.GetPtr());
        }

        public double JointERP
        {
            get => dJointGetFixedParam(jid, (int)ParamType.dParamERP);
            set => dJointSetFixedParam(jid, (int)ParamType.dParamERP, value);
        }

        // Property for joint_cfm
        public double JointCFM
        {
            get => dJointGetFixedParam(jid, (int)ParamType.dParamCFM);
            set => dJointSetFixedParam(jid, (int)ParamType.dParamCFM, value);
        }

        [DllImport("ModifyODE")]
        static extern UIntPtr dJointCreateFixed(UIntPtr world, UIntPtr joint);


        [DllImport("ModifyODE")] 
        static extern double dJointGetFixedParam(UIntPtr joint, int param);

        [DllImport("ModifyODE")]
        static extern void dJointSetFixedParam(UIntPtr joint, int parameter, double value);

        /// <summary>
        /// Call this on the fixed joint after it has been attached to
        /// remember the current desired relative offset and desired
        /// relative rotation between the bodies.
        /// </summary>
        /// <param name="joint"></param>
        [DllImport("ModifyODE")]
        static extern void dJointSetFixed(UIntPtr joint);

    }
}
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
using static RenderV2.ODE.ODEHeader;
using static RenderV2.ODE.BodyDataClass;


namespace RenderV2.ODE
{
    /// <summary>
    /// The rigid body class encapsulating the ODE body.
    /// This class represents a rigid body that has a location and orientation
    /// in space and that stores the mass properties of an object.
    /// </summary>
    public class ODEBody
    {
        UIntPtr body = UIntPtr.Zero;
        // List<Geom> GeomList = new List<Geom>();

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="world">The world in which the body should be created.</param>
        public ODEBody(ODEWorld world)
        {
            body = dBodyCreate(world is null ? UIntPtr.Zero : world.WorldPtr());
        }

        public ODEBody(UIntPtr body_)
        {
            body = body_;
        }

        public void Destroy()
        {
            if (body != UIntPtr.Zero)
            {
                dBodyDestroy(body);
                body = UIntPtr.Zero;
            }
        }

        public void DestroyAllGeoms()
        {
            if (body == UIntPtr.Zero) return;
            // destroy all geoms.
            UIntPtr geom = dBodyGetFirstGeom(body);
            while (geom != UIntPtr.Zero)
            {
                var next_geom = dBodyGetNextGeom(geom);
                Geom.StaticDestroy(geom);
                geom = next_geom;
            }
        }

        public bool BidIsNotNull() => body != UIntPtr.Zero;

        public UIntPtr bid { get => body; }

        // public List<Geom> GetGeomList() => GeomList;

        #region FunctionWrapper
        public UIntPtr BodyPtr() => body;

        public override int GetHashCode() => body.GetHashCode();

        public override bool Equals(object other)
        {
            if (other is null) return false;
            if (GetType() != other.GetType()) return false;
            return body == ((ODEBody)other).body;
        }

        public static bool operator ==(ODEBody lhs, ODEBody rhs)
        {
            if (lhs is null) return rhs is null;
            if (rhs is null) return false;
            return lhs.body == rhs.body;
        }

        public static bool operator != (ODEBody lhs, ODEBody rhs) => !(lhs == rhs);

        public void BodySetIntValue(int value)
        {
            dBodySetData(body, new IntPtr(value));
        }

        public int BodyGetIntValue()
        {
            return dBodyGetData(body).ToInt32();
        }

        public void GeomIgnoreAdd(UIntPtr other_geom)
        {
            UIntPtr geom = dBodyGetFirstGeom(body);
            while (geom != UIntPtr.Zero)
            {
                Geom.GetGeomDataImpl(geom).AddIgnore(other_geom.ToUInt64());
                geom = dBodyGetNextGeom(geom);
            }
        }

        public void BodyIgnoreAdd(ODEBody body)
        {
            if (bid == body.bid) return;
            UIntPtr geom = dBodyGetFirstGeom(body.body);
            while (geom != UIntPtr.Zero)
            {
                GeomIgnoreAdd(geom);
                geom = dBodyGetNextGeom(geom);
            }
        }

        public int AreConnected(ODEBody other) => dAreConnected(body, other.body);

        public int AreConnectedExcluding(ODEBody other, int joint_type) => dAreConnectedExcluding(body, other.BodyPtr(), joint_type);

        public int AreConnectedExcluding(ODEBody other, JointType joint_type) => dAreConnectedExcluding(body, other.BodyPtr(), (int)joint_type);

        /// <summary>
        /// Add an external force f given in absolute coordinates. The force
        // is applied at the center of mass.
        /// </summary>
        /// <param name="fx"></param>
        /// <param name="fy"></param>
        /// <param name="fz"></param>
        public void AddForce(double fx, double fy, double fz) => dBodyAddForce(body, fx, fy, fz);

        public void AddForceAtPos(double fx, double fy, double fz, double px, double py, double pz) => dBodyAddForceAtPos(body, fx, fy, fz, px, py, pz);

        public void AddForceAtRelPos(double fx, double fy, double fz, double px, double py, double pz) => dBodyAddForceAtRelPos(body, fx, fy, fz, px, py, pz);

        /// <summary>
        /// Add an external force f given in relative coordinates
        /// (relative to the body's own frame of reference). The force
        /// is applied at the center of mass.
        /// </summary>
        /// <param name="fx"></param>
        /// <param name="fy"></param>
        /// <param name="fz"></param>
        public void AddRelForce(double fx, double fy, double fz) => dBodyAddRelForce(body, fx, fy, fz);

        /// <summary>
        /// Add an external force f at position p. Both arguments must be
        /// given in absolute coordinates.
        /// </summary>
        /// <param name="fx"></param>
        /// <param name="fy"></param>
        /// <param name="fz"></param>
        /// <param name="px"></param>
        /// <param name="py"></param>
        /// <param name="pz"></param>
        public void AddRelForceAtPos(double fx, double fy, double fz, double px, double py, double pz) => dBodyAddRelForceAtPos(body, fx, fy, fz, px, py, pz);

        public void AddRelForceAtRelPos(double fx, double fy, double fz, double px, double py, double pz) => dBodyAddRelForceAtRelPos(body, fx, fy, fz, px, py, pz);

        /// <summary>
        /// Add an external torque t given in relative coordinates
        /// (relative to the body's own frame of reference).
        /// </summary>
        /// <param name="fx"></param>
        /// <param name="fy"></param>
        /// <param name="fz"></param>
        public void AddRelTorque(double fx, double fy, double fz) => dBodyAddRelTorque(body, fx, fy, fz);

        /// <summary>
        /// Add an external torque t given in absolute coordinates.
        /// </summary>
        /// <param name="fx"></param>
        /// <param name="fy"></param>
        /// <param name="fz"></param>
        public void AddTorque(double fx, double fy, double fz) => dBodyAddTorque(body, fx, fy, fz);

        /// <summary>
        /// Manually disable a body. Note that a disabled body that is connected
        /// through a joint to an enabled body will be automatically re-enabled
        // at the next simulation step.
        /// </summary>
        public void Disable() => dBodyDisable(body);

        public void Enable() => dBodyEnable(body);

        public double GetAngularDamping() => dBodyGetAngularDamping(body);

        public double GetAngularDampingThreshold() => dBodyGetAngularDampingThreshold(body);

        /// <summary>
        /// the angular velocity of the body.
        /// </summary>
        public Vector3 AngularVel
        {
            get => GetAngularVel();
            set => dBodySetAngularVel(body, value.x, value.y, value.z);
        }

        public void SetAngularVel(Vector3 w) => dBodySetAngularVel(body, w.x, w.y, w.z);

        public Vector3 GetAngularVel() => CommonFunc.Vec3FromDoublePtr(dBodyGetAngularVel(body));

        public double GetAutoDisableAngularThreshold() => dBodyGetAutoDisableAngularThreshold(body);

        public int GetAutoDisableAverageSamplesCount() => dBodyGetAutoDisableAverageSamplesCount(body);

        public int GetAutoDisableFlag() => dBodyGetAutoDisableFlag(body);

        public double GetAutoDisableLinearThreshold() => dBodyGetAutoDisableLinearThreshold(body);

        public int GetAutoDisableSteps() => dBodyGetAutoDisableSteps(body);

        public double GetAutoDisableTime() => dBodyGetAutoDisableTime(body);

        public Vector3 GetFiniteRotationAxis()
        {
            double[] p = new double[4];
            dBodyGetFiniteRotationAxis(body, p);
            return CommonFunc.Vec3FromDoubleArr(p);
        }

        public int GetFiniteRotationMode() => dBodyGetFiniteRotationMode(body);

        public UIntPtr GetFirstGeom() => dBodyGetFirstGeom(body);

        public int GetFlags() => dBodyGetFlags(body);

        public Vector3 Force
        {
            get => GetForce();
            set => dBodySetForce(body, value.x, value.y, value.z);
        }

        public Vector3 GetForce() => CommonFunc.Vec3FromDoublePtr(dBodyGetForce(body));

        public int GetGravityMode() => dBodyGetGravityMode(body);

        public int GetGyroscopicMode() => dBodyGetGyroscopicMode(body);

        public Mat33Wrapper GetInertia() => dBodyGetInertiaWrapper(body);

        public Mat33Wrapper GetInertiaInv() => dBodyGetInertiaInv(body);

        public Mat33Wrapper GetInitInertia() => dBodyGetInitInertia(body);

        public Mat33Wrapper GetInitInertiaInv() => dBodyGetInitInertiaInv(body);

        public UIntPtr GetJoint(int index) => dBodyGetJoint(body, index);

        public double GetLinearDamping() => dBodyGetLinearDamping(body);

        public double GetLinearDampingThreshold() => dBodyGetLinearDampingThreshold(body);

        public double[] GetLinearVelArr(double[] res)
        {
            IntPtr ptr = dBodyGetLinearVel(body);
            Marshal.Copy(ptr, res, 0, 3);
            return res;
        }

        public Vector3 LinearVel
        {
            get => GetLinearVel();
            set => dBodySetLinearVel(body, value.x, value.y, value.z);
        }

        public Vector3 GetLinearVel() => CommonFunc.Vec3FromDoublePtr(dBodyGetLinearVel(body));
 
        public double MassValue
        {
            get => dBodyGetMassValue(body);
        }

        public double GetBodyMassValue() => dBodyGetMassValue(body);

        public double GetBodyMaxAngularSpeed() => dBodyGetMaxAngularSpeed(body);

        public int GetBodyNumGeoms() => dBodyGetNumGeoms(body);

        public int GetBodyNumJoints() => dBodyGetNumJoints(body);

        public Vector3 GetBodyPositionVec3()
        {
            return CommonFunc.Vec3FromDoublePtr(dBodyGetPosition(body));
        }

        public double[] GetBodyPositionArray(double[] res)
        {
            IntPtr ptr = dBodyGetPosition(body);
            Marshal.Copy(ptr, res, 0, 3);
            return res;
        }

        public double[] GetBodyPositionArray() => CommonFunc.ArrayFromDoublePtr3(dBodyGetPosition(body));

        public Vector3 pos
        {
            get => CommonFunc.Vec3FromDoublePtr(dBodyGetPosition(body));
            set => SetPosition(value);
        }

        public Quaternion quat
        {
            get => GetBodyQuaternion();
            set => SetQuaternion(value);
        }

        public Quaternion GetBodyQuaternion()
        {
            var ptr = dBodyGetQuaternion(body);
            return CommonFunc.QuatFromODEPtr(ptr);
        }

        public Quat4Wrapper GetBodyQuaternionWrapper() => dBodyGetQuaternionWrapper(body);

        public Mat33Wrapper GetBodyRotation() => dBodyGetRotationWrapper(body);

        public Vector3 GetBodyTorque() => CommonFunc.Vec3FromDoublePtr(dBodyGetTorque(body));

        public UIntPtr GetBodyWorld() => dBodyGetWorld(body);

        public bool IsBodyEnabled() => dBodyIsEnabled(body) != 0;

        public bool IsBodyKinematic() => dBodyIsKinematic(body) != 0;

        public void SetBodyAngularDamping(double value) => dBodySetAngularDamping(body, value);

        public void SetBodyAngularDampingThreshold(double value) => dBodySetAngularDampingThreshold(body, value);

        public void SetAngularVelocity(double x, double y, double z) => dBodySetAngularVel(body, x, y, z);

        public void SetBodyAutoDisableAngularThreshold(double value) => dBodySetAutoDisableAngularThreshold(body, value);

        public void SetBodyAutoDisableAverageSamplesCount(int value) => dBodySetAutoDisableAverageSamplesCount(body, value);

        public void SetBodyDamping(double linearScale, double angularScale) => dBodySetDamping(body, linearScale, angularScale);

        public void SetDampingDefaults() => dBodySetDampingDefaults(body);

        /// <summary>
        ///  Set a body to the (default) "dynamic" state, instead of "kinematic".
        /// </summary>
        public void SetDynamic() => dBodySetDynamic(body);

        public void SetFiniteRotationAxis(double x, double y, double z) => dBodySetFiniteRotationAxis(body, x, y, z);

        public void SetFiniteRotationMode(int mode) => dBodySetFiniteRotationMode(body, mode);

        public void SetForce(double x, double y, double z) => dBodySetForce(body, x, y, z);

        public void SetGravityMode(int mode) => dBodySetGravityMode(body, mode);

        public void SetGyroscopicMode(int enabled) => dBodySetGyroscopicMode(body, enabled);

        /// <summary>
        /// Set the kinematic state of the body (change it into a kinematic body)
        /// Kinematic bodies behave as if they had infinite mass.This means they don't react
        /// to any force (gravity, constraints or user-supplied); they simply follow
        /// velocity to reach the next position. [from ODE wiki]
        /// </summary>
        public void SetKinematic() => dBodySetKinematic(body);

        public void SetLinearDamping(double value) => dBodySetLinearDamping(body, value);

        public void SetLinearDampingThreshold(double threshold) => dBodySetLinearDampingThreshold(body, threshold);

        /// <summary>
        /// the linear velocity of the body.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        public void SetLinearVel(double x, double y, double z) => dBodySetLinearVel(body, x, y, z);

        public void SetLinearVel(Vector3 v) => dBodySetLinearVel(body, v.x, v.y, v.z);

        public void SetMass(ref dMass mass) => dBodySetMass(body, ref mass);

        public void SetMass(ODEMass mass) => dBodySetMass(body, ref mass.dmass);

        public ODEMass GetMass()
        {
            var mass = new ODEMass();
            dBodyGetMass(body, ref mass.dmass);
            return mass;
        }

        /// <summary>
        /// You can also limit the maximum angular speed. In contrast to the damping
        /// functions, the angular velocity is affected before the body is moved.
        /// This means that it will introduce errors in joints that are forcing the
        /// body to rotate too fast.Some bodies have naturally high angular
        /// velocities (like cars' wheels), so you may want to give them a very high
        /// (like the default, dInfinity) limit.
        /// </summary>
        /// <param name="value"></param>
        public void SetMaxAngularSpeed(double value) => dBodySetMaxAngularSpeed(body, value);

        public void SetPosition(double x, double y, double z) => dBodySetPosition(body, x, y, z);

        public void SetPosition(Vector3 v) => dBodySetPosition(body, v.x, v.y, v.z);

        public void SetQuaternion(ref Quat4Wrapper quat) => dBodySetQuaternionWrapper(body, ref quat);

        public void SetQuaternion(Quaternion quat)
        {
            Quat4Wrapper q = Quat4Wrapper.FromQuat4(quat);
            dBodySetQuaternionWrapper(body, ref q);
        }

        public void SetQuaternion(double[] quat)
        {
            Quat4Wrapper q = Quat4Wrapper.FromArray(quat);
            dBodySetQuaternionWrapper(body, ref q);
        }

        public void SetQuaternion(double x, double y, double z, double w) => dBodySetQuaternionWrapper2(body, x, y, z, w);

        public void SetRotationAndQuaternion(ref Mat33Wrapper mat, ref Quat4Wrapper quat) => dBodySetRotAndQuatNoNormWrapper(body, ref mat, ref quat);

        /// <summary>
        /// Set the orientation of the body. 
        /// </summary>
        /// <param name="mat"></param>
        public void SetRotation(ref Mat33Wrapper mat) => dBodySetRotationWrapper(body, ref mat);

        public void SetTorque(double x, double y, double z) => dBodySetTorque(body, x, y, z);

        public void SetTorque(Vector3 v) => dBodySetTorque(body, v.x, v.y, v.z);

        #endregion

        #region LibImport
        [DllImport("ModifyODE")]
        static extern int dAreConnected(UIntPtr body1, UIntPtr body2);

        [DllImport("ModifyODE")]
        static extern int dAreConnectedExcluding(UIntPtr body1, UIntPtr body2, int joint_type);

        [DllImport("ModifyODE")]
        static extern void dBodyAddForce(UIntPtr body, double fx, double fy, double fz);

        [DllImport("ModifyODE")]
        static extern void dBodyAddForceAtPos(UIntPtr body, double fx, double fy, double fz, double px, double py, double pz);

        [DllImport("ModifyODE")]
        static extern void dBodyAddForceAtRelPos(UIntPtr body, double fx, double fy, double fz, double px, double py, double pz);

        [DllImport("ModifyODE")]
        static extern void dBodyAddRelForce(UIntPtr body, double fx, double fy, double fz);

        [DllImport("ModifyODE")]
        static extern void dBodyAddRelForceAtPos(UIntPtr body, double fx, double fy, double fz, double px, double py, double pz);

        [DllImport("ModifyODE")]
        static extern void dBodyAddRelForceAtRelPos(UIntPtr body, double fx, double fy, double fz, double px, double py, double pz);

        [DllImport("ModifyODE")]
        static extern void dBodyAddRelTorque(UIntPtr body, double fx, double fy, double fz);

        [DllImport("ModifyODE")]
        static extern void dBodyAddTorque(UIntPtr body, double fx, double fy, double fz);

        [DllImport("ModifyODE")]
        static extern UIntPtr dBodyCreate(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern void dBodyDestroy(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern void dBodyDisable(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern void dBodyEnable(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern double dBodyGetAngularDamping(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern double dBodyGetAngularDampingThreshold(UIntPtr body);

        

        [DllImport("ModifyODE")]
        static extern double dBodyGetAutoDisableAngularThreshold(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyGetAutoDisableAverageSamplesCount(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyGetAutoDisableFlag(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern double dBodyGetAutoDisableLinearThreshold(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyGetAutoDisableSteps(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern double dBodyGetAutoDisableTime(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern void dBodyGetFiniteRotationAxis(UIntPtr body, [Out] double[] res);

        [DllImport("ModifyODE")]
        static extern int dBodyGetFiniteRotationMode(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern UIntPtr dBodyGetFirstGeom(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyGetFlags(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern IntPtr dBodyGetForce(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyGetGravityMode(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyGetGyroscopicMode(UIntPtr body);

        [DllImport("ModifyODE")] // TODO
        static extern Mat33Wrapper dBodyGetInertiaWrapper(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern Mat33Wrapper dBodyGetInertiaInv(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern Mat33Wrapper dBodyGetInitInertia(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern Mat33Wrapper dBodyGetInitInertiaInv(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern UIntPtr dBodyGetJoint(UIntPtr body, int index);

        [DllImport("ModifyODE")]
        static extern double dBodyGetLinearDamping(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern double dBodyGetLinearDampingThreshold(UIntPtr body);

        

        [DllImport("ModifyODE")]
        static extern double dBodyGetMassValue(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern double dBodyGetMaxAngularSpeed(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern UIntPtr dBodyGetNextGeom(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dBodyGetNumGeoms(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyGetNumJoints(UIntPtr body);

        

        

        [DllImport("ModifyODE")]
        static extern Quat4Wrapper dBodyGetQuaternionWrapper(UIntPtr body);

        //[DllImport("ModifyODE")]
        //static extern void dBodyGetRelPointPos();

        //[DllImport("ModifyODE")]
        //static extern void dBodyGetRelPointVel();

        

        [DllImport("ModifyODE")]
        static extern UIntPtr dBodyGetWorld(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyIsEnabled(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern int dBodyIsKinematic(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern void dBodySetAngularDamping(UIntPtr body, double value);

        [DllImport("ModifyODE")]
        static extern void dBodySetAngularDampingThreshold(UIntPtr body, double value);

        [DllImport("ModifyODE")]
        static extern void dBodySetAngularVel(UIntPtr body, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dBodySetAutoDisableAngularThreshold(UIntPtr body, double value);

        [DllImport("ModifyODE")]
        static extern void dBodySetAutoDisableAverageSamplesCount(UIntPtr body, int value);

        [DllImport("ModifyODE")]
        static extern void dBodySetDamping(UIntPtr body, double linear_scale, double angular_scale);

        [DllImport("ModifyODE")]
        static extern void dBodySetDampingDefaults(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern void dBodySetDynamic(UIntPtr body);

        [DllImport("ModifyODE")]
        static extern void dBodySetFiniteRotationAxis(UIntPtr body, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dBodySetFiniteRotationMode(UIntPtr body, int mode);

        [DllImport("ModifyODE")]
        static extern void dBodySetForce(UIntPtr b, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dBodySetGravityMode(UIntPtr b, int mode);

        [DllImport("ModifyODE")]
        static extern void dBodySetGyroscopicMode(UIntPtr b, int enabled);

        [DllImport("ModifyODE")]
        static extern void dBodySetKinematic(UIntPtr b);

        [DllImport("ModifyODE")]
        static extern void dBodySetLinearDamping(UIntPtr b, double value);

        [DllImport("ModifyODE")] // TODO:
        static extern UIntPtr dConnectingJoint(UIntPtr body1, UIntPtr body2); // return JointID

        [DllImport("ModifyODE")]
        static extern void dBodySetLinearDampingThreshold(UIntPtr body, double threshold);

        [DllImport("ModifyODE")]
        static extern void dBodySetMass(UIntPtr body, ref dMass dmass);

        [DllImport("ModifyODE")]
        static extern void dBodyGetMass(UIntPtr body, ref dMass dmass);

        [DllImport("ModifyODE")]
        static extern void dBodySetMaxAngularSpeed(UIntPtr b, double value);

        [DllImport("ModifyODE")]
        static extern void dBodySetQuaternionWrapper(UIntPtr body, ref Quat4Wrapper quat);

        [DllImport("ModifyODE")]
        static extern void dBodySetQuaternionWrapper2(UIntPtr body, double x, double y, double z, double w);

        [DllImport("ModifyODE")]
        static extern void dBodySetRotAndQuatNoNormWrapper(UIntPtr body, ref Mat33Wrapper mat, ref Quat4Wrapper quat);

        [DllImport("ModifyODE")]
        static extern void dBodySetRotationWrapper(UIntPtr body, ref Mat33Wrapper mat);

        #endregion
    }
}
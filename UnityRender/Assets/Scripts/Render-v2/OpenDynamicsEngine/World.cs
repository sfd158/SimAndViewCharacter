using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace RenderV2.ODE
{
    /// <summary>
    /// Dynamics world.
    /// The world object is a container for rigid bodies and joints.
    /// </summary>
    public class ODEWorld
    {
        public double dt = 0.01;
        protected UIntPtr world = UIntPtr.Zero;
        public ODEWorld(double dt_ = 0.01)
        {
            if (!CommonFunc.InitCalled)
            {
                CommonFunc.InitODE();
            }
            world = dWorldCreate();
            dt = dt_;
        }

        public ODEWorld(double dt_, UIntPtr world_)
        {
            dt = dt_;
            world = world_;
        }

        public void Destroy()
        {
            if (world != UIntPtr.Zero)
            {
                dWorldDestroy(world);
                world = UIntPtr.Zero;
            }
        }

        public UIntPtr WorldPtr() => world;

        public UIntPtr wid { get => world; }

        public override int GetHashCode() => world.GetHashCode();

        public override bool Equals(object other)
        {
            if (other is null) return false;
            if (GetType() != other.GetType()) return false;
            return world == ((ODEWorld)other).world;
        }

        public static bool operator == (ODEWorld lhs, ODEWorld rhs) {

            if (lhs is null) return rhs is null;
            if (rhs is null) return false;
            return lhs.world == rhs.world; 
        }

        public static bool operator !=(ODEWorld lhs, ODEWorld rhs) { return !(lhs == rhs); }

        public double GetAngularDamping() => dWorldGetAngularDamping(world); 

        public double GetAngularDampingThreshold() => dWorldGetAngularDampingThreshold(world);

        public double GetCFM() => dWorldGetCFM(world);

        public double GetContactMaxCorrectingVel() => dWorldGetContactMaxCorrectingVel(world);

        public double GetContactSurfaceLayer() => dWorldGetContactSurfaceLayer(world);

        public double GetERP() => dWorldGetERP(world);

        //[DllImport("ModifyODE")]
        //static extern void dWorldGetFirstBody();

        //[DllImport("ModifyODE")]
        //static extern void dWorldGetFirstJoint();

        /// <summary>
        /// the world's global gravity vector.
        /// </summary>
        public Vector3 Gravity
        {
            get => GetGravity();
            set => SetGravity(value);
        }

        /// <summary>
        /// get the world's global gravity vector.
        /// </summary>
        /// <returns>Vector3 gravity</returns>
        public Vector3 GetGravity()
        {
            double[] g = new double[4];
            dWorldGetGravity(world, g);
            return CommonFunc.Vec3FromDoubleArr(g);
        }

        public double GetLinearDamping() => dWorldGetLinearDamping(world);

        public double GetLinearDampingThreshold() => dWorldGetLinearDampingThreshold(world);

        public double GetMaxAngularSpeed() => dWorldGetMaxAngularSpeed(world);

        //[DllImport("ModifyODE")]
        //static extern void dWorldGetNextBody();

        //[DllImport("ModifyODE")]
        //static extern void dWorldGetNextJoint();


        public int GetNumBallAndHingeJoints() => dWorldGetNumBallAndHingeJoints(world);

        public int NumBody => dWorldGetNumBody(world);

        public int GetNumBody() => dWorldGetNumBody(world);

        public int NumJoints => dWorldGetNumJoints(world);

        public int GetNumJoints() => dWorldGetNumJoints(world);

        public int GetQuickStepNumIterations() => dWorldGetQuickStepNumIterations(world);

        public void QuickStep(double dt)
        {
            dWorldQuickStep(world, dt);
        }

        public void SetAngularDamping(double value) => dWorldSetAngularDamping(world, value);

        public void SetAngularDampingThreshold(double value) => dWorldSetAngularDampingThreshold(world, value);

        public void SetCFM(double value) => dWorldSetCFM(world, value);

        public void SetContactMaxCorrectingVel(double value) => dWorldSetContactMaxCorrectingVel(world, value);

        public void SetContactSurfaceLayer(double depth) => dWorldSetContactSurfaceLayer(world, depth);

        public void SetDamping(double linear_scale, double angular_scale) => dWorldSetDamping(world, linear_scale, angular_scale);

        public void SetERP(double value) => dWorldSetERP(world, value);

        public void SetGravity(Vector3 g) => dWorldSetGravity(world, g.x, g.y, g.z);

        public void SetGravity(double x, double y, double z) => dWorldSetGravity(world, x, y, z);

        public void SetGravityY(double y = -9.8) => dWorldSetGravity(world, 0, y, 0);

        public void SetGravityZ(double z = -9.8) => dWorldSetGravity(world, 0, 0, z);

        public void SetLinearDampingThreshold(double value) => dWorldSetLinearDampingThreshold(world, value);

        public void SetMaxAngularSpeed(double value) => dWorldSetMaxAngularSpeed(world, value);

        public void SetQuickStepNumIterations(int num) => dWorldSetQuickStepNumIterations(world, num);

        public void Step()
        {
            if (world != UIntPtr.Zero)
            {
                dWorldStep(world, dt);
            }
        }

        public void DampedStep()
        {
            dWorldDampedStep(world, dt);
        }

        [DllImport("ModifyODE")]
        static extern UIntPtr dWorldCreate();

        [DllImport("ModifyODE")]
        static extern void dWorldDestroy(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetAngularDamping(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetAngularDampingThreshold(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetCFM(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetContactMaxCorrectingVel(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetContactSurfaceLayer(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetERP(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern void dWorldGetGravity(UIntPtr world, [Out] double[] gravity);

        [DllImport("ModifyODE")]
        static extern double dWorldGetLinearDamping(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetLinearDampingThreshold(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern double dWorldGetMaxAngularSpeed(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern int dWorldGetNumBallAndHingeJoints(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern int dWorldGetNumBody(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern int dWorldGetNumJoints(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern int dWorldGetQuickStepNumIterations(UIntPtr world);

        [DllImport("ModifyODE")]
        static extern void dWorldQuickStep(UIntPtr world, double dt);

        [DllImport("ModifyODE")]
        static extern void dWorldSetAngularDamping(UIntPtr world, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetAngularDampingThreshold(UIntPtr world, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetCFM(UIntPtr w, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetContactMaxCorrectingVel(UIntPtr w, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetContactSurfaceLayer(UIntPtr w, double depth);

        [DllImport("ModifyODE")]
        static extern void dWorldSetDamping(UIntPtr w, double linear_scale, double angular_scale);

        [DllImport("ModifyODE")]
        static extern void dWorldSetERP(UIntPtr w, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetGravity(UIntPtr w, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dWorldSetLinearDamping(UIntPtr w, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetLinearDampingThreshold(UIntPtr w, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetMaxAngularSpeed(UIntPtr world, double value);

        [DllImport("ModifyODE")]
        static extern void dWorldSetQuickStepNumIterations(UIntPtr world, int num);

        [DllImport("ModifyODE")]
        static extern void dWorldStep(UIntPtr world, double dt);

        [DllImport("ModifyODE")]
        static extern void dWorldDampedStep(UIntPtr w, double dt);
    }
}
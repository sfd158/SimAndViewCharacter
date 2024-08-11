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
using static RenderV2.ODE.ODEHeader;

namespace RenderV2.ODE
{
    public static class ODEHeader
    {

        [StructLayout(LayoutKind.Sequential)]
        public struct AABBWrapper
        {
            public double x_min, x_max;
            public double y_min, y_max;
            public double z_min, z_max;
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct Mat33Wrapper
        {
            public double x00, x01, x02,
                   x10, x11, x12,
                   x20, x21, x22;

            public readonly double[] toArr1D()
            {
                double[] ret = new double[9];
                ret[0] = x00; ret[1] = x01; ret[2] = x02;
                ret[3] = x10; ret[4] = x11; ret[5] = x12;
                ret[6] = x20; ret[7] = x21; ret[8] = x22;
                return ret;
            }

            public readonly double[,] toArr2D()
            {
                double[,] ret = new double[3, 3];
                ret[0, 0] = x00; ret[0, 1] = x01; ret[0, 2] = x02;
                ret[1, 0] = x10; ret[1, 1] = x11; ret[1, 2] = x12;
                ret[2, 0] = x20; ret[2, 1] = x21; ret[2, 2] = x22;
                return ret;
            }

            public static Mat33Wrapper fromArr1D(double[] arr)
            {
                var ret = new Mat33Wrapper();
                ret.x00 = arr[0]; ret.x01 = arr[1]; ret.x02 = arr[2];
                ret.x10 = arr[3]; ret.x11 = arr[4]; ret.x12 = arr[5];
                ret.x20 = arr[6]; ret.x21 = arr[7]; ret.x22 = arr[8];
                return ret;
            }
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct dContactGeom // assume using 64 bit System..
        {
            public dVector3 pos;       // contact position, size = 8 * 4 = 32
            public dVector3 normal;    // normal vector, size = 8 * 4 = 32
            public double depth;       // penetration depth, size = 8
            public UIntPtr g1;         // the colliding geoms, size = 8
            public UIntPtr g2;         // the colliding geoms, size = 8
            public int Side1;          // size = 4
            public int Side2;          // size = 4
        } // Total size = 32 + 32 + 8 + 8 + 8 + 4 + 4 = 96

        [StructLayout(LayoutKind.Sequential)]
        public struct dSurfaceParameters
        {
            // must always be defined
            public int mode; // size == 4
            public double mu; // size == 8

            // only defined if the corresponding flag is set in mode
            public double mu2; // size == 8
            public double bounce;  // size == 8
            public double bounce_vel;
            public double soft_erp;
            public double soft_cfm;
            public double motion1, motion2, motionN;
            public double slip1, slip2;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct dContact
        {
            public dSurfaceParameters surface;
            public dContactGeom geom; // size == 
            public dVector3 fdir1; // size == 4 * 8 = 32
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct dJointFeedback
        {
            public dVector3 f1;        // force applied to body 1, size == 4 * 8 = 32
            public dVector3 t1;        // torque applied to body 1, size == 4 * 8 == 32
            public dVector3 f2;        // force applied to body 2, size == 4 * 8 == 32
            public dVector3 t2;        // torque applied to body 2, size == 4 * 8 == 32
        }

        //[DllImport("ModifyODE")]
        //static extern void dBodyGetPointVel();

        //[DllImport("ModifyODE")]
        //static extern void dBodyGetPosRelPoint();

        //[DllImport("ModifyODE")]
        //static extern void dBodyVectorFromWorld();

        //[DllImport("ModifyODE")]
        //static extern void dBodyVectorToWorld();

        //[DllImport("ModifyODE")]
        //static extern void dClearUpperTriangle();

        //[DllImport("ModifyODE")]
        //static extern void dConnectingJointList();

        //[DllImport("ModifyODE")]
        //static extern void dCreateConvex();

        //[DllImport("ModifyODE")]
        //static extern void dCreateGeomClass();

        //[DllImport("ModifyODE")]
        //static extern void dCreateGeomTransform();

        //[DllImport("ModifyODE")]
        //static extern void dCreateHeightfield();

        //[DllImport("ModifyODE")]
        //static extern void dCreateRay();

        //[DllImport("ModifyODE")]
        //static extern void dCreateTriMesh();

        //[DllImport("ModifyODE")]
        //static extern void dGeomGetPosRelPoint();

        //[DllImport("ModifyODE")]
        //static extern void dGeomGetRelPointPos();

        /* [DllImport("ModifyODE")]
        static extern void dGeomPlaneGetNearestPointToOrigin();

        [DllImport("ModifyODE")]
        static extern void dGeomPlaneGetQuatFromZAxis();

        [DllImport("ModifyODE")]
        static extern void dGeomSetCharacterID();

        [DllImport("ModifyODE")]
        static extern void dGeomSetConvex();

        [DllImport("ModifyODE")]
        static extern void dGeomSetDrawAxisFlag();

        [DllImport("ModifyODE")]
        static extern void dGeomVectorFromWorld();

        [DllImport("ModifyODE")]
        static extern void dGeomVectorToWorld();

        [DllImport("ModifyODE")]
        static extern void dJointAddHinge2Torques();

        [DllImport("ModifyODE")]
        static extern void dJointAddPRTorque();

        [DllImport("ModifyODE")]
        static extern void dJointAddPistonForce();

        [DllImport("ModifyODE")]
        static extern void dJointAddSliderForce();

        [DllImport("ModifyODE")]
        static extern void dJointAddUniversalTorques();

        [DllImport("ModifyODE")]
        static extern void dJointCreateContact2();

        [DllImport("ModifyODE")]
        static extern void dJointCreateContact3();

        [DllImport("ModifyODE")]
        static extern void dJointCreateContactMaxForce();

        [DllImport("ModifyODE")]
        static extern void dJointCreateEmptyBall();

        [DllImport("ModifyODE")]
        static extern void dJointCreateHinge2();

        [DllImport("ModifyODE")]
        static extern void dJointCreateLMotor();

        [DllImport("ModifyODE")]
        static extern void dJointCreateNull();

        [DllImport("ModifyODE")]
        static extern void dJointCreatePR();

        [DllImport("ModifyODE")]
        static extern void dJointCreatePU();

        [DllImport("ModifyODE")]
        static extern void dJointCreatePiston();

        [DllImport("ModifyODE")]
        static extern void dJointCreatePlane2D();

        [DllImport("ModifyODE")]
        static extern void dJointCreateSlider();

        [DllImport("ModifyODE")]
        static extern void dJointCreateUniversal();
 
        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Anchor();

        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Anchor2();

        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Angle1();

        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Angle1Rate();

        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Angle2Rate();

        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Axis1();

        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Axis2();

        [DllImport("ModifyODE")]
        static extern void dJointGetHinge2Param();

        [DllImport("ModifyODE")]
        static extern void dJointSetHinge2Anchor();

        [DllImport("ModifyODE")]
        static extern void dJointSetHinge2Axis1();

        [DllImport("ModifyODE")]
        static extern void dJointSetHinge2Axis2();

        [DllImport("ModifyODE")]
        static extern void dJointSetHinge2Param(); */

        //[DllImport("ModifyODE")]
        //static extern void dJointSetHingeAnchorDelta();

        //[DllImport("ModifyODE")]
        //static extern void dMassSetTrimesh();

        //[DllImport("ModifyODE")]
        //static extern void dMassSetTrimeshTotal();

        //[DllImport("ModifyODE")]
        //static extern void dSpaceGetPlaceableAndPlaneCount();

        //[DllImport("ModifyODE")]
        //static extern void dSpaceGetPlaceableCount();
    }

    public static class BodyDataClass
    {
        [DllImport("ModifyODE")]
        public static extern IntPtr dBodyGetPosition(UIntPtr body);

        [DllImport("ModifyODE")]
        public static extern void dBodySetPosition(UIntPtr body, double x, double y, double z);

        [DllImport("ModifyODE")]
        public static extern void dBodySetTorque(UIntPtr body, double x, double y, double z);

        [DllImport("ModifyODE")]
        public static extern IntPtr dBodyGetQuaternion(UIntPtr body);

        [DllImport("ModifyODE")]
        public static extern IntPtr dBodyGetLinearVel(UIntPtr body);

        [DllImport("ModifyODE")]
        public static extern void dBodySetLinearVel(UIntPtr body, double x, double y, double z);

        [DllImport("ModifyODE")]
        public static extern IntPtr dBodyGetAngularVel(UIntPtr body);

        [DllImport("ModifyODE")]
        public static extern Mat33Wrapper dBodyGetRotationWrapper(UIntPtr body);

        [DllImport("ModifyODE")]
        public static extern IntPtr dBodyGetRotation(UIntPtr body);

        [DllImport("ModifyODE")]
        public static extern IntPtr dBodyGetTorque(UIntPtr body);

        [DllImport("ModifyODE")]
        public static extern void dBodySetData(UIntPtr body, IntPtr data);

        [DllImport("ModifyODE")]
        public static extern IntPtr dBodyGetData(UIntPtr body);
    }
}


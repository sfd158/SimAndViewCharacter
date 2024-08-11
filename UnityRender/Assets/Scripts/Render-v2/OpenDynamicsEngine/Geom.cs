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
using static RenderV2.ODE.CommonFunc;

namespace RenderV2.ODE
{
    /// <summary>
    /// base class of Geometry
    /// </summary>
    public class Geom
    {
        public UIntPtr geom;

        public Geom()
        {
            geom = UIntPtr.Zero;
        }

        public Geom(UIntPtr geom_)
        {
            geom = geom_;
        }

        public static void StaticDestroy(UIntPtr geom)
        {
            if (geom != UIntPtr.Zero)
            {
                int idx = dGeomGetData(geom).ToInt32();
                GeomPool.Dealloc(idx);
                dGeomDestroy(geom);
            }
        }

        public void Destroy()
        {
            StaticDestroy(geom);
            geom = UIntPtr.Zero;
        }

        public GeomData GetGeomData()
        {
            int idx = GeomGetIntData();
            return GeomPool.GetPool()[idx];
        }

        public static GeomData GetGeomDataImpl(UIntPtr geom)
        {
            int idx = dGeomGetData(geom).ToInt32();
            return GeomPool.GetPool()[idx];
        }

        public GeomData InitGeomData()
        {
            int idx = GeomPool.Alloc(out GeomData item);
            GeomSetIntData(idx);
            dGeomSetIndex(geom, idx);
            return item;
        }

        public void SetFriction(float value)
        {
            var data = GetGeomData();
            data.friction = value;
        }

        public void SetBounce(float value)
        {
            var data = GetGeomData();
            data.bounce = value;
        }

        public void AppendIgnore(UIntPtr geom_id)
        {
            var data = GetGeomData();
            data.AddIgnore(geom_id.ToUInt64());
        }

        public SpaceBase space
        {
            get // we need to judge space type here.
            {
                UIntPtr ptr = dGeomGetSpace(geom);
                return ptr == UIntPtr.Zero ? null : SpaceBuilder.SpaceBuild(ptr);
            }
            set
            {
                UIntPtr old_space = dGeomGetSpace(geom);
                if (old_space != UIntPtr.Zero)
                {
                    SpaceBase.dSpaceRemove(old_space, geom);
                }
                if (value != null && value.SpacePtr() != null)
                {
                    value.Add(geom);
                }
            }
        }

        public UIntPtr SpacePtr() => dGeomGetSpace(geom);

        /// <summary>
        /// the body associated with a placeable geom.
        /// </summary>
        public ODEBody body
        {
            get
            {
                UIntPtr ptr = dGeomGetBody(geom);
                if (ptr == UIntPtr.Zero) return null;
                else return new ODEBody(ptr);
            }
            set
            {
                if (value == null) dGeomSetBody(geom, UIntPtr.Zero);
                else dGeomSetBody(geom, value.BodyPtr());
            }
        }

        public UIntPtr BodyPtr() => dGeomGetBody(geom);

        /// <summary>
        /// Set the position of the geom. If the geom is attached to a body,
        /// the body's position will also be changed.
        /// </summary>
        public Vector3 pos
        {
            get => GetPositionVec3();
            set => SetPosition(value);
        }

        public Vector3 GetPositionVec3()
        {
            if (geom == UIntPtr.Zero) return Vector3.zero;
            return Vec3FromDoublePtr(dGeomGetPosition(geom));
        }

        public double[] GetPositionArray()
        {
            if (geom == UIntPtr.Zero) return null;
            return ArrayFromDoublePtr3(dGeomGetPosition(geom));
        }

        public void SetPosition(Vector3 v)
        {
            dGeomSetPosition(geom, v.x, v.y, v.z);
        }

        public void SetPosition(double[] v)
        {
            dGeomSetPosition(geom, v[0], v[1], v[2]);
        }

        public void SetPosition(double x, double y, double z)
        {
            dGeomSetPosition(geom, x, y, z);
        }

        /// <summary>
        /// Get/Set the current orientation of the geom. 
        /// Getter: If the geom is attached to a body the returned value is the body's orientation.
        /// Setter: If the geom is attached to a body, the body's orientation will also be changed.
        /// </summary>
        public Quaternion quat
        {
            get => GetQuaternion();
            set => SetQuaternion(value);
        }

        public Quaternion GetQuaternion()
        {
            double[] res = new double[4];
            dGeomGetQuaternion(geom, res);
            return QuatFromODEArr(res);
        }

        public double[] GetQuaternionArr() // TODO: check
        {
            if (geom == UIntPtr.Zero) return null;
            double[] res = new double[4];
            dGeomGetQuaternion(geom, res);
            return QuatArrFromODEArr(res, true);
        }

        public void SetQuaternion(Quaternion q) // TODO: check
        {
            if (geom == UIntPtr.Zero) return;
            double[] res = CommonFunc.QuaternionToODEArray(q);
            dGeomSetQuaternion(geom, res);
        }

        /// <summary>
        /// Get the current orientation of the geom. If the geom is attached to
        /// a body the returned value is the body's orientation.
        /// </summary>
        /// <returns></returns>
        public double[] GetRotation()
        {
            if (geom == UIntPtr.Zero) return null;
            IntPtr ptr = dGeomGetRotation(geom);
            return dMatrix3.BuildFromODEPtr(ptr).ToDenseMat();
        }

        /// <summary>
        /// Set the orientation of the geom. If the geom is attached to a body,
        /// the body's orientation will also be changed.
        /// </summary>
        /// <param name="r"></param>
        public void SetRotation(double[] r)
        {
            dGeomSetRotation(geom, dMatrix3.BuildFromDense(r).GetValues());
        }
        // 

        void CheckBody()
        {
            UIntPtr body = dGeomGetBody(geom);
            if (body == UIntPtr.Zero)
            {
                throw new InvalidOperationException("Cannot set an offset rotation on a geom before calling setBody");
            }
        }

        /// <summary>
        /// Set the offset position of the geom. The geom must be attached to a
        /// body.If the geom did not have an offset, it is automatically created.
        /// This sets up an additional (local) transformation for the geom, since
        /// geoms attached to a body share their global position and rotation.
        /// </summary>
        /// <param name="pos"></param>
        public void SetOffsetPosition(double[] pos)
        {
            CheckBody();
            dGeomSetOffsetPosition(geom, pos[0], pos[1], pos[2]);
        }

        public void SetOffsetPosition(Vector3 pos)
        {
            CheckBody();
            dGeomSetOffsetPosition(geom, pos[0], pos[1], pos[2]);
        }

        public void SetOffsetWorldPosition(double[] pos)
        {
            dGeomSetOffsetWorldPosition(geom, pos[0], pos[1], pos[2]);
        }

        public void SetOffsetWorldPosition(Vector3 pos)
        {
            dGeomSetOffsetWorldPosition(geom, pos[0], pos[1], pos[2]);
        }

        /// <summary>
        /// Get the offset position of the geom.
        /// </summary>
        /// <returns></returns>
        public double[] GetOffsetPositionArray()
        {
            IntPtr p = dGeomGetOffsetPosition(geom);
            return CommonFunc.ArrayFromDoublePtr3(p);
        }

        public Vector3 GetOffsetPositionVec3()
        {
            IntPtr p = dGeomGetOffsetPosition(geom);
            return CommonFunc.Vec3FromDoublePtr(p);
        }

        public void SetOffsetQuaternion(Quaternion q)
        {
            double[] arr = CommonFunc.QuaternionToODEArray(q);
            dGeomSetOffsetQuaternion(geom, arr);
        }

        public void SetOffsetWorldRotation(double[] rot)
        {
            CheckBody();
            var mat = dMatrix3.BuildFromDense(rot);
            dGeomSetOffsetWorldRotation(geom, mat.GetValues());
        }

        /// <summary>
        /// Set the offset rotation of the geom. The geom must be attached to a
        /// body.If the geom did not have an offset, it is automatically created.
        /// This sets up an additional (local) transformation for the geom, since
        /// geoms attached to a body share their global position and rotation.
        /// </summary>
        /// <param name="rot"></param>
        public void SetOffsetRotation(double[] rot)
        {
            CheckBody();
            var mat = dMatrix3.BuildFromDense(rot);
            dGeomSetOffsetRotation(geom, mat.GetValues());
        }

        public double[] GetOffsetRotation()
        {
            IntPtr m = dGeomGetOffsetRotation(geom);
            return dMatrix3.BuildFromODEPtr(m).ToDenseMat();
        }

        public void SetOffsetWorldQuaternion(Quaternion q)
        {
            double[] arr = CommonFunc.QuaternionToODEArray(q);
            dGeomSetOffsetWorldQuaternion(geom, arr);
        }

        #region WrapperFunc
        public override int GetHashCode()
        {
            return geom.GetHashCode();
        }

        public override bool Equals(object other)
        {
            if (other is null) return false;
            if (GetType() != other.GetType()) return false;
            return geom == ((Geom)other).geom;
        }

        public static bool operator == (Geom lhs, Geom rhs) 
        {
            if (lhs is null) return rhs is null;
            if (rhs is null) return false;
            return lhs.geom == rhs.geom;
        }

        public static bool operator !=(Geom lhs, Geom rhs) => !(lhs == rhs);

        public void DisableGeometry() => dGeomDisable(geom);

        public void EnableGeometry() => dGeomEnable(geom);

        /// <summary>
        /// Return an axis aligned bounding box that surrounds the geom.
        /// The return value is a 6-tuple(minx, maxx, miny, maxy, minz, maxz).
        /// </summary>
        /// <returns></returns>
        public double[] GetAABB()
        {
            double[] res = new double[6];
            dGeomGetAABB(geom, res);
            return res;
        }

        public AABBWrapper GetAABBWrapper() => dGeomGetAABBWrapper(geom);

        public UIntPtr GetBodyPtr() => dGeomGetBody(geom);

        public uint GetCategoryBits() => dGeomGetCategoryBits(geom);

        public int GetCharacterID() => dGeomGetCharacterID(geom);

        public int GetClass() => dGeomGetClass(geom);

        public uint GetCollideBits() => dGeomGetCollideBits(geom);

        public int GetDrawAxisFlag() => dGeomGetDrawAxisFlag(geom);

        public int GetIndex() => dGeomGetIndex(geom);

        public bool IsGeomEnabled() => dGeomIsEnabled(geom) != 0;

        public bool IsGeomOffset() => dGeomIsOffset(geom) != 0;

        public bool IsGeomPlaceable() => dGeomIsPlaceable(geom) != 0;

        public bool IsGeomSpace() => dGeomIsSpace(geom) != 0;

        public void GeomSetIntData(long value)
        {
            if (geom != UIntPtr.Zero)
            {
                dGeomSetData(geom, new IntPtr(value));
            }
        }

        public int GeomGetIntData()
        {
            return dGeomGetData(geom).ToInt32();
        }

        public void SetCategoryBits(uint bits) => dGeomSetCategoryBits(geom, bits);

        public void SetCollideBits(uint bits) => dGeomSetCollideBits(geom, bits);

        public void SetGeomIndex(int index) => dGeomSetIndex(geom, index);

        #endregion

        #region WrapperImport
        [DllImport("ModifyODE")]
        static extern void dGeomSetIndex(UIntPtr geom, int index);

        /// <summary>
        /// Disable the offset transform of the geom.
        /// </summary>
        /// <param name="geom"></param>
        [DllImport("ModifyODE")]
        static extern void dGeomClearOffset(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomDestroy(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomDisable(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomEnable(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomGetAABB(UIntPtr geom, [Out] double[] aabb);

        [DllImport("ModifyODE")]
        static extern AABBWrapper dGeomGetAABBWrapper(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern UIntPtr dGeomGetBody(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern uint dGeomGetCategoryBits(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomGetCharacterID(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomGetClass(UIntPtr geom);

        //[DllImport("ModifyODE")]
        //public static extern void dGeomGetClassData();

        [DllImport("ModifyODE")]
        static extern void dGeomSetData(UIntPtr geom, IntPtr data);

        [DllImport("ModifyODE")]
        static extern IntPtr dGeomGetData(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern uint dGeomGetCollideBits(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomGetDrawAxisFlag(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomGetIndex(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomIsEnabled(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomIsOffset(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomIsPlaceable(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dGeomIsSpace(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern UIntPtr dGeomGetSpace(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern IntPtr dGeomGetPosition(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomGetQuaternion(UIntPtr geom, [Out] double[] q);

        [DllImport("ModifyODE")]
        static extern IntPtr dGeomGetRotation(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomSetRotation(UIntPtr geom, [Out] double[] rot);

        [DllImport("ModifyODE")]
        static extern UIntPtr dGeomSetBody(UIntPtr geom, UIntPtr body);

        [DllImport("ModifyODE")]
        static extern void dGeomSetPosition(UIntPtr geom, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dGeomSetQuaternion(UIntPtr geom, [Out] double[] q);

        [DllImport("ModifyODE")]
        static extern void dGeomSetOffsetPosition(UIntPtr gid, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dGeomSetOffsetWorldPosition(UIntPtr gid, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern IntPtr dGeomGetOffsetPosition(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomSetOffsetWorldRotation(UIntPtr geom, [Out] double[] R);

        [DllImport("ModifyODE")]
        static extern void dGeomSetOffsetRotation(UIntPtr geom, [Out] double[] R);

        [DllImport("ModifyODE")]
        static extern void dGeomSetOffsetQuaternion(UIntPtr geom, [Out] double[] q);

        [DllImport("ModifyODE")]
        static extern IntPtr dGeomGetOffsetRotation(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomSetOffsetWorldQuaternion(UIntPtr geom, [In] double[] q); // TODO: Test if In works.

        [DllImport("ModifyODE")]
        static extern void dGeomSetCategoryBits(UIntPtr geom, uint bits);

        [DllImport("ModifyODE")]
        static extern void dGeomSetCollideBits(UIntPtr geom, uint bits);
        #endregion
    }

    /// <summary>
    /// This class represents a sphere centered at the origin.
    /// </summary>
    public class GeomSphere : Geom
    {
        public GeomSphere(SpaceBase space, double radius)
        {
            geom = dCreateSphere(space is null ? UIntPtr.Zero : space.SpacePtr(), radius);
            InitGeomData();
        }

        public double GetRadius() => dGeomSphereGetRadius(geom);

        public void SetRadius(double radius) => dGeomSphereSetRadius(geom, radius);

        public double radius
        {
            get => dGeomSphereGetRadius(geom);
            set => dGeomSphereSetRadius(geom, value);
        }

        [DllImport("ModifyODE")]
        static extern UIntPtr dCreateSphere(UIntPtr space, double radius);

        [DllImport("ModifyODE")]
        static extern double dGeomSphereGetRadius(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern void dGeomSpherePointDepth(UIntPtr geom, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dGeomSphereSetRadius(UIntPtr geom, double radius);
    }

    /// <summary>
    /// This class represents a capped cylinder aligned along the local Z axis
    /// and centered at the origin.
    /// </summary>
    public class GeomCapsule : Geom
    {
        public GeomCapsule(SpaceBase space, double radius, double length)
        {
            geom = dCreateCapsule(space is null ? UIntPtr.Zero : space.SpacePtr(), radius, length);
            InitGeomData();
        }

        public Vector2Wrapper GetParams() => geom == UIntPtr.Zero ? throw new NullReferenceException(): dGeomCapsuleGetParamsWrapper(geom);

        public double CapsulePointDepth(double x, double y, double z) => dGeomCapsulePointDepth(geom, x, y, z);

        public void SetParams(double radius, double length) => dGeomCapsuleSetParams(geom, radius, length);


        [DllImport("ModifyODE")]
        static extern UIntPtr dCreateCapsule(UIntPtr space, double radius, double length);

        [DllImport("ModifyODE")]
        static extern Vector2Wrapper dGeomCapsuleGetParamsWrapper(UIntPtr capsule);

        [DllImport("ModifyODE")]
        static extern double dGeomCapsulePointDepth(UIntPtr ccylinder, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dGeomCapsuleSetParams(UIntPtr ccylinder, double radius, double length);
    }

    public class Cylinder : Geom
    {
        public Cylinder(SpaceBase space, double radius, double length)
        {
            geom = dCreateCylinder(space is null ? UIntPtr.Zero : space.SpacePtr(), radius, length);
        }

        public Vector2Wrapper GetCylinderParamsWrapper() => dGeomCylinderGetParamsWrapper(geom);

        public Vector2Wrapper GetCylinderParams()
        {
            var res = new Vector2Wrapper();
            dGeomCylinderGetParams(geom, out res.x, out res.y);
            return res;
        }

        public void CylinderSetParams(double radius, double length) => dGeomCylinderSetParams(geom, radius, length);

        [DllImport("ModifyODE")]
        public static extern UIntPtr dCreateCylinder(UIntPtr space, double radius, double length);

        [DllImport("ModifyODE")]
        public static extern Vector2Wrapper dGeomCylinderGetParamsWrapper(UIntPtr geom); // return (radius, length)

        [DllImport("ModifyODE")]
        public static extern void dGeomCylinderGetParams(UIntPtr geom, out double r, out double l); // return (radius, length)

        [DllImport("ModifyODE")]
        public static extern void dGeomCylinderSetParams(UIntPtr geom, double radius, double length);
    }
}
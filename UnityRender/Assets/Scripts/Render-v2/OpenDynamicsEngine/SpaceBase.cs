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
    /// <summary>
    /// use in function dCollide
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CallBackData
    {
        public UIntPtr world;
        public UIntPtr jointgroup;
    }

    /// <summary>
    /// Space class (container for geometry objects).
    /// A Space object is a container for geometry objects which are used
    /// to do collision detection.
    /// The space does high level collision culling, which means that it
    /// can identify which pairs of geometry objects are potentially touching.
    /// This Space class can be used for both, a SimpleSpace and a HashSpace
    /// (see ODE documentation).
    /// </summary>
    public class SpaceBase
    {
        protected UIntPtr space = UIntPtr.Zero;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void CollisionCallback(IntPtr userData, UIntPtr geom1, UIntPtr geom2);

        public SpaceBase()
        {

        }

        public SpaceBase(UIntPtr space_)
        {
            this.space = space_;
        }

        public UIntPtr SpacePtr()
        {
            return space;
        }

        public void Destroy()
        {
            
        }

        public static CallBackData BuildCallBackData(ODEWorld world, JointGroup group)
        {
            return new CallBackData() { world = world.WorldPtr(), jointgroup = group.GetPtr() };
        }

        public int Length => dSpaceGetNumGeoms(space);

        public int GetNumGeoms() => dSpaceGetNumGeoms(space);

        public void ResortGeoms() => dSpaceResortGeoms(space);

        // Add a geom to a space. This does nothing if the geom is already in the space.
        public void Add(Geom geom)
        {
            if (geom != null)
            {
                dSpaceAdd(space, geom.geom);
            }   
        }

        /// <summary>
        /// Add a geom to a space. This does nothing if the geom is 
        /// already in the space.
        /// </summary>
        /// <param name="geometry"></param>
        public void Add(UIntPtr geometry)
        {
            dSpaceAdd(space, geometry); 
        }

        /// <summary>
        /// Remove a geom from a space.
        /// </summary>
        /// <param name="geom"></param>
        public void Remove(Geom geom)
        {
            if (geom != null)
            {
                dSpaceRemove(space, geom.geom);
            }
        }
        
        public void Clean()
        {
            if (space != UIntPtr.Zero)
            {
                dSpaceClean(space);
            }
        }

        /// <summary>
        /// Return True if the given geom is in the space.
        /// </summary>
        /// <param name="geom"></param>
        public bool Query(Geom geom)
        {
            return dSpaceQuery(space, geom.geom) != 0;
        }

        public int SpaceGetClass()
        {
            return dSpaceGetClass(space); 
        }

        public int GetSublevel() => dSpaceGetSublevel(space);

        public int GetManualCleanup()
        {
            return dSpaceGetManualCleanup(space);
        }

        /// <summary>
        /// callback function for collision
        /// </summary>
        /// <param name="userData">contains world and jointgroup</param>
        /// <param name="geom1">geometry 1</param>
        /// <param name="geom2">geometry 2</param>
        public static void CallBack(IntPtr userData, UIntPtr geom1, UIntPtr geom2)
        {
            // userData contains world and jointgroup.
            CallBackData data = Marshal.PtrToStructure<CallBackData>(userData);

            const int max_contact_num = 4;
            GeomData g1attr = Geom.GetGeomDataImpl(geom1);
            GeomData g2attr = Geom.GetGeomDataImpl(geom2);
            UIntPtr b1 = dGeomGetBody(geom1);
            UIntPtr b2 = dGeomGetBody(geom2);
            if (b1 == b2) return;
            // check ignore geom
            if (g1attr.IgnoreContains(geom2.ToUInt64())) return;
            if (g2attr.IgnoreContains(geom1.ToUInt64())) return;

            dContactGeom[] c = new dContactGeom[max_contact_num];
            dContact[] contact = new dContact[max_contact_num];
            // we can only use default init function for structure.
            // so, initialize values pointer manully here.
            for(int i = 0; i < max_contact_num; i++)
            {
                contact[i].fdir1.values = new double[4];
            }
            const int sizeof_dContactGeom = 96; // we cannot get sizeof struct w/o unsafe code.
            int n = dCollide(geom1, geom2, max_contact_num, c, sizeof_dContactGeom);
            for(int i = 0; i < n; i++)
            {
                contact[i].surface.mode = (int)ContactType.dContactApprox1;
                contact[i].surface.mu = Math.Min(g1attr.GetFriction(), g2attr.GetFriction());
                contact[i].surface.bounce = Math.Min(g1attr.GetBounce(), g2attr.GetBounce());
                contact[i].geom = c[i];
                UIntPtr joint = dJointCreateContact(data.world, data.jointgroup, ref contact[i]); // TODO: here has bug.
                dJointAttach(joint, b1, b2);
            }
        }

        public void Collide(ODEWorld world, JointGroup jgroup)
        {
            CallBackData data = new CallBackData() { world = world.WorldPtr(), jointgroup = jgroup.GetPtr()};
            IntPtr ptr = Marshal.AllocHGlobal(UIntPtr.Size * 2); // assume sizeof UIntPtr == 8
            Marshal.StructureToPtr(data, ptr, false);
            dSpaceCollide(space, ptr, CallBack);
            Marshal.FreeHGlobal(ptr);
        }

        /// <summary>
        /// Call a callback function one or more times, for all
        ///potentially intersecting objects in the space.The callback
        ///function takes 3 arguments:
        ///void NearCallback(arg, geom1, geom2):
        ///
        ///The arg parameter is just passed on to the callback function.
        ///Its meaning is user defined. The geom1 and geom2 arguments are
        ///the geometry objects that may be near each other.The callback
        ///function can call the function collide() (not the Space
        ///method) on geom1 and geom2, perhaps first determining
        ///whether to collide them at all based on other information.
        /// </summary>
        /// <param name="callbackData"></param>
        public void Collide(IntPtr callbackData) // CallBackData to Pointer.
        {
            dSpaceCollide(space, callbackData, CallBack);
        }

        public UIntPtr SpaceGetGeomPtr(int i)
        {
            return dSpaceGetGeom(space, i);
        }

        /// <summary>
        /// get all of geometries in space.
        /// </summary>
        /// <returns>list of geometries</returns>
        public Geom[] GetAllGeoms()
        {
            int l = Length;
            Geom[] geoms = new Geom[l];
            UIntPtr geom = dSpaceGetFirstGeom(space);
            for(int i = 0; i < l; i++)
            {
                geoms[i] = new Geom(geom);
                geom = dSpaceGetNextGeom(geom);
            }
            return geoms;
        }

        public int NumGeoms => dSpaceGetNumGeoms(space);

        [DllImport("ModifyODE")]
        static extern int dSpaceGetNumGeoms(UIntPtr space);

        [DllImport("ModifyODE")]
        static extern void dSpaceResortGeoms(UIntPtr space);

        [DllImport("ModifyODE")]
        static extern void dSpaceAdd(UIntPtr space, UIntPtr geom);

        [DllImport("ModifyODE")]
        public static extern void dSpaceRemove(UIntPtr space, UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dSpaceQuery(UIntPtr space, UIntPtr geom);

        [DllImport("ModifyODE")]
        protected static extern UIntPtr dSpaceSetCleanup(UIntPtr space, int mode);

        [DllImport("ModifyODE")]
        static extern int dSpaceGetClass(UIntPtr space);

        [DllImport("ModifyODE")]
        static extern int dSpaceGetManualCleanup(UIntPtr space);

        [DllImport("ModifyODE")]
        static extern void dSpaceCollide(UIntPtr space, IntPtr data, CollisionCallback callback);

        [DllImport("ModifyODE")]
        static extern void dSpaceCollide2(UIntPtr o1, UIntPtr o2, IntPtr data, CollisionCallback callback);

        [DllImport("ModifyODE")]
        static extern IntPtr dGeomGetData(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dCollide(UIntPtr o1, UIntPtr o2, int flags, [Out] dContactGeom[] contact, int skip);

        [DllImport("ModifyODE")]
        static extern UIntPtr dGeomGetBody(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern UIntPtr dJointCreateContact(UIntPtr w, UIntPtr jointgroup, ref dContact contact);

        [DllImport("ModifyODE")]
        static extern void dJointAttach(UIntPtr joint, UIntPtr body1, UIntPtr body2);

        [DllImport("ModifyODE")]
        static extern void dSpaceDestroy(UIntPtr space);

        [DllImport("ModifyODE")]
        static extern void dSpaceClean(UIntPtr space);

        [DllImport("ModifyODE")]
        static extern UIntPtr dSpaceGetGeom(UIntPtr space, int i);

        [DllImport("ModifyODE")]
        private static extern UIntPtr dSpaceGetFirstGeom(UIntPtr space);

        [DllImport("ModifyODE")]
        private static extern UIntPtr dSpaceGetNextGeom(UIntPtr geom);

        [DllImport("ModifyODE")]
        static extern int dSpaceGetSublevel(UIntPtr space);
    }

    /// <summary>
    /// Multi-resolution hash table space
    /// This uses an internal data structure that records how each geom
    /// overlaps cells in one of several three dimensional grids.Each
    /// grid has cubical cells of side lengths 2**i, where i is an integer
    /// that ranges from a minimum to a maximum value.The time required
    /// to do intersection testing for n objects is O(n) (as long as those
    /// objects are not clustered together too closely), as each object
    /// can be quickly paired with the objects around it.
    /// </summary>
    public class HashSpace: SpaceBase
    {
        public HashSpace(UIntPtr space_)
        {
            space = space_;
        }

        public HashSpace(SpaceBase parent = null)
        {
            space = dHashSpaceCreate(parent == null ? UIntPtr.Zero : parent.SpacePtr());
            dSpaceSetCleanup(space, 0);
        }

        [DllImport("ModifyODE")]
        static extern UIntPtr dHashSpaceCreate(UIntPtr parent);

        //[DllImport("ModifyODE")]
        // static extern void dHashSpaceGetLevels();

        /// <summary>
        /// Sets the size of the smallest and largest cell used in the
        /// hash table.The actual size will be 2^minlevel and 2^maxlevel respectively.
        /// </summary>
        /// <param name="space"></param>
        /// <param name="minlevel"></param>
        /// <param name="maxlevel"></param>
        [DllImport("ModifyODE")]
        static extern void dHashSpaceSetLevels(UIntPtr space, int minlevel, int maxlevel);
    }

    /// <summary>
    /// This does not do any collision culling - it simply checks every
    /// possible pair of geoms for intersection, and reports the pairs
    /// whose AABBs overlap.The time required to do intersection testing
    // for n objects is O(n**2). This should not be used for large numbers
    // of objects, but it can be the preferred algorithm for a small
    // number of objects.This is also useful for debugging potential
    // problems with the collision system.
    /// </summary>
    public class SimpleSpace: SpaceBase
    {
        public SimpleSpace(UIntPtr space_)
        {
            space = space_;
        }

        public SimpleSpace(SpaceBase parent = null)
        {
            if (!CommonFunc.InitCalled)
            {
                CommonFunc.InitODE();
            }
            space = dSimpleSpaceCreate(parent == null ? UIntPtr.Zero: parent.SpacePtr());
            dSpaceSetCleanup(space, 0);
        }

        [DllImport("ModifyODE")]
        static extern UIntPtr dSimpleSpaceCreate(UIntPtr parent);
    }

    public static class SpaceBuilder
    {
        public static SimpleSpace SimpleSpaceBuild(UIntPtr space)
        {
            if (space == UIntPtr.Zero) return null;
            int GType = dGeomGetClass(space);
            if (GType == (int)GeomType.dSimpleSpaceClass) return new SimpleSpace(space);
            else throw new ArgumentException("Input pointer is not simple space.");
        }

        public static HashSpace HashSpaceBuild(UIntPtr space)
        {
            if (space == UIntPtr.Zero) return null;
            int GType = dGeomGetClass(space);
            if (GType == (int)GeomType.dHashSpaceClass) return new HashSpace(space);
            else throw new ArgumentException("Input pointer is not hash space.");
        }

        public static SpaceBase SpaceBuild(UIntPtr space)
        {
            if (space == UIntPtr.Zero) return null;
            return new SpaceBase(space);
        }

        [DllImport("ModifyODE")]
        static extern int dGeomGetClass(UIntPtr g);
    }
}
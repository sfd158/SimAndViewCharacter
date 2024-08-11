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
using System.Collections.Generic;

namespace RenderV2.ODE
{
    /// <summary>
    /// This class represents an infinite plane. The plane equation is:
    /// n.x* x + n.y* y + n.z* z = dist
    /// This object can't be attached to a body.
    /// If you call getBody() on this object it always returns environment.
    /// </summary>
    public class GeomPlane : Geom
    {
        public GeomPlane(SpaceBase space, double a, double b, double c, double d)
        {
            geom = dCreatePlane(space is null ? UIntPtr.Zero : space.SpacePtr(), a, b, c, d);
            InitGeomData();
        }

        // Create space at (0, 0, 0) with y up.
        public static GeomPlane CreateDefault(SpaceBase space)
        {
            return new GeomPlane(space, 0, 1, 0, 0);
        }

        public void setParams(Vector3 normal, double dist)
        {
            if (geom != UIntPtr.Zero)
            {
                dGeomPlaneSetParams(geom, normal.x, normal.y, normal.z, dist);
            }
        }
        
        public double[] GetParamsVec4()
        {
            if (geom == UIntPtr.Zero)
            {
                throw new NullReferenceException();
            }
            double[] res = new double[4];
            dGeomPlaneGetParams(geom, res);
            // for(int i = 0; i < 4; i++) Debug.Log(res[i]);
            return res;
        }

        public (Vector3, double) GetParams()
        {
            var res = GetParamsVec4();
            return (CommonFunc.Vec3FromDoubleArr(res), res[3]);
        }

        public double PointDepth(Vector3 p) => dGeomPlanePointDepth(geom, p.x, p.y, p.z);

        [DllImport("ModifyODE")]
        static extern UIntPtr dCreatePlane(UIntPtr space, double a, double b, double c, double d);

        [DllImport("ModifyODE")]
        static extern void dGeomPlaneSetParams(UIntPtr geom, double x, double y, double z, double dist);

        [DllImport("ModifyODE", CallingConvention = CallingConvention.Cdecl)]
        static extern void dGeomPlaneGetParams(UIntPtr geom, [Out] double[] res);

        [DllImport("ModifyODE")]
        static extern double dGeomPlanePointDepth(UIntPtr plane, double x, double y, double z);
    }
}
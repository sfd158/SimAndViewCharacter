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
    using dGeomID = UIntPtr;

    /// <summary>
    /// This class represents a box centered at the origin.
    /// </summary>
    public class GeomBox: Geom
    {
        public GeomBox(SpaceBase space, double lx, double ly, double lz)
        {
            geom = dCreateBox(space is null ? UIntPtr.Zero : space.SpacePtr(), lx, ly, lz);
            InitGeomData();
        }

        public GeomBox(SpaceBase space, Vector3 l)
        {
            geom = dCreateBox(space is null ? UIntPtr.Zero : space.SpacePtr(), l.x, l.y, l.z);
            InitGeomData();
        }

        public GeomBox(SpaceBase space, double[] l)
        {
            geom = dCreateBox(space is null ? UIntPtr.Zero : space.SpacePtr(), l[0], l[1], l[2]);
            InitGeomData();
        }

        public Vector3 Shape
        {
            get => GeomBoxGetLengths();
            set => GeomBoxSetLengths(value);
        }

        public Vector3 GeomBoxGetLengths()
        {
            if (geom != UIntPtr.Zero)
            {
                dVector3 res = new dVector3();
                dGeomBoxGetLengths(this.geom, ref res);
                return res.ToVec3();
            }
            throw new NullReferenceException();
        }

        public double[] GeomBoxGetLengthsArr()
        {
            if (geom != UIntPtr.Zero)
            {
                dVector3 res = new dVector3();
                dGeomBoxGetLengths(this.geom, ref res);
                return res.ToArray();
            }
            throw new NullReferenceException();
        }

        public override string ToString()
        {
            dVector3 res = new dVector3();
            dGeomBoxGetLengths(this.geom, ref res);
            return res.ToString();
        }

        public double GeomBoxPointDepth(Vector3 l) => dGeomBoxPointDepth(geom, l.x, l.y, l.z);

        public double GeomBoxPointDepth(double x, double y, double z) => dGeomBoxPointDepth(geom, x, y, z);

        public void GeomBoxSetLengths(Vector3 l) => dGeomBoxSetLengths(geom, l.x, l.y, l.z);

        public void GeomBoxSetLengths(double x, double y, double z) => dGeomBoxSetLengths(geom, x, y, z);

        public void GeomBoxSetLengths(double[] l) => dGeomBoxSetLengths(geom, l[0], l[1], l[2]);

        [DllImport("ModifyODE")]
        static extern UIntPtr dCreateBox(UIntPtr space, double lx, double ly, double lz);

        [DllImport("ModifyODE")]
        static extern void dGeomBoxGetLengths(UIntPtr geom, ref dVector3 res);

        [DllImport("ModifyODE")]
        static extern double dGeomBoxPointDepth(dGeomID box, double x, double y, double z);

        [DllImport("ModifyODE")]
        static extern void dGeomBoxSetLengths(dGeomID box, double lx, double ly, double lz);
    }
}
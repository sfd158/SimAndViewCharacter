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
using UnityEngine;

namespace RenderV2
{
    public partial class DEnvironment
    {
        private void AddGeomToCreateBuffer(DGeomObject res)
        {
            VirtualGeom vgeom = res.GetComponentInParent<VirtualGeom>();
            vgeom.CalcAttrs();
            vgeom.SetInitialState(true);
            CreateBuffer.Add(vgeom);
        }

        /// <summary>
        /// Add Default Box Geometry to Environment
        /// </summary>
        /// <returns></returns>
        public DBoxGeom AddDefaultBox(bool AppendToBuffer = false)
        {
            var res = DBoxGeom.CreateGeom(gameObject);
            if (AppendToBuffer)
            {
                AddGeomToCreateBuffer(res);
            }
            return res;
        }

        /// <summary>
        /// Add Default Sphere Geometry to Environment
        /// </summary>
        /// <returns></returns>
        public DBallGeom AddDefaultSphere(bool AppendToBuffer = false)
        {
            var res = DBallGeom.CreateGeom(gameObject);
            if (AppendToBuffer)
            {
                AddGeomToCreateBuffer(res);
            }
            return res;
        }

        /// <summary>
        /// Add Default Capsule Geometry to Environment
        /// </summary>
        /// <returns></returns>
        public DCapsuleGeom AddDefaultCapsule(bool AppendToBuffer = false)
        {
            var res = DCapsuleGeom.CreateGeom(gameObject);
            if (AppendToBuffer)
            {
                AddGeomToCreateBuffer(res);
            }
            return res;
        }

        /// <summary>
        /// Add a box geometry to environment
        /// </summary>
        /// <param name="GeomLength"></param>
        /// <param name="GeomCenter"></param>
        /// <param name="GeomQuaternion"></param>
        /// <param name="AppendToBuffer"></param>
        /// <returns></returns>
        public DBoxGeom AddBox(Vector3 GeomLength, Vector3 GeomCenter, Quaternion GeomQuaternion, bool AppendToBuffer = false)
        {
            var res = DBoxGeom.CreateGeom(this.gameObject, 0, "BoxGeom", GeomLength, GeomCenter, GeomQuaternion);
            if (AppendToBuffer)
            {
                AddGeomToCreateBuffer(res);
            }
            return res;
        }

        /// <summary>
        /// Add a ball geometry to environment
        /// </summary>
        /// <param name="radius"></param>
        /// <param name="GeomCenter"></param>
        /// <param name="GeomQuaternion"></param>
        /// <param name="AppendToBuffer"></param>
        /// <returns></returns>
        public DBallGeom AddBall(float radius, Vector3 GeomCenter, Quaternion GeomQuaternion, bool AppendToBuffer = false)
        {
            var res = DBallGeom.CreateGeom(this.gameObject, 0, "BallGeom", radius, GeomCenter, GeomQuaternion);
            if (AppendToBuffer)
            {
                AddGeomToCreateBuffer(res);
            }
            return res;
        }

        /// <summary>
        /// Add a capsule geometry to environment
        /// </summary>
        /// <param name="radius"></param>
        /// <param name="length"></param>
        /// <param name="GeomCenter"></param>
        /// <param name="GeomQuaternion"></param>
        /// <param name="AppendToBuffer"></param>
        /// <returns></returns>
        public DCapsuleGeom AddCapsule(float radius, float length, Vector3 GeomCenter, Quaternion GeomQuaternion, bool AppendToBuffer = false)
        {
            var res = DCapsuleGeom.CreateGeom(this.gameObject, 0, "CapsuleGeom", radius, length, GeomCenter, GeomQuaternion);
            if (AppendToBuffer)
            {
                AddGeomToCreateBuffer(res);
            }
            return res;
        }
    }
}

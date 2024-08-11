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
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    public partial class DRigidBody
    {
        /// <summary>
        /// Export body information
        /// </summary>
        /// <returns></returns>
        public DBodyExportInfo ExportInfo()
        {
            DBodyExportInfo bodyInfo = new DBodyExportInfo
            {
                BodyID = IDNum,
                Name = gameObject.name,
                MassMode = MassMode.ToString(),
                Density = Density,
                Mass = Mass,
                InertiaMode = InertiaMode.ToString(),
                Inertia = BodyInertia,
                ParentBodyID = GetParentBodyID(),
                ParentJointID = GetParentJointID(),
                Position = Utils.Vector3ToArr(transform.position),
                Quaternion = Utils.QuaternionToArr(transform.rotation),
                LinearVelocity = Utils.Vector3ToArr(LinearVelocity),
                AngularVelocity = Utils.Vector3ToArr(AngularVelocity),
                Geoms = new DGeomExportInfo[transform.childCount]
                // IgnoreBodyID = new int[IgnoreCollision.Count]
            };

            // Generate Geom Export Info
            for (int geomIdx = 0; geomIdx < vGeomList.Count; geomIdx++)
            {
                DGeomExportInfo geomInfo = vGeomList[geomIdx].ExportInfo();
                bodyInfo.Geoms[geomIdx] = geomInfo;
            }
            Array.Sort(bodyInfo.Geoms);

            // Ignore Body ID
            List<int> IgnoreBodyID = new List<int>();
            if (IgnoreCollision != null)
            {
                for (int ignoreIdx = 0; ignoreIdx < IgnoreCollision.Count; ignoreIdx++)
                {
                    GameObject ignoreBodyObject = IgnoreCollision[ignoreIdx];
                    if (ignoreBodyObject != null)
                    {
                        if (ignoreBodyObject.TryGetComponent<DRigidBody>(out var body))
                        {
                            IgnoreBodyID.Add(body.IDNum);
                        }
                    }
                }
                bodyInfo.IgnoreBodyID = IgnoreBodyID.ToArray();
            }
            
            return bodyInfo;
        }
    }
}

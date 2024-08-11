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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// Box Geometry
    /// </summary>
    public class DBoxGeom : DGeomObject
    {
        [Tooltip("Length")]
        public Vector3 Length;

        public override DMass ToDMassByDensity(float density)
        {
            DMass geomMass = new DMass();
            var xyz = Length;
            geomMass.SetBox(density, xyz.x, xyz.y, xyz.z);
            return geomMass;
        }

        public override PrimitiveType ToPrimitiveType()
        {
            return PrimitiveType.Cube;
        }

        public override string ToPrimitiveTypeStr()
        {
            return ToPrimitiveType().ToString();
        }

        public override void SetScale()
        {
            transform.localScale = Length;
            Length = transform.localScale;
        }

        public override DGeomExportInfo ExportInfo()
        {
            DGeomExportInfo geomInfo = ExportInfoBase();
            geomInfo.Scale = Utils.Vector3ToArr(Length);
            return geomInfo;
        }

        public override Quaternion CalcLocalQuaternion()
        {
            return Quaternion.identity;
        }

        public override void ReCompute()
        {
            throw new System.NotImplementedException();
        }

        public override void CalcAttrs()
        {
            base.CalcAttrs();
        }

        public static DBoxGeom CreateGeom(
            GameObject parentObject,
            int GeomID,
            string GeomName,
            Vector3 geomLength,
            float frictionCoef,
            float restitutionCoef,
            Vector3 GeomCenter,
            Quaternion GeomQuaternion,
            bool isClung)
        {
            GameObject geometryObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            DBoxGeom boxGeom = geometryObject.AddComponent<DBoxGeom>();
            boxGeom.Length = geomLength;

            boxGeom.PostAddGeometry(parentObject, GeomID, GeomName, frictionCoef, restitutionCoef, GeomCenter, GeomQuaternion, isClung);
            return boxGeom;
        }

        public static DBoxGeom CreateGeom(GameObject parentObject, DGeomExportInfo info)
        {
            return CreateGeom(parentObject, info.GeomID, info.Name, Utils.ArrToVector3(info.Scale), info.Friction, info.Restitution, Utils.ArrToVector3(info.Position), Utils.ArrToQuaternion(info.Quaternion), info.ClungEnv);
        }

        public static DBoxGeom CreateGeom(
            GameObject parentObject,
            int GeomID,
            string GeomName,
            Vector3 geomLength,
            Vector3 GeomCenter,
            Quaternion GeomQuaternion)
        {
            return CreateGeom(parentObject, GeomID, GeomName, geomLength, DefaultFriction, DefaultRestitution, GeomCenter, GeomQuaternion, true);
        }

        public static DBoxGeom CreateGeom(GameObject parentObject)
        {
            return CreateGeom(parentObject, 0, "DefaultBox", Vector3.one, DefaultFriction, DefaultRestitution, Vector3.zero, Quaternion.identity, true);
        }
    }
}

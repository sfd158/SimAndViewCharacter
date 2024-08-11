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

using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// Ball Geometry
    /// </summary>
    public class DBallGeom : DGeomObject
    {
        [Tooltip("Transform scale in Unity is diameter, or 2 * radius")]
        public float Radius = 1.0F;

        public override DMass ToDMassByDensity(float density)
        {
            DMass geomMass = new DMass();
            geomMass.SetSphere(density, Radius);
            return geomMass;
        }

        public override PrimitiveType ToPrimitiveType()
        {
            return PrimitiveType.Sphere;
        }

        public override string ToPrimitiveTypeStr()
        {
            return ToPrimitiveType().ToString();
        }

        public override void SetScale()
        {
            transform.localScale = new Vector3(2.0F * Radius, 2.0F * Radius, 2.0F * Radius);
        }

        public override DGeomExportInfo ExportInfo()
        {
            DGeomExportInfo geomInfo = ExportInfoBase();
            geomInfo.Scale = new float[3] { Radius, Radius, Radius };
            return geomInfo;
        }

        public override Quaternion CalcLocalQuaternion()
        {
            return Quaternion.identity;
        }

        public override void ReCompute()
        {

        }

        public override void CalcAttrs()
        {
            base.CalcAttrs();
        }

        public static DBallGeom CreateGeom(GameObject parentObject_,
            int GeomID_,
            string GeomName_,
            float GeomRadius_,
            float frictionCoef_,
            float restitutionCoef_,
            Vector3 GeomCenter_,
            Quaternion GeomQuaternion_,
            bool isClung)
        {
            GameObject geometryObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DBallGeom ballGeom = geometryObject.AddComponent<DBallGeom>();
            ballGeom.Radius = GeomRadius_;
            ballGeom.PostAddGeometry(parentObject_, GeomID_, GeomName_, frictionCoef_, restitutionCoef_, GeomCenter_, GeomQuaternion_, isClung);
            return ballGeom;
        }

        public static DBallGeom CreateGeom(GameObject parentObject_, DGeomExportInfo info)
        {
            return CreateGeom(parentObject_, info.GeomID, info.Name, info.Scale[0], info.Friction, info.Restitution, Utils.ArrToVector3(info.Position), Utils.ArrToQuaternion(info.Quaternion), info.ClungEnv);
        }

        public static DBallGeom CreateGeom(GameObject parentObject_,
            int GeomID_,
            string GeomName_,
            float GeomRadius_,
            Vector3 GeomCenter_,
            Quaternion GeomQuaternion_)
        {
            return CreateGeom(parentObject_, GeomID_, GeomName_, GeomRadius_, DefaultFriction, DefaultRestitution, GeomCenter_, GeomQuaternion_, true);
        }

        public static DBallGeom CreateGeom(GameObject parentObject_, float radius = 1.0F)
        {
            return CreateGeom(parentObject_, 0, "BallGeom", radius, DefaultFriction, DefaultRestitution, Vector3.zero, Quaternion.identity, true);
        }
    }
}

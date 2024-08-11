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
    /// Capsule Geometry.
    /// </summary>
    public class DCapsuleGeom : DGeomObject
    {
        public float Radius = 1.0F;
        [Tooltip("Not Total Length of Capsule. Length of Middle Cylinder")]
        public float Length = 1.0F; // not Total Length. Length of Middle Cylinder.

        public AxisType CapsuleAxis = AxisType.Z; // Default Capsule Axis in ODE v0.12 is z axis.

        public override DMass ToDMassByDensity(float density)
        {
            DMass geomMass = new DMass();
            geomMass.SetCapsule(density, 3, Radius, Length);
            return geomMass;
        }

        public override PrimitiveType ToPrimitiveType()
        {
            return PrimitiveType.Capsule;
        }

        public override string ToPrimitiveTypeStr()
        {
            return ToPrimitiveType().ToString();
        }

        public override Quaternion CalcLocalQuaternion()
        {
            Vector3 capsuleAxis = Vector3.zero;
            capsuleAxis[(int)CapsuleAxis] = 1;
            return Quaternion.FromToRotation(Vector3.up, capsuleAxis);
        }

        public override void SetScale()
        {
            transform.localScale = new Vector3(2.0F * Radius, 0.5F * (Length + 2.0F * Radius), 2.0F * Radius);
        }

        public override DGeomExportInfo ExportInfo()
        {
            DGeomExportInfo geomInfo = ExportInfoBase();
            geomInfo.Scale = new float[3] { Radius, Length, 0.0F };
            return geomInfo;
        }

        public override void ReCompute()
        {
            throw new System.NotImplementedException();
        }

        public override void CalcAttrs()
        {
            base.CalcAttrs();
        }

        public static DCapsuleGeom CreateGeom(
            GameObject parentObject,
            int GeomID,
            string GeomName,
            float geomRadius,
            float geomLength,
            AxisType CapsuleAxisOrient,
            float frictionCoef,
            float restitutionCoef,
            Vector3 GeomCenter,
            Quaternion GeomQuaternion,
            bool isClung)
        {
            GameObject geometryObject = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            DCapsuleGeom capsuleGeom = geometryObject.AddComponent<DCapsuleGeom>();
            capsuleGeom.Radius = geomRadius;
            capsuleGeom.Length = geomLength;
            capsuleGeom.CapsuleAxis = CapsuleAxisOrient;

            capsuleGeom.PostAddGeometry(parentObject, GeomID, GeomName, frictionCoef, restitutionCoef, GeomCenter, GeomQuaternion, isClung);

            Vector3 CapsuleAxis = Vector3.zero;
            CapsuleAxis[(int)CapsuleAxisOrient] = 1;
            geometryObject.transform.localRotation = Quaternion.FromToRotation(Vector3.up, CapsuleAxis);
            return capsuleGeom;
        }

        public static DCapsuleGeom CreateGeom(GameObject parentObject, DGeomExportInfo info)
        {
            return CreateGeom(parentObject, info.GeomID, info.Name, info.Scale[0], info.Scale[1], DWorld.DefaultCapsuleAxis, info.Friction, info.Restitution, Utils.ArrToVector3(info.Position), Utils.ArrToQuaternion(info.Quaternion), info.ClungEnv);
        }

        public static DCapsuleGeom CreateGeom(GameObject parentObject,
            int GeomID,
            string GeomName,
            float geomRadius,
            float geomLength,
            Vector3 GeomCenter,
            Quaternion GeomQuaternion)
        {
            return CreateGeom(parentObject, GeomID, GeomName, geomRadius, geomLength, DWorld.DefaultCapsuleAxis, DefaultFriction, DefaultRestitution, GeomCenter, GeomQuaternion, true);
        }

        public static DCapsuleGeom CreateGeom(GameObject parentObject)
        {
            return CreateGeom(parentObject, 0, "DefaultCylinder", 1, 1, DWorld.DefaultCapsuleAxis, DefaultFriction, DefaultRestitution, Vector3.zero, Quaternion.identity, true);
        }

        /// <summary>
        /// View capsule geometry as 2 half ball and a cylinder
        /// We need to consider the rotation of cylinder (set the rotation of cylinder == capsule)
        /// </summary>
        public void ViewCapsuleAsBallAndCylinder()
        {
            Vector3 offset = new Vector3(0.0F, 0.5F * Length + 0.25F * Radius, 0.0F);  // offset is currect.
            Vector3 global_offset = transform.rotation * offset;

            transform.localScale = Vector3.one;
            GameObject ball0 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            ball0.transform.localScale = (2 * Radius) * Vector3.one;
            ball0.transform.position = transform.position + global_offset;
            ball0.transform.parent = transform;

            GameObject ball1 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            ball1.transform.localScale = (2 * Radius) * Vector3.one;
            ball1.transform.position = transform.position - global_offset;
            ball1.transform.parent = transform;

            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            cylinder.transform.localScale = new Vector3(2 * Radius, 0.5F * Length + 0.25F * Radius, 2 * Radius);
            cylinder.transform.position = transform.position;
            cylinder.transform.rotation = transform.rotation;
            cylinder.transform.parent = transform;
              
            // disable render the original capsule in Unity
            MeshRenderer render = GetComponent<MeshRenderer>();
            render.enabled = false;
        }

        public void ViewCapsuleAsRaw()
        {
            for(int i = 0; i < transform.childCount; i++)
            {
                MeshRenderer render = transform.GetChild(i).GetComponent<MeshRenderer>();
                render.enabled = false;
            }
            MeshRenderer renderer = GetComponent<MeshRenderer>();
            renderer.enabled = true;
        }
    }
}

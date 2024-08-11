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
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// Geometry Component for GameObject.
    /// Supported Geometry: Box, Capsule, Sphere, plane
    /// TODO: support mesh geometry, height map geometry
    /// </summary>
    [DisallowMultipleComponent]
    public abstract class DGeomObject : DBaseObject
    {
        [Tooltip("collision detection with other geometry")]
        public bool Collidable = true;

        [Tooltip("Friction")]
        public float Friction = DefaultFriction;

        [Tooltip("Restitution")]
        public float Restitution = DefaultRestitution;

        [Tooltip("if this geom contact with environment, character fall down flag in ODE will be set true")]
        public bool ClungEnv = false;

        [HideInInspector]
        public static readonly float DefaultFriction = 0.8F;

        [HideInInspector]
        public static readonly float DefaultRestitution = 1.0F;

        [HideInInspector]
        public DRigidBody dRigidBody; // rigid body of this geometry

        [HideInInspector]
        public Renderer geomRenderer;

        [HideInInspector]
        public Material initMaterial;

        public void PostAddGeometry(
            GameObject parentObject_,
            int geomID_,
            string GeomName_,
            float frictionCoef_,
            float restitutionCoef_,
            Vector3 geomCenter_,
            Quaternion GeomQuaternion_,
            bool isClung)
        {
            GameObject virtualGeomObject = new GameObject
            {
                name = "virtual_" + GeomName_
            };
            virtualGeomObject.transform.parent = parentObject_.transform;
            VirtualGeom virtualGeom = virtualGeomObject.AddComponent<VirtualGeom>();
            virtualGeom.IDNum = IDNum = geomID_;
            virtualGeom.Friction = Friction = frictionCoef_;
            virtualGeom.Restitution = Restitution = restitutionCoef_;
            virtualGeom.transform.SetPositionAndRotation(geomCenter_, GeomQuaternion_);
            virtualGeom.SetInitialPosAndQuat(geomCenter_, GeomQuaternion_);
            virtualGeom.ClungEnv = ClungEnv = isClung;

            SetScale();
            InitialPosition = geomCenter_;
            gameObject.name = GeomName_;
            // virtualGeomObject.transform.localRotation = GeomQuaternion_;
            gameObject.transform.SetPositionAndRotation(geomCenter_, GeomQuaternion_);
            transform.parent = virtualGeomObject.transform;
        }

        public override void CalcAttrs()
        {
            if (DCommonConfig.SupportGameObjectColor)
            {
                if (geomRenderer == null)
                {
                    geomRenderer = GetComponent<Renderer>();
                }
                if (initMaterial == null)
                {
                    initMaterial = geomRenderer.material;
                }
            }
        }

        public void ResetToInitMaterial()
        {
            if (geomRenderer != null && initMaterial != null && geomRenderer.material != initMaterial)
            {
                geomRenderer.material = initMaterial;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public abstract Quaternion CalcLocalQuaternion();

        public abstract PrimitiveType ToPrimitiveType();

        public abstract string ToPrimitiveTypeStr();

        /// <summary>
        /// 
        /// </summary>
        public abstract void SetScale();

        public abstract DGeomExportInfo ExportInfo();

        public DGeomExportInfo ExportInfoBase()
        {
            DGeomExportInfo geomInfo = new DGeomExportInfo
            {
                GeomID = IDNum,
                Name = gameObject.name,
                GeomType = ToPrimitiveTypeStr(),
                Collidable = Collidable,
                Friction = Friction,
                Restitution = Restitution,
                ClungEnv = ClungEnv,
                Position = Utils.Vector3ToArr(transform.position),
                Quaternion = Utils.QuaternionToArr(transform.rotation)
            };

            return geomInfo;
        }

        public abstract DMass ToDMassByDensity(float density);
    }
}

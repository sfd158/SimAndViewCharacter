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
    /// <summary>
    /// This class is used for handle difference bewteen Capsule's axis in Open Dynamics Engine (along z axis) v0.12 and in Unity (along y axis).
    /// GameObject with Component of Virtual Geom is parent of GameObject with Component of DBallGeom, DBoxGeom, DCapsuleGeom, or DPlaneGeom.
    /// For character, GameObject with Component of VirtualGeom is child of GameObject with Component of DRigidBody.
    /// TODO: VirtualGeom should be optional in Unity.
    /// </summary>
    public class VirtualGeom : DGeomObject
    {
        [HideInInspector]
        public DGeomObject childGeom;

        public override DMass ToDMassByDensity(float density)
        {
            if(childGeom == null)
            {
                CalcAttrs();
            }
            return childGeom.ToDMassByDensity(density);
        }

        /// <summary>
        /// initialize geometry
        /// </summary>
        public override void CalcAttrs()
        {
            IDNum = GetInstanceID();
            childGeom = transform.GetChild(0).GetComponent<DGeomObject>();
            childGeom.CalcAttrs();
        }

        public void CalcAttrs(DRigidBody body)
        {
            CalcAttrs();
            this.dRigidBody = body;
            if (childGeom != null)
            {
                childGeom.dRigidBody = body;
            }
        }

        public override void SetScale()
        {
            throw new System.NotImplementedException();
        }

        public override PrimitiveType ToPrimitiveType()
        {
            return transform.GetChild(0).GetComponent<DGeomObject>().ToPrimitiveType();
        }

        public override string ToPrimitiveTypeStr()
        {
            return transform.GetChild(0).GetComponent<DGeomObject>().ToPrimitiveTypeStr();
        }

        public override DGeomExportInfo ExportInfo()
        {
            DGeomExportInfo info = childGeom.ExportInfo();

            info.GeomID = IDNum;
            info.Position = Utils.Vector3ToArr(transform.position);
            info.Quaternion = Utils.QuaternionToArr(transform.rotation);
            info.Collidable &= Collidable;
            return info;
        }

        public override void ReCompute()
        {
            // Debug.Log("VGeom ReCompute");
        }

        public override Quaternion CalcLocalQuaternion()
        {
            throw new NotImplementedException();
        }

        public new void SetInitialState(bool setQuaternion = false)
        {
            base.SetInitialState(setQuaternion);
            if (childGeom != null)
            {
                childGeom.SetInitialState(setQuaternion);
                childGeom.transform.localRotation = childGeom.CalcLocalQuaternion();
            }
        }

        public new void SaveToInitialState()
        {
            base.SaveToInitialState();
            if (childGeom != null)
            {
                childGeom.transform.localRotation = childGeom.CalcLocalQuaternion();
                childGeom.SaveToInitialState();
            }
        }

        /// <summary>
        /// set contact friction
        /// </summary>
        /// <param name="value"></param>
        public void SetMu(float value)
        {
            this.Friction = value;
            this.childGeom.Friction = value;
        }
    }
}

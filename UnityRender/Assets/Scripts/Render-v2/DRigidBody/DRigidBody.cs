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
    [System.Serializable]
    public enum DRigidBodyMassMode
    {
        Density = 0,
        MassValue = 1
    };

    [System.Serializable]
    public enum DRigidBodyInertiaMode
    {
        Density = 0,
        InertiaValue = 1
    };

    /// <summary>
    /// Rigid Body. GameObject with DRigidBody Component should have parent GameObject with dCharacter Component or dJoint Component.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(ScaleChecker))]
    public partial class DRigidBody : DBaseObject, IParseUpdate<DRigidBodyUpdateInfo>
    {
        [Tooltip("Mass Mode")]
        public DRigidBodyMassMode MassMode = DRigidBodyMassMode.Density;

        // public float mass = 1.0F;
        [Tooltip("Body Density")]
        public float Density = 1000.0F;

        [Tooltip("Body Mass")]
        public float Mass = 1.0F;

        [Tooltip("Inertia Mode")]
        public DRigidBodyInertiaMode InertiaMode = DRigidBodyInertiaMode.Density;

        [Tooltip("Body Inertia")]
        public float[] BodyInertia = {1, 0, 0, 0, 1, 0, 0, 0, 1};

        /// <summary>
        /// Linear Velocity computed in simulation
        /// </summary>
        [HideInInspector]
        public Vector3 LinearVelocity = Vector3.zero; // showing in inspector is slow..

        /// <summary>
        /// Angular Velocity computed in simulation
        /// </summary>
        [HideInInspector]
        public Vector3 AngularVelocity = Vector3.zero; // showing in inspector is slows..

        [Tooltip("Ignore Collision Detection with. You can drag GameObject with DRigidBody Component to here.")]
        public List<GameObject> IgnoreCollision;

        [Tooltip("Character GameObject")]
        public GameObject character;

        [HideInInspector]
        public DCharacter dCharacter;

        [HideInInspector]
        public DJoint parentJoint;

        //[HideInInspector]
        //public DJoint childJoint;

        [HideInInspector]
        public DRigidBody parentBody;

        [HideInInspector]
        public List<VirtualGeom> vGeomList;

        private void OnDestroy()
        {
            if (vGeomList != null)
            {
                vGeomList.Clear();
            }
        }

        /// <summary>
        /// initialize DRigidBody Component
        /// </summary>
        public override void CalcAttrs()
        {
            dCharacter = character.GetComponent<DCharacter>();
            parentJoint = GetParentJoint();
            parentBody = GetParentBody();
            vGeomList = GetVirtualGeoms();
            CalcChildAttrs();
            // Only use in debug mode..
            // SetBodyPositionToCoM();
        }

        /// <summary>
        /// initialize child geometry
        /// </summary>
        private void CalcChildAttrs()
        {
            for(int i = 0; i < vGeomList.Count; i++)
            {
                vGeomList[i].CalcAttrs(this);
            }
        }

        public DJoint GetParentJoint()
        {
            GameObject parentObject = transform.parent.gameObject;
            if (parentObject.GetComponent<DCharacter>() != null)
            {
                return null;
            }
            DJoint djoint = parentObject.GetComponent<DJoint>();
            if (djoint == null)
            {
                throw new ArgumentException("DJoint Component is Required.");
            }
            return djoint;
        }

        public int GetParentJointID()
        {
            return parentJoint == null ? -1 : parentJoint.IDNum;
        }

        public DRigidBody GetParentBody()
        {
            DJoint parentJoint = GetParentJoint();

            if (parentJoint == null)
            {
                return null;
            }
            return parentJoint.GetParentBody();
        }

        public int GetParentBodyID()
        {
            return parentBody == null ? -1 : parentBody.IDNum;
        }

        public List<VirtualGeom> GetVirtualGeoms()
        {
            int childCount = transform.childCount;
            List<VirtualGeom> VGeomList = new List<VirtualGeom>(childCount);
            for(int chIdx=0; chIdx<childCount; chIdx++)
            {
                GameObject child = transform.GetChild(chIdx).gameObject;
                if (child.TryGetComponent<VirtualGeom>(out var vgeom))
                {
                    VGeomList.Add(vgeom);
                }
            }

            // debug
            // Debug.Log(name);
            // for(int chIdx=0; chIdx<childCount; chIdx++)
            // {
            //     Debug.Log(VGeomList[chIdx].IDNum);
            // }
            // VGeomList.Sort();
            // for(int chIdx=0; chIdx<childCount; chIdx++)
            // {
            //     Debug.Log(VGeomList[chIdx].IDNum);
            // }
            return VGeomList;
        }

        public List<DGeomObject> GetGeoms()
        {
            List<DGeomObject> GeomList = new List<DGeomObject>(transform.childCount);
            // Debug.Log(transform.childCount);
            for (int chIdx = 0; chIdx < transform.childCount; chIdx++)
            {
                Transform child = transform.GetChild(chIdx).GetChild(0);
                if (child.TryGetComponent<DGeomObject>(out var geom))
                {
                    GeomList.Add(geom);
                }
                else
                {
                    throw new ArgumentException("Geometry Should have DGeomObject Component.");
                }
            }
            GeomList.Sort();
            return GeomList;
        }

        public new void SetInitialState(bool setQuaternion = false)
        {
            base.SetInitialState(setQuaternion);
            foreach(var vgeom in vGeomList)
            {
                vgeom.SetInitialState(setQuaternion);
            }
        }

        public new void SaveToInitialState()
        {
            base.SaveToInitialState();
            foreach(var vgeom in vGeomList)
            {
                vgeom.SaveToInitialState();
            }
        }

        public override void ReCompute()
        {

        }

        /// <summary>
        /// Update body information
        /// </summary>
        /// <param name="info"></param>
        public void ParseUpdateInfo(DRigidBodyUpdateInfo info)
        {
            if (info.LinearVelocity != null)
            {
                LinearVelocity = Utils.ArrToVector3(info.LinearVelocity);
            }
            if (info.AngularVelocity != null)
            {
                AngularVelocity = Utils.ArrToVector3(info.AngularVelocity);
            }
            if (info.Color != null)
            {
                Color color = new Color(info.Color[0], info.Color[1], info.Color[2]);
                Material old_mat = vGeomList[0].childGeom.initMaterial;
                Material new_mat = new Material(old_mat.shader);
                new_mat.color = color;
                foreach(var vgeom in vGeomList)
                {
                    var geom = vgeom.childGeom;
                    geom.geomRenderer.material = new_mat;
                }
            }
            else
            {
                // roll back to default color material
                foreach (var vgeom in vGeomList)
                {
                    var geom = vgeom.childGeom;
                    geom.ResetToInitMaterial();
                }
            }

            if (DCommonConfig.dWorldUpdateMode == DWorldUpdateMode.ReducedCoordinate)
            {
                // DO Nothing here
            }
            else if (DCommonConfig.dWorldUpdateMode == DWorldUpdateMode.MaximalCoordinate)
            {
                Vector3 updatePos = info.Position == null ? transform.position: Utils.ArrToVector3(info.Position);
                Quaternion updateQuat = info.Quaternion == null ? transform.rotation: Utils.ArrToQuaternion(info.Quaternion);
                transform.SetPositionAndRotation(updatePos, updateQuat);
            }
        }

        public DMass ComputeBodyMassByDensity()
        {
            DMass bodyMass = new DMass();
            foreach(var vgeom in vGeomList)
            {
                DMass geomMass = vgeom.ToDMassByDensity(Density);
                bodyMass.Add(ref geomMass);
            }
            return bodyMass;
        }

        public void SetBodyPositionToCoM()
        {
            if (vGeomList.Count == 0)
            {
                return;
            }
            Vector3 com = Vector3.zero;
            float totMass = 0.0F;
            Vector3[] InitialPos = new Vector3[vGeomList.Count];
            for(int i=0; i<vGeomList.Count; i++)
            {
                VirtualGeom vgeom = vGeomList[i];
                DMass mass = vgeom.ToDMassByDensity(Density);
                float massVal = Convert.ToSingle(mass.mass);
                totMass += massVal;
                com += massVal * vgeom.transform.position;
                InitialPos[i] = vgeom.transform.position;
            }
            com /= totMass;
            transform.position = com;
            for(int i=0; i<vGeomList.Count; i++)
            {
                VirtualGeom vgeom = vGeomList[i];
                vgeom.transform.position = InitialPos[i];
            }
        }
    }
}


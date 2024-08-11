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
    /// Character Component
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(ScaleChecker))]
    public partial class DCharacter : DBaseObject, IParseUpdate<DCharacterUpdateInfo>, IExportInfo<DCharacterExportInfo>
    {
        [Tooltip("View in Playing")]
        public float TotalMass = 0.0F;

        [Tooltip("View in Playing")]
        public Vector3 CenterOfMass = Vector3.zero;

        /// <summary>
        /// Character has real root joint
        /// </summary>
        public bool HasRealRootJoint
        {
            get
            {
                return RootJoint != null && !(RootJoint is DEmptyJoint);
            }
        }

        [Tooltip("self collision detection")]
        public bool SelfCollision = true;

        [Tooltip("Ignore Collision between parent and self")]
        public bool IgnoreParentCollision = true;

        [Tooltip("Ignore Collision between grand parent and self")]
        public bool IgnoreGrandParentCollision = true;

        [Tooltip("Kinematic")]
        public bool Kinematic = false;

        [Tooltip("Label")]
        public string characterLabel = "";

        [Tooltip("Joint Render Radius")]
        public float JointRenderRadius = 0.02F;

        [Tooltip("Render Joint Position")]
        public bool IsRenderJointPosition = false;

        [Tooltip("Render Body Position")]
        public bool IsRenderBodyPosition = false;

        [Tooltip("Render Body Center of Mass")]
        public bool IsRenderBodyCoM = false;

        [Tooltip("Render CoM")]
        public bool IsRenderCoM = true;

        [Tooltip("Render Joint Torque")]
        public bool IsRenderJointTorque = false;

        [Tooltip("Default Joint Damping")]
        public float DefaultJointDamping = 0;

        [Tooltip("Default Geom Mu")]
        public float DefaultGeomMu = 0.8F;

        /// <summary>
        /// Root Joint, which can be null
        /// </summary>
        [HideInInspector]
        public DJoint RootJoint;

        [HideInInspector]
        protected DRigidBody RootBody;  // maintain all of joints of character

        [HideInInspector]
        public List<DRigidBody> BodyList;  // maintain all of bodies of character

        [HideInInspector]
        public List<DJoint> JointList;

        [HideInInspector]
        protected List<DEndJoint> EndJointList;

        [HideInInspector]
        public PDController pdController; // PD Controller parameter for dCharacter

        [HideInInspector]
        public DCharacterControl dCharacterControl; // controller for dCharacter

        public override void CalcAttrs()
        {
            CalcAttrs(out _);
        }

        /// <summary>
        /// initialize character
        /// </summary>
        /// <param name="q"></param>
        public void CalcAttrs(out List<GameObject> q)
        {
            // IDNum = GetInstanceID();

            RootJoint = null;
            pdController = null;
            dCharacterControl = GetComponent<DCharacterControl>();
            for (int i = 0; i < transform.childCount; i++)
            {
                GameObject obj = transform.GetChild(i).gameObject;
                if (obj.TryGetComponent<DJoint>(out var RootJoint_)) // Get Root Joint
                {
                    RootJoint = RootJoint_;
                }
                if (obj.TryGetComponent<PDController>(out var pdController_)) // Get PD Controller
                {
                    pdController = pdController_;
                }
                if (obj.TryGetComponent<DRigidBody>(out var drigidBody_)) // Get Root Body
                {
                    RootBody = drigidBody_;
                }
            }

            if (RootJoint != null)
            {
                RootBody = RootJoint.GetChildBody(); //RootJoint.childBody;
            }

            q = Utils.GetAllOffSpring(gameObject);
            BodyList = GetBodyList(q, true);
            JointList = GetJoints(q, false, true);
            EndJointList = GetEndJoints(q);
            CalcChildAttrs();
        }

        /// <summary>
        /// Get PDController Component in children
        /// </summary>
        /// <returns></returns>
        public PDController GetPDController()
        {
            int cnt = 0;
            PDController pdController_ = null;
            for (int chIdx = 0; chIdx < transform.childCount; chIdx++)
            {
                GameObject gameObject = transform.GetChild(chIdx).gameObject;
                if (gameObject.TryGetComponent<PDController>(out var res_))
                {
                    cnt++;
                    pdController_ = res_;
                }
            }

            if (cnt > 1)  // controller should be unique in one character
            {
                throw new ArgumentException("There should be 0 or 1 Controller Component attached to Character.");
            }

            return pdController_;
        }

        /// <summary>
        /// initialize children
        /// </summary>
        private void CalcChildAttrs()
        {
            if (RootJoint != null)
            {
                if (RootJoint is DEmptyJoint)
                {
                    RootJoint.CalcAttrs();
                }
            }

            // compute all of child attrs of joints
            for (int jointIdx=0; jointIdx<JointList.Count; jointIdx++)
            {
                DJoint dJoint = JointList[jointIdx];
                dJoint.CalcAttrs();
            }
            for(int bodyIdx=0; bodyIdx<BodyList.Count; bodyIdx++)
            {
                DRigidBody dRigidBody = BodyList[bodyIdx];
                dRigidBody.CalcAttrs();
            }
            for(int endIdx=0; endIdx<EndJointList.Count; endIdx++)
            {
                DEndJoint endJoint = EndJointList[endIdx];
                endJoint.CalcAttrs();
            }
        }

        public void AfterLoad(DRigidBody RootBody_, DJoint RootJoint_, List<DRigidBody> BodyList_, List<DJoint> JointList_, List<DEndJoint> EndJointList_, PDController pdController_)
        {
            RootBody = RootBody_;
            RootJoint = RootJoint_;
            if(RootJoint != null)
            {
                InitialPosition = RootJoint.InitialPosition;
            }
            else if (RootBody != null)
            {
                InitialPosition = RootBody.InitialPosition;
            }

            CalcAttrs();
            SetInitialState(true);

            // ReCompute();
        }

        public void JointViewer()
        {
            if (IsRenderJointPosition && JointList != null)
            {
                for(int i=0; i<JointList.Count; i++)
                {
                    Vector3 pos = JointList[i].transform.position;
                    Gizmos.DrawSphere(pos, JointRenderRadius);
                }
            }
        }

        public void BodyPosViewer()
        {
            if (IsRenderBodyPosition && BodyList != null)
            {
                for(int i=0; i<BodyList.Count; i++)
                {
                    Vector3 pos = BodyList[i].transform.position;
                    Gizmos.DrawSphere(pos, 2 * JointRenderRadius);
                }
            }
        }

        public void BodyPositionViewer2()
        {
            if (IsRenderBodyPosition && BodyList != null)
            {
                // Debug.Log("Render2");
                Gizmos.color = Color.red;
                Vector3 show_size = new Vector3(0.02f, 0.3f, 0.02f);
                for (int i = 0; i < BodyList.Count; i++)
                {
                    Vector3 pos = BodyList[i].transform.position;
                    Gizmos.DrawCube(pos, show_size);
                    // Gizmos.DrawSphere(pos, 2 * JointRenderRadius);
                }
            }
        }

        /// <summary>
        /// View joint torque as single line? or color?
        /// View body torque/force as single line? or color?
        /// </summary>
        public void JointTorqueViewer()
        {
            if (IsRenderJointTorque && JointList != null)
            {

            }
        }

        public void ComViewer()
        {
            if (IsRenderCoM)
            {
                var com = CenterOfMass;
                // Gizmos.DrawLine(com, new Vector3(com.x, 0, com.z));
                Gizmos.color = Color.cyan;
                Gizmos.DrawCube(new Vector3(com.x, 0.5F * com.y, com.z), new Vector3(0.02F, com.y, 0.02F));
            }
        }

        /// <summary>
        /// Get all of capsule component, and view as ball/capsule here..
        /// </summary>
        public void ViewCapsuleAsBallAndCylinder()
        {
            DCapsuleGeom[] capsule_list = GetComponentsInChildren<DCapsuleGeom>();
            for(int i = 0; i < capsule_list.Length; i++)
            {
                capsule_list[i].ViewCapsuleAsBallAndCylinder();
            }
        }

        public void ViewCapsuleAsRaw()
        {
            DCapsuleGeom[] capsule_list = GetComponentsInChildren<DCapsuleGeom>();
            for (int i = 0; i < capsule_list.Length; i++)
            {
                capsule_list[i].ViewCapsuleAsRaw();
            }
        }

        void Start()
        {
            ViewCapsuleAsBallAndCylinder();
        }

        private void OnDrawGizmos()
        {
            JointViewer();
            BodyPosViewer();
            BodyPositionViewer2();
            ComViewer();
        }
    }
}


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
    public enum DJointType
    {
        BallJointType,
        HingeJointType,
        EmptyJointType,
        ContactJointType,
        FixedJointType,
        EndJointType
    };

    /// <summary>
    /// Joint Class
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(ScaleChecker))] // joint.transform.Scale should always be (1, 1, 1) 
    public abstract class DJoint : DBaseObject, IExportInfo<DJointExportInfo>, IParseUpdate<DJointUpdateInfo>
    {
        // public float TorqueLimit = 0.0F;

        [Tooltip("Joint Damping")]
        public float Damping = 0.0F;

        [Tooltip("Joint Weight for computing loss and etc")]
        public float weight = 1.0F;

        //[Tooltip("JointID")]
        //public int IDNum = -1;

        [Tooltip("Character GameObject")]
        public GameObject character;

        [Tooltip("Euler Axis Object. It can be null, or joint's child GameObject with DEulerAxis Component")]
        public GameObject EulerAxisObject;

        /// <summary>
        /// Euler Angle Axis
        /// </summary>
        [HideInInspector]
        public DEulerAxis dEulerAxis;

        [HideInInspector]
        public DCharacter dCharacter;

        [HideInInspector]
        public DRigidBody parentBody;

        [HideInInspector]
        public DRigidBody childBody;

        [HideInInspector]
        public DJoint parentJoint;

        /// <summary>
        /// Euler Axis Local Rotation
        /// </summary>
        public Quaternion EulerAxisLocalRot
        {
            get
            {
                return EulerAxisObject == null ? Quaternion.identity : EulerAxisObject.transform.localRotation;
            }
        }

        public void SetParam(string jointName_, int jointID_, float damping_, GameObject character_)
        {
            gameObject.name = jointName_;
            IDNum = jointID_;

            Damping = damping_;
            character = character_;
        }

        public override void CalcAttrs()
        {
            if (character != null)
            {
                dCharacter = character.GetComponent<DCharacter>();
                parentBody = GetParentBody();
                parentJoint = GetParentJoint();
                childBody = GetChildBody();
                if(EulerAxisObject != null)
                {
                    dEulerAxis = EulerAxisObject.GetComponent<DEulerAxis>();
                }
            }
        }

        /// <summary>
        /// Get Joint Type String
        /// </summary>
        /// <returns></returns>
        public abstract string JointType();

        public abstract DJointType EnumJointType();

        public DRigidBody GetChildBody()
        {
            if (character == null)
            {
                return null;
            }
            int childBodyCount = 0;
            DRigidBody dRigidBody = null;
            for(int gameIdx=0; gameIdx<transform.childCount; gameIdx++)
            {
                GameObject gameObject = transform.GetChild(gameIdx).gameObject;
                DRigidBody dRigidBody_ = gameObject.GetComponent<DRigidBody>();
                if (dRigidBody_ != null)
                {
                    childBodyCount++;
                    dRigidBody = dRigidBody_;
                }
            }

            if (childBodyCount > 1) // Make sure there is only 0 or 1 child body
            {
                throw new ArithmeticException("dRigidBody Count != 1 " + this.name);
            }

            return dRigidBody;
        }

        public int GetChildBodyID()
        {
            return childBody == null ? -1 : childBody.IDNum;
        }

        public DJoint GetParentJoint()
        {
            if (transform.parent.TryGetComponent(out DJoint parentDJoint))
            {
                return parentDJoint;
            }
            else
            {
                return null;
            }
        }

        public int GetParentJointID()
        {
            return parentJoint == null ? -1 : parentJoint.IDNum;
        }

        public DRigidBody GetParentBody()
        {
            if (dCharacter == null)
            {
                return null;
            }
            if (transform.parent == dCharacter.transform.parent)
            {
                return null;
            }
            GameObject parentJoint = transform.parent.gameObject;
            if(parentJoint.transform == dCharacter.transform)
            {
                return null;
            }
            DJoint dJointParent = parentJoint.GetComponent<DJoint>();
            if (dJointParent == null)
            {
                throw new ArgumentException("DJoint Component is required.");
            }
            return dJointParent.GetChildBody();
        }

        public int GetParentBodyID()
        {
            return parentBody == null ? -1 : parentBody.IDNum;
        }

        public DJointExportInfo ExportInfoBase()
        {
            DJointExportInfo jointInfo = new DJointExportInfo
            {
                JointID = IDNum,
                Name = gameObject.name,
                JointType = JointType(),
                Weight = weight,
                ParentBodyID = GetParentBodyID(),
                ChildBodyID = GetChildBodyID(),
                ParentJointID = GetParentJointID(),
                Damping = Damping,
                Position = Utils.Vector3ToArr(transform.position),
                Quaternion = Utils.QuaternionToArr(transform.rotation),
                EulerAxisLocalRot = Utils.QuaternionToArr(EulerAxisLocalRot)
            };
            return jointInfo;
        }

        public abstract DJointExportInfo ExportInfo();

        public void ParseUpdateInfo(DJointUpdateInfo info)
        {
            Vector3 pos3 = Utils.ArrToVector3(info.Position);
            Quaternion q = Utils.ArrToQuaternion(info.Quaternion);
            if (DCommonConfig.dWorldUpdateMode == DWorldUpdateMode.ReducedCoordinate)
            {
                if (pos3 != null)
                {
                    transform.SetPositionAndRotation(pos3, q);
                }
                else
                {
                    transform.rotation = q;
                }
            }
            else if (DCommonConfig.dWorldUpdateMode == DWorldUpdateMode.MaximalCoordinate)
            {
                transform.SetPositionAndRotation(pos3, q);
            }
        }

        public new void SetInitialState(bool setQuaternion = false)
        {
            base.SetInitialState(setQuaternion);
            if (dEulerAxis != null)
            {
                dEulerAxis.SetInitialState(setQuaternion);
            }
        }

        public new void SaveToInitialState()
        {
            base.SaveToInitialState();
            if (dEulerAxis != null)
            {
                dEulerAxis.SaveToInitialState();
            }
        }

        public override void ReCompute()
        {
            if (transform.rotation == Quaternion.identity)
            {
                return;
            }
            if (EulerAxisObject == null)
            {
                GameObject eulerObject = new GameObject();
                eulerObject.name = gameObject.name + "Euler";
                eulerObject.transform.parent = transform;
                eulerObject.transform.localPosition = Vector3.zero;
                eulerObject.transform.localRotation = Quaternion.identity;
                dEulerAxis = eulerObject.AddComponent<DEulerAxis>();
                dEulerAxis.SaveToInitialState();
                EulerAxisObject = eulerObject;
            }
            InitialQuaternion = Quaternion.identity;
        }

        /// <summary>
        /// should run re-compute after modifing joint rotation
        /// </summary>
        public void ReComputeButton()
        {
            if (dCharacter == null)
            {
                dCharacter = Utils.GetComponentAncestor<DCharacter>(gameObject);
            }
            if (dCharacter != null)
            {
                dCharacter.ReCompute();
            }
            else if (TryGetComponent<ExtJoint>(out _))
            {
                return; // ExtJoint
            }
            else
            {
                Debug.LogError("Joint doesn't have DCharacter Ancestor.");
            }
        }

        public bool HasVisualizeComponent()
        {
            for(int childIdx=0; childIdx<transform.childCount; childIdx++)
            {
                if (transform.GetChild(childIdx).GetComponent<DJointVisualizeComponent>() != null)
                {
                    return true;
                }
            }
            return false;
        }
    }
}

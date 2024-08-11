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
    /// External Joint
    /// </summary>
    [DisallowMultipleComponent]
    public class ExtJoint : DBaseObject, IParseUpdate<ExtJointUpdateInfo>
    {
        [Tooltip("Body 0 attached to Constraint Joint")]
        public GameObject Body0;

        [Tooltip("Body 1 attached to Constraint Joint")]
        public GameObject Body1;

        [HideInInspector]
        public DRigidBody dRigidBody0;

        [HideInInspector]
        public DRigidBody dRigidBody1;

        [HideInInspector]
        public DJoint dJoint;

        public override void ReCompute()
        {
            
        }

        public override void CalcAttrs()
        {
            dRigidBody0 = Body0 == null ? null : Body0.GetComponent<DRigidBody>();
            dRigidBody1 = Body1 == null ? null : Body1.GetComponent<DRigidBody>();
            if (Body0 == null && Body1 == null)
            {
                throw new MissingReferenceException("There should be 1 or 2 bodies attached to the joint");
            }
            dJoint = GetComponent<DJoint>();
        }

        public ExtJointExportInfo ExportInfo()
        {
            return ExportInfo(dJoint);
        }

        public ExtJointExportInfo ExportInfo(DJoint joint)
        {
            DJointExportInfo jointInfo = joint.ExportInfo();
            ExtJointExportInfo info = new ExtJointExportInfo(jointInfo);
            info.Body0ID = dRigidBody0.IDNum;
            info.Character0ID = dRigidBody0.dCharacter.IDNum;

            info.Body1ID = dRigidBody1.IDNum;
            info.Character1ID = dRigidBody1.dCharacter.IDNum;

            return info;
        }

        public void ParseUpdateInfo(ExtJointUpdateInfo info)
        {
            if (info.Quaternion != null)
            {
                transform.SetPositionAndRotation(Utils.ArrToVector3(info.Position), Utils.ArrToQuaternion(info.Quaternion));
            }
            else
            {
                transform.position = Utils.ArrToVector3(info.Position);
            }
        }
    }
}

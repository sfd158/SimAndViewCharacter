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
    public partial class DCharacter
    {
        /// <summary>
        /// Export Character to Json Format
        /// </summary>
        public DCharacterExportInfo ExportInfo()
        {
            DCharacterExportInfo characterInfo = new DCharacterExportInfo
            {
                CharacterID = IDNum,
                CharacterName=gameObject.name,
                SelfCollision = SelfCollision,
                IgnoreParentCollision = IgnoreParentCollision,
                IgnoreGrandpaCollision = IgnoreGrandParentCollision,
                Kinematic = Kinematic,
                CharacterLabel = characterLabel,
                HasRealRootJoint = HasRealRootJoint,
                Bodies = new DBodyExportInfo[BodyList.Count],
                Joints = new DJointExportInfo[JointList.Count],
                EndJoints = new DEndJointExportInfo[EndJointList.Count],
                PDControlParam = pdController == null ? null : pdController.ExportInfo()
            };

            // Generate Body Export Info
            for (int bodyIdx = 0; bodyIdx < BodyList.Count; bodyIdx++)
            {
                characterInfo.Bodies[bodyIdx] = BodyList[bodyIdx].ExportInfo();
            }

            // Generate Joint Export Info
            for (int jointIdx = 0; jointIdx < JointList.Count; jointIdx++)
            {
                characterInfo.Joints[jointIdx] = JointList[jointIdx].ExportInfo();
            }

            // Generate EndJoint Export Info
            for (int endIdx = 0; endIdx < EndJointList.Count; endIdx++)
            {
                characterInfo.EndJoints[endIdx] = EndJointList[endIdx].EndJointExportInfo();
            }

            characterInfo.RootInfo = ExportRootInfo();

            return characterInfo;
        }

        DRootExportInfo ExportRootInfo()
        {
            DRootExportInfo result = new DRootExportInfo{
                Position=Utils.Vector3ToArr(RootJoint.transform.position),
                Quaternion=Utils.QuaternionToArr(RootJoint.transform.rotation)
            };
            return result;
        }
        /// <summary>
        /// Convert Export Infomation to Json Format
        /// </summary>
        /// 
        public string ExportInfoJson()
        {
            DCharacterExportInfo CharacterInfo = ExportInfo();
            return JsonUtility.ToJson(CharacterInfo);
        }
    }
}


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
    /// Add External Joint as constraints
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(ScaleChecker))]
    public class DExtJointList : DBaseContainer<ExtJoint>, IExportInfo<ExtJointListExportInfo>, IParseUpdate<ExtJointListUpdateInfo>
    {
        /// <summary>
        /// External Joint List Export Information in Unity
        /// </summary>
        /// <returns></returns>
        public ExtJointListExportInfo ExportInfo()
        {
            ExtJointListExportInfo info = new ExtJointListExportInfo
            {
                Joints = new ExtJointExportInfo[CreateBuffer.Count]
            };

            for(int jointIdx=0; jointIdx < CreateBuffer.Count; jointIdx++)
            {
                ExtJoint extJoint = CreateBuffer[jointIdx];
                info.Joints[jointIdx] = extJoint.ExportInfo();
            }

            return info;
        }

        /// <summary>
        /// External joint list remove information in Unity
        /// </summary>
        /// <returns>External joint list remove info</returns>
        public DExtJointListRemoveInfo RemoveInfo()
        {
            if (RemoveBuffer == null)
            {
                return null;
            }
            DExtJointListRemoveInfo info = new DExtJointListRemoveInfo
            {
                ExtJointID = new int[RemoveBuffer.Count]
            };
            for(int i=0; i<RemoveBuffer.Count; i++)
            {
                info.ExtJointID[i] = RemoveBuffer[i].IDNum;
            }
            return info;
        }

        /// <summary>
        /// Parse external joint list update info from server.
        /// </summary>
        /// <param name="updateInfo"></param>
        public void ParseUpdateInfo(ExtJointListUpdateInfo updateInfo)
        {
            if (updateInfo.Joints == null)
            {
                return;
            }
            
            foreach(var info in updateInfo.Joints)
            {
                if (TObjectDict.TryGetValue(info.JointID, out var extJoint))
                {
                    extJoint.ParseUpdateInfo(info);
                }
            }

            // TODO: Add / Remove ExtJoint in Python Server
        }
    }
}


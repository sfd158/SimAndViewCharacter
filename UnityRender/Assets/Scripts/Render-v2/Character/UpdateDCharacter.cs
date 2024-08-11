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
        /// Parse Character Update Information from Server
        /// </summary>
        /// <param name="UpdateInfo">Update Information from Server</param>
        public void ParseUpdateInfo(DCharacterUpdateInfo UpdateInfo)
        {
            // Parse Root Update Info
            DRootUpdateInfo RootInfo = UpdateInfo.RootInfo;
            transform.position = Utils.ArrToVector3(RootInfo.Position);
            // TODO: Rotation should set on Character or EmptyRootJoint?
            if (RootJoint != null)
            {
                RootJoint.transform.rotation = Utils.ArrToQuaternion(RootInfo.Quaternion);
            }
            else
            {
                RootBody.transform.rotation = Utils.ArrToQuaternion(RootInfo.Quaternion);
            }

            // Parse Joint Update Info
            DJointUpdateInfo[] JointInfo = UpdateInfo.JointInfo;
            if (JointList.Count != JointInfo.Length)
            {
                throw new ArgumentException("Number of Joints in UpdateInfo and JointList don't match ");
            }
            Array.Sort(JointInfo);
            for (int jointIdx = 0; jointIdx < JointInfo.Length; jointIdx++)
            {
                JointList[jointIdx].ParseUpdateInfo(JointInfo[jointIdx]);
            }

            // Parse Body Update info
            DRigidBodyUpdateInfo[] BodyInfo = UpdateInfo.BodyInfo;
            if (BodyInfo != null)
            {
                if (BodyInfo.Length != BodyList.Count)
                {
                    throw new ArgumentException("Number of Bodies in UpdateInfo and BodyList don't match ");
                }
                Array.Sort(BodyInfo);
                for(int bodyIdx = 0; bodyIdx < BodyInfo.Length; bodyIdx++)
                {
                    BodyList[bodyIdx].ParseUpdateInfo(BodyInfo[bodyIdx]);
                }
            }

            // parse the scale info..
            if (UpdateInfo.RuntimeInfo != null)
            {
                // scale all of geoms here..
                // only used for render arrow character..
                // we should also move the character here..
                // The end position should always at root..
                float[] RunTimeScale = UpdateInfo.RuntimeInfo.Scale;
                foreach(var body in BodyList)
                {
                    foreach(var vgeom in body.vGeomList)
                    {
                        var geom = vgeom.childGeom;
                        geom.transform.localScale = new Vector3(RunTimeScale[0], RunTimeScale[1], RunTimeScale[2]);
                    }
                }
            }
            else
            {
                // reset the scale here..
            }

            // Recompute CoM of character
            this.ComputeCenterOfMass();
        }

        /// <summary>
        /// Parse Character Update Information from Server in json Format
        /// </summary>
        /// <param name="message">Character Update info in json Format</param>
        public void ParseUpdateInfoJson(string message)
        {
            DCharacterUpdateInfo updateInfo = JsonUtility.FromJson<DCharacterUpdateInfo>(message);
            ParseUpdateInfo(updateInfo);
        }
    }
}
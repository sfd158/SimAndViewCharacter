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
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// Load from DCharacterExportInfo
    /// </summary>
    public class LoadCharacterExportInfo
    {
        private DCharacter dCharacter;
        private DCharacterList characterList;

        private List<DJoint> JointList;
        private List<DRigidBody> BodyList;
        private List<DEndJoint> EndJointList;

        private bool HasRealRootJoint = false;
        private DJoint RootJoint;
        private DRigidBody RootBody;

        private List<List<int>> IgnoreBodyIDList;

        public LoadCharacterExportInfo(DCharacterList characterList_)
        {
            characterList = characterList_;
        }

        /// <summary>
        /// Load joint info
        /// </summary>
        /// <param name="JointsInfo"></param>
        private void LoadJoints(DJointExportInfo[] JointsInfo)
        {
            if (!HasRealRootJoint)
            {
                RootJoint = DEmptyJoint.CreateEmptyJoint(null, dCharacter.gameObject);
            }

            JointList = new List<DJoint>(JointsInfo.Length);
            Array.Sort(JointsInfo);
            for(int i=0; i<JointsInfo.Length; i++)
            {
                DJointExportInfo info = JointsInfo[i];
                DJoint joint = DJointCreator.CreateJoint(info, dCharacter);
                JointList.Add(joint);
                if (info.ParentBodyID != -1)
                {
                    if (info.ParentJointID == -1)
                    {
                        joint.transform.parent = RootJoint.transform;
                    }
                    else
                    {
                        joint.transform.parent = JointList[info.ParentJointID].transform;
                    }
                }
                else
                {
                    joint.transform.parent = dCharacter.transform;
                    if (RootJoint != null)
                    {
                        RootJoint = joint;
                    }
                    else
                    {
                        throw new ArgumentException("Have more than 1 root joint");
                    }
                }
            }
        }

        /// <summary>
        /// Load body info
        /// </summary>
        /// <param name="BodiesInfo"></param>
        private void LoadBodies(DBodyExportInfo[] BodiesInfo)
        {
            IgnoreBodyIDList = new List<List<int>>(BodiesInfo.Length);
            BodyList = new List<DRigidBody>(BodiesInfo.Length);
            Array.Sort(BodiesInfo);
            for(int i=0; i<BodiesInfo.Length; i++)
            {
                DBodyExportInfo info = BodiesInfo[i];
                DRigidBody body = DRigidBody.CreateBody(info, dCharacter);

                BodyList.Add(body);
                IgnoreBodyIDList.Add(info.IgnoreBodyID == null ? new List<int>() : new List<int>(info.IgnoreBodyID));

                if (info.ParentBodyID == -1)
                {
                    if (RootJoint != null)
                    {
                        body.transform.parent = RootJoint.transform;
                        RootJoint.InitialPosition = body.InitialPosition;
                    }
                    else
                    {
                        body.transform.parent = dCharacter.transform;
                    }
                    
                    if (RootBody == null)
                    {
                        RootBody = body;
                    }
                    else
                    {
                        throw new ArgumentException("More than 1 RootBody");
                    }
                }
                else
                {
                    if (info.ParentJointID != -1)
                    {
                        body.transform.parent = JointList[info.ParentJointID].transform;
                    }
                    else
                    {
                        throw new ArgumentException("Joint is required");
                    }
                }
            }
            ParseIgnoreBodyID();
        }

        private void ParseIgnoreBodyID()
        {
            for(int i=0; i<BodyList.Count; i++)
            {
                List<int> IgnoreBodyIDs = IgnoreBodyIDList[i];
                List<GameObject> res = new List<GameObject>(IgnoreBodyIDs.Count);
                foreach(int IgnoreID in IgnoreBodyIDs)
                {
                    res.Add(BodyList[IgnoreID].gameObject);
                }
                BodyList[i].IgnoreCollision = res;
            }
        }

        /// <summary>
        /// load end joints info
        /// </summary>
        /// <param name="EndJoints"></param>
        private void LoadEndJoint(DEndJointExportInfo[] EndJoints)
        {
            EndJointList = new List<DEndJoint>(EndJoints.Length);
            for(int i=0; i<EndJoints.Length; i++)
            {
                var info = EndJoints[i];
                EndJointList.Add(DEndJoint.DEndJointCreate(info.Name, Utils.ArrToVector3(info.Position), dCharacter.gameObject, JointList[info.ParentJointID]));
            }
        }

        /// <summary>
        /// Parse character from DCharacterExportInfo
        /// </summary>
        /// <param name="loadInfo"></param>
        public void Parse(DCharacterExportInfo loadInfo, out DCharacter result)
        {
            dCharacter = DCharacter.CreateCharacter(characterList);
            HasRealRootJoint = loadInfo.HasRealRootJoint;

            if (loadInfo.Joints != null && loadInfo.Joints.Length > 0)
            {
                LoadJoints(loadInfo.Joints);
            }

            LoadBodies(loadInfo.Bodies);

            if (loadInfo.EndJoints != null && loadInfo.EndJoints.Length > 0)
            {
                LoadEndJoint(loadInfo.EndJoints);
            }
            if (loadInfo.PDControlParam != null) // load PD Control param
            {
                PDController pdController = PDController.PDControllerCreate(loadInfo.PDControlParam);
                pdController.transform.parent = dCharacter.transform;
            }

            dCharacter.AfterLoad(RootBody, RootJoint, BodyList, JointList, null, null);

            dCharacter.Kinematic = loadInfo.Kinematic;
            dCharacter.SelfCollision = loadInfo.SelfCollision;

            result = dCharacter;

            dCharacter = null; // deconstruct
            characterList = null;

            JointList = null;
            BodyList = null;
            EndJointList = null;

            HasRealRootJoint = false;
            RootJoint = null;
            RootBody = null;

            IgnoreBodyIDList = null;
        }
    }
}

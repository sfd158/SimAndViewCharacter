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
using System.IO;
using System.Xml;
using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// Load character from xml file
    /// </summary>
    public class XMLCharacterLoader
    {
        private readonly string XMLPath; // File Path of Character
        // public string XMLJointPath; // File Path of joint Kp and Kd param info

        private DWorld world;
        private DCharacterList CharacterList;
        private GameObject Character;
        private DCharacter dCharacter;

        private DRigidBody[] BodyList = null;
        private DJoint[] JointList = null;
        private DEndJoint[] EndJointList = null;
        private PDController pdController = null;

        private Dictionary<string, int> BodyNameToID = null;
        private Dictionary<string, int> JointNameToID = null;
        private DRigidBody RootBody;

        private SortedList<int, XmlNode> xmlSortedJoint;
        bool hasRootJoint = false;
        // DJoint RootJoint;

        private static string[] clungNames = { "toe", "heel" , "foot"};

        public XMLCharacterLoader(DWorld dWorld_, string xmlPath_)
        {
            world = dWorld_;
            XMLPath = xmlPath_;
        }

        bool isClungName(string s)
        {
            string lower = s.ToLower();
            foreach(var i in clungNames)
            {
                if (lower.Contains(i))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// parse body information
        /// </summary>
        /// <param name="xmlCharacter"></param>
        private void ParseBody(XmlNode xmlCharacter)
        {
            hasRootJoint = HasRootJoint(xmlCharacter);

            XmlNodeList xmlBodies = xmlCharacter.SelectNodes("Body");
            BodyList = new DRigidBody[xmlBodies.Count];
            BodyNameToID = new Dictionary<string, int>();

            SortedList<int, XmlNode> xmlSortedBodies = new SortedList<int, XmlNode>(); // sort bodies by BodyID
            for (int bodyIdx = 0; bodyIdx < xmlBodies.Count; bodyIdx++)
            {
                XmlNode xmlBody = xmlBodies.Item(bodyIdx);
                XmlNode xmlBodyID = xmlBody.SelectSingleNode("Id");
                int bodyID = xmlBodyID != null ? int.Parse(xmlBodyID.InnerText): bodyIdx;
                xmlSortedBodies.Add(bodyID, xmlBody);
            }

            for (int bodyIdx = 0; bodyIdx < xmlBodies.Count; bodyIdx++)
            {
                XmlNode xmlBody = xmlSortedBodies[bodyIdx];
                XmlNode xmlBodyID = xmlBody.SelectSingleNode("Id");
                int bodyID = xmlBodyID != null ? int.Parse(xmlBodyID.InnerText) : bodyIdx;
                if (bodyID != bodyIdx)
                {
                    Debug.LogError("bodyID != bodyIdx");
                }

                string xmlName = xmlBody.SelectSingleNode("Name").InnerText;
                BodyNameToID.Add(xmlName, bodyID);
                bool isClung = isClungName(xmlName);

                XmlNode xmlBodyFrame = xmlBody.SelectSingleNode("BodyFrame");
                Vector3 BodyCenter = Utils.StringToVec3(xmlBodyFrame.SelectSingleNode("Center").InnerText); 

                XmlNode xmlPhysicsProperty = xmlBody.SelectSingleNode("PhysicsProperty");
                float FrictionCoef = float.Parse(xmlPhysicsProperty.SelectSingleNode("FrictionCoef").InnerText);
                float RestitutionCoef = float.Parse(xmlPhysicsProperty.SelectSingleNode("RestitutionCoef").InnerText);
                float Density = float.Parse(xmlPhysicsProperty.SelectSingleNode("Density").InnerText);

                DRigidBody dRigidBody = DRigidBody.CreateBody(Character, bodyID, xmlName, BodyCenter);
                GameObject bodyObject = dRigidBody.gameObject;
                dRigidBody.Density = Density;

                BodyList[bodyID] = dRigidBody;

                // Parse Inertia..
                XmlNodeList xmlColliGeoms = xmlBody.SelectNodes("CollisionGeometry");
                for (int geomIdx = 0; geomIdx < xmlColliGeoms.Count; geomIdx++)
                {
                    XmlNode xmlGeom = xmlColliGeoms.Item(geomIdx);
                    XmlNode xmlGeomID = xmlGeom.SelectSingleNode("Id");
                    int geomID = xmlGeomID == null ? geomIdx : int.Parse(xmlGeomID.InnerText);

                    string geomType = xmlGeom.SelectSingleNode("Type").InnerText;
                    string geomName = xmlGeom.SelectSingleNode("Name").InnerText;

                    XmlNode xmlGeomFrame = xmlGeom.SelectSingleNode("BodyFrame");
                    Vector3 geomCenter = Utils.StringToVec3(xmlGeomFrame.SelectSingleNode("Center").InnerText);
                    Vector3 XAxis = Vector3.right, YAxis = Vector3.up, ZAxis = Vector3.forward;
                    XmlNode xmlXAxis = xmlGeomFrame.SelectSingleNode("X_Axis");
                    XmlNode xmlYAxis = xmlGeomFrame.SelectSingleNode("Y_Axis");
                    XmlNode xmlZAxis = xmlGeomFrame.SelectSingleNode("Z_Axis");
                    if (xmlXAxis != null)
                    {
                        XAxis = Utils.StringToVec3(xmlXAxis.InnerText);
                        YAxis = Utils.StringToVec3(xmlYAxis.InnerText);
                        ZAxis = Utils.StringToVec3(xmlZAxis.InnerText);
                    }

                    Quaternion geomQuaternion = Utils.OrthogonalAxisToQuaternion(XAxis, YAxis, ZAxis);

                    if (geomType == "Sphere")
                    {
                        float geomRadius = float.Parse(xmlGeom.SelectSingleNode("Radius").InnerText);
                        DBallGeom.CreateGeom(bodyObject, geomID, geomName, geomRadius, FrictionCoef, RestitutionCoef, geomCenter, geomQuaternion, isClung);
                    }
                    else if (geomType == "Box")
                    {
                        Vector3 geomLength = new Vector3(
                            float.Parse(xmlGeom.SelectSingleNode("XLength").InnerText),
                            float.Parse(xmlGeom.SelectSingleNode("YLength").InnerText),
                            float.Parse(xmlGeom.SelectSingleNode("ZLength").InnerText)
                        );
                        DBoxGeom.CreateGeom(bodyObject, geomID, geomName, geomLength, FrictionCoef, RestitutionCoef, geomCenter, geomQuaternion, isClung);
                    }
                    else if (geomType == "CCylinder")
                    {
                        float geomRadius = float.Parse(xmlGeom.SelectSingleNode("Radius").InnerText);
                        float geomLength = float.Parse(xmlGeom.SelectSingleNode("Length").InnerText);
                        DCapsuleGeom.CreateGeom(bodyObject, geomID, geomName, geomRadius, geomLength, world.CapsuleAxis, FrictionCoef, RestitutionCoef, geomCenter, geomQuaternion, isClung);
                    }
                    else
                    {
                        throw new NotImplementedException("Geom Type" + geomType + "Not Supported");
                    }
                }

                XmlNode xmlBodyParent = xmlBody.SelectSingleNode("Parent");
                if (xmlBodyParent != null && !Utils.IsNullString(xmlBodyParent.InnerText))
                {
                    string bodyParent = xmlBodyParent.InnerText;
                    GameObject parentJointObject = new GameObject();
                    bodyObject.transform.parent = parentJointObject.transform;
                    parentJointObject.transform.parent = BodyList[BodyNameToID[bodyParent]].transform.parent;
                }
                else // if (xmlBodyParent == null || Utils.IsNullString(xmlBodyParent.InnerText))
                {
                    if (!hasRootJoint)
                    {
                        DEmptyJoint.CreateEmptyJoint(dRigidBody, Character); // Create Empty Root Joint
                    }
                    else
                    {
                        GameObject parentJointObject = new GameObject();
                        parentJointObject.transform.parent = dCharacter.transform;
                        bodyObject.transform.parent = parentJointObject.transform;
                    }
                    if (RootBody == null)
                    {
                        RootBody = dRigidBody;
                    }
                    else
                    {
                        Debug.LogError("Has Multi Root Body.");
                    }
                }
            }

            ParseIgnorePair(xmlCharacter);
        }

        /// <summary>
        /// parse ignore pair
        /// </summary>
        /// <param name="xmlCharacter"></param>
        private void ParseIgnorePair(XmlNode xmlCharacter)
        {
            XmlNodeList xmlIgnorePairs = xmlCharacter.SelectSingleNode("IgnorePair").SelectNodes("Pair");
            for (int ignoreIdx = 0; ignoreIdx < xmlIgnorePairs.Count; ignoreIdx++)
            {
                XmlNode xmlIgnore = xmlIgnorePairs.Item(ignoreIdx);
                string[] ignoreStr = xmlIgnore.InnerText.Split(' ');
                int[] bodyID = { BodyNameToID[ignoreStr[0]], BodyNameToID[ignoreStr[1]] };
                DRigidBody body0 = BodyList[bodyID[0]], body1 = BodyList[bodyID[1]];
                body0.IgnoreCollision.Add(body1.gameObject);
                body1.IgnoreCollision.Add(body0.gameObject);
            }
        }

        private bool HasRootJoint(XmlNode xmlCharacter)
        {
            bool res = false;
            XmlNodeList xmlJoints = xmlCharacter.SelectNodes("Joint");
            xmlSortedJoint = new SortedList<int, XmlNode>();
            for (int jointIdx = 0; jointIdx < xmlJoints.Count; jointIdx++)
            {
                XmlNode xmlJoint = xmlJoints.Item(jointIdx);
                XmlNode xmlJointID = xmlJoint.SelectSingleNode("Id");

                int jointId = xmlJointID != null ? int.Parse(xmlJointID.InnerText): jointIdx;
                xmlSortedJoint.Add(jointId, xmlJoint);

                string jointName = xmlJoint.SelectSingleNode("Name").InnerText;
                if (jointName == Utils.RootJointName())
                {
                    res = true;
                }
            }

            return res;
        }

        private void ParseJoint()
        {
            JointList = new DJoint[xmlSortedJoint.Count];
            JointNameToID = new Dictionary<string, int>();

            for (int jointIdx = 0; jointIdx < xmlSortedJoint.Count; jointIdx++)
            {
                XmlNode xmlJoint = xmlSortedJoint[jointIdx];
                XmlNode xmlJointID = xmlJoint.SelectSingleNode("Id");
                int jointID = xmlJointID != null ? int.Parse(xmlJointID.InnerText) : jointIdx;
                if (jointID != jointIdx)
                {
                    Debug.LogError("jointId != jointIdx");
                }

                string jointType = xmlJoint.SelectSingleNode("Type").InnerText;
                string jointName = xmlJoint.SelectSingleNode("Name").InnerText;
                JointNameToID.Add(jointName, jointID);

                Vector3 jointPos = Utils.StringToVec3(xmlJoint.SelectSingleNode("Position").InnerText);
                string jointChild = xmlJoint.SelectSingleNode("Child").InnerText;
                int childBodyID = BodyNameToID[jointChild];

                XmlNodeList xmlAngleLimit = xmlJoint.SelectNodes("AngleLimit");

                GameObject childBodyObject = BodyList[childBodyID].gameObject;
                // if (childBodyObject.transform.parent == null && jointName == Utils.RootJointName())

                GameObject jointObject = childBodyObject.transform.parent.gameObject;

                if (jointType == "BallJoint")
                {
                    string eulerOrder = xmlJoint.SelectSingleNode("EulerOrder").InnerText;
                    float[] AngleLoLimit = new float[3], AngleHiLimit = new float[3];
                    for (int angleIdx = 0; angleIdx < 3; angleIdx++)
                    {
                        string[] angleLimitStr = xmlAngleLimit.Item(angleIdx).InnerText.Split(' ');
                        AngleLoLimit[angleIdx] = float.Parse(angleLimitStr[0]);
                        AngleHiLimit[angleIdx] = float.Parse(angleLimitStr[1]);
                    }

                    JointList[jointID] = DBallJoint.AddBallJoint(jointObject, jointID, jointName, 0.0F, jointPos, AngleLoLimit, AngleHiLimit, eulerOrder, Character);
                }
                else if (jointType == "HingeJoint")
                {
                    string hingeAxisStr = xmlJoint.SelectSingleNode("HingeAxis").InnerText;
                    string[] angleLimitStr = xmlAngleLimit.Item(0).InnerText.Split(' ');
                    float AngleLoLimit = float.Parse(angleLimitStr[0]), AngleHiLimit = float.Parse(angleLimitStr[1]);

                    JointList[jointID] = DHingeJoint.AddHingeJoint(jointObject, jointID, jointName, 0.0F, jointPos, AngleLoLimit, AngleHiLimit, hingeAxisStr, Character);
                }
                else
                {
                    throw new NotImplementedException("Joint Type " + jointType + "Not Supported");
                }
            }
        }

        private void ParseEndJoint(XmlNode xmlCharacter)
        {
            XmlNodeList xmlEndJoints = xmlCharacter.SelectNodes("EndPoint");
            EndJointList = new DEndJoint[xmlEndJoints.Count];
            for (int endIdx = 0; endIdx < xmlEndJoints.Count; endIdx++)
            {
                XmlNode xmlEndJoint = xmlEndJoints.Item(endIdx);
                string[] endJointStr = xmlEndJoint.InnerText.Split(' ');
                string endJointName = endJointStr[0];
                Vector3 endJointPos = new Vector3(float.Parse(endJointStr[1]), float.Parse(endJointStr[2]), float.Parse(endJointStr[3]));

                DJoint ParentJoint = BodyList[BodyNameToID[endJointName]].GetParentJoint();

                EndJointList[endIdx] = DEndJoint.DEndJointCreate(endJointName, endJointPos, Character, ParentJoint);
            }
        }

        /* public void LoadJointParamXML()
        {
            if (!File.Exists(XMLJointPath))
            {
                Debug.Log("Please Set XML Joint Param File");
                return;
            }

            XmlDocument xmlDoc = new XmlDocument();
            xmlDoc.Load(XMLJointPath);
            XmlNode xmlJointControlParam = xmlDoc.SelectSingleNode("JointControlParam");
            XmlNodeList xmlJointParams = xmlJointControlParam.SelectNodes("Joint");

            // Sort Joints by JointID
            SortedDictionary<int, XmlNode> SortedJointParam = new SortedDictionary<int, XmlNode>();
            for(int jointIdx=0; jointIdx<xmlJointParams.Count; jointIdx++)
            {
                XmlNode xmlJointParam = xmlJointParams.Item(jointIdx);
                string JointName = xmlJointParam.SelectSingleNode("Name").InnerText;

                if (JointName == Utils.RootJointName() && !dCharacter.HasRealRootJoint)
                {
                    continue;
                }

                int JointID = JointNameToID[JointName];
                SortedJointParam.Add(JointID, xmlJointParam);
            }

            // Add PDControl Component to Character
            GameObject pdControlObject = new GameObject
            {
                name = "PDControl"
            };
            pdControlObject.transform.parent = Character.transform;
            pdController = pdControlObject.AddComponent<PDController>();
            pdController.Kps = new float[SortedJointParam.Count];
            pdController.TorqueLimits = new float[SortedJointParam.Count];

            for(int jointIdx=0; jointIdx<SortedJointParam.Count; jointIdx++)
            {
                XmlNode xmlJointParam = SortedJointParam[jointIdx];
                string jointName = xmlJointParam.SelectSingleNode("Name").InnerText;
                int jointID = JointNameToID[jointName];
                if (jointID != jointIdx)
                {
                    Debug.LogError("JointID != JointIdx");
                }

                DJoint dJoint = JointList[jointID];

                float kp = float.Parse(xmlJointParam.SelectSingleNode("kp").InnerText);
                float kd = float.Parse(xmlJointParam.SelectSingleNode("kd").InnerText);
                float torqueLimit = float.Parse(xmlJointParam.SelectSingleNode("TorqueLimit").InnerText);

                pdController.Kps[jointIdx] = kp;
                dJoint.Damping = kd;
                dJoint.TorqueLimit = torqueLimit;
            }
        } */

        public void LoadFromXML()
        {
            if (!File.Exists(XMLPath))
            {
                return;
            }
            
            CharacterList = world.CharacterList.GetComponent<DCharacterList>();
            dCharacter = DCharacter.CreateCharacter(CharacterList);
            Character = dCharacter.gameObject;

            RootBody = null;

            XmlDocument xmlDoc = new XmlDocument();
            xmlDoc.Load(XMLPath);
            XmlNode xmlCharacter = xmlDoc.SelectSingleNode("Character");
            ParseBody(xmlCharacter);
            ParseJoint();
            ParseEndJoint(xmlCharacter);
            PDController.PDControllerCreate(JointList.Length, Character);

            // LoadJointParamXML();
            dCharacter.AfterLoad(RootBody, RootBody.GetComponent<DRigidBody>().GetParentJoint(),
                new List<DRigidBody>(BodyList), new List<DJoint>(JointList), new List<DEndJoint>(EndJointList), pdController);

            Character = null;
            dCharacter = null;
            BodyList = null;
            JointList = null;
            EndJointList = null;
            pdController = null;
            BodyNameToID = null;
            JointNameToID = null;
            RootBody = null;
        }
    }
}


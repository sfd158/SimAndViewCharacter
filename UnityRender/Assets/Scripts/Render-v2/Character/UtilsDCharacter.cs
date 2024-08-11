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
        /// Joint Count of character
        /// </summary>
        /// <value></value>
        public int JointCount
        {
            get
            {
                return JointList.Count;
            }
        }

        /// <summary>
        /// Body Count of character
        /// </summary>
        /// <value></value>
        public int BodyCount
        {
            get
            {
                return BodyList.Count;
            }
        }

        /// <summary>
        /// Count of end joint in character
        /// </summary>
        /// <value></value>
        public int EndJointCount
        {
            get
            {
                return EndJointList.Count;
            }
        }

        /// <summary>
        /// find all of bodies of character
        /// </summary>
        /// <param name="OffSpringList"></param>
        /// <param name="SortByID"></param>
        /// <returns></returns>
        public static List<DRigidBody> GetBodyList(List<GameObject> OffSpringList, bool SortByID = false)
        {
            List<DRigidBody> bodies = new List<DRigidBody>();
            for (int head = 0; head < OffSpringList.Count; head++)
            {
                GameObject obj = OffSpringList[head];
                if (obj.TryGetComponent<DRigidBody>(out var dRigidBody))
                {
                    bodies.Add(dRigidBody);
                }
            }

            if (SortByID) // Sort Bodies by BodyID
            {
                bodies.Sort();
            }
            return bodies;
        }

        /// <summary>
        /// find all bodies of character
        /// </summary>
        /// <param name="characterObject"></param>
        /// <param name="SortByID"></param>
        /// <returns></returns>
        public static List<DRigidBody> GetBodyList(GameObject characterObject, bool SortByID = false)
        {
            List<GameObject> q = Utils.GetAllOffSpring(characterObject);
            return GetBodyList(q, SortByID);
        }

        public static DRigidBody[] GetBodyArray(GameObject characterObject, bool SortByID = false)
        {
            return GetBodyList(characterObject, SortByID).ToArray();
        }

        public static void SetRenderGeom(GameObject characterObject, bool flag)
        {
            List<GameObject> q = Utils.GetAllOffSpring(characterObject);
            for (int head = 0; head < q.Count; head++)
            {
                GameObject g = q[head];
                MeshRenderer render = g.GetComponent<MeshRenderer>();
                if (render != null)
                {
                    render.enabled = flag;
                }
            }
        }

        /// <summary>
        /// find all Geometry of character
        /// </summary>
        /// <param name="BodyList"></param>
        /// <returns></returns>
        public static List<List<DGeomObject>> GetGeomList(List<DRigidBody> BodyList)
        {
            List<List<DGeomObject>> res = new List<List<DGeomObject>>(BodyList.Count);
            for(int bodyIdx=0; bodyIdx<BodyList.Count; bodyIdx++)
            {
                res.Add(BodyList[bodyIdx].GetGeoms());
            }
            return res;
        }

        /// <summary>
        /// find all joint of character
        /// </summary>
        /// <param name="OffSpringtList"></param>
        /// <param name="CountEmptyJoint"></param>
        /// <param name="SortByID"></param>
        /// <returns></returns>
        public static List<DJoint> GetJoints(List<GameObject> OffSpringtList, bool CountEmptyJoint = false, bool SortByID = false)
        {
            List<GameObject> q = OffSpringtList;
            List<DJoint> joints = new List<DJoint>();
            for (int head = 0; head < q.Count; head++)
            {
                GameObject obj = q[head];
                DJoint djointBase = obj.GetComponent<DJoint>();
                if (djointBase == null) continue;
                if (djointBase is DEndJoint) continue;
                if (djointBase is DEmptyJoint && !CountEmptyJoint) continue;

                joints.Add(djointBase);
            }

            if (SortByID) // Sort Joints by JointID
            {
                joints.Sort();
            }
            return joints;
        }

        /// <summary>
        /// Get all joint of character
        /// </summary>
        /// <param name="characterObject"></param>
        /// <param name="CountEmptyJoint"></param>
        /// <param name="SortByID"></param>
        /// <returns></returns>
        public static List<DJoint> GetJoints(GameObject characterObject, bool CountEmptyJoint = false, bool SortByID = false)
        {
            List<GameObject> q = Utils.GetAllOffSpring(characterObject);
            return GetJoints(q, CountEmptyJoint, SortByID);
        }

        /// <summary>
        /// Get all end joint of character
        /// </summary>
        /// <param name="OffSpringList"></param>
        /// <returns></returns>
        public static List<DEndJoint> GetEndJoints(List<GameObject> OffSpringList)
        {
            List<GameObject> q = OffSpringList;
            List<DEndJoint> endJoints = new List<DEndJoint>();
            for (int head = 0; head < q.Count; head++)
            {
                if (q[head].TryGetComponent<DEndJoint>(out var endJoint))
                {
                    endJoints.Add(endJoint);
                }
            }
            return endJoints;  // need not sort by ID
        }

        public static List<DEndJoint> GetEndJoints(GameObject characterObject)
        {
            List<GameObject> q = Utils.GetAllOffSpring(characterObject);
            return GetEndJoints(q);
        }

        public static Dictionary<DJoint, int> JointListToDict(List<DJoint> JointList)
        {
            Dictionary<DJoint, int> res = new Dictionary<DJoint, int>(JointList.Count);
            for(int jointIdx=0; jointIdx<JointList.Count; jointIdx++)
            {
                res.Add(JointList[jointIdx], jointIdx);
            }
            return res;
        }

        public static Dictionary<GameObject, int> JointListToGameObjDict(List<DJoint> JointList)
        {
            Dictionary<GameObject, int> res = new Dictionary<GameObject, int>(JointList.Count);
            for(int jointIdx=0; jointIdx<JointList.Count; jointIdx++)
            {
                res.Add(JointList[jointIdx].gameObject, jointIdx);
            }
            return res;
        }

        /// <summary>
        /// Get parent joint ID
        /// </summary>
        /// <param name="JointList"></param>
        /// <returns></returns>
        public static List<int> JointListToParentIdx(List<DJoint> JointList)
        {
            List<int> parentIdxList = new List<int>(JointList.Count);
            for(int jointIdx=0; jointIdx<JointList.Count; jointIdx++)
            {
                parentIdxList.Add(JointList[jointIdx].GetParentJointID());
            }
            return parentIdxList;
        }

        public static List<List<int>> JointListToChildIdx(List<DJoint> JointList)
        {
            List<int> parentIdxList = JointListToParentIdx(JointList);
            List<List<int>> childIdxList = new List<List<int>>(JointList.Count);
            for(int i=0; i<JointList.Count; i++)
            {
                childIdxList.Add(new List<int>());
            }

            for(int jointIdx=0; jointIdx < JointList.Count; jointIdx++)
            {
                int parentIdx = parentIdxList[jointIdx];
                if (parentIdx == -1)
                {
                    continue;
                }
                childIdxList[parentIdx].Add(jointIdx);
            }
            return childIdxList;
        }

        /// <summary>
        /// check all of joints in character
        /// </summary>
        /// <param name="JointList"></param>
        static void CheckJointList(List<DJoint> JointList)
        {
            // Joint's Parent must be Character or Joint
            foreach (DJoint dJoint in JointList)
            {
                GameObject parentObject = dJoint.transform.parent.gameObject;
                if (!parentObject.TryGetComponent<DJoint>(out _) && !parentObject.TryGetComponent<DCharacter>(out _))
                {
                    throw new ArgumentNullException("Joint's Parent must have DCharacter or DJoint Component.");
                }
            }
        }

        /// <summary>
        /// check all of bodies in character
        /// </summary>
        /// <param name="BodyList"></param>
        static void CheckBodyList(List<DRigidBody> BodyList)
        {
            // Body's Parent must be Character or Joint
            foreach (DRigidBody drigidBody in BodyList)
            {
                GameObject parentObject = drigidBody.transform.parent.gameObject;
                if (!parentObject.TryGetComponent<DJoint>(out _) && !parentObject.TryGetComponent<DCharacter>(out _))
                {
                    throw new ArgumentNullException("Body's Parent must have DCharacter or DJoint Component.");
                }
            }
        }

        /// <summary>
        /// Check each end joint
        /// </summary>
        /// <param name="EndJointList"></param>
        static void CheckEndJointList(List<DEndJoint> EndJointList)
        {
            // end Joint's parent must be Joint, and end Joint has no child.
            foreach (DEndJoint endJoint in EndJointList)
            {
                GameObject parentObject = endJoint.transform.parent.gameObject;
                if (parentObject.GetComponent<DJoint>() == null)
                {
                    Debug.LogWarning("End Joint's parent must have DJoint Component.");
                }
                if (endJoint.transform.childCount > 0)
                {
                    Debug.LogWarning("End Joint has no child");
                }
            }
        }

        /// <summary>
        /// Recompute Character's Joint ID, Body ID, Geometry ID. 
        /// Rotation of Euler Axis Object and Geometry is unchanged. 
        /// Set rotation of all joints and bodies to 0.
        /// ReCompute PD param if exists.
        ///
        /// TODO: when create with hierarchy, position may be wrong.
        /// TODO: a method is: create each geometry seperately, and add joint GameObject as parent.
        /// TODO: Then modify joint position. Finally, create character component.
        /// </summary>
        public override void ReCompute()
        {
            CalcAttrs(out var q);
            q = (q == null) ? Utils.GetAllOffSpring(gameObject): q; // q = (q == null) ? Utils... : q;

            CheckJointList(JointList);
            CheckBodyList(BodyList);
            CheckEndJointList(EndJointList);

            for (int jointIdx = 0; jointIdx < JointList.Count; jointIdx++)
            {
                DJoint dJoint = JointList[jointIdx];
                dJoint.IDNum = jointIdx; // Recompute Joint ID
                dJoint.character = gameObject;
                dJoint.dCharacter = this;
            }

            // Recompute Body ID
            for (int bodyIdx = 0; bodyIdx < BodyList.Count; bodyIdx++)
            {
                DRigidBody dRigidBody = BodyList[bodyIdx];
                GameObject bodyObject = dRigidBody.gameObject;
                dRigidBody.IDNum = bodyIdx;
                dRigidBody.character = gameObject;
                dRigidBody.CalcAttrs();

                // Recompute GeomID
                for (int geomIdx = 0; geomIdx < bodyObject.transform.childCount; geomIdx++)
                {
                    GameObject geomObject = bodyObject.transform.GetChild(geomIdx).gameObject;
                    DGeomObject dGeom = geomObject.GetComponent<DGeomObject>();
                    if (dGeom == null)
                    {
                        throw new ArgumentException("DGeomObject Component is Required. ");
                    }
                    dGeom.IDNum = geomIdx;
                }
            }

            SaveToInitialState();

            foreach(var joint in JointList)
            {
                joint.ReCompute();
            }
            for(int bodyIdx=0; bodyIdx<BodyList.Count; bodyIdx++)
            {
                DRigidBody dRigidBody = BodyList[bodyIdx];
                dRigidBody.InitialQuaternion = Quaternion.identity;
                List<DGeomObject> GeomList = dRigidBody.GetGeoms();
                for (int geomIdx = 0; geomIdx < GeomList.Count; geomIdx++)
                {
                    DGeomObject geom = GeomList[geomIdx];
                    geom.transform.localRotation = geom.CalcLocalQuaternion();
                    geom.InitialQuaternion = geom.transform.rotation;
                }
            }
            foreach (GameObject gameObj in q)
            {
                // Here we should ignore the mesh component
                // That is, if there is no joint/body/geom component, we should not remove rotation here..
                if (gameObj.TryGetComponent<DBaseObject>(out var _))
                {
                    gameObj.transform.rotation = Quaternion.identity;
                }
            }

            SetInitialState(true);
            // Recompute Kps and Torque Limit. Original Kps is unknown, so it should reset manually.
            PDController pdController = GetPDController();
            if (pdController != null)
            {
                pdController.ResetParams(JointList.Count);
            }
        }

        /// <summary>
        /// Set damping of each joint
        /// </summary>
        public void SetDefaultDamping()
        {
            CalcAttrs();
            if (JointList != null)
            {
                foreach (var joint in JointList)
                {
                    joint.Damping = DefaultJointDamping;
                }
            }
        }

        /// <summary>
        /// Set contact mu of each geometry
        /// </summary>
        public void SetDefaultMu()
        {
            CalcAttrs();
            float value = DefaultGeomMu >= 0 ? DefaultGeomMu : 0;
            if (BodyList != null)
            {
                foreach(var body in BodyList)
                {
                    foreach(var vgeom in body.vGeomList)
                    {
                        vgeom.SetMu(value);
                    }
                }
            }
        }

        /// <summary>
        /// Compute initial mass of bodies
        /// </summary>
        public void ComputeInitialMass()
        {
            // Debug.Log("In ComputeInitialMass, isPlaying = " + Application.isPlaying);
            // only call this method in running
            TotalMass = 0.0F;
            foreach(var body in this.BodyList)
            {
                // initial body mass
                if (body.MassMode == DRigidBodyMassMode.Density)
                {
                    DMass bodyMass = body.ComputeBodyMassByDensity();
                    body.Mass = Convert.ToSingle(bodyMass.mass);
                }
                TotalMass += body.Mass;
            }
        }

        public Vector3 ComputeCenterOfMass()
        {
            Vector3 result = Vector3.zero;
            foreach(var body in this.BodyList)
            {
                result += body.Mass * body.transform.position;
            }
            result /= this.TotalMass;
            CenterOfMass = result;
            return result;
        }

        /// <summary>
        /// Add child component to joint, for visualize...
        /// </summary>
        public void AddJointView(float VisualizeRadius = 0.1F)
        {
            this.CalcAttrs();
            for(int jointIdx=0; jointIdx<JointList.Count; jointIdx++)
            {
                DJoint joint = JointList[jointIdx];
                if (joint.HasVisualizeComponent())
                {
                    continue;
                }
                GameObject sphereVis = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphereVis.AddComponent<DJointVisualizeComponent>();
                sphereVis.transform.position = joint.transform.position;
                sphereVis.transform.localScale = VisualizeRadius * Vector3.one;
                sphereVis.transform.parent = joint.transform;

                // Add a flag to disable the render color..
                if (DCommonConfig.SupportGameObjectColor)
                {
                    sphereVis.GetComponent<Renderer>().material.color = Color.red;
                }
            }
        }

        public void RemoveJointView()
        {
            var VisComponents = GetComponentsInChildren<DJointVisualizeComponent>();
            for(int i=0; i<VisComponents.Length; i++)
            {
                DestroyImmediate(VisComponents[i].gameObject);
            }
        }

        /// <summary>
        /// Recompute character by ExportInfo..
        /// </summary>
        /// <param name="exportInfo"></param>
        public void UpdateShapeParameter(DCharacterExportInfo exportInfo)
        {
            //TODO: Test
            if (exportInfo == null)
            {
                return;
            }
            if (exportInfo.BodyCount > 0)
            {
                // resort body list by body id
                Array.Sort(exportInfo.Bodies);
                for(int body_index = 0; body_index < exportInfo.Bodies.Length; body_index++)
                {
                    DRigidBody dbody = BodyList[body_index];
                    DBodyExportInfo bodyExportInfo = exportInfo.Bodies[body_index];
                    if (dbody.IDNum != body_index || dbody.name != bodyExportInfo.Name)
                    {
                        throw new ArgumentException();
                    }
                    Transform dBodyTrans = dbody.transform;
                    dBodyTrans.position = Utils.ArrToVector3(bodyExportInfo.Position);
                    dBodyTrans.rotation = Utils.ArrToQuaternion(bodyExportInfo.Quaternion);

                    // reset geometries..
                    if (bodyExportInfo.NumGeoms > 0)
                    {
                        Array.Sort(bodyExportInfo.Geoms);
                        for (int geom_index = 0; geom_index < bodyExportInfo.Geoms.Length; geom_index++)
                        {
                            DGeomExportInfo geomExportInfo = bodyExportInfo.Geoms[geom_index];
                            VirtualGeom vgeom = dbody.vGeomList[geom_index];
                            if (vgeom.IDNum != geom_index || vgeom.name != geomExportInfo.Name)
                            {
                                throw new ArgumentException();
                            }
                            Vector3 newGeomPos = Utils.ArrToVector3(geomExportInfo.Position);
                            Vector3 newGeomRot = Utils.ArrToVector3(geomExportInfo.Quaternion);
                            // set geom position
                            Transform vGeomTrans = vgeom.transform;
                            // TODO: set geom rotation..
                            DGeomObject geom = vgeom.childGeom;
                            Transform dGeomTrans = geom.transform;
                            dGeomTrans.position = newGeomPos;
                        }
                    }
                }
            }
            if (exportInfo.JointCount > 0)
            {
                Array.Sort(exportInfo.Joints);
                for(int joint_index = 0; joint_index < exportInfo.Joints.Length; joint_index++)
                {
                    DJointExportInfo dJointExport = exportInfo.Joints[joint_index];
                    DJoint dJoint = JointList[joint_index];
                    if (dJoint.IDNum != joint_index || dJoint.name != dJointExport.Name)
                    {
                        throw new ArgumentException();
                    }
                }
            }
            if (exportInfo.EndJointCount > 0)
            {
                Array.Sort(exportInfo.EndJoints);
                for(int end_index = 0; end_index < exportInfo.EndJoints.Length; end_index++)
                {
                    DEndJoint dEndJoint = EndJointList[end_index];
                    DEndJointExportInfo endExportInfo = exportInfo.EndJoints[end_index];
                    if (dEndJoint.name != endExportInfo.Name)
                    {
                        throw new ArgumentException();
                    }
                    dEndJoint.InitialPosition = Utils.ArrToVector3(endExportInfo.Position);
                }
            }
        }
    }
}

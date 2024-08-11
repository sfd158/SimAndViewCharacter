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
using UnityEngine;
using UnityEngine.UI;

namespace RenderV2
{
    /// <summary>
    /// World
    /// </summary>
    [DisallowMultipleComponent]
    public partial class DWorld : MonoBehaviour, ICalcAttrs, IReCompute, IExportInfo<DWorldExportInfo>, IParseUpdate<DWorldUpdateInfo>, IRemoveInfo<DWorldRemoveInfo>
    {
        [Tooltip("World Update Mode")]
        public DWorldUpdateMode dWorldUpdateMode = DWorldUpdateMode.MaximalCoordinate;

        [HideInInspector]
        public static readonly AxisType DefaultCapsuleAxis = AxisType.Z; // in ODE v0.12, capsule axis is along z axis

        [Tooltip("Environment GameObject")]
        public GameObject Environment;

        [Tooltip("CharacterList GameObject")]
        public GameObject CharacterList;

        [Tooltip("ExtJointList GameObject")]
        public GameObject ExtJointListObject;

        [Tooltip("Contact Joint List Object")]
        public GameObject ContactListObject;

        [Tooltip("External Force List GameObject")]
        public GameObject ExtForceObject;

        [Tooltip("Information Text")]
        public GameObject InfoTextObject;

        [Tooltip("Arrow")]
        public GameObject ArrowsObject;

        [Tooltip("Capsule Axis")]
        public AxisType CapsuleAxis = DefaultCapsuleAxis;

        [Tooltip("Gravity")]
        public Vector3 Gravity = new Vector3(0, -9.8F, 0);

        [Tooltip("simulation step executed at each frame. If set to 0, (SimulateFPS // RenderFPS) times of simulation step will be executed.")]
        public int StepCount = 0;

        [Tooltip("Render FPS in Unity")]
        public int RenderFPS = 60;

        [Tooltip("Simulation FPS used in physics engine. In ODE, dWorldDampedStep(1.0 /SimulateFPS) will be called")]
        public int SimulateFPS = 120;

        [Tooltip("Use Hinge Joint")]
        public bool UseHinge = true;

        [Tooltip("Use Angle Limit")]
        public bool UseAngleLimit = true;

        [Tooltip("Self Collision Detection")]
        public bool SelfCollision = true;

        // maybe memory leak occurs when setting color for GameObject
        // disable setting color in running simply..
        // TODO: check will memory leak occur in pickle part..
        // [Tooltip("Enable character color")]
        // public bool SupportColor = false; // Simply disable color here..

        [HideInInspector]
        public DCharacterList dCharacterList;

        [HideInInspector]
        public DEnvironment dEnvironment;

        [HideInInspector]
        public DExtJointList extJointList;

        protected DContactList contactList; // Only recieve from python server. Unity will not generate by itself.

        //[HideInInspector]
        public DExtForceBase extForceBase;

        protected Text InfoText;

        protected DArrowList dArrows;

        private void Awake()
        {
            Application.targetFrameRate = RenderFPS; // set unity FPS
            Application.runInBackground = true;
            ReCompute();
        }

        public void CalcAttrs()
        {
            dCharacterList = CharacterList.GetComponent<DCharacterList>();
            dEnvironment = Environment.GetComponent<DEnvironment>();

            if (ExtJointListObject != null)
            {
                extJointList = ExtJointListObject.GetComponent<DExtJointList>();
            }

            if (ContactListObject != null)
            {
                contactList = ContactListObject.GetComponent<DContactList>();
            }

            if (ExtForceObject != null)
            {
                extForceBase = ExtForceObject.GetComponent<DExtForceBase>();
            }

            if (InfoTextObject != null)
            {
                InfoText = InfoTextObject.GetComponent<Text>();
            }

            if (ArrowsObject != null)
            {
                dArrows = ArrowsObject.GetComponent<DArrowList>();
            }
            CalcChildAttrs();
        }

        private void CalcChildAttrs()
        {
            dCharacterList.CalcAttrs();
            dEnvironment.CalcAttrs();
            if (extJointList != null)
            {
                extJointList.CalcAttrs();
            }
            if (contactList != null)
            {
                contactList.CalcAttrs();
            }
            if (dArrows != null)
            {
                dArrows.CalcAttrs();
            }
        }

        /// <summary>
        /// Remove All GameObjects in CharacterList
        /// </summary>
        public void ClearCharacterList()
        {
            List<GameObject> buf = Utils.GetAllOffSpring(CharacterList);
            for (int i = buf.Count - 1; i > 0; i--) // note: buf[0] is Character List GameObject.
            {
                DestroyImmediate(buf[i]);
            }
        }

        /// <summary>
        /// Get Remove Info in Unity
        /// from DCharacterList, DEnvironment, DExtJointList
        /// </summary>
        /// <returns></returns>
        public DWorldRemoveInfo RemoveInfo()
        {
            DWorldRemoveInfo info = new DWorldRemoveInfo();

            info.CharacterList = dCharacterList.RemoveInfo();
            info.Environment = dEnvironment.RemoveInfo();
            if (extJointList != null)
            {
                info.ExtJointList = extJointList.RemoveInfo(); // TODO
            }

            if (info.CharacterList == null && info.Environment == null && info.ExtJointList == null)
            {
                return null; // Reduce communication overhead
            }

            return info;
        }

        /// <summary>
        /// Get Remove Info in Unity
        /// from DCharacterList, DEnvironment, DExtJointList
        /// </summary>
        /// <returns>Remove Info in Unity</returns>
        public DWorldRemoveInfo RemoveInfoWithPostProcess()
        {
            DWorldRemoveInfo info = RemoveInfo();

            if (extJointList != null)
            {
                extJointList.MergeRemoveBuffer(); // remove GameObjects in Unity
            }

            dEnvironment.MergeRemoveBuffer();
            dCharacterList.MergeRemoveBuffer();

            return info;
        }

        public DWorldControlSignal GetWorldControlSignal()
        {
            DWorldControlSignal info = new DWorldControlSignal();
            info.CharacterSignals = dCharacterList.GatherCharacterControlSignal();
            return info;
        }

        /// <summary>
        /// information send to server at each frame
        /// </summary>
        /// <returns></returns>
        public DUpdateSendInfo UpdateSendInfoWithPostProcess()
        {
            DUpdateSendInfo updateSendInfo = new DUpdateSendInfo();
            updateSendInfo.ExportInfo = ExportInfoWithPostProcess();
            updateSendInfo.RemoveInfo = RemoveInfoWithPostProcess();
            // TODO: Get other Update Send Infomation
            updateSendInfo.WorldControlSignal = GetWorldControlSignal();

            return updateSendInfo;
        }

        /// <summary>
        /// Parse Remove Info from Server
        /// </summary>
        /// <param name="RemoveInfo">Remove Info From Server</param>
        public void ParseRemoveInfo(DWorldRemoveInfo RemoveInfo) // TODO
        {
            if (RemoveInfo.ExtJointList != null) // Note: External Joint List remove info shoule be parsed in front of CharacterList
            {
                
            }
            if (RemoveInfo.Environment != null)
            {
                
            }
            if (RemoveInfo.CharacterList != null)
            {
                dCharacterList.ParseRemoveInfo(RemoveInfo.CharacterList);
            }
        }

        /// <summary>
        /// Parse recieved information from server
        /// </summary>
        /// <param name="RecieveInfo"></param>
        public void ParseUpdateRecieveInfo(UpdateRecieveInfo RecieveInfo)
        {
            if (RecieveInfo.RemoveInfo != null) // Remove Info From Server
            {
                ParseRemoveInfo(RecieveInfo.RemoveInfo);
            }
            ParseUpdateInfo(RecieveInfo.WorldUpdateInfo);

            if (RecieveInfo.ExportInfo != null) // Export Info From Server
            {
                // TODO: Parse ExportInfo from Server
            }
        }

        public void ParseHelperInfo(HelperUpdateInfo HelperInfo)
        {
            if (HelperInfo == null)
            {
                return;
            }
            if (HelperInfo.Message == null)
            {
                return;
            }
            if (InfoText != null)
            {
                InfoText.text = HelperInfo.Message;
            }
        }

        /// <summary>
        /// Parse Update information recieved from server
        /// </summary>
        /// <param name="UpdateInfo"></param>
        public void ParseUpdateInfo(DWorldUpdateInfo UpdateInfo)
        {
            if (UpdateInfo == null)
            {
                return;
            }
            dEnvironment.ParseUpdateInfo(UpdateInfo.Environment); // parse environment update info
            dCharacterList.ParseUpdateInfo(UpdateInfo.CharacterList); // parse character list update info

            if (extJointList != null)
            {
                extJointList.ParseUpdateInfo(UpdateInfo.ExtJointList); // parse ext Joint Update Info
            }

            if (contactList != null)
            {
                contactList.ParseUpdateInfo(UpdateInfo.ContactList); // parse contact joint update info
            }

            ParseHelperInfo(UpdateInfo.HelperInfo);
            if (dArrows != null)
            {
                dArrows.ParseUpdateInfo(UpdateInfo.ArrowList);
            }
        }

        /// <summary>
        /// Parse update information recieved from server in json format
        /// </summary>
        /// <param name="message"></param>
        public void ParseUpdateInfoJson(string message)
        {
            DWorldUpdateInfo UpdateInfo = JsonUtility.FromJson<DWorldUpdateInfo>(message);
            ParseUpdateInfo(UpdateInfo);
        }

        /// <summary>
        /// Operation after Step function
        /// </summary>
        public void PostWorldStep()
        {
            if (extForceBase != null)
            {
                //Debug.Log("PostWorldStep func");
                extForceBase.PostWorldStep();
            }
        }

        public void ReCompute()
        {
            // recompute characterlist, environment, extjointlist.
            CalcAttrs();
            dCharacterList.ReCompute(); // recompute dCharacterList
            dEnvironment.ReCompute(); // recompute dEnvironment

            if (extJointList != null)
            {
                extJointList.ReCompute(); // recompute external joint
            }
        }
    }
}

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
    [DisallowMultipleComponent]
    public class DExtDragRoot: DExtForceBase
    {
        [Tooltip("The magnitute of push force")]
        public float PushForce = 10F;

        [Tooltip("Camera Update Mode")]
        public MonoUpdateMode UpdateMode = MonoUpdateMode.FixedUpdate;
        public List<DExtForceExportInfo> forces;

        protected DCameraController controller;
        private void Start()
        {
            forces = new List<DExtForceExportInfo>();
            controller = GetComponent<DCameraController>();
        }

        /// <summary>
        /// Get all of external force on bodies
        /// </summary>
        /// <returns></returns>
        public override DExtForceListExportInfo GetExtForceList()
        {
            DExtForceListExportInfo res = null;
            if (forces.Count > 0)
            {
                res = new DExtForceListExportInfo(forces);
                Debug.Log("Get Ext Force List" + res.Count);
            }
            return res;
        }

        /// <summary>
        /// clear operation after each update function
        /// </summary>
        public override void PostWorldStep()
        {
            // Debug.Log("Post. Count = " + forces.Count);
            forces.Clear();
        }


        private Vector3 dragOrigin;
        public DRigidBody target;
        void UpdateFunc()
        {
            if (Input.GetMouseButtonDown(0))
            {
                dragOrigin = Input.mousePosition;
                return;
            }
            if (!Input.GetMouseButton(0)) return;

            Vector3 pos = Camera.main.ScreenToViewportPoint(Input.mousePosition - dragOrigin);
            Vector3 force = 5000*pos;
            DExtForceExportInfo info = new DExtForceExportInfo();
            info.CharacterID = target.dCharacter.IDNum;
            info.BodyID = target.IDNum;
            info.Position = Utils.Vector3ToArr(pos);
            info.Force = Utils.Vector3ToArr(force);
            forces.Add(info);
        }

        private void Update()
        {
            if (UpdateMode == MonoUpdateMode.Update)
            {
                UpdateFunc();
            }
        }

        private void FixedUpdate()
        {
            if (UpdateMode == MonoUpdateMode.FixedUpdate)
            {
                UpdateFunc();
            }
        }

        private void LateUpdate()
        {
            if (UpdateMode == MonoUpdateMode.LateUpdate)
            {
                UpdateFunc();
            }
        }
    }
}

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
    public class ForceArrow : DBaseObject, IParseUpdate<ArrowUpdateInfo>
    {
        protected GameObject capsule;
        protected GameObject cone;

        protected float cone_length;
        protected float cone_radius;

        public override void CalcAttrs()
        {
            capsule = GetComponentInChildren<CapsuleCollider>().gameObject; // this is ugly..
            if (capsule == transform.GetChild(0))
            {
                cone = transform.GetChild(1).gameObject;
            }
            else
            {
                cone = transform.GetChild(0).gameObject;
            }

            // 1. get the length of cone..
            MeshFilter filter = cone.GetComponent<MeshFilter>();
            if (filter != null && filter.mesh != null)
            {
                Bounds bound = filter.mesh.bounds;
                Vector3 size = bound.size;
                // find which dimension is same..
                if (Mathf.Abs(size.x - size.y) < 1e-3f)
                {
                    cone_length = size.z;
                    cone_radius = 0.25F * (size.x + size.y);
                }
                else if (Mathf.Abs(size.x - size.z) < 1e-3f)
                {
                    cone_length = size.y;
                    cone_radius = 0.25F * (size.x + size.z);
                }
                else if (Mathf.Abs(size.y - size.z) < 1e-3f)
                {
                    cone_length = size.x;
                    cone_radius = 0.25F * (size.y + size.z);
                }
                else
                {
                    cone_length = size.z;
                    cone_radius = 0.25F * (size.x + size.y);
                }
            }

            // 2. get the length of capsule

        }

        public void GetInitStartEndPos(out Vector3 StartPos, out Vector3 EndPos)
        {
            // This function is only called one time at the running
            Vector3 cap_pos = capsule.transform.position;
            Vector3 cone_pos = cone.transform.position;
            Vector3 dir = (cap_pos - cone_pos).normalized;
            StartPos = cap_pos + capsule.transform.localScale.y * dir;
            EndPos = cone_pos;
        }

        public ArrowUpdateInfo ExportInfo()
        {
            ArrowUpdateInfo result = new ArrowUpdateInfo();
            GetInitStartEndPos(out var StartPos, out var EndPos);

            result.StartPos = Utils.Vector3ToArr(StartPos);
            result.EndPos = Utils.Vector3ToArr(EndPos);
            result.InUse = true;
            result.IDNum = IDNum;
            return result;
        }

        public void ParseUpdateInfo(ArrowUpdateInfo info)
        {
            if (info is null)
            {
                return;
            }
            gameObject.SetActive(info.InUse);
            if (!info.InUse)
            {
                return; // not update here..
            }

            Vector3 StartPos = info.StartPosVec3();
            Vector3 EndPos = info.EndPosVec3();
            // Debug.Log(StartPos);
            // Debug.Log(EndPos);
            Vector3 dir = EndPos - StartPos;
            float length = (EndPos - StartPos).magnitude;
            dir /= length;

            // change the orientation of gameobject
            transform.rotation = Quaternion.FromToRotation(Vector3.forward, dir);

            // change the position of cone
            cone.transform.position = EndPos;
            // change the position, scale of cylinder
            Vector3 scale = capsule.transform.localScale;
            scale.y = 0.5F * length;
            capsule.transform.localScale = scale;
            capsule.transform.position = 0.5F * (StartPos + EndPos);
        }

        public override void ReCompute()
        {
            throw new System.NotImplementedException();
        }
    }
}

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
using UnityEngine;

namespace RenderV1
{
    [System.Serializable]
    public class CreateBaseInfoType
    {
        public ulong[] CreateID;
        public int[] CreateType;
        public float[] CreateScale;
        public float[] CreatePos;
        public float[] CreateQuat;
        public float[] CreateChildQuat;
        public string[] CreateName;
        public float[] CreateColor;

        public void MyReshapeBase(int n)
        {
            MyReshapeNoColorBase(n);
            CreateColor = new float[3 * n];
        }

        public void MyReshapeNoColorBase(int n)
        {
            CreateID = new ulong[n];
            CreateType = new int[n];
            CreateScale = new float[3 * n];
            CreatePos = new float[3 * n];
            CreateQuat = new float[4 * n];
            CreateChildQuat = new float[4 * n];
            CreateName = new string[n];
            CreateColor = new float[0];
        }

        public void SetScale(int i, Vector3 scale)
        {
            CreateScale[3 * i] = scale[0];
            CreateScale[3 * i + 1] = scale[1];
            CreateScale[3 * i + 2] = scale[2];
        }

        public void SetPosition(int i, Vector3 pos)
        {
            CreatePos[3 * i] = pos[0];
            CreatePos[3 * i + 1] = pos[1];
            CreatePos[3 * i + 2] = pos[2];
        }

        public void SetQuaternion(int i, Quaternion q)
        {
            CreateQuat[4 * i] = q.x;
            CreateQuat[4 * i + 1] = q.y;
            CreateQuat[4 * i + 2] = q.z;
            CreateQuat[4 * i + 3] = q.w;
        }

        public void SetChildQuaternion(int i, Quaternion q)
        {
            CreateChildQuat[4 * i] = q.x;
            CreateChildQuat[4 * i + 1] = q.y;
            CreateChildQuat[4 * i + 2] = q.z;
            CreateChildQuat[4 * i + 3] = q.w;
        }
    }

    [System.Serializable]
    class BodyAndJointCreateInfo : CreateBaseInfoType
    {
        public string[] JointName;
        public float[] JointPos;
        public string[] JointType;
        public int[] Parent;

        public void MyReshapeNoColor(int n)
        {
            MyReshapeNoColorBase(n);
            JointName = new string[n];
            JointPos = new float[3 * n];
            JointType = new string[n];
            Parent = new int[n];
        }

        public void SetJointPos(int i, Vector3 pos)
        {
            JointPos[3 * i] = pos[0];
            JointPos[3 * i + 1] = pos[1];
            JointPos[3 * i + 2] = pos[2];
        }
    }
}

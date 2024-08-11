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

using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;
using UnityEngine;
using static RenderV2.ODE.CommonFunc;
using System.Text;


namespace RenderV2.ODE
{
    public class ODECharacterHandle
    {
        List<ODEBody> bodyList;
        List<ODEJoint> jointList;
        DCharacter character;

        public ODECharacterHandle()
        {

        }

        public void Destroy()
        {
            foreach (var joint in jointList) joint.Destroy();
            foreach (var body in bodyList)
            {
                body.Destroy();
                body.DestroyAllGeoms();
                body.Destroy();
            }
            bodyList.Clear();
            jointList.Clear();
        }

        public List<UIntPtr> GetBodyPtrList()
        {
            var res = new List<UIntPtr>(bodyList.Count);
            foreach (var body in bodyList)
                res.Add(body.BodyPtr());
            return res;
        }

        public List<UIntPtr> GetJointPtrList()
        {
            var res = new List<UIntPtr>(jointList.Count);
            foreach (var joint in jointList)
                res.Add(joint.GetPtr());
            return res;
        }

        public List<ODEBody> BodyList => bodyList;
        public List<ODEJoint> JointList => jointList;

        public List<ODEBody> GetBodyList() => bodyList;
        public List<ODEJoint> GetJointList() => jointList;

        public Vector3 CenterOfMass()
        {
            double total_mass = 0;
            double[] res = new double[3], pos = new double[3];
            foreach (var body in bodyList)
            {
                double mass = body.MassValue;
                total_mass += mass;
                body.GetBodyPositionArray(pos);
                for (int i = 0; i < 3; i++) res[i] += mass * pos[i];
            }
            return Vec3FromDoubleArr(res) / (float)total_mass;
        }

        /// <summary>
        /// velocity of center of mass
        /// </summary>
        /// <returns></returns>
        public Vector3 VeloCoM()
        {
            double total_mass = 0;
            double[] res = new double[3], vel = new double[3];
            foreach (var body in bodyList)
            {
                double mass = body.MassValue;
                total_mass += mass;
                body.GetLinearVelArr(vel);
                for (int i = 0; i < 3; i++) res[i] += mass * vel[i];
            }
            return Vec3FromDoubleArr(res) / (float)total_mass;
        }

        public void UpdateFromODE()
        {
            // update by full coordinate.
            for(int j = 0; j < JointList.Count; j++)
            {
                character.JointList[j].transform.SetPositionAndRotation(
                    jointList[j].GetAnchor(),
                    jointList[j].GetGlobalQuat()
                );
            }
            for(int b = 0; b < BodyList.Count; b++)
            {
                character.BodyList[b].transform.SetPositionAndRotation(
                    BodyList[b].GetBodyPositionVec3(),
                    BodyList[b].GetBodyQuaternion()
                );
            }
        }

        public string DebugString
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                for(int b = 0; b < bodyList.Count; b++)
                {
                    
                }
                return sb.ToString();
            }
        }
    }

    public class ODEWorldHandle
    {
        public List<ODECharacterHandle> characterList;
        public ODEWorld world;
        public SimpleSpace space;
        public JointGroup jointGroup;
        public GeomPlane plane;

        private CallBackData callBackData;
        private IntPtr callBackPtr;

        public ODEWorldHandle(float dt = 0.01f)
        {
            world = new ODEWorld(dt);
            space = new SimpleSpace();
            jointGroup = new JointGroup();
            callBackData = new CallBackData()
            {
                world = world.WorldPtr(),
                jointgroup = jointGroup.GetPtr()
            };
            callBackPtr = Marshal.AllocHGlobal(UIntPtr.Size * 2);
            Marshal.StructureToPtr(callBackData, callBackPtr, false);
        }

        ~ODEWorldHandle()
        {
            if (world.WorldPtr() != UIntPtr.Zero)
                Destroy();
        }

        public void Destroy()
        {
            Marshal.FreeHGlobal(callBackPtr);
            foreach (var ch in characterList) ch.Destroy();
            jointGroup.Destroy();
            plane?.Destroy();
            space.Destroy();
            world.Destroy();
        }

        public void Step()
        {
            space.Collide(callBackPtr);
            world.Step();
            space.ResortGeoms();
        }

        public void DampedStep()
        {
            space.Collide(callBackPtr);
            world.DampedStep();
            space.ResortGeoms();
            UpdateFromODE();
        }

        public void UpdateFromODE()
        {
            foreach(var ch in characterList) 
                ch.UpdateFromODE();
        }
    }

}
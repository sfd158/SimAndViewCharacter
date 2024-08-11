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
    public abstract class DBaseObject: MonoBehaviour, ICalcAttrs, IComparable<DBaseObject>, IReCompute
    {
        public int IDNum = 0;
        public Vector3 InitialPosition = Vector3.zero;
        public Quaternion InitialQuaternion = Quaternion.identity;

        public void SetInitialState(bool setQuaternion = false)
        {
            if (setQuaternion)
            {
                transform.SetPositionAndRotation(InitialPosition, InitialQuaternion);
            }
            else
            {
                transform.position = InitialPosition;
            }
        }

        public void SetInitialPosAndQuat(Vector3 Pos, Quaternion Quat)
        {
            InitialPosition = Pos;
            InitialQuaternion = Quat;
        }

        public void SaveToInitialState()
        {
            InitialPosition = transform.position;
            InitialQuaternion = transform.rotation;
        }

        public abstract void CalcAttrs();

        public int CompareTo(DBaseObject other) => IDNum.CompareTo(other.IDNum);

        public abstract void ReCompute();

        /// <summary>
        /// Destroy this GameObject and All Off-Springs
        /// </summary>
        public void DAllDestroy()
        {
            List<GameObject> q = Utils.GetAllOffSpring(gameObject);
            for (int i = q.Count - 1; i >= 0; i--)
            {
                DestroyImmediate(q[i]);
            }
        }
    }
}

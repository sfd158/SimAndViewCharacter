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
using UnityEngine;

namespace RenderV2.ODE
{
    [Serializable]
    public enum ODEStepMode
    {
        Step,
        DampedStep
    }

    [Serializable]
    public enum ODEUpdateMode
    {
        RagDoll,
        ControlVAETracking
    }

    public class UpdateByODE : MonoBehaviour
    {
        public ODEUpdateMode updateMode = ODEUpdateMode.RagDoll;
        public GameObject CharacterList;
        public GameObject EnvList;
        public ODEWorldHandle handle;
        UpdateByServer ServerUpdate;
        DWorld dworld;

        public ODEWorldHandle build(double dt_ = 0.01)
        {
            ServerUpdate = GetComponent<UpdateByServer>();
            dworld = GetComponent<DWorld>();
            /* if (ServerUpdate.enabled)
            {
                return null;
            } */
            dworld.CalcAttrs();
            handle = new ODEWorldHandle();
            if (CharacterList == null)
            {
                CharacterList = FindFirstObjectByType<DCharacterList>().gameObject;
            }
            for (int cidx = 0; cidx < CharacterList.transform.childCount; cidx++)
            {
                if (CharacterList.transform.GetChild(cidx).TryGetComponent<DCharacter>(out var ch))
                {
                    handle.characterList.Add(new CharacterBuilder(handle, ch, true).build());
                }
            }
            handle.plane = GeomPlane.CreateDefault(handle.space);
            // TODO: load env object. here only use a simple plane.
            /* if (EnvList == null)
            {
            } */
            return handle;
        }

        private void Start()
        {
            build();
        }

        private void Update()
        {
            if (ServerUpdate.enabled) return;
            if (updateMode == ODEUpdateMode.RagDoll)
            {
                handle.DampedStep();
            }
            else if (updateMode == ODEUpdateMode.ControlVAETracking)
            {

            }
        }

        /* private void OnApplicationQuit()
        {
            handle?.Destroy();
            CommonFunc.CloseODE();
        } */
    }
}

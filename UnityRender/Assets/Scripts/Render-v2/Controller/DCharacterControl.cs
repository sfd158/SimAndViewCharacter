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
    public class DCharacterControl : MonoBehaviour, IExportInfo<DControlSignal>
    {
        [HideInInspector]
        public DCharacter dCharacter = null;

        public DCharacter GetDCharacter()
        {
            if (dCharacter == null)
            {
                dCharacter = GetComponent<DCharacter>();
            }
            return dCharacter;
        }

        [HideInInspector]
        public bool CallNextPhaseTrigger = false;

        [HideInInspector]
        public float vertical = 0.0F;

        [HideInInspector]
        public float horizontal = 0.0F;

        public void CallNextPhase()
        {
            CallNextPhaseTrigger = true;
        }

        public DControlSignal ExportInfo()
        {
            var signal = new DControlSignal();
            signal.CharacterID = this.dCharacter.IDNum;
            signal.vertical = this.vertical;
            signal.horizontal = this.horizontal;
            signal.GoNextPhase = this.CallNextPhaseTrigger;
            this.CallNextPhaseTrigger = false;
            return signal;
        }

        /// <summary>
        /// Control the character with joy stick
        /// Handle the character to move.
        /// </summary>
        public void JoystickControl()
        {
            this.vertical = Input.GetAxis("Vertical");
            this.horizontal = Input.GetAxis("Horizontal");
            if (vertical != 0 || horizontal != 0)
            {
                Debug.Log("vertical " + vertical);
                Debug.Log("horizontal " + horizontal);
            }
        }

        void Update()
        {
            this.JoystickControl();
        }
    }
}

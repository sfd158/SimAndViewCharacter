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
    /// <summary>
    /// Save 
    /// </summary>
    [DisallowMultipleComponent]
    public class DArrowList : DBaseContainer<ForceArrow>, IParseUpdate<ArrowListUpdateInfo>
    {
        public ArrowListUpdateInfo ExportInfo()
        {
            ArrowListUpdateInfo result = new ArrowListUpdateInfo();
            result.ArrowList = new ArrowUpdateInfo[CreateBuffer.Count];

            for (int i = 0; i < CreateBuffer.Count; i++)
            {
                result.ArrowList[i] = CreateBuffer[i].ExportInfo();
            }
            return result;
        }

        public void ParseUpdateInfo(ArrowListUpdateInfo UpdateInfo)
        {
            if (UpdateInfo == null)
            {
                return;
            }
            Array.Sort(UpdateInfo.ArrowList); // Sort Character by CharacterID
            foreach (ArrowUpdateInfo info in UpdateInfo.ArrowList)
            {
                if (TObjectDict.TryGetValue(info.IDNum, out var arrow))
                {
                    arrow.ParseUpdateInfo(info);
                }
                else
                {
                    Debug.LogWarning("Arrow ID" + info.IDNum + " Not exist. ignore..");
                    Debug.Log(TObjectDict.Keys);
                }
            }
        }
    }
}


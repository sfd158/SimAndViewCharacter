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
    [DisallowMultipleComponent]
    [RequireComponent(typeof(ScaleChecker))]
    public partial class DCharacterList : DBaseContainer<DCharacter>, IParseUpdate<DCharacterListUpdateInfo>, IExportInfo<DCharacterListExportInfo>, IParseRemove<DCharacterListRemoveInfo>
    {
        /// <summary>
        /// append character to create buffer
        /// </summary>
        /// <param name="res"></param>
        public void CreateBufferAppend(DCharacter res)
        {
            CreateBuffer.Add(res);
        }

        /// <summary>
        /// Parse Update Infomation from Server
        /// </summary>
        /// <param name="UpdateInfo">Character List Update Info From Server</param>
        public void ParseUpdateInfo(DCharacterListUpdateInfo UpdateInfo)
        {
            if (UpdateInfo.Characters.Length != TObjectDict.Count)
            {
                throw new ArgumentException("Character Count not match." + UpdateInfo.Characters.Length + " " + TObjectDict.Count);
            }

            Array.Sort(UpdateInfo.Characters); // Sort Character by CharacterID
            foreach (DCharacterUpdateInfo info in UpdateInfo.Characters)
            {
                if (TObjectDict.TryGetValue(info.CharacterID, out var dCharacter))
                {
                    dCharacter.ParseUpdateInfo(info);
                }
                else
                {
                    Debug.LogWarning("CharacterID" + info.CharacterID + " Not exist. ignore..");
                    Debug.Log(TObjectDict.Keys);
                }
            }

            // TODO: Add character in Python server
        }

        /// <summary>
        /// Export all characters in Unity
        /// </summary>
        /// <returns>Character List Export Info</returns>
        public DCharacterListExportInfo ExportInfo()
        {
            DCharacterListExportInfo info = new DCharacterListExportInfo
            {
                Characters = new DCharacterExportInfo[CreateBuffer.Count]
            };

            for (int chIdx = 0; chIdx < CreateBuffer.Count; chIdx++)
            {
                info.Characters[chIdx] = CreateBuffer[chIdx].ExportInfo();
            }

            return info;
        }

        /// <summary>
        /// Get Remove Infomation in Unity
        /// </summary>
        /// <returns>Character List Remove Information. return null if there is no character to remove.</returns>
        public DCharacterListRemoveInfo RemoveInfo()
        {
            if (RemoveBuffer.Count == 0)
            {
                return null; // no remove info
            }

            DCharacterListRemoveInfo info = new DCharacterListRemoveInfo
            {
                CharacterID = new int[RemoveBuffer.Count]
            };

            for(int i=0; i<RemoveBuffer.Count; i++)
            {
                info.CharacterID[i] = RemoveBuffer[i].IDNum;
            }
            return info;
        }

        /// <summary>
        /// Parse Remove Infomation from Server
        /// </summary>
        /// <param name="info">Remove Information from Server</param>
        public void ParseRemoveInfo(DCharacterListRemoveInfo info)
        {
            foreach(var idnum in info.CharacterID)
            {
                if (TObjectDict.TryGetValue(idnum, out var dCharacter))
                {
                    // TODO: Ext Joint List should be checkd..
                    TObjectDict.Remove(idnum);
                    dCharacter.DAllDestroy();
                }
                else
                {
                    Debug.LogWarning(string.Format("Character ID {0} not exist. Ignore.", idnum));
                }
            }
        }

        public new void PostExportInPlaying()
        {
            foreach(var character in CreateBuffer)
            {
                character.ComputeInitialMass();
            }
            base.PostExportInPlaying();
        }

        public DControlSignal[] GatherCharacterControlSignal()
        {
            int NumCharacter = GetNumTObject();
            if (NumCharacter > 0)
            {
                DControlSignal[] CharacterSignals = new DControlSignal[NumCharacter];
                int curr_index = 0;
                foreach (KeyValuePair<int, DCharacter> pair in TObjectDict)
                {
                    DCharacter character = pair.Value;
                    if (character.dCharacterControl == null)
                    {
                        return null;
                    }
                    CharacterSignals[curr_index++] = character.dCharacterControl.ExportInfo();
                }
                return CharacterSignals;
            }
            else
            {
                return null;
            }
        }
    }
}


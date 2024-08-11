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
using Newtonsoft.Json.Linq;

namespace RenderV2
{
    /// <summary>
    /// Load Json format from file
    /// </summary>
    public class JsonCharacterLoader
    {
        private readonly DWorld dWorld;
        public JsonCharacterLoader(DWorld dWorld_)
        {
            dWorld = dWorld_;
            dWorld.dCharacterList = dWorld.CharacterList.GetComponent<DCharacterList>();
        }

        /// <summary>
        /// Load character in json format
        /// </summary>
        /// <param name="message">character in json format</param>
        public DCharacter LoadJson(string message)
        {
            // Newtonsoft.Json.JsonReader
            JToken data = JObject.Parse(message)["CharacterList"]["Characters"][0];
            DCharacterExportInfo loadInfo = data.ToObject<DCharacterExportInfo>();
            // for loading a scene from file, we can ignore world 
            LoadCharacterExportInfo loader = new LoadCharacterExportInfo(dWorld.dCharacterList);
            loader.Parse(loadInfo, out var result);
            result.IDNum = (int)data["CharacterID"];
            Debug.Log(result.IDNum);
            return result;
        }

        /// <summary>
        /// Load character in json format
        /// </summary>
        /// <param name="JsonFileName"></param>
        public DCharacter LoadJsonFromFile(string JsonFileName)
        {
            if (JsonFileName == null || !File.Exists(JsonFileName))
            {
                Debug.LogError("File Not Exist" + JsonFileName);
                return null;
            }
            string message = System.IO.File.ReadAllText(JsonFileName);
            return LoadJson(message);
        }
    }
}


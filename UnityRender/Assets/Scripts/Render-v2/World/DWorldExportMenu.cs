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
using System.IO;
using UnityEditor;
using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// Menu for export
    /// </summary>
    public class DWorldExportMenu: DMenuBase
    {
        /// <summary>
        /// Menu for export world as json format
        /// </summary>
        [MenuItem("Assets/Export/World")] // menu entrance
        public static void ExportWorldEntrance()
        {
            string fname = EditorUtility.SaveFilePanel("Save World", ".", "world", "json");
            World.ExportInfoJsonFile(fname);
        }

        [MenuItem("Assets/Export/WorldPickle")]
        public static void ExportWordPickleEntrance()
        {
            string fname = EditorUtility.SaveFilePanel("Save World", ".", "world", "pickle");
            World.ExportInfoPickle(fname);
        }

        /// <summary>
        /// Menu for export character as json format
        /// </summary>
        [MenuItem("Assets/Export/Character")] // menu entrance
        public static void ExportCharacterEntrance()
        {
            DCharacter[] characters = Selection.GetFiltered<DCharacter>(SelectionMode.TopLevel);
            if (characters == null || characters.Length == 0)
            {
                characters = GameObject.FindObjectsOfType<DCharacter>();
            }
            if (characters == null || characters.Length != 1)
            {
                EditorUtility.DisplayDialog("Select 1 Character", "Please Select 1 Character in Unity", "OK");
                return;
            }

            string fname = EditorUtility.SaveFilePanel("Save Character", ".", "character", "json");
            characters[0].ReCompute();
            string result = characters[0].ExportInfoJson();
            using (StreamWriter file = new StreamWriter(fname, false))
            {
                file.Write(result);
            }
            Debug.Log("Export World to " + fname);
        }
    }
}

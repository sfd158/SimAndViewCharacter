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
using UnityEditor;

namespace RenderV2
{
    /// <summary>
    /// load json character from menu
    /// TODO: it may have bugs..
    /// </summary>
    public class CharacterLoaderMenu: DMenuBase
    {
        /// <summary>
        /// Load Character in xml format
        /// </summary>
        [MenuItem("Assets/Load/XMLCharacter")]
        public static void LoadXMLCharacterEntrance()
        {
            string fname = EditorUtility.OpenFilePanelWithFilters("XML Character File", ".", new string[]{ "XML Character File", "xml"});
            if (fname == null)
            {
                return;
            }
            XMLCharacterLoader loader = new XMLCharacterLoader(World, fname);
            loader.LoadFromXML();
            Debug.Log(string.Format("Load XML Character from {0}", fname));
        }

        static string LoadPDControlTitle = "Load PD Control Parameter";
        static string SelectPDControlMessage = "Please Select 1 PD Controller or 1 Character with child Component of PD Controller.";

        // [MenuItem("Assets/Load/XML PD Param")]
        public static void LoadXMLPDEntrance()
        {
            PDController controller = null;
            DCharacter[] characters = Selection.GetFiltered<DCharacter>(SelectionMode.TopLevel);
            if (characters != null && characters.Length == 1)
            {
                controller = characters[0].GetPDController();
                if (controller == null)
                {
                    controller = PDController.PDControllerCreate(characters[0].JointCount, characters[0].gameObject);
                }
            }
            else
            {
                PDController[] pds;
                pds = Selection.GetFiltered<PDController>(SelectionMode.TopLevel);
                if (pds == null)
                {
                    pds = GameObject.FindObjectsOfType<PDController>();
                }
                if (pds != null && pds.Length == 1)
                {
                    controller = pds[0];
                }
            }
            if (controller == null)
            {
                Debug.LogWarning(SelectPDControlMessage);
                EditorUtility.DisplayDialog(LoadPDControlTitle, SelectPDControlMessage, "OK");
                return;
            }

            // TODO: Parse xml pd control parameter file
        }

        /// <summary>
        /// Load character in json format
        /// </summary>
        [MenuItem("Assets/Load/JsonCharacter")]
        public static void JsonCharacterLoaderEntrance()
        {
            string fname = EditorUtility.OpenFilePanelWithFilters("Json Character File", ".", new string[] { "Json Character File", "json" });
            if (fname == null)
            {
                return;
            }
            JsonCharacterLoader loader = new JsonCharacterLoader(World);
            loader.LoadJsonFromFile(fname);
            Debug.Log(string.Format("Load Json Character from {0}", fname));
        }
    }
}

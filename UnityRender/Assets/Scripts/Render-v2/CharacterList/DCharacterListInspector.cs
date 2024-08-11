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
using UnityEditor;

namespace RenderV2
{
    /// <summary>
    /// Inspector of DCharacterList
    /// </summary>
    [CustomEditor(typeof(DCharacterList))]
    public class DCharacterListInspector : Editor
    {
        DCharacterList dCharacterList;
        private void OnEnable()
        {
            dCharacterList = target as DCharacterList;
        }

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            if (GUILayout.Button("ReCompute All Characters"))
            {
                dCharacterList.ReCompute();
            }

            GUILayout.BeginHorizontal();
            GUILayout.Label("Add Character");
            if (GUILayout.Button("Sphere")) // create character with a sphere geometry
            {
                dCharacterList.AddSphereCharacter(AppendToCreateBuffer: false);
            }
            if (GUILayout.Button("Box")) // create character with a box geometry
            {
                dCharacterList.AddBoxCharacter(AppendToCreateBuffer: false);
            }
            if (GUILayout.Button("Capsule")) // create character with a capsule geometry
            {
                dCharacterList.AddCapsuleCharacter(AppendToCreateBuffer: false);
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Hide Geoms"))
            {
                MeshRenderer[] renders = dCharacterList.GetComponentsInChildren<MeshRenderer>();
                for(int i = 0; i < renders.Length; i++)
                {
                    renders[i].enabled = false;
                }
            }
            if (GUILayout.Button("Show Geoms"))
            {
                MeshRenderer[] renders = dCharacterList.GetComponentsInChildren<MeshRenderer>();
                for (int i = 0; i < renders.Length; i++)
                {
                    renders[i].enabled = true;
                }
            }
            GUILayout.EndHorizontal();
        }
    }
}


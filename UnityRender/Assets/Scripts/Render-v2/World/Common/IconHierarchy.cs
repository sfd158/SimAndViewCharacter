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
using UnityEngine;
using UnityEditor;


namespace RenderV2
{
    /// <summary>
    /// Add icon in Unity Hierarchy Window.
    /// GameObject with DCharacter, DRigidBody, DJoint, DGeomObject component has icon in Unity hierarchy window.
    /// </summary>
    [InitializeOnLoad]
    class IconHierarchy
    {
        //static Texture2D worldTexture = null;
        //static Texture2D characterListTexture = null;
        private static readonly Texture2D characterTexture = null;
        private static readonly Texture2D jointTexture = null;
        private static readonly Texture2D eulerAxisTexture = null;
        private static readonly Texture2D bodyTexture = null;
        private static readonly Texture2D geomTexture = null;
        private static readonly Texture2D controlTexture = null;

        static IconHierarchy()
        {
            // worldTexture = AssetDatabase.LoadAssetAtPath("Assets/Images/world-icon.png", typeof(Texture2D)) as Texture2D;
            // characterListTexture = AssetDatabase.LoadAssetAtPath("Assets/Images/", typeof(Texture2D)) as Texture2D;
            characterTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Images/character-icon.png"); // icon for GameObject with DCharacter Component
            jointTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Images/joint-icon.png"); // icon for GameObject with DJoint Component
            eulerAxisTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Images/euler-axis-icon.png"); // icon for GameObject with EulerAxis Component
            bodyTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Images/body-icon.png"); // icon for GameObject with DRigidBody Component
            geomTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Images/geom-icon.png"); // icon for GameObject with DGeomObject Component
            controlTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Images/control-icon.png"); // icon for GameObject with ControllerBase Component

            // EditorApplication.update += UpdateFunc; // add update function in Unity
            EditorApplication.hierarchyWindowItemOnGUI += HierarchyWindowUpdateFunc; 
        }

        static Texture2D GetTexture(GameObject gameObject)
        {
            Texture2D texture = null;
            if (gameObject == null)
            {
                return texture;
            }
            /* else if (gameObject.GetComponent<DWorld>() != null)
            {
                texture = worldTexture;
            }
            else if (gameObject.GetComponent<DCharacterList>() != null)
            {
                texture = characterListTexture;
            } */
            else if (gameObject.GetComponent<DCharacter>() != null)
            {
                texture = characterTexture;
            }
            else if (gameObject.GetComponent<DJoint>() != null)
            {
                texture = jointTexture;
            }
            else if (gameObject.GetComponent<DEulerAxis>() != null)
            {
                texture = eulerAxisTexture;
            }
            else if (gameObject.GetComponent<DRigidBody>() != null)
            {
                texture = bodyTexture;
            }
            else if (gameObject.GetComponent<DGeomObject>() != null)
            {
                texture = geomTexture;
            }
            else if (gameObject.GetComponent<ControllerBase>() != null)
            {
                texture = controlTexture;
            }

            return texture;
        }

        static bool IsAddHierarchyIcon(GameObject gameObject)
        {
            return GetTexture(gameObject) != null;
        }

        static void HierarchyWindowUpdateFunc(int instanceID, Rect selectionRect)
        {
            GameObject obj = EditorUtility.InstanceIDToObject(instanceID) as GameObject;
            Texture2D texture = GetTexture(obj);
            if (texture == null)
            {
                return;
            }

            Rect r = new Rect(selectionRect);
            // r.x = r.width;
            r.x -= 30; // icon is on the left of GameObject in Unity Hierarchy Window
            r.width = 20;
            GUI.Label(r, texture);
        }
    }
}
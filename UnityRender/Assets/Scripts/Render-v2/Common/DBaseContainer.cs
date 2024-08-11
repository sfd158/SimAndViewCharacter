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
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    public class DBaseContainer<T>: MonoBehaviour, ICalcAttrs, IReCompute
        where T: DBaseObject
    {
        /// <summary>
        /// key: Instance ID, value: T Component
        /// </summary>
        [HideInInspector]
        public Dictionary<int, T> TObjectDict;

        /// <summary>
        /// create in physics simulation engine
        /// </summary>
        [HideInInspector]
        public List<T> CreateBuffer;

        /// <summary>
        /// T Component need to be removed
        /// </summary>
        [HideInInspector]
        public List<T> RemoveBuffer;

        public void CalcAttrs()
        {
            CreateBuffer = GetTObjects();
            TObjectDict = new Dictionary<int, T>();
            CalcChildAttrs(CreateBuffer);
        }

        private void CalcChildAttrs(List<T> tobjs)
        {
            foreach (var v in tobjs)
            {
                if (v.gameObject.activeInHierarchy)
                {
                    v.CalcAttrs();
                }
            }
        }

        public void MergeRemoveBuffer()
        {
            foreach (T dObject in RemoveBuffer)
            {
                TObjectDict.Remove(dObject.IDNum);
                dObject.DAllDestroy();
            }
            RemoveBuffer.Clear();
        }

        /// <summary>
        /// Merge create buffer (T Objects created in Unity) into TObjectDict
        /// </summary>
        public void MergeCreateBuffer()
        {
            foreach (T dObject in CreateBuffer)
            {
                TObjectDict.Add(dObject.IDNum, dObject);
            }
            // Debug.Log(string.Format("Create Buffer size = {0} merged.", CreateBuffer.Count));
            CreateBuffer.Clear();
        }

        /// <summary>
        /// Callback after export information in Unity
        /// </summary>
        public void PostExportInPlaying()
        {
            // buffer should be merged into TObjectDict after calling ExportInfo when Unity is playing
            MergeCreateBuffer();
        }

        /// <summary>
        /// Convert TObjectDict to List
        /// </summary>
        /// <returns></returns>
        protected List<T> GetTObjectListFromDict()
        {
            return new List<T>(TObjectDict.Values);
        }

        public T GetNode0()
        {
            return new List<T>(TObjectDict.Values)[0];
        }

        /// <summary>
        /// Get all T Component
        /// </summary>
        /// <returns></returns>
        protected List<T> GetTObjects()
        {
            List<T> TObjList = new List<T>();
            for (int gameIdx = 0; gameIdx < transform.childCount; gameIdx++)
            {
                GameObject gameObject = transform.GetChild(gameIdx).gameObject;
                if (gameObject.activeInHierarchy && gameObject.TryGetComponent<T>(out var dObject))
                {
                    TObjList.Add(dObject);
                }
            }
            TObjList.Sort();
            return TObjList;
        }

        Dictionary<int, T> GetDObjectsDict(List<T> TObjectsList)
        {
            Dictionary<int, T> TObjDict = new Dictionary<int, T>(TObjectsList.Count);
            for (int i = 0; i < TObjectsList.Count; i++)
            {
                T dObject = TObjectsList[i];
                TObjDict.Add(dObject.IDNum, dObject);
            }
            return TObjDict;
        }

        public int GetNumTObject()
        {
            if (TObjectDict == null)
            {
                Debug.Log("TObject Dict == null");
                return 0;
            }
            else
            {
                return TObjectDict.Count;
            }
        }

        public void ReCompute()
        {
            for (int gameIdx = 0; gameIdx < transform.childCount; gameIdx++)
            {
                GameObject gameObject = transform.GetChild(gameIdx).gameObject;
                if (gameObject.TryGetComponent<T>(out var dObject))
                {
                    // dObject.IDNum = dObject.GetInstanceID(); // set id by instance id
                    dObject.ReCompute(); // Re-compute all T Component
                }
                else
                {
                    // throw new ArithmeticException("Component is Required");
                }
            }
        }
    }
}

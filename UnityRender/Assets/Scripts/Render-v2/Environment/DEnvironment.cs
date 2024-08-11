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
    /// Store all motionless Geometry
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(ScaleChecker))]
    public partial class DEnvironment : DBaseContainer<VirtualGeom>, IExportInfo<DEnvironmentExportInfo>, IParseUpdate<DEnvironmentUpdateInfo>
    {
        /// <summary>
        /// Get all VirtualGeom Component in DEnvironment's children
        /// </summary>
        /// <returns></returns>
        public List<VirtualGeom> GetVGeomList()
        {
            return GetTObjects();
        }

        /// <summary>
        /// Environment Export Information in Unity
        /// </summary>
        /// <returns>Environment Export Info</returns>
        public DEnvironmentExportInfo ExportInfo()
        {
            DEnvironmentExportInfo EnvironmentInfo = new DEnvironmentExportInfo
            {
                Geoms = new DGeomExportInfo[CreateBuffer.Count]
            };

            for(int i = 0; i < CreateBuffer.Count; i++)
            {
                EnvironmentInfo.Geoms[i] = CreateBuffer[i].ExportInfo();
            }
            return EnvironmentInfo;
        }

        /// <summary>
        /// Environment Export Information in Unity as json Format
        /// </summary>
        /// <returns>string for Environment Export Info as json Format</returns>
        public string ExportInfoJson()
        {
            DEnvironmentExportInfo EnvironmentInfo = ExportInfo();
            return JsonUtility.ToJson(EnvironmentInfo); ;
        }

        /// <summary>
        /// Environment Remove Info in Unity
        /// </summary>
        /// <returns></returns>
        public DEnvironmentRemoveInfo RemoveInfo()
        {
            if (RemoveBuffer == null)
            {
                return null;
            }

            DEnvironmentRemoveInfo info = new DEnvironmentRemoveInfo
            {
                GeomID = new int[RemoveBuffer.Count]
            };

            for(int i=0; i<RemoveBuffer.Count; i++)
            {
                info.GeomID[i] = RemoveBuffer[i].IDNum;
            }
            return info;
        }

        /// <summary>
        /// Parse Update Information from Server
        /// </summary>
        /// <param name="UpdateInfo"></param>
        public void ParseUpdateInfo(DEnvironmentUpdateInfo UpdateInfo)
        {
            // TODO: Add Geometry in python server when running
        }

        /// <summary>
        /// Parse Update Information from Server
        /// </summary>
        /// <param name="message"></param>
        public void ParseUpdateInfoJson(string message)
        {
            DEnvironmentUpdateInfo UpdateInfo = JsonUtility.FromJson<DEnvironmentUpdateInfo>(message);
            ParseUpdateInfo(UpdateInfo);
        }

    }
}


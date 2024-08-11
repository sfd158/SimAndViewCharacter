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
using System.Collections;
using UnityEngine;
using Razorvine.Pickle;

namespace RenderV2
{
    public partial class DWorld
    {
        /// <summary>
        /// get world's export information, and merge CreateBuffer of DCharacterList, DEnvironment, DExtJointList
        /// </summary>
        /// <returns></returns>
        public DWorldExportInfo ExportInfoWithPostProcess()
        {
            DWorldExportInfo info = ExportInfo();

            dCharacterList.PostExportInPlaying();
            dEnvironment.PostExportInPlaying();

            if (extJointList != null)
            {
                extJointList.PostExportInPlaying();
            }
            if (dArrows != null)
            {
                dArrows.PostExportInPlaying();
            }
            // TODO: DEnvironment, DExtJointList, Post Export In Playing
            return info;
        }

        /// <summary>
        /// get world's export information
        /// </summary>
        /// <returns></returns>
        public DWorldExportInfo ExportInfo()
        {
            DWorldExportInfo WorldInfo = new DWorldExportInfo();

            // TODO: Some Attributes can be modified in playing
            // while some attributes cannot be modified in playing.
            WorldInfo.WorldAttr = new DWorldAttr();

            WorldInfo.WorldAttr.ChangeAttr = new ChangeAbleDWordAttr{
                Gravity = Utils.Vector3ToArr(Gravity),
                StepCount = StepCount,
                RenderFPS = RenderFPS,
            };

            WorldInfo.WorldAttr.FixedAttr = new FixedDWorldAttr{
                SimulateFPS = SimulateFPS,
                UseHinge = UseHinge,
                UseAngleLimit = UseAngleLimit,
                SelfCollision = SelfCollision,
                dWorldUpdateMode = (int)dWorldUpdateMode
            };

            WorldInfo.Environment = dEnvironment.ExportInfo();
            WorldInfo.CharacterList = dCharacterList.ExportInfo();

            if (extJointList != null)
            {
                WorldInfo.ExtJointList = extJointList.ExportInfo();
            }

            if (extForceBase != null)
            {
                WorldInfo.ExtForceList = extForceBase.GetExtForceList(); // force add by mouse click/drag and etc.
                // Debug.Log("WorldInfo.ExtForceList " + WorldInfo.ExtForceList);
            }

            if (dArrows != null)
            {
                WorldInfo.ArrowList = dArrows.ExportInfo();
            }
            return WorldInfo;
        }

        /// <summary>
        /// get world's export information in json format
        /// </summary>
        /// <returns>json string</returns>
        public string ExportInfoJson()
        {
            DWorldExportInfo WorldInfo = ExportInfo();
            string JsonResult = JsonUtility.ToJson(WorldInfo);
            return JsonResult;
        }

        /// <summary>
        /// Export World infomation to json format
        /// </summary>
        /// <param name="ExportJsonPath"></param>
        public void ExportInfoJsonFile(string ExportJsonPath)
        {
            ReCompute();
            string jsonResult = ExportInfoJson();
            using (StreamWriter file = new StreamWriter(ExportJsonPath, false))
            {
                file.Write(jsonResult);
            }
            Debug.Log("Export World to " + ExportJsonPath);
        }

        public void ExportInfoPickle(string ExportPickleFile)
        {
            if (ExportPickleFile == null || ExportPickleFile.Length == 0)
            {
                return;
            }
            if (File.Exists(ExportPickleFile))
            {
                // overwrite original file
                File.Delete(ExportPickleFile);
            }
            ReCompute();
            DWorldExportInfo exportInfo = ExportInfo();
            var pickler = new Pickler(true);
            byte[] sendInfo = pickler.dumps(exportInfo.ToHashTable());
            var bstream = new BinaryWriter(new FileStream(ExportPickleFile, FileMode.CreateNew));
            bstream.Write(sendInfo);
            bstream.Flush();
            bstream.Close();
        }
    }
}
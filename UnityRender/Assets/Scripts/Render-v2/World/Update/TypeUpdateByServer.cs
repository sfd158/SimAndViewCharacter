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
    enum ServerMessageType
    {
        SUCCESS,
        FAIL,
        HIERARCHY, // Send Hierarchy infomation
        UPDATE, // Send Update infomation
        INITIAL_INSTRUCTION
    }

    [Serializable]
    public class InitialInstructionSendInfo: ISupportToHashTable
    {
        public int MessType = (int)ServerMessageType.INITIAL_INSTRUCTION;
        public InitialInstructionSendInfo() {}
        public InitialInstructionSendInfo(Hashtable table)
        {
            MessType = Convert.ToInt32(table["MessType"]);
        }
        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["MessType"] = MessType;
            return table;
        }
    }

    /// <summary>
    /// Hierarchy information send to server at start
    /// </summary>
    [Serializable]
    public class HierarchySendInfo : ISupportToHashTable
    {
        public int MessType;  // ServerMessageType
        public DWorldExportInfo WorldInfo;

        public HierarchySendInfo() { }

        public HierarchySendInfo(DWorldExportInfo WorldInfo_)
        {
            MessType = (int)ServerMessageType.HIERARCHY;
            WorldInfo = WorldInfo_;
        }

        public HierarchySendInfo(Hashtable table)
        {
            MessType = Convert.ToInt32(table["MessType"]);
            WorldInfo = new DWorldExportInfo(table["WorldInfo"] as Hashtable);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["MessType"] = MessType;
            table["WorldInfo"] = WorldInfo.ToHashTable();

            return table;
        }
    }

    /// <summary>
    /// Update information send to server at each frame
    /// </summary>
    [Serializable]
    public class DUpdateSendInfo : ISupportToHashTable
    {
        public int MessType;
        // TODO: Add new Character / Geometry / Joint; Remove Character / Geometry / Joint.
        public DWorldExportInfo ExportInfo = null; // Add new Character / Geometry / Joint
        public DWorldRemoveInfo RemoveInfo = null; // Remove Character / Geometry / Joint

        public DWorldControlSignal WorldControlSignal = null; // control each character..

        public DUpdateSendInfo()
        {
            MessType = (int)ServerMessageType.UPDATE;
        }

        public DUpdateSendInfo(Hashtable table)
        {
            MessType = Convert.ToInt32(table["MessType"]);
            if (table.ContainsKey("ExportInfo"))
            {
                ExportInfo = new DWorldExportInfo(table["ExportInfo"] as Hashtable);
            }
            if (table.ContainsKey("RemoveInfo"))
            {
                RemoveInfo = new DWorldRemoveInfo(table["RemoveInfo"] as Hashtable);
            }
            if (table.ContainsKey("DWorldControlSignal"))
            {
                WorldControlSignal = new DWorldControlSignal(table["WorldControlSignal"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["MessType"] = MessType;
            if (ExportInfo != null)
            {
                table["ExportInfo"] = ExportInfo.ToHashTable();
            }
            if (RemoveInfo != null)
            {
                table["RemoveInfo"] = RemoveInfo.ToHashTable();
            }
            if (WorldControlSignal != null)
            {
                table["WorldControlSignal"] = WorldControlSignal.ToHashTable();
            }

            return table;
        }
    }

    /// <summary>
    /// Check success
    /// </summary>
    [Serializable]
    public class CheckRecieveInfo : ISupportToHashTable
    {
        public int MessType;

        public CheckRecieveInfo() { }

        public CheckRecieveInfo(Hashtable table)
        {
            MessType = Convert.ToInt32(table["MessType"]);
        }

        public bool CheckOK()
        {
            return MessType == (int)ServerMessageType.SUCCESS;
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["MessType"] = MessType;
            return table;
        }
    }

    [Serializable]
    public class InstructionRecieveInfo: CheckRecieveInfo
    {
        public string[] DupCharacterNames;
        public string[] LoadCharacterFileName;

        public InstructionRecieveInfo() { }
        public InstructionRecieveInfo(Hashtable table): base(table)
        {
            if (table.ContainsKey("DupCharacterNames"))
            {
                DupCharacterNames = Utils.ArrayListToStringArray(table["DupCharacterNames"]);
            }
            if (table.ContainsKey("LoadCharacterFileName"))
            {
                LoadCharacterFileName = Utils.ArrayListToStringArray(table["LoadCharacterFileName"]);
            }
        }

        public int NumDuplicateCharacter
        {
            get
            {
                if (DupCharacterNames == null)
                {
                    return 0;
                }
                else
                {
                    return DupCharacterNames.Length;
                }
            }
        }

        public int NumLoadCharacter
        {
            get
            {
                if (LoadCharacterFileName == null)
                {
                    return 0;
                }
                else
                {
                    return LoadCharacterFileName.Length;
                }
            }
        }

        public new Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["MessType"] = MessType;
            table["DupCharacterNames"] = DupCharacterNames;
            table["LoadCharacterFileName"] = LoadCharacterFileName;
            return table;
        }
    }

    /// <summary>
    /// information recieved from server
    /// </summary>
    [Serializable]
    public class UpdateRecieveInfo : ISupportToHashTable
    {
        public DWorldUpdateInfo WorldUpdateInfo;

        public DWorldExportInfo ExportInfo = null; // TODO: Create Information from Server
        public DWorldRemoveInfo RemoveInfo = null; // TODO: Remove Information from Server

        public UpdateRecieveInfo() { }

        public UpdateRecieveInfo(Hashtable table)
        {
            if (table == null)
            {
                return;
            }

            if (table.ContainsKey("WorldUpdateInfo"))
            {
                WorldUpdateInfo = new DWorldUpdateInfo(table["WorldUpdateInfo"] as Hashtable);
            }

            if (table.ContainsKey("ExportInfo"))
            {
                ExportInfo = new DWorldExportInfo(table["ExportInfo"] as Hashtable);
            }

            if (table.ContainsKey("RemoveInfo"))
            {
                RemoveInfo = new DWorldRemoveInfo(table["RemoveInfo"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["WorldUpdateInfo"] = WorldUpdateInfo.ToHashTable();
            table["ExportInfo"] = ExportInfo?.ToHashTable();
            table["RemoveInfo"] = RemoveInfo?.ToHashTable();

            return table;
        }
    }
}
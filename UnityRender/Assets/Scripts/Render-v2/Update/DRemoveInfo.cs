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

namespace RenderV2
{
    /// <summary>
    /// CharacterList Remove Information
    /// 1. Get From Server;
    /// 2. Get From Unity.
    /// </summary>
    [Serializable]
    public class DCharacterListRemoveInfo : ISupportToHashTable
    {
        public int[] CharacterID = null; // character in CharacterList to be removed
        public DCharacterListRemoveInfo() { }
        public DCharacterListRemoveInfo(Hashtable table)
        {
            if (table.ContainsKey("CharacterID"))
            {
                CharacterID = Utils.ArrayListToIntArray(table["CharacterID"]);
            }
        }
        public Hashtable ToHashTable()
        {
            if (CharacterID == null || CharacterID.Length == 0)
            {
                return null;
            }
            else
            {
                Hashtable table = new Hashtable();
                table["CharacterID"] = new ArrayList(CharacterID);
                return table;
            }
        }
    }

    /// <summary>
    /// Environment Remove Information
    /// 1. Get From Server;
    /// 2. Get From Unity.
    /// </summary>
    [Serializable]
    public class DEnvironmentRemoveInfo : ISupportToHashTable
    {
        public int[] GeomID = null; // geom in Environment to be removed
        public DEnvironmentRemoveInfo() { }
        public DEnvironmentRemoveInfo(Hashtable table)
        {
            if (table.ContainsKey("GeomID"))
            {
                GeomID = Utils.ArrayListToIntArray(table["GeomID"]);
            }
        }

        /// <summary>
        /// Extract environment remove information to hashtable
        /// </summary>
        /// <returns></returns>
        public Hashtable ToHashTable()
        {
            if (GeomID == null || GeomID.Length == 0)
            {
                return null;
            }
            else
            {
                Hashtable table = new Hashtable();
                table["GeomID"] = new ArrayList(GeomID);
                return table;
            }
        }
    }

    /// <summary>
    /// External Joint List Remove Information
    /// 1. Get From Server;
    /// 2. Get From Unity.
    /// </summary>
    [Serializable]
    public class DExtJointListRemoveInfo : ISupportToHashTable
    {
        public int[] ExtJointID = null; // ext joint in ExtJointList to be removed
        public DExtJointListRemoveInfo() { }
        public DExtJointListRemoveInfo(Hashtable table)
        {
            if (table.ContainsKey("ExtJointID"))
            {
                ExtJointID = Utils.ArrayListToIntArray(table["ExtJointID"]);
            }
        }
        public Hashtable ToHashTable()
        {
            if (ExtJointID == null || ExtJointID.Length == 0)
            {
                return null;
            }
            else
            {
                Hashtable table = new Hashtable();
                table["ExtJointID"] = new ArrayList(ExtJointID);
                return table;
            }
        }
    }

    /// <summary>
    /// World Remove Infomation.
    /// 1. Get From Server;
    /// 2. Get From Unity.
    /// </summary>
    [Serializable]
    public class DWorldRemoveInfo : ISupportToHashTable
    {
        public DCharacterListRemoveInfo CharacterList;
        public DEnvironmentRemoveInfo Environment;
        public DExtJointListRemoveInfo ExtJointList;

        public DWorldRemoveInfo() { }

        /// <summary>
        /// constructor from Hashtable
        /// </summary>
        /// <param name="table"></param>
        public DWorldRemoveInfo(Hashtable table)
        {
            if (table.ContainsKey("CharacterList"))
            {
                CharacterList = new DCharacterListRemoveInfo(table["CharacterList"] as Hashtable);
            }
            if (table.ContainsKey("Environment"))
            {
                Environment = new DEnvironmentRemoveInfo(table["Environment"] as Hashtable);
            }
            if (table.ContainsKey("ExtJointList"))
            {
                ExtJointList = new DExtJointListRemoveInfo(table["ExtJointList"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable characterListRes = CharacterList?.ToHashTable();
            Hashtable environmentRes = Environment?.ToHashTable();
            Hashtable extJointListRes = ExtJointList?.ToHashTable();

            if (characterListRes == null && environmentRes == null && extJointListRes == null)
            {
                return null;
            }
            else
            {
                return new Hashtable
                {
                    ["CharacterList"] = characterListRes,
                    ["Environment"] = environmentRes,
                    ["ExtJointList"] = extJointListRes
                };
            }
        }
    }
}

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


namespace RenderV2
{
    /// <summary>
    /// Control Signal in Unity.
    /// i.e. Use keyboard to control character
    /// </summary>
    [Serializable]
    public class DControlSignal: ISupportToHashTable
    {
        public int CharacterID = 0; // which character to control
        // TODO: Add some control signals
        // e.g. press keyboard left, right, up, down button

        public float horizontal = 0.0F;
        public float vertical = 0.0F;

        // go to next phase for this character
        public bool GoNextPhase = false;
        public DControlSignal() { }
        public DControlSignal(Hashtable table)
        {
            if (table.ContainsKey("CharacterID"))
            {
                CharacterID = Convert.ToInt32(table["CharacterID"]);
            }
            if (table.ContainsKey("horizontal"))
            {
                horizontal = Convert.ToSingle(table["horizontal"]);
            }
            if (table.ContainsKey("vertical"))
            {
                vertical = Convert.ToSingle(table["vertical"]);
            }
            if (table.ContainsKey("GoNextPhase"))
            {
                GoNextPhase = Convert.ToBoolean(table["GoNextPhase"]);
            }
            else
            {
                GoNextPhase = false;
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["CharacterID"] = CharacterID;
            table["horizontal"] = horizontal;
            table["vertical"] = vertical;
            table["GoNextPhase"] = GoNextPhase;
            return table;
        }
    }

    [Serializable]
    public class DWorldControlSignal: ISupportToHashTable
    {
        public DControlSignal[] CharacterSignals;
        public DWorldControlSignal()
        {

        }

        public DWorldControlSignal(Hashtable table)
        {
            if (table.ContainsKey("CharacterSignals"))
            {
                ArrayList SignalArr = table["CharacterSignals"] as ArrayList;
                if (SignalArr != null && SignalArr.Count > 0)
                {
                    // build control signal for each character.
                    CharacterSignals = new DControlSignal[SignalArr.Count];
                    for(int i=0; i<SignalArr.Count; i++)
                    {
                        CharacterSignals[i] = new DControlSignal(SignalArr[i] as Hashtable);
                    }
                }
            }
        }

        public Hashtable ToHashTable()
        {
            if (this.CharacterSignals == null || this.CharacterSignals.Length == 0)
            {
                return null;
            }

            Hashtable table = new Hashtable();
            int SignalCount = this.CharacterSignals.Length;
            ArrayList result = new ArrayList();
            for(int i=0; i<SignalCount; i++)
            {
                result.Add(this.CharacterSignals[i].ToHashTable());
            }
            table["CharacterSignals"] = result;
            return table;
        }
    }
}

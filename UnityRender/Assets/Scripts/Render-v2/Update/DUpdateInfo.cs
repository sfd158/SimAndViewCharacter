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
    /// Base Joint Update Infomation from Server
    /// </summary>
    [Serializable]
    public class DJointUpdateInfoBase : ISupportToHashTable, IComparable<DJointUpdateInfoBase>
    {
        public int JointID = 0;
        public float[] Position = null; // Global Position

        public DJointUpdateInfoBase() { }

        public DJointUpdateInfoBase(Hashtable table)
        {
            if (table.ContainsKey("JointID"))
            {
                JointID = Convert.ToInt32(table["JointID"]);
            }
            if (table.ContainsKey("Position"))
            {
                Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable
            {
                ["JointID"] = JointID,
                ["Position"] = Position
            };
            return table;
        }

        public int CompareTo(DJointUpdateInfoBase other) => JointID.CompareTo(other.JointID);
    }

    public class DJointUpdateInfoQuatBase: DJointUpdateInfoBase
    {
        public float[] Quaternion = null; // Global Quaternion
        public DJointUpdateInfoQuatBase() { }
        public DJointUpdateInfoQuatBase(Hashtable table) : base(table)
        {
            if (table.ContainsKey("Quaternion"))
            {
                Quaternion = Utils.ArrayListToFloatArray(table["Quaternion"] as ArrayList);
            }
        }

        public new Hashtable ToHashTable()
        {
            Hashtable table = base.ToHashTable();
            table["Quaternion"] = Quaternion;
            return table;
        }
    }

    /// <summary>
    /// Joint Update Infomation from Server
    /// </summary>
    [Serializable]
    public class DJointUpdateInfo : DJointUpdateInfoQuatBase
    {
        public DJointUpdateInfo() : base() { }

        public DJointUpdateInfo(Hashtable table) : base(table) { }
    }

    /// <summary>
    /// Root Joint Update Information from Server
    /// </summary>
    [Serializable]
    public class DRootUpdateInfo: DJointUpdateInfoQuatBase
    {
        public DRootUpdateInfo() : base() { }

        public DRootUpdateInfo(Hashtable table) : base(table) { }
    }

    /// <summary>
    /// External Joint Update Information from Server
    /// </summary>
    [Serializable]
    public class ExtJointUpdateInfo: DJointUpdateInfoQuatBase
    {
        public ExtJointUpdateInfo() : base() { }

        public ExtJointUpdateInfo(Hashtable table) : base(table) { }
    }

    public class ContactJointUpdateInfo: DJointUpdateInfoBase
    {
        public float[] Force;
        public float ContactLabel = 1.0F;
        public ContactJointUpdateInfo() { }
        public ContactJointUpdateInfo(Hashtable table): base(table)
        {
            if (table.ContainsKey("Force"))
            {
                Force = Utils.ArrayListToFloatArray(table["Force"] as ArrayList);
            }
            if (table.ContainsKey("ContactLabel"))
            {
                ContactLabel = Convert.ToSingle(table["ContactLabel"]);
            }
        }

        public new Hashtable ToHashTable()
        {
            Hashtable table = base.ToHashTable();
            table["Force"] = Force;
            table["ContactLabel"] = ContactLabel;
            return table;
        }
    }

    /// <summary>
    /// Rigid Body Update Information from Server
    /// </summary>
    [Serializable]
    public class DRigidBodyUpdateInfo: ISupportToHashTable, IComparable<DRigidBodyUpdateInfo>
    {
        public int BodyID;
        // public string BodyName; // for debug..
        public float[] Position;
        public float[] Quaternion;
        public float[] LinearVelocity; // Global Linear Velocity
        public float[] AngularVelocity; // Global Angular Velocity

        public float[] Color;

        public DRigidBodyUpdateInfo() { }
        public DRigidBodyUpdateInfo(Hashtable table)
        {
            if (table.ContainsKey("BodyID"))
            {
                BodyID = Convert.ToInt32(table["BodyID"]);
            }
            if (table.ContainsKey("Position"))
            {
                Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
            }
            if (table.ContainsKey("Quaternion"))
            {
                Quaternion = Utils.ArrayListToFloatArray(table["Quaternion"] as ArrayList);
            }
            if (table.ContainsKey("LinearVelocity"))
            {
                LinearVelocity = Utils.ArrayListToFloatArray(table["LinearVelocity"] as ArrayList);
            }
            if (table.ContainsKey("AngularVelocity"))
            {
                AngularVelocity = Utils.ArrayListToFloatArray(table["AngularVelocity"] as ArrayList);
            }
            if (table.ContainsKey("Color"))
            {
                Color = Utils.ArrayListToFloatArray(table["Color"] as ArrayList);
            }
        }

        public int CompareTo(DRigidBodyUpdateInfo other) => BodyID.CompareTo(other.BodyID);

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            if (Position != null)
            {
                table["Position"] = new ArrayList(Position);
            }
            if (Quaternion != null)
            {
                table["Quaternion"] = new ArrayList(Quaternion);
            }
            if (LinearVelocity != null && LinearVelocity.Length > 0)
            {
                table["LinearVelocity"] = new ArrayList(LinearVelocity);
            }
            if (AngularVelocity != null && AngularVelocity.Length > 0)
            {
                table["AngularVelocity"] = new ArrayList(AngularVelocity);
            }
            if (Color != null && Color.Length > 0)
            {
                table["Color"] = new ArrayList(Color);
            }
            return table;
        }
    }

    /// <summary>
    /// BodyList Update Information from Server
    /// </summary>
    [Serializable]
    public class DBodyListUpdateInfo: ISupportToHashTable
    {
        public DRigidBodyUpdateInfo[] BodyList;
        public DBodyListUpdateInfo() { }
        public DBodyListUpdateInfo(Hashtable table)
        {
            ArrayList BodyArray = table["BodyList"] as ArrayList;
            BodyList = new DRigidBodyUpdateInfo[BodyArray.Count];
            for(int i=0; i<BodyArray.Count; i++)
            {
                BodyList[i] = new DRigidBodyUpdateInfo(BodyArray[i] as Hashtable);
            }
            Array.Sort(BodyList);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            ArrayList BodyArray = new ArrayList(BodyList.Length);
            for(int i=0; i<BodyList.Length; i++)
            {
                BodyArray.Add(BodyList[i].ToHashTable());
            }
            table["BodyList"] = BodyArray;
            return table;
        }
    }

    [Serializable]
    public class RuntimeUpdateInfo: ISupportToHashTable
    {
        public float[] Scale;
        public RuntimeUpdateInfo()
        {

        }

        public RuntimeUpdateInfo(Hashtable table)
        {
            if (table.ContainsKey("Scale"))
            {
                Scale = Utils.ArrayListToFloatArray(table["Scale"] as ArrayList);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["Scale"] = new ArrayList(Scale);
            return table;
        }
    }

    /// <summary>
    /// Character Update Information from Server.
    /// Update Root Position, Root Rotation, Joints' Rotation, Bodies' Linear Velocity and Angular Velocity
    /// </summary>
    [Serializable]
    public class DCharacterUpdateInfo : IComparable<DCharacterUpdateInfo>, ISupportToHashTable
    {
        public int CharacterID;
        public DRootUpdateInfo RootInfo;
        public DJointUpdateInfo[] JointInfo;
        public DRigidBodyUpdateInfo[] BodyInfo;

        // here we add some attributes directly..Scale x, y, z.
        // actually, this is unused here..
        public RuntimeUpdateInfo RuntimeInfo;

        public int CompareTo(DCharacterUpdateInfo other)
        {
            return CharacterID.CompareTo(other.CharacterID);
        }

        public DCharacterUpdateInfo() { }

        public DCharacterUpdateInfo(Hashtable table)
        {
            CharacterID = Convert.ToInt32(table["CharacterID"]);
            RootInfo = new DRootUpdateInfo(table["RootInfo"] as Hashtable);

            ArrayList jointList = table["JointInfo"] as ArrayList;
            JointInfo = new DJointUpdateInfo[jointList.Count];
            for(int i=0; i<jointList.Count; i++)
            {
                JointInfo[i] = new DJointUpdateInfo(jointList[i] as Hashtable);
            }

            if (table.ContainsKey("BodyInfo"))
            {
                ArrayList bodyList = table["BodyInfo"] as ArrayList;
                if (bodyList != null)
                {
                    BodyInfo = new DRigidBodyUpdateInfo[bodyList.Count];
                    for(int i=0; i<bodyList.Count; i++)
                    {
                        BodyInfo[i] = new DRigidBodyUpdateInfo(bodyList[i] as Hashtable);
                    }
                }
            }

            if (table.ContainsKey("RuntimeInfo"))
            {
                RuntimeInfo = new RuntimeUpdateInfo(table["RuntimeInfo"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["CharacterID"] = CharacterID;
            table["RootInfo"] = RootInfo.ToHashTable();

            ArrayList jointList = new ArrayList(JointInfo.Length);
            for(int i=0; i<JointInfo.Length; i++)
            {
                jointList.Add(JointInfo[i].ToHashTable());
            }
            table["JointInfo"] = jointList;

            if (BodyInfo != null)
            {
                ArrayList bodyList = new ArrayList(BodyInfo.Length);
                for(int i=0; i<BodyInfo.Length; i++)
                {
                    bodyList.Add(BodyInfo[i].ToHashTable());
                }
                table["BodyInfo"] = bodyList;
            }
            else
            {
                table["BodyInfo"] = null;
            }

            if (RuntimeInfo != null)
            {
                table["RuntimeInfo"] = RuntimeInfo.ToHashTable();
            }

            return table;
        }
    }

    /// <summary>
    /// Character List Update Information from Server
    /// </summary>
    [Serializable]
    public class DCharacterListUpdateInfo: ISupportToHashTable
    {
        public DCharacterUpdateInfo[] Characters;
        public DCharacterListUpdateInfo() { }
        public DCharacterListUpdateInfo(Hashtable table)
        {
            ArrayList characterList = table["Characters"] as ArrayList;
            Characters = new DCharacterUpdateInfo[characterList.Count];
            for (int i = 0; i < Characters.Length; i++)
            {
                Characters[i] = new DCharacterUpdateInfo(characterList[i] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            ArrayList characterList = new ArrayList(Characters.Length);
            for (int i = 0; i < Characters.Length; i++)
            {
                characterList.Add(Characters[i].ToHashTable());
            }
            table["Characters"] = characterList;

            return table;
        }
    }

    /// <summary>
    /// Environment Update Information from Server
    /// </summary>
    [Serializable]
    public class DEnvironmentUpdateInfo: ISupportToHashTable
    {
        public DEnvironmentUpdateInfo() { }

        public DEnvironmentUpdateInfo(Hashtable table)
        {

        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            return table;
        }
    }

    /// <summary>
    /// External Joint List Update Information from Server
    /// </summary>
    [Serializable]
    public class ExtJointListUpdateInfo : ISupportToHashTable
    {
        public ExtJointUpdateInfo[] Joints;

        public ExtJointListUpdateInfo() { }

        public ExtJointListUpdateInfo(Hashtable table)
        {
            ArrayList jointList = table["Joints"] as ArrayList;
            Joints = new ExtJointUpdateInfo[jointList.Count];
            for(int i=0; i<jointList.Count; i++)
            {
                Joints[i] = new ExtJointUpdateInfo(jointList[i] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            ArrayList jointList = new ArrayList(Joints.Length);
            for(int i=0; i<Joints.Length; i++)
            {
                jointList.Add(Joints[i].ToHashTable());
            }
            table["Joints"] = jointList;

            return table;
        }
    }

    [Serializable]
    public class ContactListUpdateInfo: ISupportToHashTable
    {
        public ContactJointUpdateInfo[] Joints;
        public ContactListUpdateInfo() { }
        public ContactListUpdateInfo(Hashtable table)
        {
            if (table.ContainsKey("Joints"))
            {
                ArrayList jointList = table["Joints"] as ArrayList;
                if (jointList != null)
                {
                    Joints = new ContactJointUpdateInfo[jointList.Count];
                    for(int i=0; i<Joints.Length; i++)
                    {
                        Joints[i] = new ContactJointUpdateInfo(jointList[i] as Hashtable);
                    }
                }
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            ArrayList jointList = new ArrayList(Joints.Length);
            for(int i=0; i<Joints.Length; i++)
            {
                jointList.Add(Joints[i].ToHashTable());
            }
            table["Joints"] = jointList;
            return table;
        }

        public int Length
        {
            get
            {
                if (Joints == null)
                {
                    return 0;
                }
                else
                {
                    return Joints.Length;
                }
            }
        }
    }

    /// <summary>
    /// Helper information
    /// </summary>
    [Serializable]
    public class HelperUpdateInfo: ISupportToHashTable
    {
        public string Message = "";
        public HelperUpdateInfo() {}
        public HelperUpdateInfo(Hashtable table)
        {
            if (table.ContainsKey("Message"))
            {
                Message = table["Message"] as string;
            }
        }

        public Hashtable ToHashTable()
        {
            if (Message == null)
            {
                return null;
            }
            Hashtable table = new Hashtable();
            table["Message"] = Message;
            return table;
        }
    }

    /// <summary>
    /// for rendering the arrow in the unity
    /// </summary>
    [Serializable]
    public class ArrowUpdateInfo: ISupportToHashTable
    {
        public float[] StartPos;
        public float[] EndPos;
        public bool InUse = true;
        public int IDNum;
        public ArrowUpdateInfo()
        {

        }

        public ArrowUpdateInfo(Hashtable table)
        {
            StartPos = Utils.ArrayListToFloatArray(table["StartPos"] as ArrayList);
            EndPos = Utils.ArrayListToFloatArray(table["EndPos"] as ArrayList);
            InUse = Convert.ToBoolean(table["InUse"]);
            IDNum = Convert.ToInt32(IDNum);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["StartPos"] = new ArrayList(StartPos);
            table["EndPos"] = new ArrayList(EndPos);
            table["InUse"] = InUse;
            table["IDNum"] = IDNum;
            return table;
        }

        public Vector3 StartPosVec3()
        {
            return new Vector3(StartPos[0], StartPos[1], StartPos[2]);
        }

        public Vector3 EndPosVec3()
        {
            return new Vector3(EndPos[0], EndPos[1], EndPos[2]);
        }
    }

    /// <summary>
    /// for rendering the arrow in the Unity
    /// </summary>
    [Serializable]
    public class ArrowListUpdateInfo: ISupportToHashTable
    {
        public ArrowUpdateInfo[] ArrowList;
        public ArrowListUpdateInfo()
        {

        }

        public ArrowListUpdateInfo(Hashtable table)
        {
            if (table.ContainsKey("ArrowList"))
            {
                ArrayList arr = table["ArrowList"] as ArrayList;
                ArrowList = new ArrowUpdateInfo[arr.Count];
                for (int i = 0; i < ArrowList.Length; i++)
                {
                    ArrowList[i] = new ArrowUpdateInfo(arr[i] as Hashtable);
                }
            }
        }

        public int Length
        {
            get
            {
                if (ArrowList == null) return 0;
                else return ArrowList.Length;
            }
        }


        public Hashtable ToHashTable()
        {
            if (ArrowList == null || ArrowList.Length == 0)
            {
                return null;
            }
            Hashtable table = new Hashtable();

            ArrayList arr = new ArrayList(ArrowList.Length);
            for (int i = 0; i < ArrowList.Length; i++)
            {
                arr.Add(ArrowList[i].ToHashTable());
            }

            table["ArrowList"] = arr;

            return table;
        }
    }

    /// <summary>
    /// World Update Information from Server
    /// </summary>
    [Serializable]
    public class DWorldUpdateInfo: ISupportToHashTable
    {
        public DEnvironmentUpdateInfo Environment; // Environment update info
        public DCharacterListUpdateInfo CharacterList; // CharacterList update info
        public ExtJointListUpdateInfo ExtJointList; // external joint list update info
        public ContactListUpdateInfo ContactList; // contact joint in simulation
        public HelperUpdateInfo HelperInfo;
        public ArrowListUpdateInfo ArrowList; // render arrows in the Unity Scene.

        public DWorldUpdateInfo() { }

        public DWorldUpdateInfo(Hashtable table)
        {
            if (table.ContainsKey("Environment"))
            {
                Environment = new DEnvironmentUpdateInfo(table["Environment"] as Hashtable);
            }

            if (table.ContainsKey("CharacterList"))
            {
                CharacterList = new DCharacterListUpdateInfo(table["CharacterList"] as Hashtable);
            }

            if (table.ContainsKey("ExtJointList"))
            {
                ExtJointList = new ExtJointListUpdateInfo(table["ExtJointList"] as Hashtable);
            }

            if (table.ContainsKey("ContactList"))
            {
                Hashtable contact_table = table["ContactList"] as Hashtable;
                if (contact_table != null)
                {
                    ContactList = new ContactListUpdateInfo(contact_table);
                }
            }

            if (table.ContainsKey("HelperInfo"))
            {
                HelperInfo = new HelperUpdateInfo(table["HelperInfo"] as Hashtable);
            }
            if (table.ContainsKey("ArrowList"))
            {
                ArrowList = new ArrowListUpdateInfo(table["ArrowList"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["Environment"] = Environment.ToHashTable();
            table["CharacterList"] = CharacterList.ToHashTable();
            if (ExtJointList != null)
            {
                table["ExtJointList"] = ExtJointList.ToHashTable();
            }

            // Ignore Contact joint when export...
            if (HelperInfo != null)
            {
                table["HelperInfo"] = HelperInfo.ToHashTable();
            }
            if (ArrowList != null)
            {
                table["ArrowList"] = ArrowList.ToHashTable();
            }
            return table;
        }
    }
}

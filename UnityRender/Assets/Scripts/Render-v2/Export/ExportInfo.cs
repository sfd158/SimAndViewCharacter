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
    [Serializable]
    public class DExtForceExportInfo: ISupportToHashTable
    {
        public int CharacterID;
        public int BodyID;
        public float[] Position; // force Position
        public float[] Force;

        public DExtForceExportInfo()
        {
            CharacterID = BodyID = 0;
            Position = Force = null;
        }

        public DExtForceExportInfo(Hashtable table)
        {
            CharacterID = Convert.ToInt32(table["CharacterID"]);
            BodyID = Convert.ToInt32(table["BodyID"]);
            Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
            Force = Utils.ArrayListToFloatArray(table["Force"] as ArrayList);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["CharacterID"] = CharacterID;
            table["BodyID"] = BodyID;
            table["Position"] = new ArrayList(Position);
            table["Force"] = new ArrayList(Force);

            return table;
        }

        public override string ToString()
        {
            return "ExtForceExportInfo: CharacterID = " + CharacterID + " BodyID = " + BodyID;
        }
    }

    [Serializable]
    public class DExtForceListExportInfo: ISupportToHashTable
    {
        public DExtForceExportInfo[] Forces;
        public DExtForceListExportInfo()
        {
            Forces = null;
        }

        public DExtForceListExportInfo(Hashtable table)
        {
            ArrayList arr = table["Forces"] as ArrayList;
            Forces = new DExtForceExportInfo[arr.Count];
            for(int i=0; i<arr.Count; i++)
            {
                Forces[i] = new DExtForceExportInfo(arr[i] as Hashtable);
            }
        }

        public DExtForceListExportInfo(List<DExtForceExportInfo> res)
        {
            Forces = res.ToArray();
        }

        public DExtForceListExportInfo(DExtForceExportInfo[] res)
        {
            Forces = res;
        }

        public int Count
        {
            get
            {
                if (Forces == null) return 0;
                else return Forces.Length;
            }
        }

        public int Length
        {
            get
            {
                if (Forces == null) return 0;
                else return Forces.Length;
            }
        }

        public Hashtable ToHashTable()
        {
            if (Forces == null || Forces.Length == 0)
            {
                return null;
            }

            Hashtable table = new Hashtable();
            ArrayList arr = new ArrayList(Forces.Length);
            for(int i=0; i<Forces.Length; i++)
            {
                arr.Add(Forces[i].ToHashTable());
            }
            table["Forces"] = arr;
            return table;
        }
    }

    [Serializable]
    public class DGeomExportInfo : IComparable<DGeomExportInfo>, ISupportToHashTable
    {
        public int GeomID = 0;
        public string Name = "geom";
        public string GeomType; // Geometry Type
        public bool Collidable = true;
        public float Friction;
        public float Restitution;
        public bool ClungEnv = false;
        public float[] Position;
        public float[] Quaternion; // (x, y, z, w)
        public float[] Scale;

        public int CompareTo(DGeomExportInfo other)
        {
            return GeomID.CompareTo(other.GeomID);
        }

        public DGeomExportInfo()
        {

        }

        public DGeomExportInfo(Hashtable table)
        {
            GeomID = Convert.ToInt32(table["GeomID"]);
            Name = table["Name"] as string;
            GeomType = table["GeomType"] as string;
            Collidable = Convert.ToBoolean(table["Collidable"]);
            Friction = Convert.ToSingle(table["Friction"]);
            Restitution = Convert.ToSingle(table["Restitution"]);
            ClungEnv = Convert.ToBoolean(table["ClungEnv"]);
            Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
            Quaternion = Utils.ArrayListToFloatArray(table["Quaternion"] as ArrayList);
            Scale = Utils.ArrayListToFloatArray(table["Scale"] as ArrayList);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["GeomID"] = GeomID;
            table["Name"] = Name;
            table["GeomType"] = GeomType;
            table["Collidable"] = Collidable;
            table["Friction"] = Friction;
            table["Restitution"] = Restitution;
            table["ClungEnv"] = ClungEnv;
            table["Position"] = new ArrayList(Position);
            table["Quaternion"] = new ArrayList(Quaternion);
            table["Scale"] = new ArrayList(Scale);
            return table;
        }
    }

    [Serializable]
    public class DBodyExportInfo : IComparable<DBodyExportInfo>, ISupportToHashTable
    {
        public int BodyID;
        public string Name;
        public string MassMode;
        public float Density; // Density of body
        public float Mass;

        public string InertiaMode;
        public float[] Inertia;

        public int ParentJointID;
        public int ParentBodyID;
        public float[] Position;
        public float[] Quaternion; // x, y, z, w
        public float[] LinearVelocity; // initial linear velocity in global coordinate
        public float[] AngularVelocity; // initial angular velocity in global coordinate
        public DGeomExportInfo[] Geoms;
        public int[] IgnoreBodyID;

        public int CompareTo(DBodyExportInfo other)
        {
            return BodyID.CompareTo(other.BodyID);
        }

        public DBodyExportInfo()
        {

        }

        public DBodyExportInfo(Hashtable table)
        {
            BodyID = Convert.ToInt32(table["BodyID"]);
            Name = table["Name"] as string;
            MassMode = table["MassMode"] as string;
            if (table.ContainsKey("Density"))
            {
                Density = Convert.ToSingle(table["Density"]);
            }
            else
            {
                Density = 0.0F;
            }

            if (table.ContainsKey("Mass"))
            {
                Mass = Convert.ToSingle(table["Mass"]);
            }
            else
            {
                Mass = 0.0F;
            }

            InertiaMode = table["InertiaMode"] as string;
            if (table.ContainsKey("Inertia"))
            {
                Inertia = Utils.ArrayListToFloatArray(table["Inertia"] as ArrayList);
            }

            ParentJointID = Convert.ToInt32(table["ParentJointID"]);
            ParentBodyID = Convert.ToInt32(table["ParentBodyID"] as ArrayList);
            Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
            Quaternion = Utils.ArrayListToFloatArray(table["Quaternion"] as ArrayList);

            if (table.ContainsKey("LinearVelocity"))
            {
                LinearVelocity = Utils.ArrayListToFloatArray(table["LinearVelocity"] as ArrayList);
            }
            if (table.ContainsKey("AngularVelocity"))
            {
                AngularVelocity = Utils.ArrayListToFloatArray(table["AngularVelocity"] as ArrayList);
            }

            ArrayList geomList = table["Geoms"] as ArrayList;
            Geoms = new DGeomExportInfo[geomList.Count];
            for(int i=0; i<Geoms.Length; i++)
            {
                Geoms[i] = new DGeomExportInfo(geomList[i] as Hashtable);
            }

            if (table.ContainsKey("IgnoreBodyID"))
            {
                IgnoreBodyID = Utils.ArrayListToIntArray(table["IgnoreBodyID"]);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["BodyID"] = BodyID;
            table["Name"] = Name;

            table["MassMode"] = MassMode;
            table["Density"] = Density;
            table["Mass"] = Mass;

            table["InertiaMode"] = InertiaMode;
            table["Inertia"] = new ArrayList(Inertia);

            table["ParentJointID"] = ParentJointID;
            table["ParentBodyID"] = ParentBodyID;
            table["Position"] = new ArrayList(Position);
            table["Quaternion"] = new ArrayList(Quaternion);

            if (LinearVelocity != null)
            {
                table["LinearVelocity"] = new ArrayList(LinearVelocity);
            }

            if (AngularVelocity != null)
            {
                table["AngularVelocity"] = new ArrayList(AngularVelocity);
            }

            ArrayList geomList = new ArrayList(Geoms.Length);
            for(int i=0; i<Geoms.Length; i++)
            {
                geomList.Add(Geoms[i].ToHashTable());
            }
            table["Geoms"] = geomList;

            if (IgnoreBodyID != null && IgnoreBodyID.Length > 0)
            {
                table["IgnoreBodyID"] = new ArrayList(IgnoreBodyID);
            }

            return table;
        }

        public int NumGeoms
        {
            get
            {
                if (Geoms == null) return 0;
                else return Geoms.Length;
            }
        }
    };

    [Serializable]
    public class DRootExportInfo: ISupportToHashTable
    {
        public float[] Position;
        public float[] Quaternion;
        public DRootExportInfo()
        {

        }

        public DRootExportInfo(Hashtable table)
        {
            if (table.ContainsKey("Position"))
            {
                Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
            }
            if (table.ContainsKey("Quaternion"))
            {
                Quaternion = Utils.ArrayListToFloatArray(table["Quaternion"] as ArrayList);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            if (Position != null)
            {
                table["Position"] = Position;
            }
            if (Quaternion != null)
            {
                table["Quaternion"] = Quaternion;
            }
            return table;
        }
    }

    [Serializable]
    public class DJointExportInfoBase : IComparable<DJointExportInfo>, ISupportToHashTable
    {
        public int JointID;
        public string Name;
        public string JointType;

        public float Damping;
        public float Weight;

        public float[] Position;
        public float[] Quaternion; // (x, y, z, w), same order in Unity Quaternion

        public float[] AngleLoLimit;
        public float[] AngleHiLimit;

        public string EulerOrder;
        public float[] EulerAxisLocalRot; // Joint Euler Axis.

        public int CompareTo(DJointExportInfo other)
        {
            return JointID.CompareTo(other.JointID);
        }

        public DJointExportInfoBase()
        {

        }

        public DJointExportInfoBase(DJointExportInfoBase rhs) // no copy
        {
            JointID = rhs.JointID;
            Name = rhs.Name;
            JointType = rhs.JointType;
            Damping = rhs.Damping;
            Weight = rhs.Weight;
            Position = rhs.Position;
            Quaternion = rhs.Quaternion;
            AngleLoLimit = rhs.AngleLoLimit;
            AngleHiLimit = rhs.AngleHiLimit;
            EulerOrder = rhs.EulerOrder;
            EulerAxisLocalRot = rhs.EulerAxisLocalRot;
        }

        public DJointExportInfoBase(Hashtable table)
        {
            JointID = Convert.ToInt32(table["JointID"]);
            Name = table["Name"] as string;
            JointType = table["JointType"] as string;

            Damping = Convert.ToSingle(table["Damping"]);
            Weight = Convert.ToSingle(table["Weight"]);

            Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
            Quaternion = Utils.ArrayListToFloatArray(table["Quaternion"] as ArrayList);

            AngleLoLimit = Utils.ArrayListToFloatArray(table["AngleLoLimit"] as ArrayList);
            AngleHiLimit = Utils.ArrayListToFloatArray(table["AngleHiLimit"] as ArrayList);

            EulerOrder = table["EulerOrder"] as string;

            if (table.ContainsKey("EulerAxisLocalRot"))
            {
                EulerAxisLocalRot = Utils.ArrayListToFloatArray(table["EulerAxisLocalRot"] as ArrayList);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["JointID"] = JointID;
            table["Name"] = Name;
            table["JointType"] = JointType;

            table["Damping"] = Damping;
            table["Weight"] = Weight;

            table["Position"] = new ArrayList(Position);
            table["Quaternion"] = new ArrayList(Quaternion);
            table["AngleLoLimit"] = new ArrayList(AngleLoLimit);
            table["AngleHiLimit"] = new ArrayList(AngleHiLimit);

            table["EulerOrder"] = EulerOrder;

            if (EulerAxisLocalRot != null)
            {
                table["EulerAxisLocalRot"] = new ArrayList(EulerAxisLocalRot);
            }

            return table;
        }
    }

    [Serializable]
    public class DJointExportInfo : DJointExportInfoBase
    {
        public int ParentBodyID;
        public int ChildBodyID;
        public int ParentJointID;

        public DJointExportInfo() : base() { }

        public DJointExportInfo(Hashtable table): base(table)
        {
            ParentBodyID = Convert.ToInt32(table["ParentBodyID"]);
            ChildBodyID = Convert.ToInt32(table["ChildBodyID"]);
            ParentJointID = Convert.ToInt32(table["ParentJointID"]);
        }

        public new Hashtable ToHashTable()
        {
            Hashtable table = base.ToHashTable();
            table["ParentBodyID"] = ParentBodyID;
            table["ChildBodyID"] = ChildBodyID;
            table["ParentJointID"] = ParentJointID;
            return table;
        }
    }

    [Serializable]
    public class DEndJointExportInfo: ISupportToHashTable
    {
        public int ParentJointID;
        public string Name;
        public float[] Position;

        public DEndJointExportInfo() { }

        public DEndJointExportInfo(Hashtable table)
        {
            ParentJointID = Convert.ToInt32(table["ParentJointID"]);
            Name = table["Name"] as string;
            Position = Utils.ArrayListToFloatArray(table["Position"] as ArrayList);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["ParentJointID"] = ParentJointID;
            table["Name"] = Name;
            table["Position"] = new ArrayList(Position);

            return table;
        }
    }

    [Serializable]
    public class DCharacterExportInfo: ISupportToHashTable
    {
        public int CharacterID;
        public string CharacterName;
        public bool SelfCollision; // self collision detection

        public bool IgnoreParentCollision = true;

        public bool IgnoreGrandpaCollision = true;

        public bool Kinematic;
        public string CharacterLabel;
        public bool HasRealRootJoint;

        public DBodyExportInfo[] Bodies;
        public DJointExportInfo[] Joints; // Optional.
        public DEndJointExportInfo[] EndJoints; // Optional. End joints.
        public PDControlExportInfo PDControlParam; // PD Controller param

        public DRootExportInfo RootInfo;
        public DCharacterExportInfo OnlyCopyNameID()
        {
            DCharacterExportInfo result = new DCharacterExportInfo{
                CharacterID=CharacterID + 0,
                CharacterName=CharacterName + "",
                SelfCollision=SelfCollision,
                IgnoreParentCollision=IgnoreParentCollision,
                IgnoreGrandpaCollision=IgnoreGrandpaCollision,
                Kinematic=Kinematic,
                CharacterLabel=CharacterLabel,
                HasRealRootJoint=HasRealRootJoint,
                Bodies=Bodies,
                Joints=Joints,
                EndJoints=EndJoints,
                PDControlParam=PDControlParam,
                RootInfo=RootInfo
            };
            return result;
        }

        public int BodyCount
        {
            get
            {
                if (Bodies == null) return 0;
                else return Bodies.Length;
            }
        }

        public int JointCount
        {
            get
            {
                if (Joints == null) return 0;
                else return Joints.Length;
            }
        }

        public int EndJointCount
        {
            get
            {
                if (EndJoints == null) return 0;
                else return EndJoints.Length;
            }
        }

        public DCharacterExportInfo() { }

        public DCharacterExportInfo(Hashtable table)
        {
            CharacterID = Convert.ToInt32(table["CharacterID"]);
            if (table.ContainsKey("CharacterName"))
            {
                CharacterName = table["CharacterName"] as string;
            }
            if (table.ContainsKey("SelfCollision"))
            {
                SelfCollision = Convert.ToBoolean(table["SelfCollision"]);
            }
            if (table.ContainsKey("IgnoreParentCollision"))
            {
                IgnoreParentCollision = Convert.ToBoolean(table["IgnoreParentCollision"]);
            }
            if (table.ContainsKey("IgnoreGrandpaCollision"))
            {
                IgnoreGrandpaCollision = Convert.ToBoolean(table["IgnoreGrandpaCollision"]);
            }
            if (table.ContainsKey("Kinematic"))
            {
                Kinematic = Convert.ToBoolean(table["Kinematic"]);
            }
            if (table.ContainsKey("CharacterLabel"))
            {
                CharacterLabel = table["CharacterLabel"] as string;
            }
            HasRealRootJoint = Convert.ToBoolean(table["HasRealRootJoint"]);

            ArrayList bodyList = table["Bodies"] as ArrayList;
            Bodies = new DBodyExportInfo[bodyList.Count];
            for (int i = 0; i < Bodies.Length; i++)
            {
                Bodies[i] = new DBodyExportInfo(bodyList[i] as Hashtable);
            }

            if (table.ContainsKey("Joints"))
            {
                ArrayList jointList = table["Joints"] as ArrayList;
                Joints = new DJointExportInfo[jointList.Count];
                for (int i = 0; i < jointList.Count; i++)
                {
                    Joints[i] = new DJointExportInfo(jointList[i] as Hashtable);
                }
            }

            if (table.ContainsKey("EndJoints"))
            {
                ArrayList endJointList = table["EndJoints"] as ArrayList;
                EndJoints = new DEndJointExportInfo[endJointList.Count];
                for (int i = 0; i < endJointList.Count; i++)
                {
                    EndJoints[i] = new DEndJointExportInfo(endJointList[i] as Hashtable);
                }
            }

            if (table.ContainsKey("ControlParam"))
            {
                PDControlParam = new PDControlExportInfo(table["ControlParam"] as Hashtable);
            }

            if (table.ContainsKey("RootInfo"))
            {
                RootInfo = new DRootExportInfo(table["RootInfo"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["CharacterID"] = CharacterID;
            table["CharacterName"] = CharacterName ?? "";
            table["SelfCollision"] = SelfCollision;

            table["IgnoreParentCollision"] = IgnoreParentCollision;
            table["IgnoreGrandpaCollision"] = IgnoreGrandpaCollision;

            table["Kinematic"] = Kinematic;
            table["CharacterLabel"] = CharacterLabel;
            table["HasRealRootJoint"] = HasRealRootJoint;

            ArrayList bodyList = new ArrayList(Bodies.Length);
            for(int i=0; i<Bodies.Length; i++)
            {
                bodyList.Add(Bodies[i].ToHashTable());
            }
            table["Bodies"] = bodyList;

            if (Joints != null && Joints.Length > 0)
            {
                ArrayList jointList = new ArrayList(Joints.Length);
                for (int i = 0; i < Joints.Length; i++)
                {
                    jointList.Add(Joints[i].ToHashTable());
                }
                table["Joints"] = jointList;
            }

            if (EndJoints != null && EndJoints.Length > 0)
            {
                ArrayList endJointList = new ArrayList(EndJoints.Length);
                for (int i = 0; i < EndJoints.Length; i++)
                {
                    endJointList.Add(EndJoints[i].ToHashTable());
                }
                table["EndJoints"] = endJointList;
            }

            if (PDControlParam != null)
            {
                table["PDControlParam"] = PDControlParam.ToHashTable();
            }

            if (RootInfo != null)
            {
                table["RootInfo"] = RootInfo.ToHashTable();
            }
            return table;
        }
    }

    [Serializable]
    public class DCharacterListExportInfo: ISupportToHashTable
    {
        public DCharacterExportInfo[] Characters = null;
        public int Count
        {
            get
            {
                if (Characters == null) return 0;
                else return Characters.Length;
            }
        }

        public void duplicate(int count, int index = 0)
        {
            DCharacterExportInfo[] newCharacters = new DCharacterExportInfo[Characters.Length + count];
            for(int i=0; i<Characters.Length; i++)
            {
                newCharacters[i] = Characters[i];
            }
            for(int i=0; i<count; i++)
            {
                newCharacters[i + Characters.Length] = Characters[index].OnlyCopyNameID();
            }
            Characters = newCharacters;
        }
        public DCharacterListExportInfo() { }
        public DCharacterListExportInfo(Hashtable table)
        {
            if (table.ContainsKey("Characters"))
            {
                ArrayList characterList = table["Characters"] as ArrayList;
                Characters = new DCharacterExportInfo[characterList.Count];
                for (int i = 0; i < characterList.Count; i++)
                {
                    Characters[i] = new DCharacterExportInfo(characterList[i] as Hashtable);
                }
            }
        }

        public Hashtable ToHashTable()
        {
            if (Characters == null || Characters.Length == 0)
            {
                return null;
            }
            else
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
    }

    [Serializable]
    public class DEnvironmentExportInfo: ISupportToHashTable
    {
        public DGeomExportInfo[] Geoms;
        public int FloorGeomID; // Geom ID of floor. TODO: add in dEnvironment ExportInfo() and python

        public int Count
        {
            get
            {
                if (Geoms == null) return 0;
                else return Geoms.Length;
            }
        }

        public DEnvironmentExportInfo() { }

        public DEnvironmentExportInfo(Hashtable table)
        {
            ArrayList geomList = table["Geoms"] as ArrayList;
            Geoms = new DGeomExportInfo[geomList.Count];
            for(int i=0; i<geomList.Count; i++)
            {
                Geoms[i] = new DGeomExportInfo(geomList[i] as Hashtable);
            }

            if (table.ContainsKey("FloorGeomID"))
            {
                FloorGeomID = Convert.ToInt32(table["FloorGeomID"]);
            }
        }

        public Hashtable ToHashTable()
        {
            if (Geoms == null || Geoms.Length == 0)
            {
                return null;
            }
            else
            {
                Hashtable table = new Hashtable();

                ArrayList geomList = new ArrayList(Geoms.Length);
                for (int i = 0; i < Geoms.Length; i++)
                {
                    geomList.Add(Geoms[i].ToHashTable());
                }
                table["Geoms"] = geomList;

                table["FloorGeomID"] = FloorGeomID;
                return table;
            }
        }
    }

    [Serializable]
    public class ChangeAbleDWordAttr: ISupportToHashTable
    {
        public float[] Gravity; // gravity in world
        public int StepCount = 0; // simulation step executed at each frame. If set to 0, (SimulateFPS // RenderFPS) times of simulation step will be executed.
        public int RenderFPS; // RenderFPS in Unity
        public ChangeAbleDWordAttr()
        {

        }

        public ChangeAbleDWordAttr(Hashtable table)
        {
            Gravity = Utils.ArrayListToFloatArray(table["Gravity"] as ArrayList);
            if (table.ContainsKey("StepCount"))
            {
                StepCount = Convert.ToInt32(table["StepCount"]);
            }
            else
            {
                StepCount = 0;
            }
            RenderFPS = Convert.ToInt32(table["RenderFPS"]);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["Gravity"] = new ArrayList(Gravity);
            table["StepCount"] = StepCount;
            table["RenderFPS"] = RenderFPS;
            return table;
        }
    }

    [Serializable]
    public class FixedDWorldAttr: ISupportToHashTable
    {
        public int SimulateFPS; // SimulateFPS in physics engine
        public bool UseHinge; // using ball joint instead of hinge joint if UseHinge == false
        public bool UseAngleLimit;
        public bool SelfCollision; // enable/disable collision detection between geometries of same character

        public int dWorldUpdateMode;
        public FixedDWorldAttr()
        {

        }

        public FixedDWorldAttr(Hashtable table)
        {
            SimulateFPS = Convert.ToInt32(table["SimulateFPS"]);
            UseHinge = Convert.ToBoolean(table["UseHinge"]);
            UseAngleLimit = Convert.ToBoolean(table["UseAngleLimit"]);
            SelfCollision = Convert.ToBoolean(table["SelfCollision"]);
            dWorldUpdateMode = Convert.ToInt32(table["dWorldUpdateMode"]);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            table["SimulateFPS"] = SimulateFPS;
            table["UseHinge"] = UseHinge;
            table["UseAngleLimit"] = UseAngleLimit;
            table["SelfCollision"] = SelfCollision;
            table["dWorldUpdateMode"] = dWorldUpdateMode;
            return table;
        }
    }


    [Serializable]
    public class DWorldAttr : ISupportToHashTable
    {
        public FixedDWorldAttr FixedAttr;
        public ChangeAbleDWordAttr ChangeAttr;
        public DWorldAttr() { }
        public DWorldAttr(Hashtable table)
        {
            if (table.ContainsKey("FixedAttr"))
            {
                FixedAttr = new FixedDWorldAttr(table["FixedAttr"] as Hashtable);
            }
            if (table.ContainsKey("ChangeAttr"))
            {
                ChangeAttr = new ChangeAbleDWordAttr(table["ChangeAttr"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();
            if (FixedAttr != null)
            {
                table["FixedAttr"] = FixedAttr.ToHashTable();
            }
            if (ChangeAttr != null)
            {
                table["ChangeAttr"] = ChangeAttr.ToHashTable();
            }
            return table;
        }
    }

    [Serializable]
    public class DWorldExportInfo: ISupportToHashTable
    {
        public DWorldAttr WorldAttr; // World Attributes
        public DEnvironmentExportInfo Environment;
        public DCharacterListExportInfo CharacterList;
        public ExtJointListExportInfo ExtJointList;
        public DExtForceListExportInfo ExtForceList;
        public ArrowListUpdateInfo ArrowList;

        public DWorldExportInfo() { }

        public DWorldExportInfo(Hashtable table)
        {
            WorldAttr = new DWorldAttr(table["WorldAttr"] as Hashtable);
            Environment = new DEnvironmentExportInfo(table["Environment"] as Hashtable);
            CharacterList = new DCharacterListExportInfo(table["CharacterList"] as Hashtable);

            if (table.ContainsKey("ExtJointList"))
            {
                ExtJointList = new ExtJointListExportInfo(table["ExtJointList"] as Hashtable);
            }

            if (table.ContainsKey("ExtForceList"))
            {
                ExtForceList = new DExtForceListExportInfo(table["ExtForceList"] as Hashtable);
            }

            if (table.ContainsKey("ArrowList"))
            {
                ArrowList = new ArrowListUpdateInfo(table["ArrowList"] as Hashtable);
            }
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable();

            table["WorldAttr"] = WorldAttr.ToHashTable();

            if (Environment != null && Environment.Count > 0)
            {
                table["Environment"] = Environment.ToHashTable();
            }

            if (CharacterList != null && CharacterList.Count > 0)
            {
                table["CharacterList"] = CharacterList.ToHashTable();
            }

            if (ExtJointList != null && ExtJointList.Count > 0)
            {
                table["ExtJointList"] = ExtJointList.ToHashTable();
            }

            if (ExtForceList != null && ExtForceList.Count > 0)
            {
                table["ExtForceList"] = ExtForceList.ToHashTable();
            }

            if (ArrowList != null)
            {
                table["ArrowList"] = ArrowList.ToHashTable();
            }
            return table;
        }
    }

    [Serializable]
    public class PDControlExportInfo: ISupportToHashTable
    {
        public float[] Kps; // Kp parameter for each joint
        public float[] TorqueLimit; // Torque Limit of each joint

        public PDControlExportInfo() { }

        public PDControlExportInfo(Hashtable table)
        {
            Kps = Utils.ArrayListToFloatArray(table["Kps"] as ArrayList);
            TorqueLimit = Utils.ArrayListToFloatArray(table["TorqueLimit"] as ArrayList);
        }

        public Hashtable ToHashTable()
        {
            Hashtable table = new Hashtable
            {
                ["Kps"] = new ArrayList(Kps),
                ["TorqueLimit"] = new ArrayList(TorqueLimit)
            };

            return table;
        }
    }

    [Serializable]
    public class ExtJointExportInfo: DJointExportInfoBase
    {
        public int Character0ID;
        public int Body0ID;
        public int Character1ID;
        public int Body1ID;

        public ExtJointExportInfo() { }

        public ExtJointExportInfo(DJointExportInfoBase rhs): base(rhs)
        {

        }

        public ExtJointExportInfo(Hashtable table): base(table)
        {
            Character0ID = Convert.ToInt32(table["Character0ID"]);
            Body0ID = Convert.ToInt32(table["Body0ID"]);
            Character1ID = Convert.ToInt32(table["Character1ID"]);
            Body1ID = Convert.ToInt32(table["Body1ID"]);
        }

        public new Hashtable ToHashTable()
        {
            Hashtable table = base.ToHashTable();

            table["Character0ID"] = Character0ID;
            table["Body0ID"] = Body0ID;
            table["Character1ID"] = Character1ID;
            table["Body1ID"] = Body1ID;

            return table;
        }
    }

    [Serializable]
    public class ExtJointListExportInfo: ISupportToHashTable
    {
        public ExtJointExportInfo[] Joints;

        public int Count
        {
            get
            {
                if (Joints == null) return 0;
                else return Joints.Length;
            }
        }

        public ExtJointListExportInfo() { }

        public ExtJointListExportInfo(Hashtable table)
        {
            if (table.ContainsKey("Joints"))
            {
                ArrayList JointsArray = table["Joints"] as ArrayList;
                Joints = new ExtJointExportInfo[JointsArray.Count];
                for (int i = 0; i < JointsArray.Count; i++)
                {
                    Joints[i] = new ExtJointExportInfo(JointsArray[i] as Hashtable);
                }
            }
        }

        public Hashtable ToHashTable()
        {
            if (Joints == null || Joints.Length == 0)
            {
                return null;
            }
            else
            {
                Hashtable table = new Hashtable();

                ArrayList JointsArray = new ArrayList(Joints.Length);
                for (int i = 0; i < Joints.Length; i++)
                {
                    JointsArray.Add(Joints[i].ToHashTable());
                }
                table["Joints"] = JointsArray;

                return table;
            }
        }
    }
}

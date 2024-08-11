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

namespace RenderV1
{
    using System.IO;
    using System.Net.Sockets;
    using System.Collections.Generic;
    using UnityEngine;
    using UnityEditor;
    using Razorvine.Pickle;
    using System.Collections;
    using System;

    public class RecieveODEDbg : MonoBehaviour
    {
        public string IPAddr = "localhost";
        public int IPPort = 8888;
        public int fps = 30;
        public float jointRadius = 0.025F;

        NetworkStream MyNetStream;
        BinaryWriter MySocketWriter;
        BinaryReader MySocketReader;
        TcpClient MyTcpSocket;

        Dictionary<ulong, GameObject> ObjDict = new Dictionary<ulong, GameObject>();
        Transform BaseTrans;
        MyInfoType UpdateInfo;

        public void ClearInfo()
        {
            BaseTrans = GetComponent<Transform>();
            List<GameObject> buf = new List<GameObject>();
            int head = 0;
            buf.Add(BaseTrans.gameObject);
            while (head < buf.Count)
            {
                GameObject obj = buf[head++];
                for (int i = 0; i < obj.transform.childCount; i++)
                {
                    buf.Add(obj.transform.GetChild(i).gameObject);
                }
            }
            for (int i = buf.Count - 1; i > 0; i--)
            {
                DestroyImmediate(buf[i]);
            }
            ObjDict.Clear();
        }

        public Dictionary<string, object> GetHierarchyInfoStr()
        {
            Dictionary<string, object> res = new Dictionary<string, object>
        {
            { "type", "GetHierarchyInfo" },
            { "JointRadius", jointRadius }
        };
            return res;
            //return JsonUtility.ToJson(new Serialization<string, string>(res)).Replace("\n", "");
        }

        public Hashtable GetMessageFromServer()
        {
            // Format of information:
            //      4 bytes, bufLen, length of buffer 
            // bufLen bytes, result, data with dict format in python

            int bufLen = MySocketReader.ReadInt32();
            byte[] buf = MySocketReader.ReadBytes(bufLen);
            var unpickler = new Unpickler();
            Hashtable result = unpickler.loads(buf) as Hashtable;
            return result;
        }

        public void SendMessageToServer(object message)
        {
            // Format of information:
            //      4 bytes, bufLen, length of buffer
            // bufLen bytes, message

            var pickler = new Pickler(true);
            byte[] sendInfo = pickler.dumps(message); // unpickle result is Hashtable
            byte[] bufLen = BitConverter.GetBytes(Convert.ToInt32(sendInfo.Length));
            MySocketWriter.Write(bufLen);
            MySocketWriter.Write(sendInfo);
            MySocketWriter.Flush();
        }

        public void ODEConnect()
        {
            ClearInfo();
            try
            {
                MyTcpSocket = new TcpClient(IPAddr, IPPort);
                MyNetStream = MyTcpSocket.GetStream();
                MySocketWriter = new BinaryWriter(MyNetStream);
                MySocketReader = new BinaryReader(MyNetStream);

                Debug.Log("Connect");
                SendMessageToServer(GetHierarchyInfoStr());
                GetInfoFromServer();
            }
            catch (IOException e)
            {
                Debug.LogError(e);
                EditorApplication.isPlaying = false;
            }
        }

        public void ODEClose()
        {
            ClearInfo();
            if (MySocketWriter != null)
            {
                MySocketWriter.Close();
            }
            if (MySocketReader != null)
            {
                MySocketReader.Close();
            }
            if (MyTcpSocket != null)
            {
                MyTcpSocket.Close();
            }
            Debug.Log("Close");
        }

        public void ODEStep()
        {
            try
            {
                GetInfoFromServer();
            }
            catch (IOException e)
            {
                Debug.LogError(e);
                EditorApplication.isPlaying = false;
            }
        }

        void Awake()
        {
            Application.targetFrameRate = fps;
            Application.runInBackground = true;
            ODEConnect();
        }

        private void OnApplicationQuit()
        {
            ODEClose();
        }

        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            ODEStep();
        }

        Vector3 ArrToVec3(float[] arr, int offset)
        {
            return new Vector3(arr[offset], arr[offset + 1], arr[offset + 2]);
        }

        Quaternion ArrToQuat(float[] arr, int offset)
        {
            return new Quaternion(arr[offset], arr[offset + 1], arr[offset + 2], arr[offset + 3]);
        }

        void CreateGameObject()
        {
            if (UpdateInfo.CreateType is null)
            {
                return;
            }

            int cnt = UpdateInfo.CreateType.Length;
            if (cnt != UpdateInfo.CreateID.Length || 3 * cnt != UpdateInfo.CreateScale.Length || 3 * cnt != UpdateInfo.CreatePos.Length || 4 * cnt != UpdateInfo.CreateQuat.Length)
            {
                Debug.LogWarning("Length Not Match");
                return;
            }

            for (int i = 0; i < cnt; i++)
            {
                PrimitiveType res = (PrimitiveType)UpdateInfo.CreateType[i];
                ulong id = UpdateInfo.CreateID[i];
                if (ObjDict.ContainsKey(id))
                {
                    Debug.LogWarning("Create GameObject: ID " + id.ToString() + "exist. Ignore.");
                    continue;
                }

                GameObject obj;
                if (res != PrimitiveType.Plane)
                {
                    obj = GameObject.CreatePrimitive(res);
                }
                else
                {
                    obj = GameObject.CreatePrimitive(PrimitiveType.Cube);
                }

                Vector3 scale = ArrToVec3(UpdateInfo.CreateScale, 3 * i);
                switch (res)
                {
                    case PrimitiveType.Sphere:  // (radius, radius, radius)
                        obj.transform.localScale = 2 * scale;
                        break;
                    case PrimitiveType.Capsule: // (radius, length, radius)
                        obj.transform.localScale = new Vector3(2 * scale[0], scale[0] + 0.5F * scale[1], 2 * scale[2]);
                        break;
                    case PrimitiveType.Cylinder:
                        obj.transform.localScale = new Vector3(2 * scale[0], 0.5F * scale[1], 2 * scale[2]);
                        break;
                    case PrimitiveType.Cube:
                        obj.transform.localScale = scale;
                        break;
                    case PrimitiveType.Plane:
                        obj.transform.localScale = scale;
                        break;
                    default:
                        break;
                }

                // Add an empty parent.
                GameObject trans_pa = new GameObject();
                trans_pa.transform.position = ArrToVec3(UpdateInfo.CreatePos, 3 * i);
                trans_pa.transform.parent = BaseTrans;
                trans_pa.transform.localRotation = ArrToQuat(UpdateInfo.CreateQuat, 4 * i);

                obj.transform.parent = trans_pa.transform;
                obj.transform.position = ArrToVec3(UpdateInfo.CreatePos, 3 * i);
                obj.transform.localRotation = ArrToQuat(UpdateInfo.CreateChildQuat, 4 * i);

                if (UpdateInfo.CreateName[i].Length > 0)
                {
                    obj.name = UpdateInfo.CreateName[i];
                }
                trans_pa.name = obj.name + "TransPa";

                MeshRenderer render = obj.GetComponent<MeshRenderer>();
                render.sharedMaterial.color = new Color(UpdateInfo.CreateColor[3 * i], UpdateInfo.CreateColor[3 * i + 1], UpdateInfo.CreateColor[3 * i + 2]);
                // render.material.color
                //Debug.Log(render.material.color);
                render.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                ObjDict.Add(id, obj);
            }
        }

        void ModifyGameObject()
        {
            if (UpdateInfo.ModifyID is null)
            {
                return;
            }

            int cnt = UpdateInfo.ModifyID.Length;
            if (3 * cnt != UpdateInfo.ModifyPos.Length || 4 * cnt != UpdateInfo.ModifyQuat.Length)
            {
                return;
            }

            for (int i = 0; i < cnt; i++)
            {
                ulong id = UpdateInfo.ModifyID[i];
                if (!ObjDict.ContainsKey(id))
                {
                    Debug.LogWarning("Modify GameObject: ID " + id.ToString() + "not exists. Ignore.");
                    continue;
                }

                GameObject obj = ObjDict[id];
                GameObject trans_pa = obj.transform.parent.gameObject;

                obj.transform.position = ArrToVec3(UpdateInfo.ModifyPos, 3 * i);
                trans_pa.transform.position = ArrToVec3(UpdateInfo.ModifyPos, 3 * i);
                trans_pa.transform.rotation = ArrToQuat(UpdateInfo.ModifyQuat, 4 * i);
            }
        }

        void RemoveGameObject()
        {
            if (UpdateInfo.RemoveID is null)
            {
                return;
            }
            for (int i = 0; i < UpdateInfo.RemoveID.Length; i++)
            {
                ulong id = UpdateInfo.RemoveID[i];
                if (!ObjDict.ContainsKey(id))
                {
                    Debug.LogWarning("Remove GameObject: ID " + id.ToString() + "not exists. Ignore.");
                    continue;
                }
                GameObject obj = ObjDict[id];
                ObjDict.Remove(id);
                DestroyImmediate(obj.transform.parent.gameObject);
                DestroyImmediate(obj); // TODO:Test
            }
        }

        Dictionary<string, object> GetUpdateSendMess()
        {
            Dictionary<string, object> mess = new Dictionary<string, object>
        {
            { "type", "GetUpdateInfo" },
            { "JointRadius", jointRadius }
        };
            return mess;
            // return JsonUtility.ToJson(new Serialization<string, string>(mess)).Replace("\n", "");
        }

        ulong[] ArrayListToULongArray(object rhs)
        {
            ArrayList arr = rhs as ArrayList;
            ulong[] res = new ulong[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = Convert.ToUInt64(arr[i]);
            }
            return res;
        }

        int[] ArrayListToIntArray(object rhs)
        {
            ArrayList arr = rhs as ArrayList;
            int[] res = new int[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = Convert.ToInt32(arr[i]);
            }
            return res;
        }

        float[] ArrayListToFloatArray(object rhs)
        {
            ArrayList arr = rhs as ArrayList;
            float[] res = new float[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = Convert.ToSingle(arr[i]);
            }
            return res;
        }

        string[] ArrayListToStringArray(object rhs)
        {
            ArrayList arr = rhs as ArrayList;
            string[] res = new string[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = arr[i] as string;
            }
            return res;
        }

        void GetInfoFromServer()
        {
            try
            {
                SendMessageToServer(GetUpdateSendMess());
                Hashtable result = GetMessageFromServer();

                UpdateInfo = new MyInfoType
                {
                    CreateID = ArrayListToULongArray(result["CreateID"]),
                    CreateType = ArrayListToIntArray(result["CreateType"]),
                    CreateScale = ArrayListToFloatArray(result["CreateScale"]),
                    CreatePos = ArrayListToFloatArray(result["CreatePos"]),
                    CreateQuat = ArrayListToFloatArray(result["CreateQuat"]),
                    CreateChildQuat = ArrayListToFloatArray(result["CreateChildQuat"]),
                    CreateName = ArrayListToStringArray(result["CreateName"]),
                    CreateColor = ArrayListToFloatArray(result["CreateColor"]),
                    ModifyID = ArrayListToULongArray(result["ModifyID"]),
                    ModifyPos = ArrayListToFloatArray(result["ModifyPos"]),
                    ModifyQuat = ArrayListToFloatArray(result["ModifyQuat"]),
                    RemoveID = ArrayListToULongArray(result["RemoveID"])
                };

                // UpdateInfo = JsonUtility.FromJson<MyInfoType>(recieve);

                CreateGameObject();
                ModifyGameObject();
                RemoveGameObject();
            }
            catch (IOException e)
            {
                Debug.Log(e);
                ODEClose();
            }
        }
    }

}
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
using System.IO;
using System.Net.Sockets;
using UnityEditor;
using UnityEngine;
using Razorvine.Pickle;

namespace RenderV2
{
    /// <summary>
    /// Update the world by Server
    /// </summary>
    public class UpdateByServer : UpdateBase
    {
        [Tooltip("Server IP Address")]
        public string IPAddr = "localhost";

        [Tooltip("Server IP Port")]
        public int IPPort = 8888;

        NetworkStream netStream;
        BinaryWriter socketWriter;
        BinaryReader socketReader;
        TcpClient tcpSocket; // communicate by TCP socket

        DWorld dWorld;

        void Start()
        {
            ConnectToServer();
        }

        private void FixedUpdate()
        {
            if (UpdateMode == MonoUpdateMode.FixedUpdate)
            {
                DWorldStep();
            }
        }

        private void Update()
        {
            if (UpdateMode == MonoUpdateMode.Update)
            {
                DWorldStep();
            }
        }

        private void LateUpdate()
        {
            if (UpdateMode == MonoUpdateMode.LateUpdate)
            {
                DWorldStep();
            }
        }

        private void OnApplicationQuit()
        {
            CloseServer();
        }

        /// <summary>
        /// Get pickled message from python server
        /// </summary>
        /// <returns>Hashtable</returns>
        protected Hashtable GetMessageFromServer()
        {
            // Format of information:
            //      4 bytes, bufLen, length of buffer
            // bufLen bytes, result, data with dict format in python

            Hashtable result = null;
            try
            {
                int bufLen = socketReader.ReadInt32(); // length of buffer
                byte[] buf = socketReader.ReadBytes(bufLen); // recieve buffer
                var unpickler = new Unpickler();
                result = unpickler.loads(buf) as Hashtable;
            }
            catch(Exception e)
            {
                Debug.LogError(e);
                EditorApplication.isPlaying = false;
            }

            return result;
        }

        /// <summary>
        /// Send Hierarchy / Update infomation to Server
        /// </summary>
        /// <param name="message"></param>
        protected void SendMessageToServer(ISupportToHashTable message)
        {
            // Format of information:
            //      4 bytes, bufLen, length of buffer
            // bufLen bytes, message

            try
            {
                var pickler = new Pickler(true);
                byte[] sendInfo = pickler.dumps(message.ToHashTable()); // unpickle result is Hashtable
                byte[] bufLen = BitConverter.GetBytes(Convert.ToInt32(sendInfo.Length)); // length of send information
                socketWriter.Write(bufLen);
                socketWriter.Write(sendInfo);
            }
            catch(Exception e)
            {
                Debug.LogError(e);
                EditorApplication.isPlaying = false;
            }
        }

        /// <summary>
        /// simulate in server, then get update info from server
        /// </summary>
        public override void DWorldStep()
        {
            DUpdateSendInfo updateSendInfo = dWorld.UpdateSendInfoWithPostProcess();
            SendMessageToServer(updateSendInfo);
            Hashtable result = GetMessageFromServer();
            UpdateRecieveInfo updateInfo = new UpdateRecieveInfo(result);
            dWorld.ParseUpdateRecieveInfo(updateInfo); // Parse Update Information
            dWorld.PostWorldStep();
        }

        /// <summary>
        /// Connect to server when start
        /// </summary>
        public void ConnectToServer()
        {
            dWorld = GetComponent<DWorld>();
            DCommonConfig.dWorldUpdateMode = dWorld.dWorldUpdateMode;
            try
            {
                tcpSocket = new TcpClient(IPAddr, IPPort);
                netStream = tcpSocket.GetStream();
                socketWriter = new BinaryWriter(netStream);
                socketReader = new BinaryReader(netStream);
                Debug.Log("Connect to Server " + IPAddr + ":" + IPPort.ToString());
            }
            catch (Exception e)
            {
                Debug.LogError(e);
                EditorApplication.isPlaying = false;
                EditorUtility.DisplayDialog("Connect Failed", "Connect Failed. Please check the server", "OK", "Cancel");
                // Application.Quit(1);
            }

            SendHierarchyInfo();
        }

        /// <summary>
        /// Support:
        /// 1. duplicate character in Unity, driven by python server
        /// 2. TODO: add offset to characters
        /// </summary>
        void GetInstructionInfoFromServer(out int DupCount, out int loadCount)
        {
            InitialInstructionSendInfo sendInfo = new InitialInstructionSendInfo();
            SendMessageToServer(sendInfo);
            Hashtable result = GetMessageFromServer();
            InstructionRecieveInfo instruction = new InstructionRecieveInfo(result);
            // Note: here we should add json file name to load...
            if (!instruction.CheckOK())
            {
                throw new IOException("Error in getting instruction from python server");
            }

            DupCount = instruction.NumDuplicateCharacter;
            DCharacterList dCharacterList = this.dWorld.dCharacterList;

            // duplicate characters
            if (DupCount > 0)
            {
                Debug.Log("Load duplicate character");
                DCharacter dCharacter0 = dCharacterList.transform.GetChild(0).GetComponent<DCharacter>();
                if (dCharacter0 == null)
                {
                    throw new ArgumentNullException("There is no character");
                }
                GameObject character0 = dCharacter0.gameObject;
                character0.name = instruction.DupCharacterNames[0];

                // duplicate characters
                for (int dup_idx = 1; dup_idx < DupCount; dup_idx++)
                {
                    GameObject characterDup = Instantiate(character0);
                    DCharacter dCharacterDup = characterDup.GetComponent<DCharacter>();
                    dCharacterDup.IDNum = dCharacterDup.gameObject.GetInstanceID();
                    characterDup.transform.parent = dCharacterList.transform;
                    characterDup.name = instruction.DupCharacterNames[dup_idx];

                    // set render color for different characters
                    if (DCommonConfig.SupportGameObjectColor)
                    {
                        var renderObjs = characterDup.GetComponentsInChildren<Renderer>(true);
                        Color objColor = new Color(
                            UnityEngine.Random.Range(0f, 1f),
                            UnityEngine.Random.Range(0f, 1f),
                            UnityEngine.Random.Range(0f, 1f)
                        );
                        foreach (var obj in renderObjs)
                        {
                            obj.material.color = objColor;
                        }
                    }
                    else
                    {
                        // Not use set color here..
                    }
                }
            }

            // Here we should load characters from json scene..
            // we should use abs file path here...
            loadCount = instruction.NumLoadCharacter;
            if (loadCount > 0)
            {
                var loader = new JsonCharacterLoader(this.dWorld);
                for(int i = 0; i < loadCount; i++)
                {
                    Debug.Log("Load character from file " + instruction.LoadCharacterFileName[i]);
                    DCharacter character = loader.LoadJsonFromFile(instruction.LoadCharacterFileName[i]);
                    if (character == null)
                    {
                        continue;
                    }
                    // Debug.Log(character.IDNum);
                    dCharacterList.TObjectDict.Add(character.IDNum, character);
                    // do we need to update scene here...?
                    // No, the scene is re-computed after.

                    if (i == 0)
                    {
                        // set the camera..
                        DCameraController cameraControl = FindObjectOfType<DCameraController>();
                        cameraControl.target = character.transform;
                        cameraControl.targetDirTransform = character.transform;
                    }
                }
            }

            Debug.Log("Num of children in CharacterList is " + dCharacterList.transform.childCount);
            dWorld.CalcAttrs();
        }

        /// <summary>
        /// Send Initial Hierarchy infomation to Server
        /// </summary>
        protected void SendHierarchyInfo()
        {
            // TODO: recieve instruction info from server
            DWorldExportInfo exportInfo = dWorld.ExportInfoWithPostProcess();
            GetInstructionInfoFromServer(out int dup_count, out int load_count);
            if (dup_count > 1)
            {
                exportInfo.CharacterList.duplicate(dup_count - 1);
                for(int dup_idx = 0; dup_idx < dup_count; dup_idx++)
                {
                    var info = exportInfo.CharacterList.Characters[dup_idx];
                    var create_buf = dWorld.dCharacterList.CreateBuffer[dup_idx];
                    info.CharacterName = create_buf.name;
                    info.CharacterID = create_buf.IDNum;
                }
            }

            dWorld.dCharacterList.PostExportInPlaying();
            Debug.Log("Find " + exportInfo.CharacterList.Characters.Length + " characters in exportInfo");
            HierarchySendInfo sendInfo = new HierarchySendInfo(exportInfo); // send to python server

            SendMessageToServer(sendInfo);

            Hashtable result = GetMessageFromServer();
            CheckRecieveInfo checkInfo = new CheckRecieveInfo(result);  // recieve from python server
            if (!checkInfo.CheckOK())
            {
                throw new IOException("Set Hierarchy Wrong. ");
            }
        }

        /// <summary>
        /// close connection to server when exiting
        /// </summary>
        private void CloseServer()
        {
            if (socketWriter != null)
            {
                socketWriter.Close();
            }
            if (socketReader != null)
            {
                socketReader.Close();
            }
            if (tcpSocket != null)
            {
                tcpSocket.Close();
            }
            Debug.Log("Connect Closed.");
        }
    }
}
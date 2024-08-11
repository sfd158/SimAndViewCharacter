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
    public partial class DRigidBody
    {
        /// <summary>
        /// Create Empty Body
        /// </summary>
        /// <param name="character">body's character</param>
        /// <param name="bodyID">body id</param>
        /// <param name="bodyName">body's name</param>
        /// <param name="BodyCenter">Position of body's center</param>
        /// <returns>DRigidBody Component</returns>
        public static DRigidBody CreateBody(GameObject character, int bodyID = 0, string bodyName = "Body", Vector3 BodyCenter = new Vector3())
        {
            GameObject bodyObject = new GameObject
            {
                name = bodyName
            };

            DRigidBody drigidBody = bodyObject.AddComponent<DRigidBody>();
            drigidBody.IDNum = bodyID;

            drigidBody.IgnoreCollision = new List<GameObject>();
            drigidBody.InitialPosition = BodyCenter;
            drigidBody.character = character;
            return drigidBody;
        }

        /// <summary>
        /// Create GameObject with DRigidBody component by DBodyExportInfo
        /// </summary>
        /// <param name="info"></param>
        /// <param name="dCharacter"></param>
        /// <returns>DRigidBody Component</returns>
        public static DRigidBody CreateBody(DBodyExportInfo info, DCharacter dCharacter)
        {
            DRigidBody body = DRigidBody.CreateBody(dCharacter.gameObject, info.BodyID, info.Name, Utils.ArrToVector3(info.Position));
            body.InitialQuaternion = Utils.ArrToQuaternion(info.Quaternion);
            body.Density = info.Density;

            foreach (var geominfo in info.Geoms)
            {
                DGeomObject geom;
                switch (geominfo.GeomType)
                {
                    case "Sphere":
                    case "sphere":
                    case "Ball":
                    case "ball":
                        {
                            geom = DBallGeom.CreateGeom(body.gameObject, geominfo);
                            break;
                        }
                    case "Box":
                    case "box":
                    case "Cube":
                    case "cube":
                        {
                            geom = DBoxGeom.CreateGeom(body.gameObject, geominfo);
                            break;
                        }
                    case "Capsule":
                    case "capsule":
                    case "CCylinder":
                    case "ccylinder":
                        {
                            geom = DCapsuleGeom.CreateGeom(body.gameObject, geominfo);
                            break;
                        }
                    default:
                        throw new ArgumentException("Geometry type not supported.");
                }
            }

            return body;
        }

        /// <summary>
        /// Create body with a single ball geometry
        /// </summary>
        /// <param name="character"></param>
        /// <param name="bodyID"></param>
        /// <param name="bodyName"></param>
        /// <param name="BodyCenter"></param>
        /// <param name="BallRadius"></param>
        /// <returns></returns>
        public static DRigidBody CreateBallBody(GameObject character, int bodyID = 0, string bodyName = "BallBody", Vector3 BodyCenter = new Vector3(),
            float BallRadius = 1.0F)
        {
            DRigidBody body = CreateBody(character, bodyID, bodyName, BodyCenter);
            DBallGeom.CreateGeom(body.gameObject, 0, "BallGeom", BallRadius, BodyCenter, Quaternion.identity);
            body.InitialPosition = BodyCenter;
            return body;
        }

        /// <summary>
        /// Create body with a box geometry
        /// </summary>
        /// <param name="character"></param>
        /// <param name="bodyID"></param>
        /// <param name="bodyName"></param>
        /// <param name="BodyCenter"></param>
        /// <param name="GeomLength"></param>
        /// <returns></returns>
        public static DRigidBody CreateBoxBody(GameObject character, int bodyID = 0, string bodyName = "BoxBody", Vector3 BodyCenter = new Vector3(),
            Vector3 GeomLength = new Vector3())
        {
            DRigidBody body = CreateBody(character, bodyID, bodyName, BodyCenter);
            DBoxGeom.CreateGeom(body.gameObject, 0, "BoxGeom", GeomLength, BodyCenter, Quaternion.identity);
            return body;
        }

        /// <summary>
        /// Create body with capsule geometry
        /// </summary>
        /// <param name="character"></param>
        /// <param name="bodyID"></param>
        /// <param name="bodyName"></param>
        /// <param name="BodyCenter"></param>
        /// <param name="capsuleRadius"></param>
        /// <param name="capsuleLength"></param>
        /// <returns></returns>
        public static DRigidBody CreateCapsuleBody(GameObject character, int bodyID = 0, string bodyName = "BallBody", Vector3 BodyCenter = new Vector3(),
            float capsuleRadius = 1.0F, float capsuleLength = 1.0F)
        {
            DRigidBody body = CreateBody(character, bodyID, bodyName, BodyCenter);
            DCapsuleGeom.CreateGeom(body.gameObject, 0, "CapsuleGeom", capsuleRadius, capsuleLength, BodyCenter, Quaternion.identity);
            return body;
        }
    }
}

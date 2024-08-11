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
using UnityEngine;

namespace RenderV2
{
    // TODO: set initial linear velocity and angular velocity
    public partial class DCharacter
    {
        /// <summary>
        /// Create Empty character
        /// </summary>
        /// <param name="CharacterList">CharacterList</param>
        /// <returns>DCharacter</returns>
        public static DCharacter CreateCharacter(DCharacterList CharacterList)
        {
            GameObject Character = new GameObject
            {
                name = "DCharacter" + CharacterList.transform.childCount
            };
            Character.transform.parent = CharacterList.transform;
            DCharacter dCharacter = Character.AddComponent<DCharacter>();
            return dCharacter;
        }

        static DCharacter PreCreateCharacter(DCharacterList characterList, Vector3 BodyCenter)
        {
            DCharacter dCharacter = CreateCharacter(characterList);
            dCharacter.InitialPosition = BodyCenter;
            return dCharacter;
        }

        static DCharacter PostCreateCharacter(DCharacter dCharacter, DRigidBody body)
        {
            body.transform.parent = dCharacter.transform;
            dCharacter.CalcAttrs();
            dCharacter.SetInitialState(true);
            dCharacter.ReCompute();

            return dCharacter;
        }
        /// <summary>
        /// Create character with a ball geometry
        /// </summary>
        /// <param name="characterList"></param>
        /// <returns></returns>
        public static DCharacter CreateBallCharacter(DCharacterList characterList, Vector3 BodyCenter = default, float Radius = 1)
        {
            DCharacter dCharacter = PreCreateCharacter(characterList, BodyCenter);
            DRigidBody body = DRigidBody.CreateBallBody(dCharacter.gameObject, 0, "BallBody", BodyCenter, Radius);
            return PostCreateCharacter(dCharacter, body);
        }

        /// <summary>
        /// Create character with a box geometry
        /// </summary>
        /// <param name="characterList"></param>
        /// <returns></returns>
        public static DCharacter CreateBoxCharacter(DCharacterList characterList, Vector3 BodyCenter = default, float lx = 1, float ly = 1, float lz = 1)
        {
            DCharacter dCharacter = PreCreateCharacter(characterList, BodyCenter);
            DRigidBody body = DRigidBody.CreateBoxBody(dCharacter.gameObject, 0, "BoxBody", BodyCenter, new Vector3(lx, ly, lz));
            return PostCreateCharacter(dCharacter, body);
        }

        /// <summary>
        /// Create character with a capsule geometry
        /// </summary>
        /// <param name="characterList"></param>
        /// <returns></returns>
        public static DCharacter CreateCapsuleCharacter(DCharacterList characterList, Vector3 BodyCenter = default, float radius = 1, float length = 1)
        {
            DCharacter dCharacter = PreCreateCharacter(characterList, BodyCenter);
            DRigidBody body = DRigidBody.CreateCapsuleBody(dCharacter.gameObject, BodyCenter: BodyCenter, capsuleRadius: radius, capsuleLength: length);
            return PostCreateCharacter(dCharacter, body);
        }
    }
}

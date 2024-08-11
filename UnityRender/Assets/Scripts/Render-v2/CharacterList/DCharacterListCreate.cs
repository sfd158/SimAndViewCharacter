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
    public partial class DCharacterList
    {
        /// <summary>
        /// Add Character with a sphere geometry in CharacterList
        /// </summary>
        /// <param name="BodyCenter">new Character's initial position</param>
        /// <param name="Radius">Ball's Radius</param>
        /// <param name="AppendToCreateBuffer"></param>
        /// <returns>created new character</returns>
        public DCharacter AddSphereCharacter(Vector3 BodyCenter = default, float Radius = 1, bool AppendToCreateBuffer = false)
        {
            var res = DCharacter.CreateBallCharacter(this, BodyCenter, Radius);
            if (AppendToCreateBuffer)
            {
                CreateBufferAppend(res);
            }
            return res;
        }

        /// <summary>
        /// Add Character with a box geometry in CharacterList
        /// </summary>
        /// <param name="BodyCenter">new Character</param>
        /// <param name="lx">length x</param>
        /// <param name="ly">length y</param>
        /// <param name="lz">length z</param>
        /// <param name="AppendToCreateBuffer"></param>
        /// <returns>created new character</returns>
        public DCharacter AddBoxCharacter(Vector3 BodyCenter = default, float lx = 1, float ly = 1, float lz = 1, bool AppendToCreateBuffer = false)
        {
            var res = DCharacter.CreateBoxCharacter(this, BodyCenter, lx, ly, lz);
            if (AppendToCreateBuffer)
            {
                CreateBufferAppend(res);
            }
            return res;
        }

        /// <summary>
        /// Add Character with a Capsule Geometry in CharacterList
        /// </summary>
        /// <param name="BodyCenter"></param>
        /// <param name="radius"></param>
        /// <param name="length"></param>
        /// <param name="AppendToCreateBuffer"></param>
        /// <returns></returns>
        public DCharacter AddCapsuleCharacter(Vector3 BodyCenter = default, float radius = 1, float length = 1, bool AppendToCreateBuffer = false)
        {
            var res = DCharacter.CreateCapsuleCharacter(this, BodyCenter, radius, length);
            if (AppendToCreateBuffer)
            {
                CreateBufferAppend(res);
            }
            return res;
        }
    }
}

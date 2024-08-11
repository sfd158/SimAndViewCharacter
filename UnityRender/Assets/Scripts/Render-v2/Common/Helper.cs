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

using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    public static class Helper
    {
        public static bool IsList<T>()
        {
            return typeof(T).IsGenericType && typeof(T).GetGenericTypeDefinition() == typeof(List<>);
        }

        public static List<string> DeepCopyList(List<string> origin, int start = 0, int end = -1)
        {
            if (end == -1) end = origin.Count;
            // return origin.GetRange(start, end);
            List<string> res = new List<string>(end - start);
            for(int i = start; i < end; i++) 
                res.Add(origin[i] == null ? null : new string(origin[i].ToCharArray()));
            return res;
        }

        public static List<int> DeepCopyList(List<int> origin, int start = 0, int end = -1)
        {
            if (end == -1) end = origin.Count;
            return origin.GetRange(start, end - start);
        }

        public static List<float> DeepCopyList(List<float> origin, int start = 0, int end = -1)
        {
            if (end == -1) end = origin.Count;
            return origin.GetRange(start, end - start);
        }

        public static List<Vector3> DeepCopyList(List<Vector3> origin, int start = 0, int end = -1)
        {
            if (end == -1) end = origin.Count;
            var res = new List<Vector3>(end - start);
            for(int i = start; i < end; i++)
                res.Add(new Vector3(origin[i].x, origin[i].y, origin[i].z));
            return res;
        }

        public static List<Quaternion> DeepCopyList(List<Quaternion> origin, int start = 0, int end = -1)
        {
            if (end == -1) end = origin.Count;
            var res = new List<Quaternion>(end - start);
            for (int i = start; i < end; i++) 
                res.Add(new Quaternion(origin[i].x, origin[i].y, origin[i].z, origin[i].w));
            return res;
        }

        public static List<List<Vector3>> DeepCopyList(List<List<Vector3>> origin, int start = 0, int end = -1)
        {
            if (end == -1) end = origin.Count;
            var res = new List<List<Vector3>>(end - start);
            for (int i = start; i < end; i++)
                res.Add(DeepCopyList(origin[i], start, end));
            return res;
        }

        public static List<List<Quaternion>> DeepCopyList(List<List<Quaternion>> origin, int start = 0, int end = -1)
        {
            if (end == -1) end = origin.Count;
            var res = new List<List<Quaternion>>(origin.Count);
            for (int i = start; i < end; i++)
                res.Add(DeepCopyList(origin[i], start, end));
            return res;
        }
    }
}
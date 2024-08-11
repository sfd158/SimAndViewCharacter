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
using System.Runtime.InteropServices;
using UnityEngine;

using static RenderV2.ODE.BodyDataClass;
using static RenderV2.ODE.CommonFunc;
using Tensor = RenderV2.Eigen.Tensor;

namespace RenderV2.ODE
{

    public class ChatacterState
    {
        public Tensor pos;
        ChatacterState()
        {

        }
    }

    public class ControlVAEObs
    {
        /*
        pos = obs[...,0:3*num_body]
        rot = obs[...,3*num_body:9*num_body]
        vel = obs[...,9*num_body:12*num_body]
        avel = obs[...,12*num_body:15*num_body]
        height = obs[...,15*num_body:16*num_body]
        up_dir = obs[...,16*num_body:]
        */
        public static float w_pos;
        public static float w_rot;
        public static float w_vel;
        public static float w_avel;
        public static float w_height;
        public static float w_updir;

        public Tensor pos;
        public Tensor rot;
        public Tensor vel;
        public Tensor avel;
        public Tensor height;
        public Tensor up_dir;
        public ControlVAEObs() { }

        public ControlVAEObs(Tensor obs)
        {
            int nbody = (obs.cols - 3) / 16;
            pos = new Tensor(nbody, 3);
            for (int i = 0; i < nbody; i++)
                for (int j = 0; j < 3; j++)
                    pos.SetValue(i, j, obs.GetValue(0, 3 * i + j));
            int off = 3;

            rot = new Tensor(nbody, 6);
            for(int i = 0; i < nbody; i++)
                for (int j = 0; j < 6; j++)
                    rot.SetValue(i, j, obs.GetValue(0, off * nbody + 6 * i + j));
            off += 6;

            vel = new Tensor(nbody, 3);
            for (int i = 0; i < nbody; i++)
                for (int j = 0; j < 3; j++)
                    vel.SetValue(i, j, obs.GetValue(0, off * nbody + 3 * i + j));
            off += 3;

            avel = new Tensor(nbody, 3);
            for (int i = 0; i < nbody; i++)
                for (int j = 0; j < 3; j++)
                    avel.SetValue(i, j, obs.GetValue(0, off * nbody + 3 * i + j));
            off += 3;

            height = new Tensor(nbody, 1);
            for (int i = 0; i < nbody; i++) height.SetValue(i, 0, off * nbody + i);
            off += 1;

            up_dir = new Tensor(3, 1);
            for (int j = 0; j < 3; j++) up_dir.SetValue(j, 0, obs.GetValue(0, off * nbody + j));
        }

        public float CalcLoss(ControlVAEObs other)
        {
            throw new NotImplementedException();
        }

        public float CalcLoss(Tensor obs)
        {
            var obs_ = new ControlVAEObs(obs);
            return CalcLoss(obs_);
        }
    }

    public class DataUtils<T> where T : struct
    {
        List<UIntPtr> BodyList;
        List<UIntPtr> JointList;
        readonly int[] ode_vec6d_index = new int[] { 0, 1, 4, 5, 8, 9 };
        readonly int[] vec6d_index = new int[] { 0, 1, 3, 4, 6, 7 };

        public DataUtils()
        {
            // var x = typeof(T);
        }

        public Tensor GetBodyPos()
        {
            int nbody = BodyList.Count;
            Tensor ret = new Tensor(nbody, 3);
            double[] res = new double[3];
            for (int i = 0; i < nbody; i++)
            {
                IntPtr ptr = dBodyGetPosition(BodyList[i]);
                Marshal.Copy(ptr, res, 0, 3);
                for (int j = 0; j < 3; j++) ret.SetValue(i, j, res[j]);
            }
            return ret;
        }

        public Tensor GetBodyVec6d()
        {
            int nbody = BodyList.Count;
            Tensor ret = new Tensor(nbody, 6);
            double[] mat = new double[12];
            for (int i = 0; i < nbody; i++)
            {
                IntPtr ptr = dBodyGetRotation(BodyList[i]);
                Marshal.Copy(ptr, mat, 0, 12);
                // [0], [1], 2, 3
                // [4], [5], 6, 7
                // [8], [9], 10, 11
                for (int j = 0; j < 6; j++) ret.SetValue(i, j, mat[ode_vec6d_index[j]]);
            }
            return ret;
        }

        public Tensor GetBodyQuat()
        {
            int nbody = BodyList.Count;
            Tensor ret = new Tensor(nbody, 4);
            double[] res = new double[4];
            for (int i = 0; i < nbody; i++)
            {
                IntPtr ptr = dBodyGetQuaternion(BodyList[i]);
                Marshal.Copy(ptr, res, 0, 4);
                QuatArrFromODEArr(res, true);
                for (int j = 0; j < 4; j++) ret.SetValue(i, j, res[j]);
            }
            return ret;
        }

        public Tensor GetBodyLinearVel()
        {
            int nbody = BodyList.Count;
            Tensor ret = new Tensor(nbody, 3);
            double[] res = new double[3];
            for (int i = 0; i < nbody; i++)
            {
                IntPtr ptr = dBodyGetLinearVel(BodyList[i]);
                Marshal.Copy(ptr, res, 0, 3);
                for (int j = 0; j < 3; j++) ret.SetValue(i, j, res[j]);
            }
            return ret;
        }

        public Tensor GetBodyAngularVel()
        {
            int nbody = BodyList.Count;
            Tensor ret = new Tensor(nbody, 3);
            double[] res = new double[3];
            for (int i = 0; i < nbody; i++)
            {
                IntPtr ptr = dBodyGetAngularVel(BodyList[i]);
                Marshal.Copy(ptr, res, 0, 3);
                for (int j = 0; j < 3; j++) ret.SetValue(i, j, res[j]);
            }
            return ret;
        }

        public Tensor GetControlVAEObs()
        {
            int nbody = BodyList.Count;
            Tensor res = new Tensor(1, 0);
            UIntPtr root = BodyList[0];
            Vector3 root_pos = Vec3FromDoublePtr(dBodyGetPosition(root));
            Quaternion root_q = QuatFromODEPtr(dBodyGetQuaternion(root));
            var root_q_inv = Quaternion.Inverse(root_q);
            var y_up = new Vector3(0, 1, 0);

            for(int i = 0; i < nbody; i++)
            {
                /*
                 * pos = obs[...,0:3*num_body]
                rot = obs[...,3*num_body:9*num_body]
                vel = obs[...,9*num_body:12*num_body]
                avel = obs[...,12*num_body:15*num_body]
                height = obs[...,15*num_body:16*num_body]
                up_dir = obs[...,16*num_body:]
                */
                var ptr = BodyList[i];
                var ode_pos = Vec3FromDoublePtr(dBodyGetPosition(ptr));
                var pos = root_q_inv * (ode_pos - root_pos);
                for (int j = 0; j < 3; j++) res.SetValue(0, 3 * i + j, pos[j]);
                int off = 3;

                var mat = new Eigen.Matrix3f(root_q_inv * QuatFromODEPtr(dBodyGetQuaternion(ptr)));
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 2; k++)
                        res.SetValue(0, off * nbody + 6 * i + 3 * j + k, mat.GetValue(j, k));
                off += 6;

                var vel = root_q_inv * Vec3FromDoublePtr(dBodyGetLinearVel(ptr));
                for (int j = 0; j < 3; j++) res.SetValue(0, off * nbody + 3 * i + j, vel[j]);
                off += 3;

                var avel = root_q_inv * Vec3FromDoublePtr(dBodyGetAngularVel(ptr));
                for (int j = 0; j < 3; j++) res.SetValue(0, off * nbody + 3 * i + j, avel[j]);
                off += 3;

                res.SetValue(0, off * nbody + i, ode_pos.y);
            }
            var up = root_q * y_up;
            for (int j = 0; j < 3; j++) res.SetValue(0, 16 * nbody, up[j]);
            return res;
        }

        public Tensor GetJointPos()
        {
            return null;
        }
    }
}
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
using System.Runtime.InteropServices;
using UnityEngine.Assertions;

namespace RenderV2.Eigen
{
    // implementation of https://github.com/heyuanYao-pku/Control-VAE
    public class ControlVAE
    {
        IntPtr net = IntPtr.Zero;
        public static readonly int nJoint = 19;
        public static readonly int nBody = 20;
        public static readonly int zDim = 64;
        public static readonly int ObsDim = 323;

        public ControlVAE(string weight_fname)
        {
            net = ControlVAE_new();
            ControlVAE_load_weights(net, weight_fname);
        }

        ~ControlVAE()
        {
            if (net != IntPtr.Zero)
            {
                ControlVAE_del(net);
                net = IntPtr.Zero;
            }
        }

        // get action through prior distribution
        public Tensor act_prior_quat(Tensor obs, Tensor res)
        {
            Assert.IsFalse(obs is null || res is null);
            Assert.IsTrue(obs.rows == 1 && obs.cols == ObsDim);
            Assert.IsTrue(res.rows == nJoint &&  res.cols == 4);
            ControlVAE_act_prior_quat(net, obs.GetPtr(), res.GetPtr());
            return res;
        }

        // get tracking action through posterior distribution
        public Tensor act_tracking_quat(Tensor obs, Tensor target, Tensor res)
        {
            Assert.IsFalse (obs is null || target is null || res is null);
            Assert.IsTrue(obs.rows == 1 && obs.cols == ObsDim);
            Assert.IsTrue(target.rows == 1 && target.cols == ObsDim);
            Assert.IsTrue(res.rows == nJoint && res.cols == 4);
            ControlVAE_act_tracking_quat(net, obs.GetPtr(), target.GetPtr(), res.GetPtr());
            return res;
        }

        #region ImportWrapper
        [DllImport("EigenForUnity")]
        static extern IntPtr ControlVAE_new();

        [DllImport("EigenForUnity")]
        static extern void ControlVAE_del(IntPtr controlvae);

        [DllImport("EigenForUnity")]
        static extern ulong ControlVAE_load_weights(IntPtr controlvae, string fname);

        [DllImport("EigenForUnity")]
        static extern void ControlVAE_act_prior(IntPtr controlvae, IntPtr obs, IntPtr res);

        [DllImport("EigenForUnity")]
        static extern void ControlVAE_act_tracking(IntPtr controlvae, IntPtr obs, IntPtr target, IntPtr res);

        [DllImport("EigenForUnity")]
        static extern void ControlVAE_act_prior_quat(IntPtr controlvae, IntPtr obs, IntPtr res);

        [DllImport("EigenForUnity")]
        static extern void ControlVAE_act_tracking_quat(IntPtr controlvae, IntPtr obs, IntPtr target, IntPtr res);
        #endregion
    }
}
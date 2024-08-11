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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    /// <summary>
    /// Parameter for PD Controller. 
    /// </summary>
    public class PDController : ControllerBase, IExportInfo<PDControlExportInfo>
    {
        [Tooltip("Kp parameter used in PD Controller")]
        public float[] Kps;

        [Tooltip("magnitude of torque is clipped, to avoid numerical explosion")]
        public float[] TorqueLimits;

        [Tooltip("Default Kp parameter for each joint")]
        public float DefaultKp = 0.0F;

        [Tooltip("Default torque limit for each joint")]
        public float DefaultTorqueLimit = 0.0F;

        public void RemoveAt(int index)
        {
            if (Kps != null)
            {
                float[] NewKps = new float[Kps.Length - 1];
                for(int i=0, j=0; i<Kps.Length; i++)
                {
                    if (i != index)
                    {
                        NewKps[j++] = Kps[i];
                    }
                }
                Kps = NewKps;
            }
            if (TorqueLimits != null)
            {
                float[] NewLimits = new float[TorqueLimits.Length - 1];
                for(int i=0, j=0; i<TorqueLimits.Length; i++)
                {
                    if (i != index)
                    {
                        NewLimits[j++] = TorqueLimits[i];
                    }
                }
                TorqueLimits = NewLimits;
            }
        }

        public void ExtendToSize(int new_size)
        {
            float[] NewKps = new float[new_size];
            if (Kps != null)
            {
                int copy_length = Mathf.Min(new_size, Kps.Length);
                for(int i=0; i<copy_length; i++)
                {
                    NewKps[i] = Kps[i];
                }
            }
            for (int i = Kps == null ? 0 : Kps.Length; i < new_size; i++)
            {
                NewKps[i] = DefaultKp;
            }

            float[] NewTorlim = new float[new_size];
            if (NewTorlim != null)
            {
                int copy_length = Mathf.Min(new_size, TorqueLimits.Length);
                for (int i = 0; i < copy_length; i++)
                {
                    NewTorlim[i] = TorqueLimits[i];
                }
            }
            for (int i = TorqueLimits == null ? 0 : TorqueLimits.Length; i < new_size; i++)
            {
                NewTorlim[i] = DefaultTorqueLimit;
            }

            Kps = NewKps;
            TorqueLimits = NewTorlim;
        }

        /// <summary>
        /// Set All Kp Parameter to default Kp
        /// </summary>
        public void SetDefaultKp()
        {
            if (Kps == null)
            {
                return;
            }

            for(int i=0; i<Kps.Length; i++)
            {
                Kps[i] = DefaultKp;
            }
        }

        /// <summary>
        /// Set All Torque Limit to default Torque Limit
        /// </summary>
        public void SetDefaultTorqueLimit()
        {
            if (TorqueLimits == null)
            {
                return;
            }

            for(int i=0; i<TorqueLimits.Length; i++)
            {
                TorqueLimits[i] = DefaultTorqueLimit;
            }
        }

        public void ResetParams(int length)
        {
            if (Kps == null || Kps.Length != length)
            {
                Kps = new float[length];
                SetDefaultKp();
            }

            if (TorqueLimits == null || TorqueLimits.Length != length)
            {
                TorqueLimits = new float[length];
                SetDefaultTorqueLimit();
            }
        }

        public PDControlExportInfo ExportInfo()
        {
            return new PDControlExportInfo
            {
                Kps = Kps,
                TorqueLimit = TorqueLimits
            };
        }

        /// <summary>
        /// Create GameObject with PDController component by PDControlExportInfo
        /// </summary>
        /// <param name="info"></param>
        /// <returns></returns>
        public static PDController PDControllerCreate(PDControlExportInfo info)
        {
            if (info.Kps.Length != info.TorqueLimit.Length)
            {
                throw new System.ArgumentException("Kps and TorqueLimit Length not match");
            }
            GameObject gameObject = new GameObject { name = "PDController" };
            PDController pdController = gameObject.AddComponent<PDController>();

            pdController.Kps = info.Kps; // Note: not copied
            pdController.TorqueLimits = info.TorqueLimit;

            return pdController;
        }

        /// <summary>
        /// Create GameObject with PDController component
        /// </summary>
        /// <param name="size_"></param>
        /// <param name="Character"></param>
        /// <returns></returns>
        public static PDController PDControllerCreate(int size_, GameObject Character)
        {
            GameObject pdControlObject = new GameObject
            {
                name = "PDControl"
            };
            pdControlObject.transform.parent = Character.transform;
            PDController pdController = pdControlObject.AddComponent<PDController>();
            pdController.Kps = new float[size_];
            pdController.TorqueLimits = new float[size_];

            return pdController;
        }
    }
}

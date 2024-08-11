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
    public static class MathHelper
    {
        public static Quaternion IntegrateQuaternion(Quaternion q, Vector3 omega, float dt)
        {
            var wq = new Quaternion(omega.x, omega.y, omega.z, 0) * q;
            var res = new Quaternion();
            for (int i = 0; i < 4; i++) res[i] = q[i] + 0.5f * dt;
            return res.normalized;
        }

        public static Quaternion DecomposeRotation(Quaternion q, Vector3 vb)
        {
            Vector3 va = (q * vb).normalized;
            Vector3 rotAxis = Vector3.Cross(va, vb).normalized;
            float rotAngle = -Mathf.Acos(Mathf.Clamp(Vector3.Dot(va, vb), -1f, 1f));
            return Quaternion.AngleAxis(rotAngle, -rotAxis) * q;
        }

        // Function to compute the quaternion between two vectors
        public static Quaternion QuatBetween(Vector3 a, Vector3 b)
        {
            var crossRes = Vector3.Cross(a, b);
            float w_ = Mathf.Sqrt(a.sqrMagnitude * b.sqrMagnitude) + Vector3.Dot(a, b);
            return new Quaternion(crossRes.x, crossRes.y, crossRes.z, w_).normalized;
        }

        public static List<List<Quaternion>> Resample(List<List<Quaternion>> rotation, float freq1, float freq2)
        {
            int numTimeSamples = rotation.Count;
            int numJoints = rotation[0].Count;

            // Calculate the total time duration based on the original frequency
            float totalTime = (numTimeSamples - 1) / freq1;

            // Determine the number of samples required for the new frequency
            int newNumTimeSamples = Mathf.RoundToInt(totalTime * freq2) + 1;

            List<List<Quaternion>> resampledRotation = new List<List<Quaternion>>();

            for (int i = 0; i < newNumTimeSamples; i++)
            {
                // Calculate the corresponding time in the original sequence
                float t = i / freq2;
                float originalTimeIndex = t * freq1;

                int index1 = Mathf.FloorToInt(originalTimeIndex);
                int index2 = Mathf.Clamp(index1 + 1, 0, numTimeSamples - 1);

                float interpFactor = originalTimeIndex - index1;

                List<Quaternion> resampledFrame = new List<Quaternion>();

                for (int j = 0; j < numJoints; j++)
                {
                    Quaternion q1 = rotation[index1][j];
                    Quaternion q2 = rotation[index2][j];

                    Quaternion resampledQuat = Quaternion.Slerp(q1, q2, interpFactor);
                    resampledFrame.Add(resampledQuat);
                }

                resampledRotation.Add(resampledFrame);
            }

            return resampledRotation;
        }

        /// <summary>
        /// Resamples a quaternion sequence based on input and output frequencies.
        /// </summary>
        /// <param name="rotation">Input sequence of quaternions.</param>
        /// <param name="freq1">The original frequency of the quaternion sequence.</param>
        /// <param name="freq2">The desired output frequency of the quaternion sequence.</param>
        /// <returns>A new sequence of quaternions resampled along the time dimension.</returns>
        public static List<Quaternion> Resample(List<Quaternion> rotation, float freq1, float freq2)
        {
            int numTimeSamples = rotation.Count;

            // Calculate the total time duration based on the original frequency
            float totalTime = (numTimeSamples - 1) / freq1;

            // Determine the number of samples required for the new frequency
            int newNumTimeSamples = Mathf.RoundToInt(totalTime * freq2) + 1;

            List<Quaternion> resampledRotation = new List<Quaternion>(newNumTimeSamples);

            for (int i = 0; i < newNumTimeSamples; i++)
            {
                // Calculate the corresponding time in the original sequence
                float t = i / freq2;
                float originalTimeIndex = t * freq1;

                int index1 = Mathf.FloorToInt(originalTimeIndex);
                int index2 = Mathf.Clamp(index1 + 1, 0, numTimeSamples - 1);

                float interpFactor = originalTimeIndex - index1;

                Quaternion q1 = rotation[index1];
                Quaternion q2 = rotation[index2];

                Quaternion resampledQuat = Quaternion.Slerp(q1, q2, interpFactor);
                resampledRotation.Add(resampledQuat);
            }

            return resampledRotation;
        }

        /// <summary>
        /// Resamples a position sequence based on input and output frequencies.
        /// </summary>
        /// <param name="position">Input sequence of positions.</param>
        /// <param name="freq1">The original frequency of the position sequence.</param>
        /// <param name="freq2">The desired output frequency of the position sequence.</param>
        /// <returns>A new sequence of positions resampled along the time dimension.</returns>
        public static List<Vector3> Resample(List<Vector3> position, float freq1, float freq2)
        {
            int numTimeSamples = position.Count;

            // Calculate the total time duration based on the original frequency
            float totalTime = (numTimeSamples - 1) / freq1;

            // Determine the number of samples required for the new frequency
            int newNumTimeSamples = Mathf.RoundToInt(totalTime * freq2) + 1;

            List<Vector3> resampledPosition = new List<Vector3>(newNumTimeSamples);

            for (int i = 0; i < newNumTimeSamples; i++)
            {
                // Calculate the corresponding time in the original sequence
                float t = i / freq2;
                float originalTimeIndex = t * freq1;

                int index1 = Mathf.FloorToInt(originalTimeIndex);
                int index2 = Mathf.Clamp(index1 + 1, 0, numTimeSamples - 1);

                float interpFactor = originalTimeIndex - index1;

                Vector3 p1 = position[index1];
                Vector3 p2 = position[index2];

                Vector3 resampledPos = Vector3.Lerp(p1, p2, interpFactor);
                resampledPosition.Add(resampledPos);
            }

            return resampledPosition;
        }

        public static List<Quaternion> AlignQuaternions(List<Quaternion> quaternions, bool inplace = false)
        {
            var qt = inplace ? quaternions : new List<Quaternion>(quaternions);

            if (qt.Count == 1) return qt; // do nothing since there is only one quaternion

            float[] sign = new float[qt.Count - 1];
            for (int i = 0; i < qt.Count - 1; i++)
            {
                sign[i] = Quaternion.Dot(qt[i], qt[i + 1]);
                sign[i] = sign[i] < 0 ? -1 : 1;
            }

            for (int i = 1; i < sign.Length; i++)
            {
                sign[i] *= sign[i - 1];
            }

            for (int i = 1; i < qt.Count; i++)
            {
                if (sign[i - 1] < 0)
                {
                    qt[i] = new Quaternion(-qt[i].x, -qt[i].y, -qt[i].z, -qt[i].w);
                }
            }

            return qt;
        }

        public static double[] CreateGaussianKernel(int radius, double sigma)
        {
            int size = 2 * radius + 1;
            double[] kernel = new double[size];
            double norm = 1.0 / (Math.Sqrt(2 * Math.PI) * sigma);
            double coefficient = 2.0 * sigma * sigma;
            double sum = 0.0;

            for (int i = -radius; i <= radius; i++)
            {
                kernel[i + radius] = norm * Math.Exp(-(i * i) / coefficient);
                sum += kernel[i + radius];
            }

            // Normalize the kernel
            for (int i = 0; i < kernel.Length; i++) kernel[i] /= sum;
            return kernel;
        }

        public static double[] ApplyGaussianFilter1D(double[] data, double sigma)
        {
            int radius = (int)Math.Ceiling(3 * sigma);
            double[] kernel = CreateGaussianKernel(radius, sigma);
            double[] result = new double[data.Length];

            // Apply the Gaussian filter
            for (int i = 0; i < data.Length; i++)
            {
                double sum = 0.0;
                for (int j = -radius; j <= radius; j++)
                {
                    int index = i + j;
                    if (index < 0) index = 0; // Nearest edge handling
                    if (index >= data.Length) index = data.Length - 1; // Nearest edge handling
                    sum += data[index] * kernel[j + radius];
                }
                result[i] = sum;
            }

            return result;
        }

        // TODO: check in scipy.
        public static Vector3 QuaternionToEuler(Quaternion q, string order, bool is_degree = false)
        {
            Vector3 euler = Vector3.zero;

            float sqw = q.w * q.w;
            float sqx = q.x * q.x;
            float sqy = q.y * q.y;
            float sqz = q.z * q.z;

            switch (order)
            {
                case "XYZ":
                    euler.x = Mathf.Atan2(2f * (q.w * q.x - q.y * q.z), sqw - sqx - sqy + sqz);
                    euler.y = Mathf.Asin(2f * (q.w * q.y + q.z * q.x));
                    euler.z = Mathf.Atan2(2f * (q.w * q.z - q.x * q.y), sqw + sqx - sqy - sqz);
                    break;

                case "XZY":
                    euler.x = Mathf.Atan2(2f * (q.w * q.x + q.y * q.z), sqw - sqx + sqy - sqz);
                    euler.z = Mathf.Asin(-2f * (q.x * q.z - q.w * q.y));
                    euler.y = Mathf.Atan2(2f * (q.w * q.z + q.x * q.y), sqw + sqx - sqy - sqz);
                    break;

                case "YZX":
                    euler.y = Mathf.Atan2(2f * (q.w * q.y - q.x * q.z), sqw - sqx + sqy - sqz);
                    euler.z = Mathf.Asin(2f * (q.w * q.z + q.x * q.y));
                    euler.x = Mathf.Atan2(2f * (q.w * q.x - q.y * q.z), sqw + sqx - sqy - sqz);
                    break;

                case "YXZ":
                    euler.y = Mathf.Atan2(2f * (q.w * q.y + q.x * q.z), sqw - sqx - sqy + sqz);
                    euler.x = Mathf.Asin(-2f * (q.y * q.z - q.w * q.x));
                    euler.z = Mathf.Atan2(2f * (q.w * q.z + q.y * q.x), sqw + sqx - sqy - sqz);
                    break;

                case "ZYX":
                    euler.z = Mathf.Atan2(2f * (q.w * q.z + q.x * q.y), sqw - sqx - sqy + sqz);
                    euler.y = Mathf.Asin(-2f * (q.x * q.z - q.w * q.y));
                    euler.x = Mathf.Atan2(2f * (q.w * q.x + q.y * q.z), sqw - sqx + sqy - sqz);
                    break;

                case "ZXY":
                    euler.z = Mathf.Atan2(2f * (q.w * q.z - q.x * q.y), sqw - sqx + sqy - sqz);
                    euler.x = Mathf.Asin(2f * (q.w * q.x + q.y * q.z));
                    euler.y = Mathf.Atan2(2f * (q.w * q.y - q.x * q.z), sqw + sqx - sqy - sqz);
                    break;

                default:
                    throw new ArgumentException("Invalid Euler order: " + order);
            }

            // Convert from radians to degrees
            if (is_degree) euler *= Mathf.Rad2Deg;

            return euler;
        }

        public static Quaternion BuildQuat(float angle, Vector3 axis)
        {
            // (u sin theta / 2, cos theta / 2)
            var u = axis * Mathf.Sin(0.5f * angle);
            return new Quaternion(u.x, u.y, u.z, Mathf.Cos(0.5f * angle));
        }

        public static Quaternion EulerToQuaternion(Vector3 euler, string order)
        {
            // euler *= Mathf.Deg2Rad; // Convert degrees to radians
            var qx = BuildQuat(euler.x, Vector3.right);
            var qy = BuildQuat(euler.y, Vector3.up);
            var qz = BuildQuat(euler.z, Vector3.forward);

            switch (order)
            {
                case "XYZ": return qx * qy * qz;
                case "XZY": return qx * qz * qy;
                case "YXZ": return qy * qx * qz;
                case "YZX": return qy * qz * qx;
                case "ZXY": return qz * qx * qy;
                case "ZYX": return qz * qy * qx;
                default:
                    throw new ArgumentException("Invalid Euler order: " + order);
            }
        }

    }
}
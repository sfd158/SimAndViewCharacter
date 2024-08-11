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
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;

namespace RenderV2
{
    /// <summary>
    /// utils for math operation
    /// </summary>
    public class Utils
    {
        static public string[] NullStr = { "null", "no", "not", "empty", "nullptr", "", "nothing" };

        /// <summary>
        /// judge whether the input string represents null
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        static public bool IsNullString(string s)
        {
            string ss = s.ToLower();
            for (int i = 0; i < NullStr.Length; i++)
            {
                if (ss == NullStr[i])
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Adapted from: open dynamics engine v0.12, src/joints/hinge.cpp
        /// </summary>
        /// <param name="qrel">Relative Quaternion</param>
        /// <param name="axis">Rotate Axis</param>
        /// <returns>Rotate angle</returns>
        static public float GetHingeAngleFromRelativeQuat(Quaternion qrel, Vector3 axis)
        {
            // the angle between the two bodies is extracted from the quaternion that
            // represents the relative rotation between them. recall that a quaternion
            // q is:
            //    [s,v] = [ cos(theta/2) , sin(theta/2) * u ]
            // where s is a scalar and v is a 3-vector. u is a unit length axis and
            // theta is a rotation along that axis. we can get theta/2 by:
            //    theta/2 = atan2 ( sin(theta/2) , cos(theta/2) )
            // but we can't get sin(theta/2) directly, only its absolute value, i.e.:
            //    |v| = |sin(theta/2)| * |u|
            //        = |sin(theta/2)|
            // using this value will have a strange effect. recall that there are two
            // quaternion representations of a given rotation, q and -q. typically as
            // a body rotates along the axis it will go through a complete cycle using
            // one representation and then the next cycle will use the other
            // representation. this corresponds to u pointing in the direction of the
            // hinge axis and then in the opposite direction. the result is that theta
            // will appear to go "backwards" every other cycle. here is a fix: if u
            // points "away" from the direction of the hinge (motor) axis (i.e. more
            // than 90 degrees) then use -q instead of q. this represents the same
            // rotation, but results in the cos(theta/2) value being sign inverted.

            // extract the angle from the quaternion. cost2 = cos(theta/2),
            // sint2 = |sin(theta/2)|
            float cost2 = qrel.w;
            Vector3 q_xyz = new Vector3(qrel.x, qrel.y, qrel.z);
            float sint2 = q_xyz.magnitude;
            
            float theta = (Vector3.Dot(q_xyz, axis) >= 0) ? // @@@ padding assumptions
                          (2 * Mathf.Atan2(sint2, cost2)) :  // if u points in direction of axis
                          (2 * Mathf.Atan2(sint2, -cost2));  // if u points in opposite direction

            // the angle we get will be between 0..2*pi, but we want to return angles
            // between -pi..pi
            if (theta > Math.PI) theta -= Convert.ToSingle(2 * Math.PI);

            // Zhenhua Song: in ODE comment, `the angle we've just extracted has the wrong sign`. However, I think the sign is correct.
            // theta = -theta;

            return theta;
        }

        /// <summary>
        /// Convert Rotation Matrix to Quaternion
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        static public Quaternion QuaternionFromMatrix(Matrix4x4 m)
        {
            // Adapted from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
            Quaternion q = new Quaternion
            {
                w = Mathf.Sqrt(Mathf.Max(0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2,
                x = Mathf.Sqrt(Mathf.Max(0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2,
                y = Mathf.Sqrt(Mathf.Max(0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2,
                z = Mathf.Sqrt(Mathf.Max(0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
            };
            q.x *= Mathf.Sign(q.x * (m[2, 1] - m[1, 2]));
            q.y *= Mathf.Sign(q.y * (m[0, 2] - m[2, 0]));
            q.z *= Mathf.Sign(q.z * (m[1, 0] - m[0, 1]));
            return q;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <returns></returns>
        static public Matrix4x4 OrthogonalAxisToRotMat(Vector3 x, Vector3 y, Vector3 z)
        {
            Matrix4x4 mat = new Matrix4x4();
            for (int i = 0; i < 3; i++)
            {
                mat[0, i] = x[i];
            }
            for (int i = 0; i < 3; i++)
            {
                mat[1, i] = y[i];
            }
            for (int i = 0; i < 3; i++)
            {
                mat[2, i] = z[i];
            }
            return mat;
        }

        static public Quaternion OrthogonalAxisToQuaternion(Vector3 x, Vector3 y, Vector3 z)
        {
            return QuaternionFromMatrix(OrthogonalAxisToRotMat(x, y, z));
        }

        /// <summary>
        /// Root Joint's Name
        /// </summary>
        /// <returns></returns>
        static public string RootJointName()
        {
            return "RootJoint";
        }

        /// <summary>
        /// Convert Vector3 to float[] array
        /// </summary>
        /// <param name="vec">Vector3</param>
        /// <returns></returns>
        static public float[] Vector3ToArr(Vector3 vec)
        {
            float[] res = new float[3];
            res[0] = vec[0];
            res[1] = vec[1];
            res[2] = vec[2];
            return res;
        }

        /// <summary>
        /// Convert Quaternion to Array
        /// </summary>
        /// <param name="q"></param>
        /// <returns></returns>
        static public float[] QuaternionToArr(Quaternion q)
        {
            float[] res = new float[4];
            res[0] = q[0];
            res[1] = q[1];
            res[2] = q[2];
            res[3] = q[3];
            return res;
        }

        /// <summary>
        /// Convert float array to Vector3
        /// </summary>
        /// <param name="arr">float array</param>
        /// <returns>Vector3</returns>
        static public Vector3 ArrToVector3(float[] arr)
        {
            return new Vector3(arr[0], arr[1], arr[2]);
        }

        /// <summary>
        /// Convert float array to Quaternion
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        static public Quaternion ArrToQuaternion(float[] arr)
        {
            return new Quaternion(arr[0], arr[1], arr[2], arr[3]);
        }

        /// <summary>
        /// Get all off spring of input GameObject
        /// </summary>
        /// <param name="gameObject"></param>
        /// <returns></returns>
        static public List<GameObject> GetAllOffSpring(GameObject gameObject)
        {
            List<GameObject> q = new List<GameObject>
            {
                gameObject
            };
            int head = 0;
            while (head < q.Count)
            {
                GameObject g = q[head++];
                if (!g.activeInHierarchy)
                {
                    continue;
                }
                for (int ch = 0; ch < g.transform.childCount; ch++)
                {
                    q.Add(g.transform.GetChild(ch).gameObject);
                }
            }

            return q;
        }

        public static ulong[] ArrayListToULongArray(object rhs)
        {
            if (rhs == null)
            {
                return null;
            }
            ArrayList arr = rhs as ArrayList;
            ulong[] res = new ulong[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = Convert.ToUInt64(arr[i]);
            }
            return res;
        }

        public static int[] ArrayListToIntArray(object rhs)
        {
            if (rhs == null)
            {
                return null;
            }
            ArrayList arr = rhs as ArrayList;
            int[] res = new int[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = Convert.ToInt32(arr[i]);
            }
            return res;
        }

        /// <summary>
        /// Convert ArrayList to float array
        /// </summary>
        /// <param name="arr">Input ArrayList</param>
        /// <returns>float array</returns>
        public static float[] ArrayListToFloatArray(ArrayList arr)
        {
            if (arr == null)
            {
                return null;
            }
            float[] res = new float[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = Convert.ToSingle(arr[i]);
            }
            return res;
        }

        /// <summary>
        /// Convert ArrayList to string array
        /// </summary>
        /// <param name="rhs">Input ArrayList</param>
        /// <returns>array of string</returns>
        public static string[] ArrayListToStringArray(object rhs)
        {
            if (rhs == null)
            {
                return null;
            }
            ArrayList arr = rhs as ArrayList;
            string[] res = new string[arr.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = arr[i] as string;
            }
            return res;
        }

        public static char[] SplitChars = new char[2]{ ' ', '\t'};

        /// <summary>
        /// Split string to Vector3
        /// </summary>
        /// <param name="s">Input string</param>
        /// <param name="offset">offset</param>
        /// <returns>split result of input string</returns>
        public static Vector3 StringToVec3(string s, int offset = 0)
        {
            string[] buf = s.Split(SplitChars);
            Vector3 res = Vector3.zero;
            int length = Math.Min(buf.Length, 3);
            for(int i=offset; i<length; i++)
            {
                res[i] = float.Parse(buf[i]);
            }
            return res;
        }

        /// <summary>
        /// Get GameObject's ancestor with Component of type T 
        /// </summary>
        /// <typeparam name="T">class type</typeparam>
        /// <param name="obj">GameObject</param>
        /// <returns>Component of type T</returns>
        public static T GetComponentAncestor<T>(GameObject obj) where T: MonoBehaviour
        {
            if (obj == null)
            {
                return null;
            }
            for(Transform trans = obj.transform; trans != null; trans = trans.parent)
            {
                if (trans.TryGetComponent<T>(out var res))
                {
                    return res;
                }
            }
            return null;
        }

        public static DWorld GetDWorld()
        {
            DWorld[] worlds;
            worlds = Selection.GetFiltered<DWorld>(SelectionMode.TopLevel);
            if (worlds == null)
            {
                worlds = GameObject.FindObjectsOfType<DWorld>();
            }
            
            if (worlds == null || worlds.Length != 1)
            {
                Debug.LogWarning("There should be only 1 DWorld Component in Unity Scene");
            }
            return worlds[0];
        }

        /// <summary>
        /// Get DWorld Component in Scene
        /// </summary>
        /// <returns></returns>
        public static DWorld GetDWorldByScene()
        {
            Scene scene = SceneManager.GetActiveScene();
            GameObject[] objs = scene.GetRootGameObjects();
            foreach(GameObject obj in objs)
            {
                if (obj.TryGetComponent<DWorld>(out var dWorld))
                {
                    return dWorld;
                }
            }
            Debug.LogWarning("There should be only 1 DWorld Component in Unity Scene");
            return null;
        }

        public static Vector3 TransformPivot(Vector3 pivotA, Transform transA, Transform transB)
        {
            Vector3 pivotB;
            Vector3 pivotAScaled = new Vector3(pivotA.x / transA.localScale.x,
                                               pivotA.y / transA.localScale.y,
                                               pivotA.z / transA.localScale.z);
            pivotB = transB.InverseTransformPoint(transA.TransformPoint(pivotAScaled));
            pivotB.Scale(transB.localScale);
            return pivotB;
        }

        public static Vector3 PointToWorld(Vector3 pointLocal, Transform trans)
        {
            Vector3 point = new Vector3(
                pointLocal.x / trans.localScale.x,
                pointLocal.y / trans.localScale.y,
                pointLocal.z / trans.localScale.z);
            return trans.TransformPoint(point);
        }

        public static Vector3 PointFromWorld(Vector3 pointWorld, Transform trans)
        {
            Vector3 point = trans.InverseTransformPoint(pointWorld);
            point.x = point.x * trans.localScale.x;
            point.y = point.y * trans.localScale.y;
            point.z = point.z * trans.localScale.z;
            return point;
        }

        public static Vector3 FindUpFromForward(Vector3 forward)
        {
            float minDot = Mathf.Abs(Vector3.Dot(forward, Vector3.up));
            Vector3 minDotAxis = Vector3.up;

            float dot = Mathf.Abs(Vector3.Dot(forward, Vector3.right));
            if (dot < minDot)
            {
                minDot = dot;
                minDotAxis = Vector3.right;
            }

            dot = Mathf.Abs(Vector3.Dot(forward, Vector3.forward));
            if (dot < minDot)
            {
                minDot = dot;
                minDotAxis = Vector3.forward;
            }

            Vector3 up = Vector3.Cross(forward, minDotAxis);
            up.Normalize();
            return up;
        }

        public static Vector3 GetMatrix4x4Position(Matrix4x4 m)
        {
            Vector3 posColumn = m.GetColumn(3);
            return new Vector3(posColumn.x, posColumn.y, posColumn.z);
        }

        public static Quaternion GetMatrix4x4Rotation(Matrix4x4 m)
        {
            return m.rotation;
        }

        public static Vector3 GetMatrix4x4Scale(Matrix4x4 m)
        {
            return m.lossyScale;
        }

        public static bool isVectorValid(Vector3 vec)
        {
            if (float.IsInfinity(vec.x) || float.IsInfinity(vec.y) || float.IsInfinity(vec.z))
                return false;
            if (float.IsNaN(vec.x) || float.IsNaN(vec.y) || float.IsNaN(vec.z))
                return false;

            return true;
        }
    }
}

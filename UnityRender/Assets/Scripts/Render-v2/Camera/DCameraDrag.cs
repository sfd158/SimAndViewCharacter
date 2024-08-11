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
    public struct DRayHitInfo
    {
        public Vector3 baryCentricCoord;
        public float distance;
        public Vector2 lightmapCoord;
        public Vector3 normal;
        public Vector3 point;
        public Vector2 textureCoord;
        public GameObject gameObject;
        // public int triangleIndex;
        // public IntPtr hitBody;
        // public IntPtr hitBase;
        // public int hitLinkIndex;
    }

    public class DCameraDrag : MonoBehaviour
    {
        public bool m_dragPhysXBody = false;

        public GameObject m_fingerPrefab = null;

        private GameObject m_fingerVisualizer = null;

        private Vector3 m_lastTouchPosition;
        private Vector3 m_lastMousePosition;

        private Vector3 touchPosition;
        private Transform hitObj;
        private float dragDepth;

        private Vector3 localHitPointPosition;
        [SerializeField]
        private LayerMask ignoredLayers;

        void Start()
        {
            m_lastTouchPosition = m_lastMousePosition = Input.mousePosition;
        }

        void Update()
        {

        }

        public void HandleInputBegin(Vector3 screenPosition)
        {
            Ray ray = Camera.main.ScreenPointToRay(screenPosition);
            // StartAPEDrag(screenPosition, ray);
            if (m_dragPhysXBody)
            {
                StartPhysXDrag(ray);
            }
        }

        public void HandleInputEnd()
        {
            if (m_fingerVisualizer != null)
            {
                Destroy(m_fingerVisualizer);
                m_fingerVisualizer = null;
            }
        }

        private void HitPlaneRayCast(Ray ray)
        {
            //Plane p = new Plane(-Camera.main.transform.forward, hitPlaneOrigin.position);
        }

        /* private bool HitPlaneRayCast(Ray ray, out CTntRayHitInfo hitInfo)
        {
            hitInfo = new CTntRayHitInfo();

            //if (!includeNearestLinks || hitPlaneOrigin == null)
            //{
            //    return false;
            //}

            Plane p = new Plane(-Camera.main.transform.forward, hitPlaneOrigin.position);
            float enter = 0.0f;

            if (p.Raycast(ray, out enter))
            {
                Vector3 hitPoint = ray.GetPoint(enter);

                Collider[] hitColliders = Physics.OverlapSphere(hitPoint, distToNearestLinks);
                if (hitColliders.Length > 0)
                {
                    List<Transform> validObjects = new List<Transform>();
                    foreach (Collider col in hitColliders)
                    {
                        if (ignoredLayers != (ignoredLayers | (1 << col.gameObject.layer)))
                        {
                            Transform obj = col.transform;
                            if ((obj.parent != null) && (obj.parent.GetComponent<tntCompoundCollider>() != null))
                                obj = obj.parent;

                            tntRigidBody rb = obj.GetComponent<tntRigidBody>();
                            tntLink lnk = obj.GetComponent<tntLink>();
                            if (rb != null)
                            {
                                if (rb.m_rigidBody != IntPtr.Zero)
                                    validObjects.Add(obj);
                            }
                            else if (lnk != null)
                            {
                                if (lnk.m_base != IntPtr.Zero)
                                    validObjects.Add(obj);
                            }
                        }
                    }

                    float minDist = 100000;
                    Transform closestObj = null;
                    foreach (Transform obj in validObjects)
                    {
                        float dist = Vector3.Distance(hitPoint, obj.position);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            closestObj = obj;
                        }
                    }

                    if (closestObj == null)
                        return false;

                    Debug.Log(closestObj.gameObject.name);
                    hitInfo.gameObject = closestObj.gameObject;
                    hitInfo.point = hitPoint;
                    tntRigidBody body = closestObj.GetComponent<tntRigidBody>();
                    if (body != null)
                        hitInfo.hitBody = body.m_rigidBody;

                    tntLink link = closestObj.GetComponent<tntLink>();
                    if (link != null)
                        hitInfo.hitBase = link.m_base;

                    return true;

                }
                else
                {
                    Debug.Log("no nearest links");
                    return false;
                }
            }

            return false;
        } */

        private void StartPhysXDrag(Ray ray)
        {
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                if (ignoredLayers != (ignoredLayers | (1 << hit.transform.gameObject.layer)))
                {
                    dragDepth = CameraPlane.CameraToPointDepth(Camera.main, hit.point);
                    // jointTrans = AttachJoint(hit.rigidbody, hit.point);
                    hitObj = hit.transform;
                    localHitPointPosition = Utils.PointFromWorld(hit.point, hit.transform);
                }
            }
        }
        void UpdateFunc()
        {
            if (GetInputDown())
            {

            }
            else if (GetInputDownContinue())
            {

            }
            else if (GetInputUp())
            {

            }
        }

        private void InitFingerVisualizer(Vector3 HitPoint)
        {
            if (m_fingerPrefab != null)
            {
                m_fingerVisualizer = Instantiate(m_fingerPrefab);
                m_fingerVisualizer.transform.position = HitPoint;
            }
        }

        private bool GetInputDown()
        {
            touchPosition = Input.mousePosition;
            return Input.GetMouseButtonDown(0);
        }

        private bool GetInputUp()
        {
            touchPosition = Input.mousePosition;
            return Input.GetMouseButtonUp(0);
        }

        private bool GetInputDownContinue()
        {
            touchPosition = Input.mousePosition;
            return Input.GetMouseButton(0);
        }
    }
}


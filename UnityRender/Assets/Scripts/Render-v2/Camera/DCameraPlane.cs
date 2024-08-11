
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
	/// Utility class for working with planes relative to a camera.
	/// </summary>
	public static class CameraPlane
	{
		private static Vector2 m_dimensionCalibrator;

		public static Vector3 ViewportToCameraPlanePoint(Camera theCamera, float zDepth, Vector2 viewportCord)
		{
			theCamera.ResetAspect();
			float oppositeX = Mathf.Tan(theCamera.fieldOfView / 2 * Mathf.Deg2Rad);
			float oppositeY = oppositeX;

			float xProportion = ((viewportCord.x - .5f) / .5f);
			float yProportion = ((viewportCord.y - .5f) / .5f);

			//Debug.Log ("ViewportCoord:" + viewportCord + " normalized:" + new Vector2 (xProportion, yProportion));
			//Debug.Log ("fieldofView=" + theCamera.fieldOfView + " aspect=" + theCamera.aspect);
			float xOffset = oppositeX * xProportion * zDepth;
			float yOffset = oppositeY * yProportion * zDepth;
			return new Vector3(xOffset, yOffset, zDepth);
		}

		/// <summary>
		/// Returns world space position at a given viewport coordinate for a given depth.
		/// </summary>
		public static Vector3 ViewportToWorldPlanePoint(Camera theCamera, float zDepth, Vector2 viewportCord)
		{
			Vector3 computedCameraPlanePos = ViewportToCameraPlanePoint(theCamera, zDepth, viewportCord);
			computedCameraPlanePos.x *= m_dimensionCalibrator.x;
			computedCameraPlanePos.y *= m_dimensionCalibrator.y;
			return theCamera.transform.TransformPoint(computedCameraPlanePos);
		}

		public static Vector3 ScreenToWorldPlanePoint(Camera camera, float zDepth, Vector3 screenCoord)
		{
			var point = Camera.main.ScreenToViewportPoint(screenCoord);
			return ViewportToWorldPlanePoint(camera, zDepth, point);
		}

		/// <summary>
		/// Distance between the camera and a plane parallel to the viewport that passes through a given point.
		/// </summary>
		public static float CameraToPointDepth(Camera cam, Vector3 point)
		{
			Vector3 localPosition = cam.transform.InverseTransformPoint(point);
			return localPosition.z;
		}

		public static void CalibrateDimension(Camera camera, float zDepth, Vector3 screenCoord, Vector3 cameraPlanePos)
		{
			var point = Camera.main.ScreenToViewportPoint(screenCoord);
			Vector3 computedCameraPlanePos = ViewportToCameraPlanePoint(camera, zDepth, point);
			m_dimensionCalibrator.x = cameraPlanePos.x / computedCameraPlanePos.x;
			m_dimensionCalibrator.y = cameraPlanePos.y / computedCameraPlanePos.y;
		}
	}
}
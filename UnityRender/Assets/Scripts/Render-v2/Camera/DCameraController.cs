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
using UnityEngine;
using System.Collections;
namespace RenderV2
{
	using CameraUpdateMode = MonoUpdateMode;

	/// <summary>
	/// 3rd person camera controller.
	/// </summary>
	[DisallowMultipleComponent]
	public class DCameraController : MonoBehaviour
	{
		public Transform target; // The target Transform to follow
		public CameraUpdateMode updateMode = CameraUpdateMode.LateUpdate; // When to update the camera?
		public bool lockCursor = false; // If true, the mouse will be locked to screen center and hidden
		public bool smoothFollow; // If > 0, camera will smoothly interpolate towards the target
		public bool tankControlCam; // Use camera scheme for tank controls
		public float followSpeed = 10f; // Smooth follow speed
		public float distance = 10.0f; // The current distance to target
		public float minDistance = 2; // The minimum distance to target
		public float maxDistance = 10; // The maximum distance to target
		public float zoomSpeed = 10f; // The speed of interpolating the distance
		public float zoomSensitivity = 1f; // The sensitivity of mouse zoom
		public float rotationXSensitivity = 3.5f; // The sensitivity of rotation
		public float rotationYSensitivity = 3.0f; // The sensitivity of rotation
		public float tankCamDamping = 0.005f; // Sensitivity of rotation in tank control mode
		public float yMinLimit = -20; // Min vertical angle
		public float yMaxLimit = 80; // Max vertical angle
		public Vector3 offset = new Vector3(0, 0.5f, 0.0f); // The offset from target relative to camera rotation
		public bool rotateAlways = false; // Always rotate to mouse?
		public bool rotateOnLeftButton; // Rotate to mouse when left button is pressed?
		public bool rotateOnRightButton = true; // Rotate to mouse when right button is pressed?
		public bool rotateOnMiddleButton; // Rotate to mouse when middle button is pressed?

		public bool EnableControlCameraByKeyboard = false;

		public float x { get; private set; } // The current x rotation of the camera
		public float y { get; private set; } // The current y rotation of the camera
		public float distanceTarget { get; private set; } // Get/set distance
		public Transform targetDirTransform;

		private Vector3 position;
		private Quaternion rotation = Quaternion.identity;
		private Vector3 smoothPosition;
		private Camera cam;

		// Initiate, set the params to the current transformation of the camera relative to the target
		protected virtual void Awake()
		{
			Vector3 angles = transform.eulerAngles;
			x = angles.y;
			y = angles.x;

			distanceTarget = distance;
			smoothPosition = transform.position;

			cam = GetComponent<Camera>();
		}

		protected virtual void Update()
		{
			if (updateMode == CameraUpdateMode.Update)
			{
				UpdateTransform();
			}
		}

		protected virtual void FixedUpdate()
		{
			if (updateMode == CameraUpdateMode.FixedUpdate)
			{
				UpdateTransform();
			}
		}

		protected virtual void LateUpdate()
		{
			UpdateInput();

			if (updateMode == CameraUpdateMode.LateUpdate)
			{
				UpdateTransform();
			}
		}

		// Read the user input
		public void UpdateInput()
		{
			if (target == null || !cam.enabled)
			{
				return;
			}

			// Cursors
			Cursor.lockState = lockCursor ? CursorLockMode.Locked : CursorLockMode.None;
			Cursor.visible = !lockCursor;

			// Should we rotate the camera?
			bool mouseRotate = rotateAlways || (rotateOnLeftButton && Input.GetMouseButton(0)) || (rotateOnRightButton && Input.GetMouseButton(1)) || (rotateOnMiddleButton && Input.GetMouseButton(2));
			bool camRotating = false;

			// delta rotation
			if (mouseRotate)
			{
				x += Input.GetAxis("Mouse X") * rotationXSensitivity;
				y = ClampAngle(y - Input.GetAxis("Mouse Y") * rotationYSensitivity, yMinLimit, yMaxLimit);
				camRotating = true;
			}

			// Note: If we need to control the character by keyboard or joystick,
			// we should not use Up/Down/Left/Right arrow to control the camera
			if (EnableControlCameraByKeyboard && Input.anyKey)
			{
				if (Input.GetKey(KeyCode.UpArrow))
				{
					y = ClampAngle(y - 0.2f * rotationYSensitivity, yMinLimit, yMaxLimit);
					camRotating = true;
				}
				if (Input.GetKey(KeyCode.DownArrow))
				{
					y = ClampAngle(y + 0.2f * rotationYSensitivity, yMinLimit, yMaxLimit);
					camRotating = true;
				}
				if (Input.GetKey(KeyCode.LeftArrow))
				{
					x -= 0.2f * rotationXSensitivity;
					camRotating = true;
				}
				if (Input.GetKey(KeyCode.RightArrow))
				{
					x += 0.2f * rotationXSensitivity;
					camRotating = true;
				}
			}

			if (tankControlCam)
			{
				if (!camRotating)
				{
					x = Mathf.LerpAngle(x, targetDirTransform.eulerAngles.y, tankCamDamping);
					y = Mathf.LerpAngle(y, 15, tankCamDamping);
				}
			}

			// Distance
			distanceTarget = Mathf.Clamp(distanceTarget + zoomAdd, minDistance, maxDistance);
		}

		// Update the camera transform
		public void UpdateTransform()
		{
			UpdateTransform(Time.deltaTime);
		}

		public void UpdateTransform(float deltaTime)
		{
			if (target == null || !cam.enabled)
			{
				return;
			}

			// Distance
			distance += (distanceTarget - distance) * zoomSpeed * deltaTime;
            

			// Rotation
			rotation = Quaternion.AngleAxis(x, Vector3.up) * Quaternion.AngleAxis(y, Vector3.right);

			// Smooth follow
			if (!smoothFollow)
			{
				smoothPosition = target.position;
			}
			else
			{
				smoothPosition = Vector3.Lerp(smoothPosition, target.position, deltaTime * followSpeed);
			}

			// Position
            
			position = smoothPosition + rotation * (offset - Vector3.forward * distance);

            if(position[1] - smoothPosition[1] < 0.1 && position[1] - smoothPosition[1] > -0.1)
            {
                position[1] = smoothPosition[1];
            }

			//distance = Mathf.Clamp((Time.time - 15), 0, 1) * (maxDistance - minDistance) + minDistance;
			//// Position
			//position = smoothPosition + rotation * (offset - Vector3.forward * distance - Vector3.up * offset.y * (1 - ((distance - minDistance) / (maxDistance - minDistance))));        

			// Translating the camera
			transform.position = position;
			transform.rotation = rotation; // camera rotation
		}

		public Vector3 CamForward
		{
			get
			{
				return transform.rotation * Vector3.forward;
			}
		}

		// Zoom input
		private float zoomAdd
		{
			get
			{
				float scrollAxis = Input.GetAxis("Mouse ScrollWheel");
				if (scrollAxis > 0)
				{
					return -zoomSensitivity;
				}
				else if (scrollAxis < 0)
				{
					return zoomSensitivity;
				}
				else
				{
					return 0;
				}
			}
		}

		// Clamping Euler angles
		private float ClampAngle(float angle, float min, float max)
		{
			if (angle < -360)
			{
				angle += 360;
			}
			else if (angle > 360)
			{
				angle -= 360;
			}
			return Mathf.Clamp(angle, min, max);
		}

	}
}

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
	public static class DrawArrow
	{
		public static void ForGizmo(Vector3 pos, Vector3 direction, float arrowHeadLength = 0.5f, float arrowHeadAngle = 20.0f, float arrowPosition = 1f)
		{
			ForGizmo(pos, direction, Gizmos.color, arrowHeadLength, arrowHeadAngle, arrowPosition);
		}

		public static void ForGizmo(Vector3 pos, Vector3 direction, Color color, float arrowHeadLength = 0.5f, float arrowHeadAngle = 20.0f, float arrowPosition = 1f)
		{
			Gizmos.color = color;
			Gizmos.DrawRay(pos, direction);
			DrawArrowEnd(true, pos, direction, color, arrowHeadLength, arrowHeadAngle, arrowPosition);
		}

		public static void ForDebug(Vector3 pos, Vector3 direction, float arrowHeadLength = 0.5f, float arrowHeadAngle = 20.0f, float arrowPosition = 1f)
		{
			ForDebug(pos, direction, Color.white, arrowHeadLength, arrowHeadAngle, arrowPosition);
		}

		public static void ForDebug(Vector3 pos, Vector3 direction, Color color, float arrowHeadLength = 0.5f, float arrowHeadAngle = 20.0f, float arrowPosition = 1f)
		{
			if (Mathf.Approximately(direction.sqrMagnitude, 0))
				return;
			Debug.DrawRay(pos, direction, color);
			DrawArrowEnd(false, pos, direction, color, arrowHeadLength, arrowHeadAngle, arrowPosition);
		}
		private static void DrawArrowEnd(bool gizmos, Vector3 pos, Vector3 direction, Color color, float arrowHeadLength = 0.5f, float arrowHeadAngle = 20.0f, float arrowPosition = 1f)
		{
			if (Mathf.Abs(direction.magnitude) < 1e-3)
				return;
			Vector3 right = (Quaternion.LookRotation(direction) * Quaternion.Euler(arrowHeadAngle, 0, 0) * Vector3.back) * arrowHeadLength;
			Vector3 left = (Quaternion.LookRotation(direction) * Quaternion.Euler(-arrowHeadAngle, 0, 0) * Vector3.back) * arrowHeadLength;
			Vector3 up = (Quaternion.LookRotation(direction) * Quaternion.Euler(0, arrowHeadAngle, 0) * Vector3.back) * arrowHeadLength;
			Vector3 down = (Quaternion.LookRotation(direction) * Quaternion.Euler(0, -arrowHeadAngle, 0) * Vector3.back) * arrowHeadLength;

			Vector3 arrowTip = pos + (direction * arrowPosition);

			if (gizmos)
			{
				Gizmos.color = color;
				Gizmos.DrawRay(arrowTip, right);
				Gizmos.DrawRay(arrowTip, left);
				Gizmos.DrawRay(arrowTip, up);
				Gizmos.DrawRay(arrowTip, down);
			}
			else
			{
				Debug.DrawRay(arrowTip, right, color);
				Debug.DrawRay(arrowTip, left, color);
				Debug.DrawRay(arrowTip, up, color);
				Debug.DrawRay(arrowTip, down, color);
			}
		}
	}
}
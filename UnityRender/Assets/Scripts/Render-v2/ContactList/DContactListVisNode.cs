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
using UnityEditor;

namespace RenderV2
{

    public class DContactListVisNode : MonoBehaviour
    {
        public static float default_ball_radius = 0.1F;

        GameObject ball_child; // for visualize position
        Renderer ball_render;

        GameObject capsule_child; // for visualize orientation
        Renderer capsule_render;


        public void set_render_color(Color new_color)
        {
            if (ball_render != null)
            {
                ball_render.material.color = new_color;
            }
            if (capsule_render != null)
            {
                capsule_render.material.color = new_color;
            }
        }

        public float get_ball_radius()
        {
            if (ball_child == null)
            {
                return default_ball_radius;
            }
            else
            {
                return ball_child.transform.localScale.x;
            }
        }

        public void set_node_visible(bool flag)
        {
            gameObject.SetActive(flag);
            if (ball_child != null)
            {
                ball_child.SetActive(flag);
            }
            if (capsule_child != null)
            {
                capsule_child.SetActive(flag);
            }
        }

        public void set_pos_force(Vector3 new_pos, Vector3 new_force)
        {
            set_node_visible(true);

            transform.position = new_pos;
            ball_child.transform.position = new_pos;
            Quaternion global_rot = Quaternion.FromToRotation(Vector3.up, new_force);

            Vector3 scale = capsule_child.transform.localScale;
            scale.y = new_force.magnitude;
            capsule_child.transform.localScale = scale;
            Vector3 global_pos = new_pos + global_rot * new Vector3(0.0F, scale.y, 0.0F);
            capsule_child.transform.SetPositionAndRotation(global_pos, global_rot);
        }

        public void set_render_radius(float radius)
        {
            if (ball_child != null)
            {
                ball_child.transform.localScale = radius * Vector3.one;
            }
            if (capsule_child != null)
            {
                Vector3 scale = capsule_child.transform.localScale;
                capsule_child.transform.localScale = new Vector3(radius, scale.y, radius);
            }
        }

        public void clear_child()
        {
            if (ball_child != null)
            {
                DestroyImmediate(ball_child);
                ball_child = null;
            }
            if (capsule_child != null)
            {
                DestroyImmediate(capsule_child);
                capsule_child = null;
            }
        }

        public void create_child(float radius)
        {
            clear_child();
            ball_child = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            ball_child.transform.parent = transform;
            ball_child.transform.localScale = radius * Vector3.one;
            if (DCommonConfig.SupportGameObjectColor)
            {
                ball_render = ball_child.GetComponent<Renderer>();
            }
            else
            {
                ball_render = null;
            }

            capsule_child = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            capsule_child.transform.parent = transform;
            capsule_child.transform.localScale = new Vector3(radius, 1.0F, radius);
            if (DCommonConfig.SupportGameObjectColor)
            {
                capsule_render = capsule_child.GetComponent<Renderer>();
            }
            else
            {
                capsule_render = null;
            }
        }
    }
}
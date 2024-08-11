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
    public enum ContactVisType
    {
        UseGizmo, // render the contact by gizmo
        UseGameObject // render the contact by create game object
    };

    // use for rendering contact force in Unity
    public class DContactList : MonoBehaviour, IParseUpdate<ContactListUpdateInfo>, ICalcAttrs
    {
        public float ForceClipRatio = 0.1F;
        public float default_radius = 0.05F;

        public ContactVisType contact_vis_type = ContactVisType.UseGameObject; // we should not change this attribute at running.

        public Color vis_color = Color.red;

        // for visualize contact force..
        public List<Vector3> ForceVisualize = new List<Vector3>();

        public List<Vector3> PosVisualize = new List<Vector3>();

        public List<float> ContactLabelVisualize = new List<float>();

        [HideInInspector]
        public List<bool> ContactHighLight = new List<bool>();

        ContactListUpdateInfo cinfo_data;

        List<DContactListVisNode> VisNode;

        // We can visualize contact by a simple buffer (saves a ball and a capsule)
        public void CalcAttrs()
        {
            if (contact_vis_type == ContactVisType.UseGameObject)
            {
                initialize_render_buffer(50, default_radius);
            }
        }

        void set_node_color()
        {
            if (VisNode == null)
            {
                return;
            }
            for(int i = 0; i < VisNode.Count; i++)
            {
                VisNode[i].set_render_color(vis_color);
            }
        }

        void create_node_to_buffer(int i, float radius)
        {
            GameObject vis_obj = new GameObject("contact" + i);
            vis_obj.transform.parent = transform;
            DContactListVisNode node = vis_obj.AddComponent<DContactListVisNode>();
            node.create_child(radius);
            node.set_node_visible(false);
            node.set_render_color(vis_color);
            VisNode.Add(node);
        }

        void initialize_render_buffer(int max_buf_num, float radius)
        {
            if (VisNode == null)
            {
                VisNode = new List<DContactListVisNode>(max_buf_num);
            }
            else
            {
                VisNode.Clear();
                VisNode.Capacity = max_buf_num;
            }
            for(int i = 0; i < max_buf_num; i++) // we need not save GameObject here,
            {
                create_node_to_buffer(i, radius);
            }
        }

        void extend_render_buffer(int new_size, float radius)
        {
            if (VisNode == null)
            {
                initialize_render_buffer(new_size, radius);
            }
            else
            {
                VisNode.Capacity = new_size;
                int curr_count = VisNode.Count;
                for(int i = curr_count; i < new_size; i++)
                {
                    create_node_to_buffer(i, radius);
                }
            }
        }

        void set_vis_node_visible(bool flag, int start_index = 0)
        {
            if (VisNode == null)
            {
                return;
            }
            for(int i = start_index; i < VisNode.Count; i++)
            {
                DContactListVisNode node = VisNode[i];
                node.set_node_visible(flag);
            }
        }

        void update_render_buffer(ContactListUpdateInfo cinfo)
        {
            if (cinfo == null || cinfo.Length == 0)
            {
                set_vis_node_visible(false, 0);
                return;
            }
            int length = cinfo.Length;
            if (VisNode == null)
            {
                initialize_render_buffer(length, default_radius);
            }
            float radius = VisNode[0].get_ball_radius();
            if (length > VisNode.Count)
            {
                extend_render_buffer(length, radius);
            }
            set_vis_node_visible(false, 0);
            for(int i = 0; i < length; i++)
            {
                Vector3 pos = Utils.ArrToVector3(cinfo.Joints[i].Position);
                Vector3 force = ForceClipRatio * Utils.ArrToVector3(cinfo.Joints[i].Force);
                DContactListVisNode node = VisNode[i];
                node.set_pos_force(pos, force);
            }
        }

        public void set_for_vis()
        {
            if (cinfo_data != null)
            {
                if (ForceVisualize == null)
                {
                    ForceVisualize = new List<Vector3>();
                }
                if (PosVisualize == null)
                {
                    PosVisualize = new List<Vector3>();
                }
                if (ContactLabelVisualize == null)
                {
                    ContactLabelVisualize = new List<float>();
                }
                ForceVisualize.Clear();
                PosVisualize.Clear();
                ContactLabelVisualize.Clear();

                for(int i = 0; i < cinfo_data.Joints.Length; i++)
                {
                    Vector3 pos = Utils.ArrToVector3(cinfo_data.Joints[i].Position);
                    Vector3 force = Utils.ArrToVector3(cinfo_data.Joints[i].Force);
                    ForceVisualize.Add(force);
                    PosVisualize.Add(pos);
                    ContactLabelVisualize.Add(cinfo_data.Joints[i].ContactLabel);
                }
            }
            else
            {
                // do nothing here..
            }
        }

        // Visualize by Gizmo is ugly..
        public void OnDrawGizmos()
        {
            if (contact_vis_type != ContactVisType.UseGizmo)
            {
                return;
            }

            Gizmos.color = vis_color;
            for (int i = 0; i < PosVisualize.Count; i++)
            {
                Gizmos.DrawLine(PosVisualize[i], PosVisualize[i] + ForceVisualize[i] * ForceClipRatio);
            }
        }

        public void ParseUpdateInfo(ContactListUpdateInfo info)
        {
            cinfo_data = info; // should we add lock..?
            set_for_vis();
            if (contact_vis_type == ContactVisType.UseGameObject)
            {
                update_render_buffer(info);
            }
        }
    }
}


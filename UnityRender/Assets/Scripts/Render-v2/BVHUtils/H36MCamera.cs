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

using RenderV2.Eigen;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;

namespace RenderV2
{
    public class H36MCameraIn
    {
        public int cam_id;
        public Vector2 center;
        public Vector2 focal;
        public float focal_avg;
        public Vector2 res_w_h;
        public H36MCameraIn(int cam_id_, Vector2 center_, Vector2 focal_, Vector2 res_w_h_)
        {
            cam_id = cam_id_;
            focal = focal_;
            focal_avg = (focal.x + focal.y) / res_w_h_.x;
            res_w_h = res_w_h_;
            center = normalize_screen_coordinates(center_);
        }

        public Vector2 normalize_screen_coordinates(Vector2 vec)
        {
            return 2 * vec / res_w_h.x - new Vector2(1, res_w_h.y / res_w_h.x);
        }

        public Vector2 image_coordinates(Vector2 vec)
        {
            return (vec + new Vector2(1, res_w_h.y / res_w_h.x)) * res_w_h.x / 2;
        }

        /* def image_coordinates(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == 2
        w, h = self.res_w, self.res_h
        res: np.ndarray = (x + [1, h / w]) * w / 2

        return res*/

        /* def normalize_screen_coordinates(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == 2
        w, h = self.res_w, self.res_h
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return x / w* 2 - np.array([1, h / w], dtype= x.dtype) */
    }

    public class H36MCameraEx
    {
        public Vector3 pos;
        public Quaternion quat;
        public H36MCameraEx(Vector3 pos_, Quaternion quat_)
        {
            var mat_quat_inv = new Eigen.Matrix3f(Quaternion.Inverse(quat_));
            var mat_r = new Eigen.Matrix3f(1, 0, 0, 0, 0, -1, 0, 1, 0);
            var mat_r_inv = new Eigen.Matrix3f(1, 0, 0, 0, 0, 1, 0, -1, 0);
            var tmp_mat = new Eigen.Matrix3f(1, 0, 0, 0, -1, 0, 0, 0, 1);

            var quat_inv = (tmp_mat * mat_quat_inv * mat_r).ToQuatUnity();
            quat = Quaternion.Inverse(quat_inv);
            pos = mat_r_inv * (1e-3f * pos_);
            /* 
            // q_inv.apply(y_up_to_z_up(p) - trans)
            // Q^{-1} @ (R @ p - trans)
            // (Q^{-1} @ R) (p - R^{-1} @ trans) 
            // emm..we should flip y component. matrix is (1, 0, 0, 0, -1, 0, 0, 0, 1)
             */
        }
    }

    public enum H36MSubject
    {
        S1,
        S5,
        S6,
        S7,
        S8,
        S9,
        S11
    }

    public enum H36MCamID
    {
        id54138969 = 54138969,
        id55011271 = 55011271,
        id58860488 = 58860488,
        id60457274 = 60457274
    }

    public class CamList
    {
        public H36MCameraIn[] cam_in = {
            new H36MCameraIn(54138969, new Vector2(512.54150390625f, 515.4514770507812f), new Vector2(1145.0494384765625f, 1143.7811279296875f), new Vector2(1000, 1002)),
            new H36MCameraIn(55011271, new Vector2(508.8486328125f, 508.0649108886719f), new Vector2(1149.6756591796875f, 1147.5916748046875f), new Vector2(1000, 1000)),
            new H36MCameraIn(58860488, new Vector2(519.8158569335938f, 501.40264892578125f), new Vector2(1149.1407470703125f, 1148.7989501953125f), new Vector2(1000, 1000)),
            new H36MCameraIn(60457274, new Vector2(514.9682006835938f, 501.88201904296875f), new Vector2(1145.5113525390625f, 1144.77392578125f), new Vector2(1000, 1002))
        };

        public int[] cam_ids = { 54138969, 55011271, 58860488, 60457274 };

        public H36MCameraEx[] cam_ex_s1 = {
            new H36MCameraEx(new Vector3(1841.1070556640625f, 4955.28466796875f, 1563.4454345703125f), new Quaternion(-0.1500701755285263f, -0.755240797996521f, 0.6223280429840088f, 0.1407056450843811f)),
            new H36MCameraEx(new Vector3(1761.278564453125f, -5078.0068359375f, 1606.2650146484375f), new Quaternion(-0.764836311340332f, -0.14833825826644897f, 0.11794740706682205f, 0.6157187819480896f)),
            new H36MCameraEx(new Vector3(-1846.7777099609375f, 5215.04638671875f, 1491.972412109375f), new Quaternion(-0.14647851884365082f, 0.7653023600578308f, -0.6094175577163696f, 0.14651472866535187f)),
            new H36MCameraEx(new Vector3(-1794.7896728515625f, -3722.698974609375f, 1574.8927001953125f), new Quaternion(-0.7853162288665771f, 0.14548823237419128f, -0.14749594032764435f, 0.5834008455276489f))
        };

        public H36MCameraEx[] cam_ex_s5 = {
            new H36MCameraEx(new Vector3(2097.3916015625f, 4880.94482421875f, 1605.732421875f), new Quaternion(-0.162370964884758f, -0.7551892995834351f, 0.6178938746452332f, 0.1467377245426178f)),
            new H36MCameraEx(new Vector3(2031.7008056640625f, -5167.93310546875f, 1612.923095703125f), new Quaternion( -0.7626792192459106f, -0.15728192031383514f, 0.1189815029501915f, 0.6159758567810059f)),
            new H36MCameraEx(new Vector3(-1620.5948486328125f, 5171.65869140625f, 1496.43701171875f), new Quaternion(-0.12907841801643372f, 0.7678384780883789f, -0.6110143065452576f, 0.14291371405124664f)),
            new H36MCameraEx(new Vector3(-1637.1737060546875f, -3867.3173828125f, 1547.033203125f), new Quaternion(-0.7814217805862427f, 0.1274748593568802f, -0.15036417543888092f, 0.5920479893684387f))
        };

        public H36MCameraEx[] cam_ex_s6 = {
            new H36MCameraEx(new Vector3(1935.4517822265625f, 4950.24560546875f, 1618.0838623046875f), new Quaternion(-0.15692396461963654f, -0.7571090459823608f, 0.6198879480361938f, 0.1337897777557373f)),
            new H36MCameraEx(new Vector3(1969.803955078125f, -5128.73876953125f, 1632.77880859375f), new Quaternion(-0.7628812789916992f, -0.16174767911434174f, 0.11819244921207428f, 0.6147197484970093f)),
            new H36MCameraEx(new Vector3(-1769.596435546875f, 5185.361328125f, 1476.993408203125f), new Quaternion(-0.13529130816459656f, 0.7646096348762512f, -0.6112781167030334f, 0.1529948115348816f)),
            new H36MCameraEx(new Vector3(-1721.668701171875f, -3884.13134765625f, 1540.4879150390625f), new Quaternion(-0.7804774045944214f, 0.12832270562648773f, -0.1561593860387802f, 0.5916101336479187f)),
        };

        public H36MCameraEx[] cam_ex_s7 = {
            new H36MCameraEx(new Vector3(1974.512939453125f, 4926.3544921875f, 1597.8326416015625f), new Quaternion(-0.1631336808204651f, -0.7548328638076782f, 0.6188824772834778f, 0.1435241848230362f)),
            new H36MCameraEx(new Vector3(1937.0584716796875f, -5119.7900390625f, 1631.5665283203125f), new Quaternion(-0.7638262510299683f, -0.1596645563840866f, 0.1177929937839508f, 0.6141672730445862f)),
            new H36MCameraEx(new Vector3(-1741.8111572265625f, 5208.24951171875f, 1464.8245849609375f), new Quaternion(-0.12874816358089447f, 0.7660516500473022f, -0.6127139329910278f, 0.14550060033798218f)),
            new H36MCameraEx(new Vector3(-1734.7105712890625f, -3832.42138671875f, 1548.5830078125f), new Quaternion(-0.7821764349937439f, 0.12445473670959473f, -0.15196487307548523f, 0.5912848114967346f)),
        };

        public H36MCameraEx[] cam_ex_s8 = {
            new H36MCameraEx(new Vector3(2150.65185546875f, 4896.1611328125f, 1611.9046630859375f), new Quaternion(-0.15589867532253265f, -0.7561917304992676f, 0.619644045829773f, 0.14110587537288666f)),
            new H36MCameraEx(new Vector3(2219.965576171875f, -5148.453125f, 1613.0440673828125f), new Quaternion(-0.7647668123245239f, -0.14846350252628326f, 0.11158157885074615f, 0.6169601678848267f)),
            new H36MCameraEx(new Vector3(-1571.2215576171875f, 5137.0185546875f, 1498.1761474609375f), new Quaternion(-0.13377119600772858f, 0.7670128345489502f, -0.6100369691848755f, 0.1471444070339203f)),
            new H36MCameraEx(new Vector3(-1476.913330078125f, -3896.7412109375f, 1547.97216796875f), new Quaternion(-0.7825870513916016f, 0.12147816270589828f, -0.14631995558738708f, 0.5927824378013611f)),
        };

        public H36MCameraEx[] cam_ex_s9 = {
            new H36MCameraEx(new Vector3(2044.45849609375f, 4935.1171875f, 1481.2275390625f), new Quaternion(-0.15548215806484222f, -0.7532095313072205f, 0.6199594736099243f, 0.15540587902069092f)),
            new H36MCameraEx(new Vector3(1990.959716796875f, -5123.810546875f, 1568.8048095703125f), new Quaternion(-0.7634735107421875f, -0.14132238924503326f, 0.11933968216180801f, 0.618784487247467f)),
            new H36MCameraEx(new Vector3(-1670.9921875f, 5211.98583984375f, 1528.387939453125f), new Quaternion(-0.7634735107421875f, 0.7634735107421875f, -0.7634735107421875f, 0.7634735107421875f)),
            new H36MCameraEx(new Vector3(-1696.04345703125f, -3827.099853515625f, 1591.4127197265625f), new Quaternion(-0.7634735107421875f, 0.7634735107421875f, -0.7634735107421875f, 0.7634735107421875f)),
        };

        public H36MCameraEx[] cam_ex_s11 = {
            new H36MCameraEx(new Vector3(2098.440185546875f, 4926.5546875f, 1500.278564453125f), new Quaternion(-0.15442320704460144f, -0.7547563314437866f, 0.6191070079803467f, 0.15232472121715546f)),
            new H36MCameraEx(new Vector3(2083.182373046875f, -4912.1728515625f, 1561.07861328125f), new Quaternion(-0.7600917220115662f, -0.15300633013248444f, 0.1255258321762085f, 0.6189449429512024f)),
            new H36MCameraEx(new Vector3(-1609.8153076171875f, 5177.3359375f, 1537.896728515625f), new Quaternion(-0.15650227665901184f, 0.7681233882904053f, -0.6026304364204407f, 0.14943228662014008f)),
            new H36MCameraEx(new Vector3(-1590.738037109375f, -3854.1689453125f, 1578.017578125f), new Quaternion(-0.7818877100944519f, 0.13991211354732513f, -0.14715361595153809f, 0.5894251465797424f)),
        };

        public Dictionary<H36MSubject, H36MCameraEx[]> cam_ex = new Dictionary<H36MSubject, H36MCameraEx[]>(7);

        public CamList()
        {
            cam_ex.Add(H36MSubject.S1, cam_ex_s1);
            cam_ex.Add(H36MSubject.S5, cam_ex_s5);
            cam_ex.Add(H36MSubject.S6, cam_ex_s6);
            cam_ex.Add(H36MSubject.S7, cam_ex_s7);
            cam_ex.Add(H36MSubject.S8, cam_ex_s8);
            cam_ex.Add(H36MSubject.S9, cam_ex_s9);
            cam_ex.Add(H36MSubject.S11, cam_ex_s11);
        }
        
        public void SetToCamera(Camera camera, H36MSubject subject, H36MCamID cam_id)
        {
            int sel_index = -1;
            for (int i = 0; i < 4; i++)
            {
                if (cam_ids[i] == ((int)cam_id))
                {
                    sel_index = i;
                    break;
                }
            }
            H36MCameraIn in_param = cam_in[sel_index];
            H36MCameraEx ex_param = cam_ex[subject][sel_index];
            camera.usePhysicalProperties = false;
            camera.fieldOfView = Mathf.Rad2Deg * 2 * Mathf.Atan2(1, in_param.focal_avg);
            camera.transform.SetPositionAndRotation(ex_param.pos, ex_param.quat);

            // here we should set the center.
        }
    }

    public class H36MCamera : MonoBehaviour
    {
        public H36MSubject h36m_subject = H36MSubject.S1;
        public H36MCamID h36m_id = H36MCamID.id54138969;
        public GameObject camera_obj;

        CamList camlist = new CamList();
        Camera camera_handle;

        private void Start()
        {
            if (camera_obj != null)
            {
                camera_handle = camera_obj.GetComponent<Camera>();
                camlist.SetToCamera(camera_handle, h36m_subject, h36m_id);
            }
        }

        private void Update()
        {
            
        }

        public void LoadFromFile()
        {
            string video_dir = EditorUtility.OpenFilePanelWithFilters("Video File", ".", new string[] { "Video File", "mp4" });
            if (video_dir is null || video_dir.Length == 0)
            {
                EditorUtility.DisplayDialog("Info", "File not exist", "OK");
                return;
            }
            string bvh_dir = EditorUtility.OpenFilePanelWithFilters("BVH Mocap File", ".", new string[] { "BVH Mocap File", "bvh" });
            if (bvh_dir is null || bvh_dir.Length == 0)
            {
                EditorUtility.DisplayDialog("Info", "File not exist", "OK");
                return;
            }

        }
    }
}

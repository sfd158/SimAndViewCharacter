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
using System.Runtime.InteropServices;

namespace RenderV2.Eigen
{
    public class LinearLayer
    {
        IntPtr layer;
        int input_size_, output_size_;
        public LinearLayer(int input_size, int output_size)
        {
            input_size_ = input_size;
            output_size_ = output_size;
            layer = LinearLayer_new(input_size, output_size);
        }

        ~LinearLayer()
        {
            if (layer != IntPtr.Zero)
            {
                LinearLayer_del(layer);
                layer = IntPtr.Zero;
            }
        }

        public int input_size => input_size_;

        public int output_size => output_size_;

        public Tensor forward(Tensor x)
        {
            if (x == null || x.Ptr == IntPtr.Zero) throw new NullReferenceException();
            if (x.rows != input_size_ && x.cols != 1) throw new ArgumentException();
            Tensor res = new Tensor(output_size_, 1);
            LinearLayer_forward(layer, x.GetPtr(), res.GetPtr());
            return res;
        }

        [DllImport("EigenForUnity")]
        static extern IntPtr LinearLayer_new(int input_size, int output_size);

        [DllImport("EigenForUnity")]
        static extern void LinearLayer_del(IntPtr ptr);

        [DllImport("EigenForUnity")]
        static extern void LinearLayer_forward(IntPtr obj, IntPtr x, IntPtr res);
    }   
}
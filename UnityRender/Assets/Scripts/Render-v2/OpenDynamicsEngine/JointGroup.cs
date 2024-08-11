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
using UnityEngine;

namespace RenderV2.ODE
{
    /// <summary>
    /// joint group
    /// </summary>
    public class JointGroup
    {
        protected UIntPtr _gid = UIntPtr.Zero;
        public JointGroup()
        {
            _gid = dJointGroupCreate(0);
        }

        public JointGroup(UIntPtr gid)
        {
            _gid = gid;
        }

        public void Empty()
        {
            dJointGroupEmpty(_gid);
        }

        /// <summary>
        ///  Destroy all joints in the group.
        /// </summary>
        public void Destroy()
        {
            if (_gid != UIntPtr.Zero)
            {
                dJointGroupDestroy(_gid);
                _gid = UIntPtr.Zero;
            }
        }

        public override bool Equals(object other)
        {
            if (other is null) return false;
            if (GetType() != other.GetType()) return false;
            return _gid == ((JointGroup)other)._gid;
        }

        public override int GetHashCode() => _gid.GetHashCode();

        public static bool operator ==(JointGroup lhs, JointGroup rhs)
        {
            if (lhs is null) return rhs is null;
            if (rhs is null) return false;
            return lhs._gid == rhs._gid;
        }

        public static bool operator !=(JointGroup lhs, JointGroup rhs) => !(lhs == rhs);

        public UIntPtr gid { get => _gid;  }

        public UIntPtr GetPtr() => _gid;

        public void AddJoint(ODEJoint joint)
        {
            if (joint == null) throw new ArgumentNullException();
            
        }

        public bool JointInGroup(UIntPtr joint)
        {
            return dJointInGroup(_gid, joint) != 0;
        }

        public bool JointInGroup(ODEJoint joint)
        {
            return dJointInGroup(_gid, joint.GetPtr()) != 0;
        }

        [DllImport("ModifyODE")]
        private static extern UIntPtr dJointGroupCreate(int max_size);

        [DllImport("ModifyODE")]
        private static extern void dJointGroupEmpty(UIntPtr gid);

        [DllImport("ModifyODE")]
        private static extern void dJointGroupDestroy(UIntPtr gid);

        [DllImport("ModifyODE")]
        private static extern int dJointInGroup(UIntPtr group, UIntPtr joint);
    }
}
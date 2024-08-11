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

using static RenderV2.ODE.ODEHeader;
using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace RenderV2.ODE
{
    /// <summary>
    /// This class represents a contact between two bodies in one point.
    /// A Contact object stores all the input parameters for a ContactJoint.
    /// This class wraps the ODE dContact structure which has 3 components::
    /// 
    ///struct dContact
    ///{
    ///    dSurfaceParameters surface;
    ///    dContactGeom geom;
    ///    dVector3 fdir1;
    ///};
    /// This wrapper class provides methods to get and set the items of those structures.
    /// </summary>
    public class Contact
    {
        public dContact _contact;

        public Contact()
        {
            _contact.surface.mode = (int) ContactType.dContactApprox1;
            _contact.surface.mu = CommonFunc.dInfinity;
            _contact.surface.bounce = 0.0;
        }

        public void EnableSoftCfmErp(double cfm, double erp)
        {
            _contact.surface.soft_cfm = cfm;
            _contact.surface.soft_erp = erp;
            _contact.surface.mode |= (int)ContactType.dContactSoftCFM | (int)ContactType.dContactSoftERP;
        }

        public void EnableContactSlip(double slip)
        {
            _contact.surface.slip1 = slip;
            _contact.surface.slip2 = slip;
            _contact.surface.mode |= (int)ContactType.dContactSlip1 | (int)ContactType.dContactSlip2;
        }

        public bool Slip1Enabled
        {
            get => (_contact.surface.mode & (int)ContactType.dContactSlip1) > 0;
            set => _contact.surface.mode |= (int)ContactType.dContactSlip1;
        }

        public bool Slip2Enabled
        {
            get => (_contact.surface.mode & (int)ContactType.dContactSlip2) > 0;
            set => _contact.surface.mode |= (int)ContactType.dContactSlip2;
        }

        public int mode
        {
            get => _contact.surface.mode;
            set => _contact.surface.mode = value;
        }

        /// <summary>
        /// Return the Coulomb friction coefficient.
        /// </summary>
        public double mu
        {
            get => _contact.surface.mu;
            set => _contact.surface.mu = value;
        }

        /// <summary>
        /// Return the optional Coulomb friction coefficient for direction 2.
        /// </summary>
        public double mu2
        {
            get => _contact.surface.mu2;
            set => _contact.surface.mu2 = value;
        }

        /// <summary>
        /// the restitution parameter.
        /// </summary>
        public double bounce
        {
            get => _contact.surface.bounce;
            set => _contact.surface.bounce = value;
        }

        /// <summary>
        /// the minimum incoming velocity necessary for bounce.
        /// </summary>
        public double bounceVel
        {
            get => _contact.surface.bounce_vel;
            set => _contact.surface.bounce_vel = value;
        }

        /// <summary>
        /// the contact normal softness parameter.
        /// </summary>
        public double SoftERP
        {
            get => _contact.surface.soft_erp;
            set => _contact.surface.soft_erp = value;
        }

        /// <summary>
        /// the contact normal "softness" parameter.
        /// </summary>
        public double SoftCFM
        {
            get => _contact.surface.soft_cfm;
            set => _contact.surface.soft_cfm = value;
        }

        /// <summary>
        /// the surface velocity in friction direction 1.
        /// </summary>
        public double Motion1
        {
            get => _contact.surface.motion1;
            set => _contact.surface.motion1 = value;
        }

        /// <summary>
        /// the surface velocity in friction direction 2.
        /// </summary>
        public double Motion2
        {
            get => _contact.surface.motion2;
            set => _contact.surface.motion2 = value;
        }

        //the coefficient of force-dependent-slip (FDS) for friction direction 1.
        public double Slip1
        {
            get => _contact.surface.slip1;
            set => _contact.surface.slip1 = value;
        }

        /// <summary>
        /// the coefficient of force-dependent-slip (FDS) for friction direction 2.
        /// </summary>
        public double Slip2
        {
            get => _contact.surface.slip2;
            set => _contact.surface.slip2 = value;
        }

        /// <summary>
        /// the "first friction direction" vector that defines a direction
        /// along which frictional force is applied.
        /// </summary>
        public dVector3 Fdir1
        {
            get => _contact.fdir1;
            set => _contact.fdir1 = value;
        }

        public double[] ContactPos
        {
            get => new[] { _contact.geom.pos.x, _contact.geom.pos.y, _contact.geom.pos.z };
            set => _contact.geom.pos.setValue(value);
        }

        /// <summary>
        /// the normal vector of contact
        /// </summary>
        public double[] ContactNormal
        {
            get => new[] { _contact.geom.normal.x, _contact.geom.normal.y, _contact.geom.normal.z };
            set => _contact.geom.normal.setValue(value);
        }

        /// <summary>
        /// Depth of contact
        /// </summary>
        public double ContactDepth
        {
            get => _contact.geom.depth;
            set => _contact.geom.depth = value;
        }

        /// <summary>
        /// Contact Geom 1
        /// </summary>
        public UIntPtr Contactgeom1
        {
            get => _contact.geom.g1;
            set => _contact.geom.g1 = value;
        }

        public UIntPtr Contactgeom2
        {
            get => _contact.geom.g2;
            set => _contact.geom.g2 = value;
        }

        public (double[], double[], double, UIntPtr, UIntPtr) GetContactgeomParams()
        {
            return (ContactPos, ContactNormal, ContactDepth, Contactgeom1, Contactgeom2);
        }

        public void SetContactgeomParams(double[] pos, double[] normal, double depth, UIntPtr g1 = default, UIntPtr g2 = default)
        {
            ContactPos = pos;
            ContactNormal = normal;
            ContactDepth = depth;
            Contactgeom1 = g1;
            Contactgeom2 = g2;
        }
    }

    /// <summary>
    /// base class for all types of contact joint
    /// </summary>
    public class ContactJointBase: ODEJoint
    {
        Contact _contact;
        public ContactJointBase()
        {

        }

        public Contact contact => _contact;

        public double joint_erp
        {
            get => dJointGetContactParam(jid, (int)ParamType.dParamERP);
            set => dJointSetContactParam(jid, (int)ParamType.dParamERP, value);
        }

        public double joint_cfm
        {
            get => dJointGetContactParam(jid, (int)ParamType.dParamCFM);
            set => dJointSetContactParam(jid, (int)ParamType.dParamCFM, value);
        }

        /* public double slip1
        {
            get => _contact.surface.slip1;
        }*/

        [DllImport("ModifyODE")]
        static extern UIntPtr dJointCreateContact(UIntPtr w, UIntPtr jointgroup, ref dContact contact);

        [DllImport("ModifyODE")]
        static extern double dJointGetContactParam(UIntPtr joint, int param);

        [DllImport("ModifyODE")]
        static extern void dJointSetContactParam(UIntPtr joint, int param, double val);
    }
    /*
     cdef class ContactJointBase(Joint):
    """
    base class for all types of contact joint
    """
    cdef Contact _contact

    def __cinit__(self, *a, **kw):
        pass

    def __init__(self, *a, **kw):
        raise ValueError("Don't use base class directly.")

    @property
    def joint_erp(self) -> dReal:
        return dJointGetContactParam(self.jid, dParamERP)

    @joint_erp.setter
    def joint_erp(self, dReal value):
        dJointSetContactParam(self.jid, dParamERP, value)

    @property
    def joint_cfm(self) -> dReal:
        return dJointGetContactParam(self.jid, dParamCFM)

    @joint_cfm.setter
    def joint_cfm(self, dReal value):
        dJointSetContactParam(self.jid, dParamCFM, value)

    @property
    def joint_slip1(self):
        return self._contact.surface.slip1

    @joint_slip1.setter
    def joint_slip1(self, dReal value):
        self._contact.surface.slip1 = value

    @property
    def joint_slip2(self):
        return self._contact.surface.slip2

    @joint_slip2.setter
    def joint_slip2(self, dReal value):
        self._contact.surface.slip1 = value

    @property
    def contact(self) -> Contact:
        return self._contact

    @property
    def mode(self) -> int:
        return self._contact.mode

    @property
    def mu(self) -> dReal:
        return self._contact.mu

    @property
    def bounce(self) -> dReal:
        return self._contact.bounce

    @property
    def contactPosNumpy(self) -> np.ndarray:
        return self._contact.contactPosNumpy

    @property
    def contactNormalNumpy(self) -> np.ndarray:
        return self._contact.contactNormalNumpy

    @property
    def contactDepth(self) -> dReal:
        return self._contact.contactDepth

    @property
    def contactGeom1(self) -> GeomObject:
        return self._contact.contactGeom1

    @property
    def contactGeom2(self) -> GeomObject:
        return self._contact.contactGeom2
     */
}
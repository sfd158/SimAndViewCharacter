# cython: language_level=3, emit_code_comments=True, embedsignature=True
######################################################################
# Python Open Dynamics Engine Wrapper
# Copyright (C) 2004 PyODE developers (see file AUTHORS)
# All rights reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of EITHER:
#   (1) The GNU Lesser General Public License as published by the Free
#       Software Foundation; either version 2.1 of the License, or (at
#       your option) any later version. The text of the GNU Lesser
#       General Public License is included with this library in the
#       file LICENSE.
#   (2) The BSD-style license that is included with this library in
#       the file LICENSE-BSD.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
# LICENSE and LICENSE-BSD for more details.
######################################################################

from ModifyODE cimport *
import numpy as np
cimport numpy as np
cimport numpy as cnp
cimport cython
from libcpp.vector cimport vector as std_vector
from libc.math cimport abs as cmath_abs

ContactApprox1 = dContactApprox1
ContactSlip1 = dContactSlip1
ContactSlip2 = dContactSlip2

AMotorUser = dAMotorUser
AMotorEuler = dAMotorEuler

Infinity = dInfinity

import weakref
# Note: if a = weakref.proxy(b), c = a, then type(c) == weakproxy, sys.getrefcount(b) will NOT be increased.
# However, if a = weakref.ref(b), c = a(), then type(c) == type(b), sys.getrefcount(b) will be increased.

cnp.import_array()

# here we should make sure sizeof(size_t) == 8,
# that is, compile at 64 bit.
assert sizeof(size_t) == 8


cdef class GeomTypes:
    Sphere = dSphereClass
    Box = dBoxClass
    Capsule = dCapsuleClass
    Cylinder = dCylinderClass
    Plane = dPlaneClass

cdef class JointTypes:
    JointNone = dJointTypeNone
    Ball = dJointTypeBall
    Hinge = dJointTypeHinge
    Slider = dJointTypeSlider
    Contact = dJointTypeContact
    Universal = dJointTypeUniversal
    Hinge2 = dJointTypeHinge2
    Fixed = dJointTypeFixed
    JointNull = dJointTypeNull
    Amotor = dJointTypeAMotor
    Lmotor = dJointTypeLMotor
    Plane2D = dJointTypePlane2D
    PR = dJointTypePR
    PU = dJointTypePU
    Piston = dJointTypePiston
    Contact2 = dJointTypeContact2

"""
enum {
  dSphereClass = 0,
  dBoxClass,
  dCapsuleClass,
  dCylinderClass,
  dPlaneClass,
  dRayClass,
  dConvexClass,
  dGeomTransformClass,
  dTriMeshClass,
  dHeightfieldClass,

  dFirstSpaceClass,
  dSimpleSpaceClass = dFirstSpaceClass,
  dHashSpaceClass,
  dSweepAndPruneSpaceClass, // SAP
  dQuadTreeSpaceClass,
  dLastSpaceClass = dQuadTreeSpaceClass,

  dFirstUserClass,
  dLastUserClass = dFirstUserClass + dMaxUserClasses - 1,
  dGeomNumClasses
}
"""
 
cdef class JointParam:
    ParamLoStop = dParamLoStop
    ParamHiStop = dParamHiStop
    ParamVel = dParamVel
    ParamFMax = dParamFMax
    ParamFudgeFactor = dParamFudgeFactor
    ParamBounce = dParamBounce
    ParamCFM = dParamCFM
    ParamStopERP = dParamStopERP
    ParamStopCFM = dParamStopCFM
    ParamSuspensionERP = dParamSuspensionERP
    ParamSuspensionCFM = dParamSuspensionCFM

    ParamLoStop2 = dParamLoStop2
    ParamHiStop2 = dParamHiStop2
    ParamVel2 = dParamVel2
    ParamFMax2 = dParamFMax2
    ParamFudgeFactor2 = dParamFudgeFactor2
    ParamBounce2 = dParamBounce2
    ParamCFM2 = dParamCFM2
    ParamStopERP2 = dParamStopERP2
    ParamStopCFM2 = dParamStopCFM2
    ParamSuspensionERP2 = dParamSuspensionERP2
    ParamSuspensionCFM2 = dParamSuspensionCFM2

    ParamLoStop3 = dParamLoStop3
    ParamHiStop3 = dParamHiStop3
    ParamVel3 = dParamVel3
    ParamFMax3 = dParamFMax3
    ParamFudgeFactor3 = dParamFudgeFactor3
    ParamBounce3 = dParamBounce3
    ParamCFM3 = dParamCFM3
    ParamStopERP3 = dParamStopERP3
    ParamStopCFM3 = dParamStopCFM3
    ParamSuspensionERP3 = dParamSuspensionERP3
    ParamSuspensionCFM3 = dParamSuspensionCFM3

# paramLoStop        = 0
# paramHiStop        = 1
# paramVel           = 2
# paramLoVel         = 3
# paramHiVel         = 4
# paramFMax          = 5
# paramFudgeFactor   = 6
# paramBounce        = 7
# paramCFM           = 8
# paramStopERP       = 9
# paramStopCFM       = 10
# paramSuspensionERP = 11
# paramSuspensionCFM = 12
# paramERP           = 13

# ParamLoStop        = 0
# ParamHiStop        = 1
# ParamVel           = 2
# aramLoVel         = 3
# ParamHiVel         = 4
# ParamFMax          = 5
# ParamFudgeFactor   = 6
# ParamBounce        = 7
# ParamCFM           = 8
# ParamStopERP       = 9
# ParamStopCFM       = 10
# ParamSuspensionERP = 11
# ParamSuspensionCFM = 12
# ParamERP           = 13

# ParamLoStop2        = 256 + 0
# ParamHiStop2        = 256 + 1
# ParamVel2           = 256 + 2
# ParamLoVel2         = 256 + 3
# ParamHiVel2         = 256 + 4
# ParamFMax2          = 256 + 5
# ParamFudgeFactor2   = 256 + 6
# ParamBounce2        = 256 + 7
# ParamCFM2           = 256 + 8
# ParamStopERP2       = 256 + 9
# ParamStopCFM2       = 256 + 10
# ParamSuspensionERP2 = 256 + 11
# ParamSuspensionCFM2 = 256 + 12
# ParamERP2           = 256 + 13

# ParamLoStop3        = 512 + 0
# ParamHiStop3        = 512 + 1
# ParamVel3           = 512 + 2
# ParamLoVel3         = 512 + 3
# ParamHiVel3         = 512 + 4
# ParamFMax3          = 512 + 5
# ParamFudgeFactor3   = 512 + 6
# ParamBounce3        = 512 + 7
# ParamCFM3           = 512 + 8
# ParamStopERP3       = 512 + 9
# ParamStopCFM3       = 512 + 10
# ParamSuspensionERP3 = 512 + 11
# ParamSuspensionCFM3 = 512 + 12
# ParamERP3           = 512 + 13

# ParamGroup = 256

# ContactMu2          = 0x001
# ContactAxisDep      = 0x001
# ContactFDir1        = 0x002
# ContactBounce       = 0x004
# ContactSoftERP      = 0x008
# ContactSoftCFM      = 0x010
# ContactMotion1      = 0x020
# ContactMotion2      = 0x040
# ContactMotionN      = 0x080
# ContactSlip1        = 0x100
# ContactSlip2        = 0x200
# ContactRolling      = 0x400

# ContactApprox0      = 0x0000
# ContactApprox1_1    = 0x1000
# ContactApprox1_2    = 0x2000

class SimulationFailError(ValueError):
    def __init__(self, *args):
        super().__init__(*args)


cdef void _init_aabb_impl(dReal * aabb_res):
    aabb_res[0] = dInfinity
    aabb_res[1] = -dInfinity
    aabb_res[2] = dInfinity
    aabb_res[3] = -dInfinity
    aabb_res[4] = dInfinity
    aabb_res[5] = -dInfinity


cdef void _get_body_aabb_impl(dBodyID b, dReal * aabb_res):
    """
    compute AABB bounding box of rigid body
    """
    cdef dGeomID g = dBodyGetFirstGeom(b)
    cdef dReal aabb[6]
    while g != NULL:
        dGeomGetAABB(g, aabb)
        if aabb_res[0] > aabb[0]:
            aabb_res[0] = aabb[0]

        if aabb_res[1] < aabb[1]:
            aabb_res[1] = aabb[1]

        if aabb_res[2] > aabb[2]:
            aabb_res[2] = aabb[2]

        if aabb_res[3] < aabb[3]:
            aabb_res[3] = aabb[3]

        if aabb_res[4] > aabb[4]:
            aabb_res[4] = aabb[4]

        if aabb_res[5] < aabb[5]:
            aabb_res[5] = aabb[5]

        g = dGeomGetBodyNext(g)


cdef class Mass:
    """Mass parameters of a rigid body.

    This class stores mass parameters of a rigid body which can be
    accessed through the following attributes:

    - mass: The total mass of the body (float)
    - c:    The center of gravity position in body frame (3-tuple of floats)
    - I:    The 3x3 inertia tensor in body frame (3-tuple of 3-tuples)

    This class wraps the dMass structure from the C API.

    @ivar mass: The total mass of the body
    @ivar c: The center of gravity position in body frame (cx, cy, cz)
    @ivar I: The 3x3 inertia tensor in body frame ((I11, I12, I13), (I12, I22, I23), (I13, I23, I33))
    @type mass: float
    @type c: 3-tuple of floats
    @type I: 3-tuple of 3-tuples of floats
    """
    cdef dMass _mass

    def __cinit__(self):
        dMassSetZero(&self._mass)

    def setZero(self):
        """setZero()

        Set all the mass parameters to zero."""
        dMassSetZero(&self._mass)
        return self

    def setParameters(self, dReal mass, dReal cgx, dReal cgy, dReal cgz, dReal I11, dReal I22, dReal I33, dReal I12, dReal I13, dReal I23):
        """setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23)

        Set the mass parameters to the given values.

        @param mass: Total mass of the body.
        @param cgx: Center of gravity position in the body frame (x component).
        @param cgy: Center of gravity position in the body frame (y component).
        @param cgz: Center of gravity position in the body frame (z component).
        @param I11: Inertia tensor
        @param I22: Inertia tensor
        @param I33: Inertia tensor
        @param I12: Inertia tensor
        @param I13: Inertia tensor
        @param I23: Inertia tensor
        @type mass: float
        @type cgx: float
        @type cgy: float
        @type cgz: float
        @type I11: float
        @type I22: float
        @type I33: float
        @type I12: float
        @type I13: float
        @type I23: float
        """
        dMassSetParameters(&self._mass, mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23)
        return self

    def setSphere(self, dReal density, dReal radius):
        """setSphere(density, radius)

        Set the mass parameters to represent a sphere of the given radius
        and density, with the center of mass at (0,0,0) relative to the body.

        @param density: The density of the sphere
        @param radius: The radius of the sphere
        @type density: float
        @type radius: float
        """
        dMassSetSphere(&self._mass, density, radius)
        return self

    def setSphereTotal(self, dReal total_mass, dReal radius):
        """setSphereTotal(total_mass, radius)

        Set the mass parameters to represent a sphere of the given radius
        and mass, with the center of mass at (0,0,0) relative to the body.

        @param total_mass: The total mass of the sphere
        @param radius: The radius of the sphere
        @type total_mass: float
        @type radius: float
        """
        dMassSetSphereTotal(&self._mass, total_mass, radius)
        return self

    def setCapsule(self, dReal density, int direction, dReal radius, dReal length):
        """setCapsule(density, direction, radius, length)

        Set the mass parameters to represent a capsule of the given parameters
        and density, with the center of mass at (0,0,0) relative to the body.
        The radius of the cylinder (and the spherical cap) is radius. The length
        of the cylinder (not counting the spherical cap) is length. The
        cylinder's long axis is oriented along the body's x, y or z axis
        according to the value of direction (1=x, 2=y, 3=z). The first function
        accepts the density of the object, the second accepts its total mass.

        @param density: The density of the capsule
        @param direction: The direction of the capsule's cylinder (1=x axis, 2=y axis, 3=z axis)
        @param radius: The radius of the capsule's cylinder
        @param length: The length of the capsule's cylinder (without the caps)
        @type density: float
        @type direction: int
        @type radius: float
        @type length: float
        """
        dMassSetCapsule(&self._mass, density, direction, radius, length)
        return self

    def setCapsuleTotal(self, dReal total_mass, int direction, dReal radius, dReal length):
        """setCapsuleTotal(total_mass, direction, radius, length)

        Set the mass parameters to represent a capsule of the given parameters
        and mass, with the center of mass at (0,0,0) relative to the body. The
        radius of the cylinder (and the spherical cap) is radius. The length of
        the cylinder (not counting the spherical cap) is length. The cylinder's
        long axis is oriented along the body's x, y or z axis according to the
        value of direction (1=x, 2=y, 3=z). The first function accepts the
        density of the object, the second accepts its total mass.

        @param total_mass: The total mass of the capsule
        @param direction: The direction of the capsule's cylinder (1=x axis, 2=y axis, 3=z axis)
        @param radius: The radius of the capsule's cylinder
        @param length: The length of the capsule's cylinder (without the caps)
        @type total_mass: float
        @type direction: int
        @type radius: float
        @type length: float
        """
        dMassSetCapsuleTotal(&self._mass, total_mass, direction, radius, length)
        return self

    def setCylinder(self, dReal density, int direction, dReal r, dReal h):
        """setCylinder(density, direction, r, h)

        Set the mass parameters to represent a flat-ended cylinder of
        the given parameters and density, with the center of mass at
        (0,0,0) relative to the body. The radius of the cylinder is r.
        The length of the cylinder is h. The cylinder's long axis is
        oriented along the body's x, y or z axis according to the value
        of direction (1=x, 2=y, 3=z).

        @param density: The density of the cylinder
        @param direction: The direction of the cylinder (1=x axis, 2=y axis, 3=z axis)
        @param r: The radius of the cylinder
        @param h: The length of the cylinder
        @type density: float
        @type direction: int
        @type r: float
        @type h: float
        """
        dMassSetCylinder(&self._mass, density, direction, r, h)
        return self

    def setCylinderTotal(self, dReal total_mass, int direction, dReal r, dReal h):
        """setCylinderTotal(total_mass, direction, r, h)

        Set the mass parameters to represent a flat-ended cylinder of
        the given parameters and mass, with the center of mass at
        (0,0,0) relative to the body. The radius of the cylinder is r.
        The length of the cylinder is h. The cylinder's long axis is
        oriented along the body's x, y or z axis according to the value
        of direction (1=x, 2=y, 3=z).

        @param total_mass: The total mass of the cylinder
        @param direction: The direction of the cylinder (1=x axis, 2=y axis, 3=z axis)
        @param r: The radius of the cylinder
        @param h: The length of the cylinder
        @type total_mass: float
        @type direction: int
        @type r: float
        @type h: float
        """
        dMassSetCylinderTotal(&self._mass, total_mass, direction, r, h)
        return self

    def setBox(self, dReal density, dReal lx, dReal ly, dReal lz):
        """setBox(density, lx, ly, lz)

        Set the mass parameters to represent a box of the given
        dimensions and density, with the center of mass at (0,0,0)
        relative to the body. The side lengths of the box along the x,
        y and z axes are lx, ly and lz.

        @param density: The density of the box
        @param lx: The length along the x axis
        @param ly: The length along the y axis
        @param lz: The length along the z axis
        @type density: float
        @type lx: float
        @type ly: float
        @type lz: float
        """
        dMassSetBox(&self._mass, density, lx, ly, lz)
        return self

    def setBoxTotal(self, dReal total_mass, dReal lx, dReal ly, dReal lz):
        """setBoxTotal(total_mass, lx, ly, lz)

        Set the mass parameters to represent a box of the given
        dimensions and mass, with the center of mass at (0,0,0)
        relative to the body. The side lengths of the box along the x,
        y and z axes are lx, ly and lz.

        @param total_mass: The total mass of the box
        @param lx: The length along the x axis
        @param ly: The length along the y axis
        @param lz: The length along the z axis
        @type total_mass: float
        @type lx: float
        @type ly: float
        @type lz: float
        """
        dMassSetBoxTotal(&self._mass, total_mass, lx, ly, lz)
        return self

    # Zhen Wu: has error, shouldn't be used!!!!
    def setTriMesh(self, dReal density, GeomObject g):
        dMassSetTrimesh(&self._mass, density, g.gid)

    # Zhen Wu: has error, shouldn't be used!!!!
    def setTriMeshTotal(self, dReal total_mass, GeomObject g):
        dMassSetTrimesh(&self._mass, total_mass, g.gid)

    def adjust(self, dReal newmass):
        """adjust(newmass)

        Adjust the total mass. Given mass parameters for some object,
        adjust them so the total mass is now newmass. This is useful
        when using the setXyz() methods to set the mass parameters for
        certain objects - they take the object density, not the total
        mass.

        @param newmass: The new total mass
        @type newmass: float
        """
        dMassAdjust(&self._mass, newmass)
        return self

    # Add by Zhenhua Song
    def rotate(self, R):
        raise NotImplementedError

    # Add by Zhenhua Song. mass.c will be modified in dMassRotate
    def rotateNumpy(self, np.ndarray[np.float64_t, ndim = 1] Rot):
        raise NotImplementedError

    # Comment by Zhenhua Song: mass.c will be modified in dMassTranslate
    def translate(self, t):
        """translate(t)

        Adjust mass parameters. Given mass parameters for some object,
        adjust them to represent the object displaced by (x,y,z)
        relative to the body frame.

        @param t: Translation vector (x, y, z)
        @type t: 3-tuple of floats
        """
        raise NotImplementedError

    def add(self, Mass b):
        """add(b)

        Add the mass b to the mass object. Masses can also be added using
        the + operator.

        @param b: The mass to add to this mass
        @type b: Mass
        """
        dMassAdd(&self._mass, &b._mass)
        return self

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getCNumpy(self):
        cdef cnp.ndarray[np.float64_t, ndim=1] np_c = cnp.zeros(3)
        cdef dReal * res_c = <dReal *> np_c.data
        res_c[0] = self._mass.c[0]
        res_c[1] = self._mass.c[1]
        res_c[2] = self._mass.c[2]
        return np_c

    # Add by Zhenhua Song
    @property
    def inertia(self) -> np.ndarray:
        return self.getINumpy()

    # Add by Zhenhua Song
    @inertia.setter
    def inertia(self, np.ndarray[np.float64_t, ndim = 1] I):
        self.setINumpy(I)

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getINumpy(self) -> np.ndarray:
        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(9)
        cdef dReal * res = <dReal *> np_res.data
        ODEMat3ToDenseMat3(self._mass.I, res, 0)

        return np_res

    # Add by Zhenhua Song
    def setINumpy(self, np.ndarray[np.float64_t, ndim = 1] Inertia_in):
        cdef np.ndarray[np.float64_t, ndim=1] I = np.ascontiguousarray(Inertia_in)
        cdef const dReal * res = <const dReal *> I.data
        DenseMat3ToODEMat3(self._mass.I, res, 0)

    # Add by Zhenhua Song
    @property
    def mass(self) -> dReal:
        return self._mass.mass

    # Add by Zhenhua Song
    @mass.setter
    def mass(self, dReal value):
        self._mass.mass = value;

    def __add__(self, Mass b):
        self.add(b)
        return self

    def __str__(self):
        m = str(self._mass.mass)
        sc0 = str(self._mass.c[0])
        sc1 = str(self._mass.c[1])
        sc2 = str(self._mass.c[2])
        I11 = str(self._mass.I[0])
        I22 = str(self._mass.I[5])
        I33 = str(self._mass.I[10])
        I12 = str(self._mass.I[1])
        I13 = str(self._mass.I[2])
        I23 = str(self._mass.I[6])
        return ("Mass=%s\n"
                "Cg=(%s, %s, %s)\n"
                "I11=%s I22=%s I33=%s\n"
                "I12=%s I13=%s I23=%s" %
                (m, sc0, sc1, sc2, I11, I22, I33, I12, I13, I23))

    # Add by Zhenhua Song
    def copy(self):
        cdef Mass res = Mass()
        res._mass.mass = self._mass.mass
        memcpy(&(res._mass), &(self._mass), sizeof(dMass))
        return res


cpdef translate_inertia(np.ndarray[np.float64_t, ndim=1] data, dReal mass, dReal tx, dReal ty, dReal tz):
    cdef dReal t[9]
    cdef dReal * I = <dReal *> data.data
    cdef int i = 0
    for i in range(9):
        t[i] = 0

    t[0] = mass*(ty*ty+tz*tz)
    t[1] = -mass*tx*ty
    t[2] = -mass*tx*tz;
    t[3] = t[1]
    t[4] = mass*(tx*tx+tz*tz)
    t[5] = -mass*ty*tz
    t[6] = t[2]
    t[7] = t[5]
    t[8] = mass*(tx*tx+ty*ty)

    for i in range(9):
        I[i] += t[i]
    
    return data


# Add by Zhenhua Song
# Calc Inertia
cdef class Inertia:
    cdef np.ndarray I
    cdef dReal mass
    def __cinit__(self):
        self.I = np.zeros(9)
        self.mass = 0.0

    # def __dealloc__(self):
    #    del self.I

    def TransInertia(self, dReal tx, dReal ty, dReal tz):
        """
        when a rigid body is translated by (tx, ty, tz), the inertia is also modified
        """
        translate_inertia(self.I, self.mass, tx, ty, tz)
        return self.I

    def TransInertiaNumpy(self, np.ndarray[np.float64_t, ndim = 1] t):
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.ascontiguousarray(t)
        cdef const dReal * t_res = <const dReal *> np_buff.data
        return self.TransInertia(t_res[0], t_res[1], t_res[2])

    def RotInertia(self, np.ndarray[np.float64_t, ndim = 1] np_rot):
        """
        when a rigid body is rotated, the inertia is also modified
        """
        cdef np.ndarray[np.float64_t, ndim=1] np_buff = np.ascontiguousarray(np_rot)
        cdef const dReal * R = <const dReal *> np_buff.data
        cdef dReal * I = <dReal *> self.I.data
        # newI = R*I*R^t

        cdef dReal bI[9]
        cdef dReal * pI = NULL
        cdef size_t i, j, k
        # range will convert to `for loop` of cplusplus.
        # So, don't worry about the speed.
        for i in range(3):
            for j in range(3):
                pI = &(bI[i * 3 + j])
                pI[0] = 0.0
                for k in range(3):
                    pI[0] += R[i*3+k]*I[k*3+j]

        # I = bI*R^t
        for i in range(3):
            for j in range(3):
                pI = &I[i*3+j]
                pI[0] = 0.0
                for k in range(3):
                    pI[0] += bI[i*3+k]*R[j*3+k]

        return self.I

    def setMass(self, dReal mass):
        """
        set mass value
        """
        self.mass = mass

    def getMass(self):
        return self.mass

    def getI(self):
        return self.I

    def setI(self, np.ndarray[np.float64_t, ndim = 1] I):
        self.I = np.ascontiguousarray(I)

    def setMassAndInertia(self, dReal mass, np.ndarray[np.float64_t, ndim = 1] I):
        self.setMass(mass)
        self.setI(I)

    def setFromMassClass(self, Mass m):
        self.mass = m._mass.mass
        self.I = m.getINumpy()

    def add(self, Inertia o):
        cdef dReal * I1 = <dReal *>self.I.data
        cdef dReal * I2 = <dReal *>o.I.data

        for i in range(9):
            I1[i] += I2[i];

        self.mass += o.mass
        return self.I

    def toMass(self):
        """
        convert to Mass object
        """
        cdef Mass m = Mass()
        m._mass.mass = self.mass
        m.setINumpy(self.I)
        return m


cdef class Contact:
    """This class represents a contact between two bodies in one point.

    A Contact object stores all the input parameters for a ContactJoint.
    This class wraps the ODE dContact structure which has 3 components::

    struct dContact {
        dSurfaceParameters surface;
        dContactGeom geom;
        dVector3 fdir1;
    };

    This wrapper class provides methods to get and set the items of those
    structures.
    """

    cdef dContact _contact

    def __cinit__(self):
        self._contact.surface.mode = dContactApprox1
        self._contact.surface.mu = dInfinity

        self._contact.surface.bounce = 0.0

    def __init__(self):
        pass

    # Add by Zhenhua Song
    def enable_soft_cfm_erp(self, dReal cfm, dReal erp):
        self._contact.surface.soft_cfm = cfm
        self._contact.surface.soft_erp = erp
        self._contact.surface.mode = self._contact.surface.mode | dContactSoftCFM | dContactSoftERP

    # Add by Zhenhua Song
    def enable_contact_slip(self, dReal slip):
        self._contact.surface.slip1 = slip
        self._contact.surface.slip2 = slip
        self._contact.surface.mode = self._contact.surface.mode | dContactSlip1 | dContactSlip2

    # Add by Zhenhua Song
    @property
    def slip1_enabled(self):
        return self._contact.surface.mode | dContactSlip1 > 0

    # Add by Zhenhua Song
    @slip1_enabled.setter
    def slip1_enabled(self, value):
       self._contact.surface.mode |= dContactSlip1

    # Add by Zhenhua Song
    @property
    def slip2_enabled(self):
        return self._contact.surface.mode | dContactSlip2 > 0

    # Add by Zhenhua Song
    @slip2_enabled.setter
    def slip2_enabled(self, value):
        pass

    # Modify by Zhenhua Song
    @property
    def mode(self) -> int:
        """getMode() -> flags

        Return the contact flags.
        """
        return self._contact.surface.mode

    # Modify by Zhenhua Song
    @mode.setter
    def mode(self, int flags):
        """setMode(flags)

        Set the contact flags. The argument m is a combination of the
        ContactXyz flags (ContactMu2, ContactBounce, ...).

        @param flags: Contact flags
        @type flags: int
        """
        self._contact.surface.mode = flags

    # Modify by Zhenhua Song
    @property
    def mu(self) -> dReal:
        """getMu() -> float

        Return the Coulomb friction coefficient.
        """
        return self._contact.surface.mu

    # Modify by Zhenhua Song
    @mu.setter
    def mu(self, dReal mu):
        """setMu(mu)

        Set the Coulomb friction coefficient.

        @param mu: Coulomb friction coefficient (0..Infinity)
        @type mu: float
        """
        self._contact.surface.mu = mu

    # Modify by Zhenhua Song
    @property
    def mu2(self) -> dReal:
        """getMu2() -> float

        Return the optional Coulomb friction coefficient for direction 2.
        """
        return self._contact.surface.mu2

    # Modify by Zhenhua Song
    @mu2.setter
    def mu2(self, dReal mu):
        """setMu2(mu)

        Set the optional Coulomb friction coefficient for direction 2.

        @param mu: Coulomb friction coefficient (0..Infinity)
        @type mu: float
        """
        self._contact.surface.mu2 = mu

    # Modify by Zhenhua Song
    @property
    def bounce(self) -> dReal:
        """getBounce() -> float

        Return the restitution parameter.
        """
        return self._contact.surface.bounce

    # Modify by Zhenhua Song
    @bounce.setter
    def bounce(self, dReal b):
        """setBounce(b)

        @param b: Restitution parameter (0..1)
        @type b: float
        """
        self._contact.surface.bounce = b

    # Modify by Zhenhua Song
    @property
    def bounceVel(self) -> dReal:
        """getBounceVel() -> float

        Return the minimum incoming velocity necessary for bounce.
        """
        return self._contact.surface.bounce_vel

    # Modify by Zhenhua Song
    @bounceVel.setter
    def bounceVel(self, dReal bv):
        """setBounceVel(bv)

        Set the minimum incoming velocity necessary for bounce. Incoming
        velocities below this will effectively have a bounce parameter of 0.

        @param bv: Velocity
        @type bv: float
        """
        self._contact.surface.bounce_vel = bv

    # Modify by Zhenhua Song
    @property
    def SoftERP(self) -> dReal:
        """getSoftERP() -> float

        Return the contact normal "softness" parameter.
        """
        return self._contact.surface.soft_erp

    # Modify by Zhenhua Song
    @SoftERP.setter
    def SoftERP(self, dReal erp):
        """setSoftERP(erp)

        Set the contact normal "softness" parameter.

        @param erp: Softness parameter
        @type erp: float
        """
        self._contact.surface.soft_erp = erp

    # Modify by Zhenhua Song
    @property
    def SoftCFM(self) -> dReal:
        """getSoftCFM() -> float

        Return the contact normal "softness" parameter.
        """
        return self._contact.surface.soft_cfm

    # Modify by Zhenhua Song
    @SoftCFM.setter
    def SoftCFM(self, dReal cfm):
        """setSoftCFM(cfm)

        Set the contact normal "softness" parameter.

        @param cfm: Softness parameter
        @type cfm: float
        """
        self._contact.surface.soft_cfm = cfm

    # Modify by Zhenhua Song
    @property
    def Motion1(self) -> dReal:
        """getMotion1() -> float

        Get the surface velocity in friction direction 1.
        """
        return self._contact.surface.motion1

    # Modify by Zhenhua Song
    @Motion1.setter
    def Motion1(self, dReal m):
        """setMotion1(m)

        Set the surface velocity in friction direction 1.

        @param m: Surface velocity
        @type m: float
        """
        self._contact.surface.motion1 = m

    # Modify by Zhenhua Song
    @property
    def Motion2(self):
        """getMotion2() -> float

        Get the surface velocity in friction direction 2.
        """
        return self._contact.surface.motion2

    # Modify by Zhenhua Song
    @Motion2.setter
    def Motion2(self, m):
        """setMotion2(m)

        Set the surface velocity in friction direction 2.

        @param m: Surface velocity
        @type m: float
        """
        self._contact.surface.motion2 = m

    # Modify by Zhenhua Song
    @property
    def Slip1(self):
        """getSlip1() -> float

        Get the coefficient of force-dependent-slip (FDS) for friction
        direction 1.
        """
        return self._contact.surface.slip1

    # Modify by Zhenhua Song
    @Slip1.setter
    def Slip1(self, s):
        """setSlip1(s)

        Set the coefficient of force-dependent-slip (FDS) for friction
        direction 1.

        @param s: FDS coefficient
        @type s: float
        """
        self._contact.surface.slip1 = s

    # Modify by Zhenhua Song
    @property
    def Slip2(self) -> dReal:
        """getSlip2() -> float

        Get the coefficient of force-dependent-slip (FDS) for friction
        direction 2.
        """
        return self._contact.surface.slip2

    # Modify by Zhenhua Song
    @Slip2.setter
    def Slip2(self, dReal s):
        """setSlip2(s)

        Set the coefficient of force-dependent-slip (FDS) for friction
        direction 1.

        @param s: FDS coefficient
        @type s: float
        """
        self._contact.surface.slip2 = s

    # # Modify by Zhenhua Song
    @property
    def FDir1(self):
        """getFDir1() -> (x, y, z)

        Get the "first friction direction" vector that defines a direction
        along which frictional force is applied.
        """
        return (self._contact.fdir1[0],
                self._contact.fdir1[1],
                self._contact.fdir1[2])

    # Modify by Zhenhua Song
    @FDir1.setter
    def FDir1(self, fdir):
        """setFDir1(fdir)

        Set the "first friction direction" vector that defines a direction
        along which frictional force is applied. It must be of unit length
        and perpendicular to the contact normal (so it is typically
        tangential to the contact surface).

        @param fdir: Friction direction
        @type fdir: 3-sequence of floats
        """
        self._contact.fdir1[0] = fdir[0]
        self._contact.fdir1[1] = fdir[1]
        self._contact.fdir1[2] = fdir[2]

    # Modify by Zhenhua Song
    @property
    def contactPosNumpy(self) -> np.ndarray:
        cdef np.ndarray[np.float64_t, ndim = 1] res = np.zeros(3)
        memcpy(<dReal*> res.data, self._contact.geom.pos, sizeof(dReal) * 3)
        return res

    # Modify by Zhenhua Song
    @contactPosNumpy.setter
    def contactPosNumpy(self, np.ndarray[np.float64_t, ndim=1] res):
        memcpy(self._contact.geom.pos, <dReal *> res.data, sizeof(dReal) * 3)

    # Modify by Zhenhua Song
    @property
    def contactNormalNumpy(self) -> np.ndarray:
        """
        get the normal vector of contact
        """
        cdef np.ndarray[np.float64_t, ndim = 1] res = np.zeros(3)
        memcpy(<dReal*> res.data, self._contact.geom.normal, sizeof(dReal) * 3)
        return res

    # Modify by Zhenhua Song
    @contactNormalNumpy.setter
    def contactNormalNumpy(self, np.ndarray[np.float64_t, ndim=1] res):
        memcpy(self._contact.geom.normal, <dReal *> res.data, sizeof(dReal) * 3)

    # Modify by Zhenhua Song
    @property
    def contactDepth(self) -> dReal:
        """
        Depth of contact
        """
        return self._contact.geom.depth

    # Modify by Zhenhua Song
    @contactDepth.setter
    def contactDepth(self, dReal data):
        """
        setter Depth of contact
        """
        self._contact.geom.depth = data

    # Modify by Zhenhua Song
    @property
    def contactGeom1(self) -> GeomObject:
        """
        Contact Geom 1
        """
        return <GeomObject> dGeomGetData(self._contact.geom.g1)

    # Modify by Zhenhua Song
    @contactGeom1.setter
    def contactGeom1(self, GeomObject value):
        if value is not None:
            self._contact.geom.g1 = value.gid
        else:
            self._contact.geom.g1 = NULL

    # Modify by Zhenhua Song
    @property
    def contactGeom2(self) -> GeomObject:
        """
        """
        return <GeomObject> dGeomGetData(self._contact.geom.g2)

    # Modify by Zhenhua Song
    @contactGeom2.setter
    def contactGeom2(self, GeomObject value):
        if value is not None:
            self._contact.geom.g2 = value.gid
        else:
            self._contact.geom.g2 = value.gid

    # Modify by Zhenhua Song
    def getContactGeomParams(self):
        """getContactGeomParams() -> (pos, normal, depth, geom1, geom2)

        Get the ContactGeom structure of the contact.

        The return value is a tuple (pos, normal, depth, geom1, geom2)
        where pos and normal are 3-tuples of floats and depth is a single
        float. geom1 and geom2 are the Geom objects of the geoms in contact.
        """
        return self.contactPosNumpy, self.contactNormalNumpy, self.contactDepth, self.contactGeom1, self.contactGeom2

    # Modify by Zhenhua Song
    def setContactGeomParams(self,
        np.ndarray[np.float64_t, ndim=1] pos,
        np.ndarray[np.float64_t, ndim=1] normal,
        dReal depth,
        GeomObject g1=None,
        GeomObject g2=None
    ):
        """setContactGeomParams(pos, normal, depth, geom1=None, geom2=None)

        Set the ContactGeom structure of the contact.
        """
        self.contactPosNumpy = pos
        self.contactNormalNumpy = normal
        self.contactDepth = depth
        self.contactGeom1 = g1
        self.contactGeom2 = g2

# World
cdef class World:
    """Dynamics world.
    The world object is a container for rigid bodies and joints.
    Constructor::World()
    """

    # Add by Zhenhua Song
    # for fast collision detection
    cdef dJointGroupWithdWorld contact_group

    cdef dWorldID wid  # pointer of the world
    cdef dReal _simu_fail_eps

    def __cinit__(self):
        self.wid = dWorldCreate()

        # Add by Zhenhua Song
        self.contact_group.max_contact_num = 4
        self.contact_group.use_max_force_contact = 0
        self.contact_group.use_soft_contact = 0
        self.contact_group.soft_cfm = 1e-10
        self.contact_group.soft_erp = 0.2
        self.contact_group.self_collision = 0
        self.contact_group.group = dJointGroupCreate(512)  # use a large number.
        self.contact_group.world = self.wid

        self._simu_fail_eps = 5e-2

    # Modify by Zhenhua Song
    def __dealloc__(self):
        self.destroy_immediate()

    # Add by Zhenhua Song
    cpdef destroy_immediate(self):
        if self.wid != NULL:
            dWorldDestroy(self.wid)
            self.wid = NULL

        if self.contact_group.group != NULL:
            dJointGroupDestroy(self.contact_group.group)
            self.contact_group.group = NULL
            self.contact_group.world = NULL

    @property
    def max_contact_num(self) -> int:
        return self.contact_group.max_contact_num

    @max_contact_num.setter
    def max_contact_num(self, int value):
        self.contact_group.max_contact_num = value

    @property
    def use_max_force_contact(self) -> int:
        return self.contact_group.use_max_force_contact

    @use_max_force_contact.setter
    def use_max_force_contact(self, int value):
        self.contact_group.use_max_force_contact = value

    @property
    def self_collision(self):
        return self.contact_group.self_collision

    @self_collision.setter
    def self_collision(self, value):
        self.contact_group.self_collision = value

    @property
    def use_soft_contact(self):
        return self.contact_group.use_soft_contact

    @use_soft_contact.setter
    def use_soft_contact(self, value):
        self.contact_group.use_soft_contact = value

    @property
    def soft_cfm(self):
        return self.contact_group.soft_cfm

    @soft_cfm.setter
    def soft_cfm(self, value):
        self.contact_group.soft_cfm = value

    @property
    def soft_erp(self):
        return self.contact_group.soft_erp

    @soft_erp.setter
    def soft_erp(self, value):
        self.contact_group.soft_erp = value
    
    @property
    def simu_fail_eps(self):
        return self._simu_fail_eps  # for check in simulation..
    
    @simu_fail_eps.setter
    def simu_fail_eps(self, dReal value):
        self._simu_fail_eps = value

    # Add by Zhenhua Song
    def __eq__(self, World other):
        return self.wid == other.wid

    # Add by Zhenhua Song, get the pointer of the world
    def get_wid(self):
        return <size_t>self.wid

    def setGravity(self, gravity):
        """setGravity(gravity)

        Set the world's global gravity vector.

        @param gravity: Gravity vector
        @type gravity: 3-sequence of floats
        """
        dWorldSetGravity(self.wid, gravity[0], gravity[1], gravity[2])

    # Add by Zhenhua Song
    def setGravityYEarth(self):
        dWorldSetGravity(self.wid, 0, -9.81, 0)

    # Add by Zhenhua Song
    def getGravityNumpy(self) -> np.ndarray:
        cdef dVector3 g
        dWorldGetGravity(self.wid, g)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.ones(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = g[0]
        res[1] = g[1]
        res[2] = g[2]
        return np_buff

    def getGravity(self):
        """getGravity() -> 3-tuple

        Return the world's global gravity vector as a 3-tuple of floats.
        """
        cdef dVector3 g
        dWorldGetGravity(self.wid, g)
        return g[0], g[1], g[2]

    # Modify by Zhenhua Song
    @property
    def ERP(self) -> dReal:
        """getERP() -> float

        Get the global ERP value, that controls how much error
        correction is performed in each time step. Typical values are
        in the range 0.1-0.8. The default is 0.2.
        """
        return dWorldGetERP(self.wid)

    # Modify by Zhenhua Song
    @ERP.setter
    def ERP(self, dReal erp):
        """setERP(erp)

        Set the global ERP value, that controls how much error
        correction is performed in each time step. Typical values are
        in the range 0.1-0.8. The default is 0.2.

        @param erp: Global ERP value
        @type erp: float
        """
        dWorldSetERP(self.wid, erp)

    # Modify by Zhenhua Song
    @property
    def CFM(self) -> dReal:
        """getCFM() -> float

        Get the global CFM (constraint force mixing) value. Typical
        values are in the range 10E-9 - 1. The default is 10E-5 if
        single precision is being used, or 10E-10 if double precision
        is being used.
        """
        return dWorldGetCFM(self.wid)

    # Modify by Zhenhua Song
    @CFM.setter
    def CFM(self, dReal cfm):
        """setCFM(cfm)

        Set the global CFM (constraint force mixing) value. Typical
        values are in the range 10E-9 - 1. The default is 10E-5 if
        single precision is being used, or 10E-10 if double precision
        is being used.

        @param cfm: Constraint force mixing value
        @type cfm: float
        """
        dWorldSetCFM(self.wid, cfm)

    # Add by Zhenhua Song
    def dampedStep(self, dReal stepsize):
        # Add Damping in Simulation
        dWorldDampedStep(self.wid, stepsize)  # Add by Libin Liu in C++ code
        self.check_simulation()

    # Add by Zhenhua Song
    def damped_step_fast_collision(self, SpaceBase space, dReal stepsize):
        space.fast_collide(&(self.contact_group))  # collision detection
        dWorldDampedStep(self.wid, stepsize)  # forward simulation
        dJointGroupEmpty(self.contact_group.group)  # clear the contact joint
        dSpaceResortGeoms(space.sid)  # resort geometries, make sure simulation result is same when state is same
        self.check_simulation()

    def step(self, dReal stepsize):
        """step(stepsize)

        Step the world. This uses a "big matrix" method that takes
        time on the order of O(m3) and memory on the order of O(m2), where m
        is the total number of constraint rows.

        For large systems this will use a lot of memory and can be
        very slow, but this is currently the most accurate method.

        @param stepsize: Time step
        @type stepsize: float
        """
        # 1. Add Gravity
        # 2. Calc constraints of Joints and LCP
        # 3. Calc Jacobian Matrix as J
        # 4. Calc J^{-1} M J
        # 5. Solve LCP
        # 6. Calc Constraint Force
        # 7. Calc Acc, Velocity, Position
        dWorldStep(self.wid, stepsize)
        self.check_simulation()

    # Add by Zhenhua Song. Collision detection is done in cython, not in python.
    def step_fast_collision(self, SpaceBase space, dReal stepsize):
        # This will accelerate by 1.2 times
        space.fast_collide(&self.contact_group)
        dWorldStep(self.wid, stepsize)
        dJointGroupEmpty(self.contact_group.group)
        dSpaceResortGeoms(space.sid)  # resort geometries, make sure simulation result is same when state is same
        self.check_simulation()

    def quickStep(self, dReal stepsize):
        """quickStep(stepsize)

        Step the world. This uses an iterative method that takes time
        on the order of O(m*N) and memory on the order of O(m), where m is
        the total number of constraint rows and N is the number of
        iterations.

        For large systems this is a lot faster than dWorldStep, but it
        is less accurate.

        @param stepsize: Time step
        @type stepsize: float
        """
        dWorldQuickStep(self.wid, stepsize)
        self.check_simulation()

    @property
    def QuickStepNumIterations(self):
        """getQuickStepNumIterations() -> int

        Get the number of iterations that the QuickStep method
        performs per step. More iterations will give a more accurate
        solution, but will take longer to compute. The default is 20
        iterations.
        """
        return dWorldGetQuickStepNumIterations(self.wid)

    @QuickStepNumIterations.setter
    def QuickStepNumIterations(self, int num):
        """setQuickStepNumIterations(num)

        Set the number of iterations that the QuickStep method
        performs per step. More iterations will give a more accurate
        solution, but will take longer to compute. The default is 20
        iterations.

        @param num: Number of iterations
        @type num: int
        """
        dWorldSetQuickStepNumIterations(self.wid, num)

    @property
    def ContactMaxCorrectingVel(self) -> dReal:
        """getContactMaxCorrectingVel() -> float

        Get the maximum correcting velocity that contacts are allowed
        to generate. The default value is infinity (i.e. no
        limit). Reducing this value can help prevent "popping" of
        deeply embedded objects.

        """
        return dWorldGetContactMaxCorrectingVel(self.wid)

    @ContactMaxCorrectingVel.setter
    def ContactMaxCorrectingVel(self, dReal vel):
        """setContactMaxCorrectingVel(vel)

        Set the maximum correcting velocity that contacts are allowed
        to generate. The default value is infinity (i.e. no
        limit). Reducing this value can help prevent "popping" of
        deeply embedded objects.

        @param vel: Maximum correcting velocity
        @type vel: float
        """
        dWorldSetContactMaxCorrectingVel(self.wid, vel)

    @property
    def ContactSurfaceLayer(self):
        """getContactSurfaceLayer()

        Get the depth of the surface layer around all geometry
        objects. Contacts are allowed to sink into the surface layer
        up to the given depth before coming to rest. The default value
        is zero. Increasing this to some small value (e.g. 0.001) can
        help prevent jittering problems due to contacts being
        repeatedly made and broken.
        """
        return dWorldGetContactSurfaceLayer(self.wid)

    @ContactSurfaceLayer.setter
    def ContactSurfaceLayer(self, dReal depth):
        """setContactSurfaceLayer(depth)

        Set the depth of the surface layer around all geometry
        objects. Contacts are allowed to sink into the surface layer
        up to the given depth before coming to rest. The default value
        is zero. Increasing this to some small value (e.g. 0.001) can
        help prevent jittering problems due to contacts being
        repeatedly made and broken.

        @param depth: Surface layer depth
        @type depth: float
        """
        dWorldSetContactSurfaceLayer(self.wid, depth)

    @property
    def AutoDisableFlag(self) -> int:
        """getAutoDisableFlag() -> bool

        Get the default auto-disable flag for newly created bodies.
        """
        return dWorldGetAutoDisableFlag(self.wid)

    @AutoDisableFlag.setter
    def AutoDisableFlag(self, int flag):
        """setAutoDisableFlag(flag)

        Set the default auto-disable flag for newly created bodies.

        @param flag: True = Do auto disable
        @type flag: bool
        """
        dWorldSetAutoDisableFlag(self.wid, flag)

    @property
    def AutoDisableLinearThreshold(self) -> dReal:
        """getAutoDisableLinearThreshold() -> float

        Get the default auto-disable linear threshold for newly created
        bodies.
        """
        return dWorldGetAutoDisableLinearThreshold(self.wid)

    @AutoDisableLinearThreshold.setter
    def AutoDisableLinearThreshold(self, dReal threshold):
        """setAutoDisableLinearThreshold(threshold)

        Set the default auto-disable linear threshold for newly created
        bodies.

        @param threshold: Linear threshold
        @type threshold: float
        """
        dWorldSetAutoDisableLinearThreshold(self.wid, threshold)

    @property
    def AutoDisableAngularThreshold(self):
        """getAutoDisableAngularThreshold() -> float

        Get the default auto-disable angular threshold for newly created
        bodies.
        """
        return dWorldGetAutoDisableAngularThreshold(self.wid)

    @AutoDisableAngularThreshold.setter
    def AutoDisableAngularThreshold(self, threshold):
        """setAutoDisableAngularThreshold(threshold)

        Set the default auto-disable angular threshold for newly created
        bodies.

        @param threshold: Angular threshold
        @type threshold: float
        """
        dWorldSetAutoDisableAngularThreshold(self.wid, threshold)

    @property
    def AutoDisableSteps(self) -> int:
        """getAutoDisableSteps() -> int

        Get the default auto-disable steps for newly created bodies.
        """
        return dWorldGetAutoDisableSteps(self.wid)

    @AutoDisableSteps.setter
    def AutoDisableSteps(self, int steps):
        """setAutoDisableSteps(steps)

        Set the default auto-disable steps for newly created bodies.

        @param steps: Auto disable steps
        @type steps: int
        """
        dWorldSetAutoDisableSteps(self.wid, steps)

    @property
    def AutoDisableTime(self) -> dReal:
        """getAutoDisableTime() -> float

        Get the default auto-disable time for newly created bodies.
        """
        return dWorldGetAutoDisableTime(self.wid)

    @AutoDisableTime.setter
    def AutoDisableTime(self, dReal time):
        """setAutoDisableTime(time)

        Set the default auto-disable time for newly created bodies.

        @param time: Auto disable time
        @type time: float
        """
        dWorldSetAutoDisableTime(self.wid, time)

    # Add by Zhenhua Song
    @property
    def LinearDamping(self):
        return self.getLinearDamping()

    # Add by Zhenhua Song
    @LinearDamping.setter
    def LinearDamping(self, dReal scale):
        self.setLinearDamping(scale)

    cpdef setLinearDamping(self, dReal scale):
        """setLinearDamping(scale)

        Set the world's linear damping scale.
                @param scale The linear damping scale that is to be applied to bodies.
                Default is 0 (no damping). Should be in the interval [0, 1].
        @type scale: float
        """
        dWorldSetLinearDamping(self.wid, scale)

    def getLinearDamping(self) -> dReal:
        """getLinearDamping() -> float

        Get the world's linear damping scale.
        """
        return dWorldGetLinearDamping(self.wid)

    # Add by Zhenhua Song
    @property
    def AngularDamping(self):
        return self.getAngularDamping()

    # Add by Zhenhua Song
    @AngularDamping.setter
    def AngularDamping(self, dReal scale):
        self.setAngularDamping(scale)

    def setAngularDamping(self, dReal scale):
        """setAngularDamping(scale)

        Set the world's angular damping scale.
                @param scale The angular damping scale that is to be applied to bodies.
                Default is 0 (no damping). Should be in the interval [0, 1].
        @type scale: float
        """
        dWorldSetAngularDamping(self.wid, scale)

    def getAngularDamping(self) -> dReal:
        """getAngularDamping() -> float

        Get the world's angular damping scale.
        """
        return dWorldGetAngularDamping(self.wid)

    def impulseToForce(self, dReal stepsize, impulse):
        """impulseToForce(stepsize, impulse) -> 3-tuple

        If you want to apply a linear or angular impulse to a rigid
        body, instead of a force or a torque, then you can use this
        function to convert the desired impulse into a force/torque
        vector before calling the dBodyAdd... function.

        @param stepsize: Time step
        @param impulse: Impulse vector
        @type stepsize: float
        @type impulse: 3-tuple of floats
        """
        cdef dVector3 force
        dWorldImpulseToForce(self.wid, stepsize, impulse[0], impulse[1], impulse[2], force)
        return force[0], force[1], force[2]

    # Add by Zhenhua Song. Get the total number of joints.
    @property
    def NumJoints(self) -> int:
        return dWorldGetNumJoints(self.wid)

    # Add by Zhenhua Song. Get the number of ball and hinge joints
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getNumBallAndHingeJoints(self):
        return dWorldGetNumBallAndHingeJoints(self.wid)

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBallAndHingeInfos(self):
        cdef int cnt = dWorldGetNumBallAndHingeJoints(self.wid)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_id = np.zeros(cnt, np.uint64)
        cdef np.ndarray[np.float64_t, ndim = 1] np_pos = np.zeros(3 * cnt)
        cdef size_t * res_id = <size_t *> np_id.data
        cdef dReal * res_pos = <dReal *> np_pos.data
        cdef dJointID j = dWorldGetFirstJoint(self.wid)
        cdef int idx = 0
        cdef dVector3 result
        cdef int j_type

        while j != NULL:
            j_type = dJointGetType(j)
            if j_type == dJointTypeBall:
                res_id[idx] = <size_t> j
                dJointGetBallAnchor(j, result)
                res_pos[3 * idx + 0] = result[0]
                res_pos[3 * idx + 1] = result[1]
                res_pos[3 * idx + 2] = result[2]
                idx += 1
            elif j_type == dJointTypeHinge:
                res_id[idx] = <size_t> j
                dJointGetHingeAnchor(j, result)
                res_pos[3 * idx + 0] = result[0]
                res_pos[3 * idx + 1] = result[1]
                res_pos[3 * idx + 2] = result[2]
                idx += 1
            else:
                pass

            j = dWorldGetNextJoint(j)

        return np_id, np_pos

    # Add by Zhenhua Song, Get number of rigid bodies
    @property
    def NumBody(self) -> int:
        return dWorldGetNumBody(self.wid)

    # Add by Zhenhua Song. Get pointer of bodies
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def bodyListToNumpy(self, list body_list):
        cdef size_t cnt = len(body_list)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_id = np.zeros(cnt, np.uint64)
        cdef size_t * res_id = <size_t *> np_id.data
        cdef size_t idx = 0
        cdef Body b
        while idx < cnt:
            b = body_list[idx]
            res_id[idx] = <size_t> (b.bid)
            idx += 1

        return np_id

    # Add by Zhenhua Song. Get pointer of joints.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jointListToNumpy(self, list joint_list):
        cdef size_t cnt = len(joint_list)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_id = np.zeros(cnt, np.uint64)
        cdef size_t * res_id = <size_t *> np_id.data
        cdef size_t idx = 0
        cdef Joint j
        while idx < cnt:
            j = joint_list[idx]
            res_id[idx] = <size_t> (j.jid)
            idx += 1

        return np_id

    # Add by Zhenhua Song.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBodyGeomCount(self, np.ndarray[np.float64_t, ndim = 1] np_id) -> int:
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef size_t cnt = np_id.size
        cdef size_t geom_cnt = 0
        cdef size_t idx = 0
        cdef dBodyID b = NULL
        while idx < cnt:
            b = res_id[idx]
            geom_cnt += dBodyGetNumGeoms(b)
            idx += 1

        return geom_cnt

    # Add by Zhenhua Song
    # input: array of body
    # output: array of geom
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBodyGeomIDs(self, np.ndarray[np.uint64_t, ndim = 1] np_id) -> np.ndarray:
        cdef geom_cnt = self.getBodyGeomCount(np_id)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_geom = np.zeros(geom_cnt, np.uint64)
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef size_t * res_geom = <size_t *> np_geom.data
        cdef size_t cnt = np_id.size
        cdef size_t geom_idx = 0
        cdef size_t idx = 0
        cdef size_t g_idx = 0
        cdef dBodyID b = NULL
        cdef dGeomID g = NULL
        while idx < cnt:
            b = res_id[idx]
            g = dBodyGetFirstGeom(b)
            while g != NULL:
                res_geom[geom_idx] = <size_t> g
                geom_idx += 1

                g = dGeomGetBodyNext(g)

            idx += 1
        return np_geom

    # Add by Zhenhua Song
    # Compute center of mass of bodies
    # return np.ndarray in shape (3,)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_body_com(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef size_t idx = 0, cnt = np_id.size
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef np.ndarray[np.float64_t, ndim=1] com_result = np.zeros(3)
        cdef dReal * com = <dReal *> com_result.data
        cdef dBodyID body = NULL
        cdef const dReal * p = NULL
        cdef dReal total_mass = 0.0, mass = 0.0
        while idx < cnt:
            body = res_id[idx]
            p = dBodyGetPosition(body)
            mass = dBodyGetMassValue(body)
            total_mass += mass
            com[0] += mass * p[0]
            com[1] += mass * p[1]
            com[2] += mass * p[2]
            # print(idx)
            idx += 1
        if cnt > 0:
            total_mass = 1.0 / total_mass
            com[0] *= total_mass
            com[1] *= total_mass
            com[2] *= total_mass

        return com_result

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyPos(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef size_t cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim=1] np_pos = np.zeros(3 * cnt)
        cdef dReal * res_pos = <dReal *> np_pos.data

        cdef dBodyID * res_id = <dBodyID *> np_id.data

        cdef size_t idx = 0
        cdef dBodyID b

        cdef const dReal* p

        while idx < cnt:
            b = res_id[idx]

            p  = dBodyGetPosition(b)
            res_pos[3 * idx + 0] = p[0]
            res_pos[3 * idx + 1] = p[1]
            res_pos[3 * idx + 2] = p[2]

            idx += 1

        return np_pos

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 4, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyQuatScipy(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef int cnt = <int>(np_id.size)
        cdef np.ndarray[np.float64_t, ndim = 1] np_quat = np.zeros(4 * cnt, np.float64)
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef dReal * res_quat = <dReal *> np_quat.data

        cdef int idx = 0
        cdef dBodyID b = NULL
        cdef const dReal * q_ode = NULL

        while idx < cnt:
            b = res_id[idx]
            if b == NULL:
                res_quat[4 * idx + 0] = 0
                res_quat[4 * idx + 1] = 0
                res_quat[4 * idx + 2] = 0
                res_quat[4 * idx + 3] = 1
            else:
                q_ode = dBodyGetQuaternion(b)
                res_quat[4 * idx + 0] = q_ode[1]
                res_quat[4 * idx + 1] = q_ode[2]
                res_quat[4 * idx + 2] = q_ode[3]
                res_quat[4 * idx + 3] = q_ode[0]

            idx += 1

        return np_quat

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3 * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyRot(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef int cnt = np_id.size
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef np.ndarray[np.float64_t, ndim = 1] np_rot = np.zeros(9 * cnt)
        cdef dReal * res_rot = <dReal *> np_rot.data
        cdef int idx = 0
        cdef dBodyID b = NULL
        cdef const dReal * m = NULL
        while idx < cnt:
            b = res_id[idx]

            if b != NULL:
                m = dBodyGetRotation(b)
                ODEMat3ToDenseMat3(m, res_rot, 9 * idx)
            else:
                memset(res_rot + 9 * idx, 0, sizeof(dReal) * 9)
                res_rot[9 * idx + 0] = res_rot[9 * idx + 4] = res_rot[9 * idx + 8] = 1

            idx += 1

        return np_rot

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyLinVel(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim = 1] np_lin_vel = np.zeros(3 * cnt)
        cdef dReal * res_lin_vel = <dReal *> np_lin_vel.data
        cdef dBodyID * res_id = <dBodyID *> np_id.data

        cdef const dReal * linvel = NULL
        cdef int idx = 0
        cdef dBodyID b

        while idx < cnt:
            b = res_id[idx]

            if b != NULL:
                linvel = dBodyGetLinearVel(b)
                res_lin_vel[3 * idx + 0] = linvel[0]
                res_lin_vel[3 * idx + 1] = linvel[1]
                res_lin_vel[3 * idx + 2] = linvel[2]
            else:
                res_lin_vel[3 * idx + 0] = res_lin_vel[3 * idx + 1] = res_lin_vel[3 * idx + 2] = 0

            idx += 1

        return np_lin_vel

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyAngVel(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim = 1] np_ang_vel = np.zeros(3 * cnt)
        cdef dReal * res_ang_vel = <dReal *> np_ang_vel.data
        cdef dBodyID * res_id = <dBodyID *> np_id.data

        cdef const dReal * angvel = NULL
        cdef int idx = 0
        cdef dBodyID b

        while idx < cnt:
            b = res_id[idx]

            if b != NULL:
                angvel = dBodyGetAngularVel(b)
                res_ang_vel[3 * idx + 0] = angvel[0]
                res_ang_vel[3 * idx + 1] = angvel[1]
                res_ang_vel[3 * idx + 2] = angvel[2]
            else:
                res_ang_vel[3 * idx + 0] = res_ang_vel[3 * idx + 1] = res_ang_vel[3 * idx + 2] = 0

            idx += 1
        return np_ang_vel

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyForce(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim=1] np_force = np.zeros(3 * cnt)
        cdef dReal * np_force_ptr = <dReal*> np_force.data
        cdef dBodyID * res_id = <dBodyID *> np_id.data

        cdef int idx = 0
        cdef dBodyID b
        cdef const dReal * force
        while idx < cnt:
            b = res_id[idx]
            if b != NULL:
                force = dBodyGetForce(b)
                memcpy(np_force_ptr + 3 * idx, force, sizeof(dReal) * 3)
            idx += 1

        return np_force

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyTorque(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef int cnt = <int>(np_id.size)
        cdef np.ndarray[np.float64_t, ndim=1] np_torque = np.zeros(3 * cnt)
        cdef dReal * np_torque_ptr = <dReal*> np_torque.data
        cdef dBodyID * res_id = <dBodyID *> np_id.data

        cdef int idx = 0
        cdef dBodyID b = NULL
        cdef const dReal * torque = NULL
        while idx < cnt:
            b = res_id[idx]
            if b != NULL:
                torque = dBodyGetTorque(b)
                memcpy(np_torque_ptr + 3 * idx, torque, sizeof(dReal) * 3)
            idx += 1

        return np_torque

    # Add by Zhenhua Song
    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3 * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyInertia(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef int cnt = np_id.size
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef np.ndarray[np.float64_t, ndim=1] np_inertia = np.zeros(cnt * 3 * 3)
        cdef dReal * np_inertia_ptr = <dReal *> np_inertia.data
        cdef int idx = 0
        cdef dBodyID b
        cdef dMatrix3 res
        while idx < cnt:
            b = res_id[idx]
            if b != NULL:
                dBodyGetInertia(b, res)
                ODEMat3ToDenseMat3(res, np_inertia_ptr, 9 * idx)
            idx += 1

        return np_inertia

    # input: np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # return: np.ndarray in shape (num_body * 3 * 3, )
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getBodyInertiaInv(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef int cnt = np_id.size
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef np.ndarray[np.float64_t, ndim=1] np_inertia_inv = np.zeros(cnt * 3 * 3)
        cdef dReal * np_inertia_inv_ptr = <dReal *> np_inertia_inv.data
        cdef int idx = 0
        cdef dBodyID b
        cdef dMatrix3 res
        while idx < cnt:
            b = res_id[idx]
            if b != NULL:
                dBodyGetInertiaInv(b, res)
                ODEMat3ToDenseMat3(res, np_inertia_inv_ptr, 9 * idx)
            idx += 1

        return np_inertia_inv

    # Add by Zhenhua Song
    # return np_id, np_pos, np_quat, np_rot, np_lin_vel, np_ang_vel
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def getBodyInfos(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef np.ndarray[np.float64_t, ndim=1] np_pos = self.getBodyPos(np_id)
        cdef np.ndarray[np.float64_t, ndim=1] np_quat = self.getBodyQuatScipy(np_id)
        cdef np.ndarray[np.float64_t, ndim=1] np_rot = self.getBodyRot(np_id)
        cdef np.ndarray[np.float64_t, ndim=1] np_lin_vel = self.getBodyLinVel(np_id)
        cdef np.ndarray[np.float64_t, ndim=1] np_ang_vel = self.getBodyAngVel(np_id)

        return np_id, np_pos, np_quat, np_rot, np_lin_vel, np_ang_vel

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef check_simulation(self):
        # check if simulation is failed.
        cdef const dReal * data[2]
        cdef int i = 0, j = 0
        cdef dBodyID b = dWorldGetFirstBody(self.wid)

        while b != NULL:
            data[0] = dBodyGetAngularVel(b)
            data[1] = dBodyGetLinearVel(b)
            for i in range(2):
                for j in range(3):
                    if (data[i][j] < -10000.0) or (data[i][j] > 10000.0):
                        raise SimulationFailError("Simulation failed")
            
            b = dWorldGetNextBody(b)
        
        # here we should also check joint position computed by each body..
        # use inf norm for simple implementation..
        cdef dJointID joint = dWorldGetFirstJoint(self.wid)
        cdef int jtype = 0
        cdef dVector3 v0, v1
        cdef dReal delta_len = 0.0
        while joint != NULL:
            # ignore contact joint here..
            jtype = dJointGetType(joint)
            if jtype == dJointTypeBall:
                dJointGetBallAnchor(joint, v0)
                dJointGetBallAnchor2(joint, v1)
                # print(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2])
                for i in range(3):
                    delta_len = cmath_abs(v0[i] - v1[i])
                    if delta_len > self._simu_fail_eps:
                        raise SimulationFailError("Joint Distance is too large.", delta_len)
            elif jtype == dJointTypeHinge:
                dJointGetHingeAnchor(joint, v0)
                dJointGetHingeAnchor2(joint, v1)
                # print(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2])
                for i in range(3):
                    delta_len = cmath_abs(v0[i] - v1[i])
                    if cmath_abs(v0[i] - v1[i]) > self._simu_fail_eps:
                        raise SimulationFailError("Joint Distance is too large.", delta_len)

            joint = dWorldGetNextJoint(joint)

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef getAllBodyID(self):
        cdef int cnt = dWorldGetNumBody(self.wid)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_id = np.zeros(cnt, np.uint64)
        cdef dBodyID * res_id = <dBodyID *> np_id.data
        cdef dBodyID b = dWorldGetFirstBody(self.wid)
        cdef int idx = 0

        while b != NULL:
            res_id[idx] = b
            b = dWorldGetNextBody(b)
            idx += 1

        return np_id

    # Add by Zhenhua Song
    # return np_id, np_pos, np_quat, np_rot, np_lin_vel, np_ang_vel
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def getAllBodyInfos(self):
        cdef np.ndarray[np.uint64_t, ndim = 1] np_id = self.getAllBodyID()
        return self.getBodyInfos(np_id)

    # Add by Zhenhua Song
    # input
    # - np_id : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_pos: np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef loadBodyPos(self, np.ndarray[np.uint64_t, ndim = 1] np_id, np.ndarray[np.float64_t, ndim = 1] np_pos):
        cdef np.ndarray[np.uint64_t, ndim = 1] np_id_buff = np.ascontiguousarray(np_id)
        cdef np.ndarray[np.float64_t, ndim = 1] np_pos_buff = np.ascontiguousarray(np_pos)
        cdef dBodyID * res_id = <dBodyID*> np_id_buff.data
        cdef const dReal * res_pos = <const dReal *> np_pos_buff.data

        cdef int idx = 0
        cdef int cnt = np_id_buff.size

        while idx < cnt:
            b = res_id[idx]
            dBodySetPosition(b, res_pos[3 * idx], res_pos[3 * idx + 1], res_pos[3 * idx + 2])
            idx += 1

    # Add by Zhenhua Song
    # input
    # - np_id  : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_quat: np.ndarray in shape (num_body * 4, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef loadBodyQuat(self, np.ndarray[np.uint64_t, ndim = 1] np_id, np.ndarray[np.float64_t, ndim = 1] np_quat):
        cdef np.ndarray[np.uint64_t, ndim = 1] np_id_buff = np.ascontiguousarray(np_id)
        cdef np.ndarray[np.float64_t, ndim = 1] np_quat_buff = np.ascontiguousarray(np_quat)
        cdef dBodyID * res_id = <dBodyID*> np_id_buff.data
        cdef const dReal * res_quat = <const dReal *> np_quat_buff.data
        cdef int idx = 0
        cdef int cnt = np_id_buff.size
        cdef dQuaternion q_ode
        while idx < cnt:
            b = res_id[idx]

            q_scipy = &res_quat[4 * idx]
            q_ode[0] = q_scipy[3]
            q_ode[1] = q_scipy[0]
            q_ode[2] = q_scipy[1]
            q_ode[3] = q_scipy[2]
            dBodySetQuaternion(b, q_ode)

            idx += 1

    # Add by Zhenhua Song
    # input
    # - np_id  : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_quat: np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    # - np_rot : np.ndarray in shape (num_body * 3 * 3) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef loadBodyQuatAndRotNoNorm(
        self, np.ndarray[np.uint64_t, ndim = 1] np_id, np.ndarray[np.float64_t, ndim = 1] np_quat, np.ndarray[np.float64_t, ndim = 1] np_rot
    ):
        cdef np.ndarray[np.float64_t, ndim = 1] np_quat_buff = np.ascontiguousarray(np_quat)
        cdef np.ndarray[np.float64_t, ndim = 1] np_rot_buff = np.ascontiguousarray(np_rot)
        cdef dBodyID * res_id = <dBodyID*> np_id.data
        cdef const dReal * res_quat = <const dReal *> np_quat_buff.data
        cdef const dReal * res_rot = <const dReal *> np_rot_buff.data

        cdef int idx = 0
        cdef int cnt = np_id.size
        cdef dMatrix3 m

        cdef dBodyID b = NULL
        cdef const dReal * R = NULL
        cdef dQuaternion q_ode
        cdef const dReal * q_scipy = NULL

        while idx < cnt:
            b = res_id[idx]
            DenseMat3ToODEMat3(m, res_rot, 9 * idx)

            q_scipy = &res_quat[4 * idx]
            q_ode[0] = q_scipy[3]
            q_ode[1] = q_scipy[0]
            q_ode[2] = q_scipy[1]
            q_ode[3] = q_scipy[2]
            dBodySetRotAndQuatNoNorm(b, m, q_ode)

            idx += 1

    # Add by Zhenhua Song
    # input
    # - np_id     : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_lin_vel: np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef loadBodyLinVel(self, np.ndarray[np.uint64_t, ndim = 1] np_id, np.ndarray[np.float64_t, ndim = 1] np_lin_vel):
        cdef np.ndarray[np.float64_t, ndim = 1] np_linvel_buff = np.ascontiguousarray(np_lin_vel)
        cdef const dReal * res_linvel = <const dReal *> np_linvel_buff.data
        cdef dBodyID * res_id = <dBodyID*> np_id.data
        cdef dBodyID b = NULL
        cdef int idx = 0
        cdef int cnt = np_id.size
        while idx < cnt:
            b = res_id[idx]
            dBodySetLinearVel(b, res_linvel[3 * idx + 0], res_linvel[3 * idx + 1], res_linvel[3 * idx + 2])
            idx += 1

    # Add by Zhenhua Song
    # input
    # - np_id     : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_ang_vel: np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef loadBodyAngVel(self, np.ndarray[np.uint64_t, ndim = 1] np_id, np.ndarray[np.float64_t, ndim = 1] np_ang_vel):
        cdef np.ndarray[np.float64_t, ndim = 1] np_angvel_buff = np.ascontiguousarray(np_ang_vel)
        cdef dBodyID * res_id = <dBodyID*> np_id.data
        cdef const dReal * res_angvel = <const dReal *> np_angvel_buff.data
        cdef dBodyID b = NULL
        cdef int idx = 0
        cdef int cnt = np_id.size
        while idx < cnt:
            b = res_id[idx]
            dBodySetAngularVel(b, res_angvel[3 * idx + 0], res_angvel[3 * idx + 1], res_angvel[3 * idx + 2])
            idx += 1

    # Add by Zhenhua Song
    # input
    # - np_id     : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_ang_vel: np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef loadBodyForce(self, np.ndarray[np.uint64_t, ndim=1] np_id, np.ndarray[np.float64_t, ndim=1] np_force):
        cdef np.ndarray[np.float64_t, ndim = 1] np_force_buff = np.ascontiguousarray(np_force)
        cdef dBodyID * res_id = <dBodyID*> np_id.data
        cdef const dReal * res_force = <const dReal *> np_force_buff.data
        cdef dBodyID b = NULL
        cdef int idx = 0, cnt = np_id.size
        while idx < cnt:
            b = res_id[idx]
            dBodySetForce(b, res_force[3 * idx + 0], res_force[3 * idx + 1], res_force[3 * idx + 2])
            idx += 1

    # Add by Zhenhua Song
    # - np_id    : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_torque: np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef loadBodyTorque(self, np.ndarray[np.uint64_t, ndim = 1] np_id, np.ndarray[np.float64_t, ndim = 1] np_torque):
        cdef np.ndarray[np.float64_t, ndim = 1] np_torque_buff = np.ascontiguousarray(np_torque)
        cdef dBodyID * res_id = <dBodyID*> np_id.data
        cdef const dReal * res_torque = <const dReal *> np_torque_buff.data
        cdef dBodyID b = NULL
        cdef int idx = 0, cnt = np_id.size
        while idx < cnt:
            b = res_id[idx]
            dBodySetTorque(b, res_torque[3 * idx + 0], res_torque[3 * idx + 1], res_torque[3 * idx + 2])
            idx += 1

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def loadBodyInfos(self, np.ndarray[np.uint64_t, ndim = 1] np_id,
                    np.ndarray[np.float64_t, ndim = 1] np_pos,
                    np.ndarray[np.float64_t, ndim = 1] np_quat,
                    np.ndarray[np.float64_t, ndim = 1] np_rot,
                    np.ndarray[np.float64_t, ndim = 1] np_lin_vel,
                    np.ndarray[np.float64_t, ndim = 1] np_ang_vel,
                    np.ndarray[np.float64_t, ndim = 1] np_force,
                    np.ndarray[np.float64_t, ndim = 1] np_torque):
        self.loadBodyPos(np_id, np_pos)
        self.loadBodyQuatAndRotNoNorm(np_id, np_quat, np_rot)
        self.loadBodyLinVel(np_id, np_lin_vel)
        self.loadBodyAngVel(np_id, np_ang_vel)

        if np_force is not None:
            self.loadBodyForce(np_id, np_force)
        if np_torque is not None:
            self.loadBodyTorque(np_id, np_torque)

    # Add by Zhenhua Song
    # - np_id    : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_force: np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def addBodyForce(self, np.ndarray[np.uint64_t, ndim = 1] np_id, np.ndarray[np.float64_t, ndim = 1] np_force):
        assert np_force.dtype == np.float64
        cdef np.ndarray[np.float64_t, ndim = 1] np_force_buf = np.ascontiguousarray(np_force)
        cdef dBodyID * res_id = <dBodyID*> np_id.data
        cdef const dReal * res_f = <const dReal *> np_force_buf.data
        cdef dBodyID b = NULL
        cdef int idx = 0
        cdef int cnt = np_id.size
        while idx < cnt:
            b = res_id[idx]
            dBodyAddForce(b, res_f[3 * idx + 0], res_f[3 * idx + 1], res_f[3 * idx + 2])
            idx += 1

    # Add by Zhenhua Song
    # - np_id  : np.ndarray in shape (num_body,) with dtype == np.uint64 for body pointer
    # - np_tor : np.ndarray in shape (num_body * 3, ) with dtype == np.float64
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def addBodyTorque(self,
        np.ndarray[np.uint64_t, ndim=1] np_id,
        np.ndarray[np.float64_t, ndim=2] np_tor
    ):
        cdef np.ndarray[np.float64_t, ndim=2] np_tor_buf = np.ascontiguousarray(np_tor)
        cdef const dReal * res_tor = <const dReal *> np_tor_buf.data
        cdef dBodyID * res_id = <dBodyID*> np_id.data
        cdef dBodyID b = NULL
        cdef int idx = 0
        cdef int cnt = <int>(np_id.size)
        # print(np_id, np_tor)
        while idx < cnt:
            b = res_id[idx]
            dBodyAddTorque(b, res_tor[3 * idx + 0], res_tor[3 * idx + 1], res_tor[3 * idx + 2])
            idx += 1

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getJointFeedBackForce(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef dJointFeedback* fb = NULL
        cdef dJointID joint = NULL
        cdef int nj = <int>(np_id.size), i = 0, j = 0
        cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((nj, 3), np.float64)
        cdef double * res_ptr = <double *> (res.data)
        cdef const dJointID * jid_ptr = <const dJointID *> (np_id.data)
        for i in range(nj):
            joint = jid_ptr[i]
            fb = dJointGetFeedback(joint)
            for j in range(3):
                res_ptr[i * 3 + j] = fb.f1[j]
        return res

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getJointFeedBackTorque(self, np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef dJointFeedback* fb = NULL
        cdef dJointID joint = NULL
        cdef int nj = <int>(np_id.size), i = 0, j = 0
        cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((nj, 3), np.float64)
        cdef double * res_ptr = <double *> (res.data)
        cdef const dJointID * jid_ptr = <const dJointID *> (np_id.data)
        for i in range(nj):
            joint = jid_ptr[i]
            fb = dJointGetFeedback(joint)
            for j in range(3):
                res_ptr[i * 3 + j] = fb.t1[j]
        return res

    # Add by Zhenhua Song
    # return raw anchor1, raw anchor 2
    # joint type must be ball or hinge
    # Note: only support dJointID as input.
    # if your input isn't dJointID, the program will crash or fall in dead cycle
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBallAndHingeRawAnchor1(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef const dJointID * res_id = <dJointID *> np_id.data
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim = 1] np_anc1 = np.zeros(3 * cnt)

        cdef dReal * res_anc1 = <dReal *> np_anc1.data
        cdef const dReal * anc1_ptr

        cdef int idx = 0
        cdef int joint_type
        cdef dJointID j
        while idx < cnt:
            j = res_id[idx]
            joint_type = dJointGetType(j)
            if joint_type == dJointTypeBall:
                anc1_ptr = dJointGetBallAnchor1Raw(j)
            elif joint_type == dJointTypeHinge:
                anc1_ptr = dJointGetHingeAnchor1Raw(j)
            else:
                raise NotImplementedError

            res_anc1[3 * idx + 0] = anc1_ptr[0]
            res_anc1[3 * idx + 1] = anc1_ptr[1]
            res_anc1[3 * idx + 2] = anc1_ptr[2]

            idx += 1

        return np_anc1

    # Add by Zhenhua Song
    # - np_id: np.ndarray in shape (num_joint,) with dtype == np.uint64 for joint pointer
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBallAndHingeRawAnchor2(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef const dJointID * res_id = <dJointID *> np_id.data
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim = 1] np_anc2 = np.zeros(3 * cnt)
        cdef dReal * res_anc2 = <dReal *> np_anc2.data

        cdef const dReal * anc2_ptr

        cdef int idx = 0
        cdef int joint_type
        cdef dJointID j
        while idx < cnt:
            j = res_id[idx]
            joint_type = dJointGetType(j)
            if joint_type == dJointTypeBall:
                anc2_ptr = dJointGetBallAnchor2Raw(j)
            elif joint_type == dJointTypeHinge:
                anc2_ptr = dJointGetHingeAnchor2Raw(j)
            else:
                raise NotImplementedError

            res_anc2[3 * idx + 0] = anc2_ptr[0]
            res_anc2[3 * idx + 1] = anc2_ptr[1]
            res_anc2[3 * idx + 2] = anc2_ptr[2]
            idx += 1
        return np_anc2

    # - np_id: np.ndarray in shape (num_joint,) with dtype == np.uint64 for joint pointer
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBallAndHingeRawAnchor(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        return self.getBallAndHingeRawAnchor1(np_id), self.getBallAndHingeRawAnchor2(np_id)

    # Add by Zhenhua Song
    # - np_id: np.ndarray in shape (num_joint,) with dtype == np.uint64 for joint pointer
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBallAndHingeAnchor1(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef const dJointID * res_id = <dJointID *> np_id.data
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim = 1] np_anc = np.zeros(3 * cnt)
        cdef dReal * res_anc = <dReal *> np_anc.data
        cdef dVector3 anc_ptr
        cdef int joint_type, idx = 0
        cdef dJointID j
        while idx < cnt:
            j = res_id[idx]
            joint_type = dJointGetType(j)
            if joint_type == dJointTypeBall:
                dJointGetBallAnchor(j, anc_ptr)
            elif joint_type == dJointTypeHinge:
                dJointGetHingeAnchor(j, anc_ptr)
            else:
                raise NotImplementedError

            res_anc[3 * idx + 0] = anc_ptr[0]
            res_anc[3 * idx + 1] = anc_ptr[1]
            res_anc[3 * idx + 2] = anc_ptr[2]

            idx += 1
        return np_anc

    # Add by Zhenhua Song
    # - np_id: np.ndarray in shape (num_joint,) with dtype == np.uint64 for joint pointer
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBallAndHingeAnchor2(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef const dJointID * res_id = <dJointID *> np_id.data
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim = 1] np_anc = np.zeros(3 * cnt)
        cdef dReal * res_anc = <dReal *> np_anc.data
        cdef dVector3 anc_ptr
        cdef int joint_type, idx = 0
        cdef dJointID j
        while idx < cnt:
            j = res_id[idx]
            joint_type = dJointGetType(j)
            if joint_type == dJointTypeBall:
                dJointGetBallAnchor2(j, anc_ptr)
            elif joint_type == dJointTypeHinge:
                dJointGetHingeAnchor2(j, anc_ptr)
            else:
                raise NotImplementedError

            res_anc[3 * idx + 0] = anc_ptr[0]
            res_anc[3 * idx + 1] = anc_ptr[1]
            res_anc[3 * idx + 2] = anc_ptr[2]

            idx += 1
        return np_anc

    # Add by Zhenhua Song
    # - np_id: np.ndarray in shape (num_joint,) with dtype == np.uint64 for joint pointer
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getBallAndHingeAnchorAvg(self, np.ndarray[np.uint64_t, ndim = 1] np_id):
        cdef const dJointID * res_id = <dJointID *> np_id.data
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim = 1] np_anc = np.zeros(3 * cnt)
        cdef dReal * res_anc = <dReal *> np_anc.data
        cdef dVector3 anc_ptr1, anc_ptr2
        cdef int joint_type, idx = 0
        cdef dJointID j
        while idx < cnt:
            j = res_id[idx]
            joint_type = dJointGetType(j)
            if joint_type == dJointTypeBall:
                dJointGetBallAnchor(j, anc_ptr1)
                dJointGetBallAnchor2(j, anc_ptr2)
            elif joint_type == dJointTypeHinge:
                dJointGetHingeAnchor(j, anc_ptr1)
                dJointGetHingeAnchor2(j, anc_ptr2)
            else:
                raise NotImplementedError

            res_anc[3 * idx + 0] = 0.5 * (anc_ptr1[0] + anc_ptr2[0])
            res_anc[3 * idx + 1] = 0.5 * (anc_ptr1[1] + anc_ptr2[1])
            res_anc[3 * idx + 2] = 0.5 * (anc_ptr1[2] + anc_ptr2[2])
            idx += 1
        return np_anc

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_global_torque(self,
        np.ndarray[np.float64_t, ndim=2] global_torque,
        np.ndarray[np.uint64_t, ndim=1] parent_body_id,
        np.ndarray[np.uint64_t, ndim=1] child_body_id
    ):
        cdef np.ndarray[np.float64_t, ndim=2] torque_buf = np.ascontiguousarray(global_torque)
        cdef np.ndarray[np.uint64_t, ndim=1] pa_id_buf = np.ascontiguousarray(parent_body_id)
        cdef np.ndarray[np.uint64_t, ndim=1] child_id_buf = np.ascontiguousarray(child_body_id)

        cdef dReal * torque_res = <dReal *> torque_buf.data
        cdef dBodyID * pa_body_res = <dBodyID*> pa_id_buf.data
        cdef dBodyID * ch_body_res = <dBodyID *> child_id_buf.data
        cdef int cnt = <int>(pa_id_buf.size)
        cdef int idx = 0
        cdef dBodyID pa_body = NULL
        cdef dBodyID ch_body = NULL
        while idx < cnt:
            pa_body = pa_body_res[idx]
            ch_body = ch_body_res[idx]
            if ch_body == NULL:
                raise ValueError("Child body id at %d is NULL" % idx)
            dBodyAddTorque(ch_body, torque_res[3 * idx + 0], torque_res[3 * idx + 1], torque_res[3 * idx + 2])
            if pa_body != NULL:
                dBodyAddTorque(pa_body, -torque_res[3 * idx + 0], -torque_res[3 * idx + 1], -torque_res[3 * idx + 2])

            idx += 1

    # Add by Zhenhua Song
    # - np_id: np.ndarray in shape (num_joint,) with dtype == np.uint64 for joint pointer
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_all_hinge_angle(self, np.ndarray[size_t, ndim=1] hinge_id):
        cdef int i, cnt = hinge_id.size
        cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(cnt)
        for i in range(cnt):
            res[i] = dJointGetHingeAngle(<dJointID>hinge_id[i])
        return res

    # Add by Zhenhua Song
    # - joint_id: np.ndarray in shape (num_joint,) with dtype == np.uint64 for joint pointer
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_all_joint_local_angle(self, np.ndarray[np.uint64_t, ndim=1] joint_id):
        cdef int joint_count = joint_id.size
        cdef np.ndarray[np.float64_t, ndim=2] parent_qs = np.zeros((joint_count, 4))
        cdef np.ndarray[np.float64_t, ndim=2] child_qs = np.zeros((joint_count, 4))
        cdef np.ndarray[np.float64_t, ndim=2] local_qs = np.zeros((joint_count, 4))
        cdef np.ndarray[np.float64_t, ndim=2] parent_qs_inv = np.zeros((joint_count, 4))
        get_joint_local_quat_batch(
            <dJointID*>joint_id.data,
            joint_count,
            <dReal*> parent_qs.data,
            <dReal*> child_qs.data,
            <dReal*> local_qs.data,
            <dReal*> parent_qs_inv.data,
            1)

        return parent_qs, child_qs, local_qs, parent_qs_inv

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_pd_control_torque(self,
                              np.ndarray[np.uint64_t, ndim=1] joint_id,
                              np.ndarray[np.float64_t, ndim=2] local_target_quat_in,
                              np.ndarray[np.float64_t, ndim=1] kps_in,
                              np.ndarray[np.float64_t, ndim=1] tor_lim_in):
        # assert joint_id.dtype == np.uint64
        cdef int joint_count = joint_id.size
        cdef np.ndarray[np.float64_t, ndim=2] local_torque = np.zeros((joint_count, 3), np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] global_torque = np.zeros((joint_count, 3), np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] local_target = np.ascontiguousarray(local_target_quat_in, np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] kps = np.ascontiguousarray(kps_in, np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] tor_lim = np.ascontiguousarray(tor_lim_in, np.float64)
        pd_control_batch(
            <dJointID*> joint_id.data,
            joint_count,
            <const dReal*> local_target.data,
            <const dReal*> kps.data,
            NULL,
            <const dReal*> tor_lim.data,
            <dReal*> local_torque.data,
            <dReal*> global_torque.data,
            1
        )

        # cdef dReal total_power = compute_total_power(<dJointID*> joint_id.data, joint_count, <dReal*> global_torque.data)

        return local_torque, global_torque

    # Add by Zhenhua Song
    def createBody(self):
        return Body(self)

    # Add by Zhenhua Song
    def createBallJoint(self, jointgroup=None):
        return BallJoint(self, jointgroup)

    # Add by Zhenhua Song
    def createHingeJoint(self, jointgroup=None):
        return HingeJoint(self, jointgroup)

    # createHinge2Joint
    def createHinge2Joint(self, jointgroup=None):
        return Hinge2Joint(self, jointgroup)

    # Add by Zhenhua Song
    def createSliderJoint(self, jointgroup=None):
        return SliderJoint(self, jointgroup)

    # Add by Zhenhua Song
    def createFixedJoint(self, jointgroup=None):
        return FixedJoint(self, jointgroup)

    # Add by Zhenhua Song
    def createContactJoint(self, jointgroup, contact):
        return ContactJoint(self, jointgroup, contact)

    # Add by Zhenhua Song
    def createContactJointMaxForce(self, jointgroup, contact):
        return ContactJointMaxForce(self, jointgroup, contact)


# Body
cdef class Body:
    """The rigid body class encapsulating the ODE body.

    This class represents a rigid body that has a location and orientation
    in space and that stores the mass properties of an object.

    When creating a Body object you have to pass the world it belongs to
    as argument to the constructor::

    >>> import ode
    >>> w = ode.World()
    >>> b = ode.Body(w)
    """

    cdef dBodyID bid
    # A reference to the world so that the world won't be destroyed while
    # there are still joints using it.
    cdef World _world  # Modify by Zhenhua Song

    cdef str _name
    cdef list _geoms  # Add by Zhenhua Song

    cdef int _instance_id  # Add by Zhenhua Song
    cdef int _offset # Add by Heyuan Yao
    # _instance_id is body index in a character

    cdef object __weakref__  # Add by Zhenhua Song

    def __cinit__(self, World world not None):
        self.bid = dBodyCreate(world.wid)
        self._instance_id = 0
        self._offset = 0

    def __init__(self, World world not None):
        """Constructor.

        @param world: The world in which the body should be created.
        @type world: World
        """
        self._world = world
        self._name = ""
        # self._joints = list()
        self._geoms = list()

        self._setData(self) # Add by Zhenhua Song.
        # sys.getrefcount() will not be increased, because it's C++ code.
        # DO NOT use weakref.ref or weakref.proxy in self._setData()

    # Add by Zhenhua Song
    def copy_body(self, SpaceBase space = None):
        cdef Body result = Body(self._world)
        # TODO: copy body position, rotation, and quaternion
        # copy all of geoms
        cdef size_t i = 0, num_geom = len(self._geoms)
        cdef GeomObject old_geom, new_geom
        for i in range(num_geom):
            old_geom = self._geoms[i]
            new_geom = old_geom.copy_geom(result, space)
        return result

    # Add by Zhenhua Song
    cdef int _bid_is_not_null(self):
        return self.bid != NULL

    # Modify by Zhenhua Song
    def __dealloc__(self):
        self.destroy_immediate()

    # Add by Zhenhua Song
    cpdef destroy_immediate(self):
        if self.bid != NULL:
            dBodyDestroy(self.bid)
            self.bid = NULL

    # Add by Zhenhua Song
    def __eq__(self, Body other):
        return self.bid == other.bid

    # Add by Yulong Zhang
    def set_draw_local_axis(self, x):
        for geom in self._geoms:
            geom.set_draw_local_axis(x)

    # Add by Yulong Zhang. Get the 0-th geometry
    @property
    def geom0(self):
        return self._geoms[0]

    # Add by Zhenhua Song
    @property
    def world(self):
        return self._world

    # Add by Zhenhua Song
    @property
    def name(self):
        return self._name

    # Add by Zhenhua Song
    @name.setter
    def name(self, str value):
        """
        set Body's name
        """
        self._name = value

    # Add by Zhenhua Song
    @property
    def body_flags(self):
        return dBodyGetFlags(self.bid)

    # Add by Zhenhua Song
    @property
    def mass_val(self) -> dReal:
        return dBodyGetMassValue(self.bid)

    # Add by Zhenhua Song. Get the initial inertia.
    # return np.ndarray in shape (9,)
    @property
    def init_inertia(self) -> np.ndarray:
        cdef dReal * res = dBodyGetInitInertia(self.bid)
        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(9)
        ODEMat3ToDenseMat3(res, <dReal *>np_res.data, 0)
        return np_res

    # Add by Zhenhua Song
    # return np.ndarray in shape (9,)
    @property
    def init_inertia_inv(self) -> np.ndarray:
        cdef dReal * res = dBodyGetInitInertiaInv(self.bid)
        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(9)
        ODEMat3ToDenseMat3(res, <dReal *> np_res.data, 0)
        return np_res

    # Add by Zhenhua Song
    @property
    def instance_id(self):
        return self._instance_id

    # Add by Heyuan Yao
    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, int offset):
        self._offset = offset

    @property
    def offset_instance_id(self):
        return self._offset + self.instance_id

    # Add by Zhenhua Song
    @instance_id.setter
    def instance_id(self, int value):
        self._instance_id = value

    # Add by Zhenhua Song
    # @property
    def geom_iter(self):
        return iter(self._geoms)
     
    def geom_list(self):
        return self._geoms

    # Add by Zhenhua Song
    def get_bid(self):
        return <size_t>self.bid

    @property
    def Position(self):
        return self.PositionNumpy

    @Position.setter
    def Position(self, pos):
        dBodySetPosition(self.bid, pos[0], pos[1], pos[2])

    # Add by Zhenhua Song. Get global position as np.ndarray in shape (3,)
    @property
    def PositionNumpy(self):
        cdef const dReal* p  = <const dReal*>dBodyGetPosition(self.bid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]
        return np_buff

    # Add by Zhenhua Song
    # param: pos: np.ndarray in shape (3,)
    @PositionNumpy.setter
    def PositionNumpy(self, np.ndarray[np.float64_t, ndim = 1] pos):
        # As size of Position is small, create a new np.ndarray may cost more times..
        cdef const dReal * res = <const dReal *> pos.data
        dBodySetPosition(self.bid, res[0], res[1], res[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setRotAndQuatNoNormScipy(self, np.ndarray[np.float64_t, ndim = 1] Rot, np.ndarray[np.float64_t, ndim = 1] quat):
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.ascontiguousarray(Rot)
        cdef dReal * R = <dReal*> np_buff.data
        cdef dMatrix3 m
        DenseMat3ToODEMat3(m, R, 0)

        cdef const dReal * q_scipy = <const dReal *> quat.data
        cdef dQuaternion q_ode
        q_ode[0] = q_scipy[3]
        q_ode[1] = q_scipy[0]
        q_ode[2] = q_scipy[1]
        q_ode[3] = q_scipy[2]
        dBodySetRotAndQuatNoNorm(self.bid, m, q_ode)

    # Add by Zhenhua Song
    def setRotationNumpy(self, np.ndarray[np.float64_t, ndim = 1] Rot):
        """setRotationNumpy(Rot)

        Set the orientation of the body. The rotation matrix must be
        given as a sequence of 9 floats which are the elements of the
        matrix in row-major order.

        @param R: Rotation matrix
        @type R: 9-sequence of floats
        """
        # Rot needs to be continuous...
        # if A is continuous, np.ascontiguousarray(A) has same memory address as A
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.ascontiguousarray(Rot)
        cdef dReal * R = <dReal*> np_buff.data
        cdef dMatrix3 m
        DenseMat3ToODEMat3(m, R, 0)
        dBodySetRotation(self.bid, m)

    # Add by Zhenhua Song
    @property
    def odeRotation(self):
        return self.getRotation()

    def getRotation(self):
        """getRotation() -> 9-tuple

        Return the current rotation matrix as a tuple of 9 floats (row-major
        order).
        """
        cdef const dReal* m
        # The "const" in the original return value is cast away
        m = <const dReal*>dBodyGetRotation(self.bid)
        return m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]

    # Add by Zhenhua Song. Return the current rotation matrix as np.ndarray with shape (9,) (row-major order)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    def getRotationNumpy(self):
        cdef const dReal * m = <const dReal*>dBodyGetRotation(self.bid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(9)
        ODEMat3ToDenseMat3(m, <dReal*> np_buff.data , 0)

        return np_buff

    # Add by Zhenhua Song. Return the current rotation quaternion in (x, y, z, w)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    def getQuaternionScipy(self):
        # Quaternion in ode: (w, x, y, z)
        # Quaternion in scipy: (x, y, z, w)
        cdef const dReal * q_ode = dBodyGetQuaternion(self.bid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(4)
        cdef dReal * res = <dReal*> np_buff.data
        res[0] = q_ode[1]
        res[1] = q_ode[2]
        res[2] = q_ode[3]
        res[3] = q_ode[0]
        return np_buff

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setQuaternionScipy(self, np.ndarray[np.float64_t, ndim = 1] q):
        # Quaternion in ode: (w, x, y, z)
        # Quaternion in scipy: (x, y, z, w)
        cdef const dReal * q_scipy = <const dReal *> q.data
        cdef dQuaternion q_ode
        q_ode[0] = q_scipy[3]
        q_ode[1] = q_scipy[0]
        q_ode[2] = q_scipy[1]
        q_ode[3] = q_scipy[2]
        dBodySetQuaternion(self.bid, q_ode)
        # dBodySetQuaternion(self.bid, res[3], res[0], res[1], res[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getRotationVec6d(self):
        cdef const dReal * q_ode = dBodyGetQuaternion(self.bid)
        cdef dReal res[4]
        res[0] = q_ode[1]
        res[1] = q_ode[2]
        res[2] = q_ode[3]
        res[3] = q_ode[0]
        cdef np.ndarray[np.float64_t, ndim = 1] vec6d = np.empty(6)
        quat_to_vec6d_single(res, <dReal *> vec6d.data)
        return vec6d

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getRotationAxisAngle(self):
        cdef const dReal * q_ode = dBodyGetQuaternion(self.bid)
        cdef dReal res[4]
        res[0] = q_ode[1]
        res[1] = q_ode[2]
        res[2] = q_ode[3]
        res[3] = q_ode[0]
        cdef np.ndarray[np.float64_t, ndim = 1] rotvec = np.empty(3)
        cdef dReal angle = 0.0
        quat_to_rotvec_single(res, angle, <dReal *> rotvec.data)
        return rotvec

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getFacingQuaternion(self):
        """
        get the decomposed rotation (remove y rotation component)
        """
        cdef const dReal * q_ode = dBodyGetQuaternion(self.bid)
        cdef dReal res[4]
        res[0] = q_ode[1]
        res[1] = q_ode[2]
        res[2] = q_ode[3]
        res[3] = q_ode[0]
        cdef dReal y_axis[3]
        y_axis[0] = 0.0
        y_axis[1] = 1.0
        y_axis[2] = 0.0
        cdef dReal quat_y[4]
        cdef np.ndarray[np.float64_t, ndim = 1] quat_xz = np.empty(4)
        decompose_rotation_pair_single(res, y_axis, quat_y, <dReal *> quat_xz.data)
        return quat_xz

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getFacingVec6d(self):  # get facing rotation in 6d representation
        cdef const dReal * q_ode = dBodyGetQuaternion(self.bid)
        cdef dReal res[4]
        res[0] = q_ode[1]
        res[1] = q_ode[2]
        res[2] = q_ode[3]
        res[3] = q_ode[0]
        cdef dReal y_axis[3]
        y_axis[0] = 0
        y_axis[1] = 1
        y_axis[2] = 0
        cdef dReal quat_y[4]
        cdef dReal quat_xz[4]
        decompose_rotation_pair_single(res, y_axis, quat_y, quat_xz)
        cdef np.ndarray[np.float64_t, ndim = 1] vec6d = np.empty(6)
        quat_to_vec6d_single(quat_xz, <dReal *> vec6d.data)
        return vec6d

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getFacingRotVec(self):
        # convert to the order of (x, y, z, w)
        cdef const dReal * q_ode = dBodyGetQuaternion(self.bid)
        cdef dReal res[4]
        res[0] = q_ode[1]
        res[1] = q_ode[2]
        res[2] = q_ode[3]
        res[3] = q_ode[0]

        cdef dReal y_axis[3]
        y_axis[0] = 0
        y_axis[1] = 1
        y_axis[2] = 0

        # decompose rotation. res = quat_y * quat_xz
        cdef dReal quat_y[4]
        cdef dReal quat_xz[4]
        decompose_rotation_pair_single(res, y_axis, quat_y, quat_xz)
        cdef np.ndarray[np.float64_t, ndim = 1] rotvec = np.empty(3)
        cdef dReal angle = 0.0
        quat_to_rotvec_single(quat_xz, angle, <dReal *> rotvec.data)  # convert facing quaternion into axis angle
        return rotvec

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getFacingInfo(self, Body root_body = None, int is_vec6d = 1):
        # get facing rotation, linear velocity, angular velocity
        cdef dBodyID root_id = self.bid
        if root_body is not None:
            root_id = root_body.bid

        # convert to the order of (x, y, z, w)
        cdef const dReal * q_ode = dBodyGetQuaternion(root_id)
        cdef dReal res[4]
        res[0] = q_ode[1]
        res[1] = q_ode[2]
        res[2] = q_ode[3]
        res[3] = q_ode[0]
        cdef dReal quat_y[4]
        cdef dReal quat_xz[4]
        cdef dReal quat_y_inv[4]

        cdef dReal y_axis[3]
        y_axis[0] = 0
        y_axis[1] = 1
        y_axis[2] = 0

        decompose_rotation_pair_single(res, y_axis, quat_y, quat_xz)
        # compute inverse of y rotation
        quat_inv_single(quat_y, quat_y_inv)

        # convert linear velocity and angular velocity into facing coordinate
        cdef np.ndarray[np.float64_t, ndim=1] facing_velo = np.empty(3)
        cdef np.ndarray[np.float64_t, ndim=1] facing_angular = np.empty(3)
        quat_apply_single(quat_y_inv, <dReal*>dBodyGetLinearVel(self.bid), <dReal *>facing_velo.data)  # rotate the linear velocity
        quat_apply_single(quat_y_inv, <dReal*>dBodyGetAngularVel(self.bid), <dReal *>facing_angular.data)  # rotate the angular velocity

        # convert quaternion into target rotation representation
        cdef np.ndarray[np.float64_t, ndim=1] ret_rot
        cdef dReal angle = 0.0
        if is_vec6d:  # convert rotation to 6d representation
            ret_rot = np.empty(6)
            quat_to_vec6d_single(quat_xz, <dReal *> ret_rot.data)
        else:  # convert rotation to axis angle representation
            ret_rot = np.empty(3)
            quat_to_rotvec_single(quat_xz, angle, <dReal *> ret_rot.data)
        return ret_rot, facing_velo, facing_angular

    def setLinearVel(self, vel):
        """setLinearVel(vel)

        Set the linear velocity of the body.

        @param vel: New velocity
        @type vel: 3-sequence of floats
        """
        dBodySetLinearVel(self.bid, vel[0], vel[1], vel[2])

    # Add by Zhenhua Song
    @property
    def odeLinearVel(self):
        return self.getLinearVel()

    # getLinearVel
    def getLinearVel(self):
        """getLinearVel() -> 3-tuple

        Get the current linear velocity of the body.
        """
        cdef dReal* p
        # The "const" in the original return value is cast away
        p = <dReal*>dBodyGetLinearVel(self.bid)
        return p[0], p[1], p[2]

    # Add by Zhenhua Song
    @property
    def LinearVelNumpy(self) -> np.ndarray:
        cdef const dReal* v = dBodyGetLinearVel(self.bid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = v[0]
        res[1] = v[1]
        res[2] = v[2]
        return np_buff

    # Add by Zhenhua Song
    @LinearVelNumpy.setter
    def LinearVelNumpy(self, np.ndarray[np.float64_t, ndim = 1] v):
        cdef const dReal * vel = <const dReal *> (v.data)
        dBodySetLinearVel(self.bid, vel[0], vel[1], vel[2])

    def setAngularVel(self, vel):
        """setAngularVel(vel)

        Set the angular velocity of the body.

        @param vel: New angular velocity
        @type vel: 3-sequence of floats
        """
        dBodySetAngularVel(self.bid, vel[0], vel[1], vel[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAngularVelNumpy(self, np.ndarray[np.float64_t, ndim = 1] ang_vel):
        cdef const dReal * vel = <const dReal *> ang_vel.data
        dBodySetAngularVel(self.bid, vel[0], vel[1], vel[2])

    # Add by Zhenhua Song
    @property
    def odeAngularVel(self):
        return self.getAngularVel()

    def getAngularVel(self):
        """getAngularVel() -> 3-tuple

        Get the current angular velocity of the body.
        """
        cdef dReal* p
        # The "const" in the original return value is cast away
        p = <dReal*>dBodyGetAngularVel(self.bid)
        return p[0], p[1], p[2]

    # Add by Zhenhua Song. return np.ndarray in shape (3,)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngularVelNumpy(self):
        cdef const dReal* v = dBodyGetAngularVel(self.bid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = v[0]
        res[1] = v[1]
        res[2] = v[2]
        return np_buff

    # Modify by Zhenhua Song. set the mass of rigid body
    def setMass(self, Mass mass):
        """setMass(mass)

        Set the mass properties of the body. The argument mass must be
        an instance of a Mass object.

        @param mass: Mass properties
        @type mass: Mass
        """
        dBodySetMass(self.bid, &mass._mass)

    def getMass(self) -> Mass:
        """getMass() -> mass

        Return the mass properties as a Mass object.
        """
        cdef Mass m = Mass()
        dBodyGetMass(self.bid, &m._mass)
        return m

    def addForce(self, f):
        """addForce(f)

        Add an external force f given in absolute coordinates. The force
        is applied at the center of mass.

        @param f: Force
        @type f: 3-sequence of floats
        """
        dBodyAddForce(self.bid, f[0], f[1], f[2])

    # Add by Zhenhua Song
    # param: f np.ndarray
    def addForceNumpy(self, np.ndarray[np.float64_t, ndim = 1] f):
        cdef const dReal * res = <const dReal*> f.data
        dBodyAddForce(self.bid, res[0], res[1], res[2])

    def addTorque(self, t):
        """addTorque(t)

        Add an external torque t given in absolute coordinates.

        @param t: Torque
        @type t: 3-sequence of floats
        """
        dBodyAddTorque(self.bid, t[0], t[1], t[2])

    # Add by Zhenhua Song
    def addTorqueNumpy(self, np.ndarray[np.float64_t, ndim = 1] t):
        cdef const dReal * res = <const dReal*> t.data
        dBodyAddTorque(self.bid, res[0], res[1], res[2])

    def addRelForce(self, f):
        """addRelForce(f)

        Add an external force f given in relative coordinates
        (relative to the body's own frame of reference). The force
        is applied at the center of mass.

        @param f: Force
        @type f: 3-sequence of floats
        """
        dBodyAddRelForce(self.bid, f[0], f[1], f[2])

    def addRelTorque(self, t):
        """addRelTorque(t)

        Add an external torque t given in relative coordinates
        (relative to the body's own frame of reference).

        @param t: Torque
        @type t: 3-sequence of floats
        """
        dBodyAddRelTorque(self.bid, t[0], t[1], t[2])

    def addForceAtPos(self, f, p):
        """addForceAtPos(f, p)

        Add an external force f at position p. Both arguments must be
        given in absolute coordinates.

        @param f: Force
        @param p: Position
        @type f: 3-sequence of floats
        @type p: 3-sequence of floats
        """
        dBodyAddForceAtPos(self.bid, f[0], f[1], f[2], p[0], p[1], p[2])

    def addForceAtPosNumpy(self, np.ndarray[np.float64_t, ndim = 1] f, np.ndarray[np.float64_t, ndim = 1] p):
        """
        param: f: np.ndarray in shape (3,)
        p: np.ndarray in shape (3,)
        """
        dBodyAddForceAtPos(self.bid, f[0], f[1], f[2], p[0], p[1], p[2])

    def addForceAtRelPos(self, f, p):
        """addForceAtRelPos(f, p)

        Add an external force f at position p. f is given in absolute
        coordinates and p in absolute coordinates.

        @param f: Force
        @param p: Position
        @type f: 3-sequence of floats
        @type p: 3-sequence of floats
        """
        dBodyAddForceAtRelPos(self.bid, f[0], f[1], f[2], p[0], p[1], p[2])

    def addRelForceAtPos(self, f, p):
        """addRelForceAtPos(f, p)

        Add an external force f at position p. f is given in relative
        coordinates and p in relative coordinates.

        @param f: Force
        @param p: Position
        @type f: 3-sequence of floats
        @type p: 3-sequence of floats
        """
        dBodyAddRelForceAtPos(self.bid, f[0], f[1], f[2], p[0], p[1], p[2])

    def addRelForceAtRelPos(self, f, p):
        """addRelForceAtRelPos(f, p)

        Add an external force f at position p. Both arguments must be
        given in relative coordinates.

        @param f: Force
        @param p: Position
        @type f: 3-sequence of floats
        @type p: 3-sequence of floats
        """
        dBodyAddRelForceAtRelPos(self.bid, f[0], f[1], f[2], p[0], p[1], p[2])

    # Add by Zhenhua Song
    @property
    def odeForce(self):
        return self.getForce()

    def getForce(self):
        """getForce() -> 3-tuple

        Return the current accumulated force.
        """
        cdef dReal* f
        # The "const" in the original return value is cast away
        f = <dReal*>dBodyGetForce(self.bid)
        return f[0], f[1], f[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getForceNumpy(self):
        cdef const dReal* f = <dReal*>dBodyGetForce(self.bid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = f[0]
        res[1] = f[1]
        res[2] = f[2]
        return np_buff

    # Add by Zhenhua Song
    @property
    def odeTorque(self):
        return self.getTorque()

    def getTorque(self):
        """getTorque() -> 3-tuple

        Return the current accumulated torque.
        """
        cdef dReal* f
        # The "const" in the original return value is cast away
        f = <dReal*>dBodyGetTorque(self.bid)
        return f[0], f[1], f[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getTorqueNumpy(self):
        cdef dReal* t = <dReal*>dBodyGetTorque(self.bid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = t[0]
        res[1] = t[1]
        res[2] = t[2]
        return np_buff

    def setForce(self, f):
        """setForce(f)

        Set the body force accumulation vector.

        @param f: Force
        @type f: 3-tuple of floats
        """
        dBodySetForce(self.bid, f[0], f[1], f[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setForceNumpy(self, np.ndarray[np.float64_t, ndim = 1] f):
        cdef const dReal * res = <const dReal *> f.data
        dBodySetForce(self.bid, res[0], res[1], res[2])

    def setTorque(self, t):
        """setTorque(t)

        Set the body torque accumulation vector.

        @param t: Torque
        @type t: 3-tuple of floats
        """
        dBodySetTorque(self.bid, t[0], t[1], t[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setTorqueNumpy(self, np.ndarray[np.float64_t, ndim = 1] t):
        cdef const dReal * res = <const dReal *> t.data
        dBodySetTorque(self.bid, res[0], res[1], res[2])

    def getRelPointPos(self, p):
        """getRelPointPos(p) -> 3-tuple

        Utility function that takes a point p on a body and returns
        that point's position in global coordinates. The point p
        must be given in body relative coordinates.

        @param p: Body point (local coordinates)
        @type p: 3-sequence of floats
        """

        cdef dVector3 res
        dBodyGetRelPointPos(self.bid, p[0], p[1], p[2], res)
        return res[0], res[1], res[2]

    def getRelPointVel(self, p):
        """getRelPointVel(p) -> 3-tuple

        Utility function that takes a point p on a body and returns
        that point's velocity in global coordinates. The point p
        must be given in body relative coordinates.

        @param p: Body point (local coordinates)
        @type p: 3-sequence of floats
        """
        cdef dVector3 res
        dBodyGetRelPointVel(self.bid, p[0], p[1], p[2], res)
        return res[0], res[1], res[2]

    def getPointVel(self, p):
        """getPointVel(p) -> 3-tuple

        Utility function that takes a point p on a body and returns
        that point's velocity in global coordinates. The point p
        must be given in global coordinates.

        @param p: Body point (global coordinates)
        @type p: 3-sequence of floats
        """
        cdef dVector3 res
        dBodyGetPointVel(self.bid, p[0], p[1], p[2], res)
        return res[0], res[1], res[2]

    def getPosRelPoint(self, p):
        """getPosRelPoint(p) -> 3-tuple

        This is the inverse of getRelPointPos(). It takes a point p in
        global coordinates and returns the point's position in
        body-relative coordinates.

        @param p: Body point (global coordinates)
        @type p: 3-sequence of floats
        """
        cdef dVector3 res
        dBodyGetPosRelPoint(self.bid, p[0], p[1], p[2], res)
        return res[0], res[1], res[2]

    def getPosRelPointNumpy(self, np.ndarray[np.float64_t, ndim = 1] pos):
        """
        It takes a point p in global coordinates
        and returns the point's position in body-relative coordinates.

        @param p: Body point (global coordinates)
        """
        cdef np.ndarray[np.float64_t, ndim=1] np_result = np.zeros(3)
        dBodyGetPosRelPoint(self.bid, pos[0], pos[1], pos[2], <dReal *>(np_result.data))
        return np_result

    def vectorToWorld(self, v):
        """vectorToWorld(v) -> 3-tuple

        Given a vector v expressed in the body coordinate system, rotate
        it to the world coordinate system.

        @param v: Vector in body coordinate system
        @type v: 3-sequence of floats
        """
        cdef dVector3 res
        dBodyVectorToWorld(self.bid, v[0], v[1], v[2], res)
        return res[0], res[1], res[2]

    def vectorFromWorld(self, v):
        """vectorFromWorld(v) -> 3-tuple

        Given a vector v expressed in the world coordinate system, rotate
        it to the body coordinate system.

        @param v: Vector in world coordinate system
        @type v: 3-sequence of floats
        """
        cdef dVector3 res
        dBodyVectorFromWorld(self.bid, v[0], v[1], v[2], res)
        return res[0], res[1], res[2]

    def enable(self):
        """enable()

        Manually enable a body.
        """
        dBodyEnable(self.bid)

    def disable(self):
        """disable()

        Manually disable a body. Note that a disabled body that is connected
        through a joint to an enabled body will be automatically re-enabled
        at the next simulation step.
        """
        dBodyDisable(self.bid)

    @property
    def isEnabled(self) -> int:
        """isEnabled() -> bool

        Check if a body is currently enabled.
        """
        return dBodyIsEnabled(self.bid)

    def setFiniteRotationMode(self, int mode):
        """setFiniteRotationMode(mode)

        This function controls the way a body's orientation is updated at
        each time step. The mode argument can be:

        - 0: An "infinitesimal" orientation update is used. This is
        fast to compute, but it can occasionally cause inaccuracies
        for bodies that are rotating at high speed, especially when
        those bodies are joined to other bodies. This is the default
        for every new body that is created.

        - 1: A "finite" orientation update is used. This is more
        costly to compute, but will be more accurate for high speed
        rotations. Note however that high speed rotations can result
        in many types of error in a simulation, and this mode will
        only fix one of those sources of error.

        @param mode: Rotation mode (0/1)
        @type mode: int
        """
        dBodySetFiniteRotationMode(self.bid, mode)

    def getFiniteRotationMode(self) -> int:
        """getFiniteRotationMode() -> mode (0/1)

        Return the current finite rotation mode of a body (0 or 1).
        See setFiniteRotationMode().
        """
        return dBodyGetFiniteRotationMode(self.bid)

    def setFiniteRotationAxis(self, a):
        """setFiniteRotationAxis(a)

        Set the finite rotation axis of the body.  This axis only has a
        meaning when the finite rotation mode is set
        (see setFiniteRotationMode()).

        @param a: Axis
        @type a: 3-sequence of floats
        """
        dBodySetFiniteRotationAxis(self.bid, a[0], a[1], a[2])

    def getFiniteRotationAxis(self):
        """getFiniteRotationAxis() -> 3-tuple

        Return the current finite rotation axis of the body.
        """
        cdef dVector3 p
        # The "const" in the original return value is cast away
        dBodyGetFiniteRotationAxis(self.bid, p)
        return p[0], p[1], p[2]

    def getNumJoints(self):
        """getNumJoints() -> int

        Return the number of joints that are attached to this body.
        """
        return dBodyGetNumJoints(self.bid)

    # Add by Zhenhua Song
    def getJoint(self, int idx):
        return <size_t>dBodyGetJoint(self.bid, idx)

    def setGravityMode(self, mode):
        """setGravityMode(mode)

        Set whether the body is influenced by the world's gravity
        or not. If mode is True it is, otherwise it isn't.
        Newly created bodies are always influenced by the world's gravity.

        @param mode: Gravity mode
        @type mode: bool
        """
        dBodySetGravityMode(self.bid, mode)

    def getGravityMode(self) -> int:
        """getGravityMode() -> bool

        Return True if the body is influenced by the world's gravity.
        """
        return dBodyGetGravityMode(self.bid)

    def setDynamic(self):
        """setDynamic()

        Set a body to the (default) "dynamic" state, instead of "kinematic".
        See setKinematic() for more information.
        """
        dBodySetDynamic(self.bid)

    def setKinematic(self):
        """setKinematic()

        Set the kinematic state of the body (change it into a kinematic body)

        Kinematic bodies behave as if they had infinite mass. This means they don't react
        to any force (gravity, constraints or user-supplied); they simply follow 
        velocity to reach the next position. [from ODE wiki]

        """
        dBodySetKinematic(self.bid)

    def isKinematic(self) -> int:
        """isKinematic() -> bool

        Return True if the body is kinematic (not influenced by other forces).

        Kinematic bodies behave as if they had infinite mass. This means they don't react
        to any force (gravity, constraints or user-supplied); they simply follow
        velocity to reach the next position. [from ODE wiki]

        """
        return dBodyIsKinematic(self.bid)

    def setMaxAngularSpeed(self, dReal max_speed):
        """setMaxAngularSpeed(max_speed)

        You can also limit the maximum angular speed. In contrast to the damping
        functions, the angular velocity is affected before the body is moved.
        This means that it will introduce errors in joints that are forcing the
        body to rotate too fast. Some bodies have naturally high angular
        velocities (like cars' wheels), so you may want to give them a very high
        (like the default, dInfinity) limit.

        """
        dBodySetMaxAngularSpeed(self.bid, max_speed)

    # Add by Zhenhua Song
    def getNumGeom(self) -> int:
        return dBodyGetNumGeoms(self.bid)

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getGeomIDNumpy(self):
        cdef int cnt = dBodyGetNumGeoms(self.bid)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_buff = np.zeros(cnt, np.uint64)
        cdef size_t * res = <size_t*> np_buff.data
        cdef dGeomID g = dBodyGetFirstGeom(self.bid)
        cdef int idx = 0
        while g != NULL:
            res[idx] = <size_t> g
            g = dGeomGetBodyNext (g)
            idx += 1

        return np_buff

    # Add by Zhenhua Song
    def _setData(self, data):
        cdef void * res
        res = <void*> data
        dBodySetData(self.bid, res)

    # Add by Zhenhua Song
    def _getData(self):
        cdef void * res
        cdef object obj
        res = dBodyGetData(self.bid)
        obj = <object> res
        return obj

    # Add by Zhenhua Song
    def _getBodyData(self):
        return <Body> dBodyGetData(self.bid)

    # Add by Zhenhua Song:
    # compute aabb for each geometry on this body
    def get_aabb(self) -> np.ndarray:
        cdef np.ndarray[np.float64_t, ndim=1] ret = np.zeros(6)
        _init_aabb_impl(<dReal *> ret.data)
        _get_body_aabb_impl(self.bid, <dReal *> ret.data)
        return ret


cdef class JointGroup:
    """Joint group.

    Constructor::JointGroup()
    """

    # JointGroup ID
    cdef dJointGroupID gid
    # A list of Python joints that were added to the group
    cdef list jointlist  # modify by Zhenhua Song

    def __cinit__(self):
        self.gid = dJointGroupCreate(0)

    def __init__(self):
        self.jointlist = list()

    # Modify by Zhenhua Song
    def __dealloc__(self):
        self.destroy_immediate()

    def __len__(self):
        return len(self.jointlist)

    # Add by Zhenhua Song
    def destroy_immediate(self):  # for contact joints, first destroy joint group and C++ joint object, then destroy self.jointlist automatically
        if self.gid != NULL:
            for j in self.jointlist:
                j._destroyed()
            dJointGroupDestroy(self.gid)
            self.gid = NULL

    # Add by Zhenhua Song
    @property
    def joints(self) -> list:
        return self.jointlist

    def empty(self):
        """empty()

        Destroy all joints in the group.
        """
        for j in self.jointlist:
            j.pre_clear()
        dJointGroupEmpty(self.gid)
        for j in self.jointlist:
            j._destroyed()
        self.jointlist.clear()  # This will deconstruct all joints in self.jointlist
        # print("Call joint group empty")

    cdef _addjoint(self, Joint j):
        """_addjoint(j)

        Add a joint to the group.  This is an internal method that is
        called by the joints.  The group has to know the Python
        wrappers because it has to notify them when the group is
        emptied (so that the ODE joints won't get destroyed
        twice). The notification is done by calling _destroyed() on
        the Python joints.

        @param j: The joint to add
        @type j: Joint
        """
        self.jointlist.append(j)


######################################################################

cdef class Joint:
    """Base class for all joint classes."""

    # Joint id as returned by dJointCreateXxx()
    cdef dJointID jid
    # A reference to the world so that the world won't be destroyed while
    # there are still joints using it.
    cdef World _world
    # The feedback buffer
    cdef dJointFeedback* feedback

    cdef Body _body1
    cdef Body _body2
    cdef str _name  # Add by Zhenhua Song
    cdef str _euler_order  # Add by Zhenhua Song

    # cdef np.ndarray _euler_axis # Add by Zhenhua Song

    cdef int _instance_id # Add by Zhenhua Song, instance id in Unity client
    # instance_id and _joint_index not same..
    # for external joint, _instance_id is instance id in Unity client
    # for character, _instance_id is joint index in a character

    # cdef object __weakref__  # Add by Zhenhua Song

    def __cinit__(self, *a, **kw):
        self.jid = NULL
        self._world = None
        self.feedback = NULL
        self._body1 = None
        self._body2 = None
        self._name = ""
        self._euler_order = ""

        self._instance_id = 0

    def __init__(self, *a, **kw):
        raise NotImplementedError("Joint base class can't be used directly")

    # Modify by Zhenhua Song
    def __dealloc__(self):
        self.destroy_immediate()

    # Add by Heyuan Yao
    def enable_implicit_damping(self):
        dJointEnableImplicitDamping(self.jid)

    # Add by Zhenhua Song
    def disable_implicit_damping(self):
        dJointDisableImplicitDamping(self.jid)

    # Add by Zhenhua Song
    def destroy_immediate(self):
        if self.jid != NULL:
            self.setFeedback(False)
            self.detach()
            dJointDestroy(self.jid)
            self.jid = NULL

    # Add by Zhenhua Song
    def __eq__(self, Joint other):
        return self.jid == other.jid

    def pre_clear(self):
        self.setFeedback(False)
        dJointSetData(self.jid, NULL)
        self.attach(None, None)

    def _destroyed(self):
        """Notify the joint object about an external destruction of the ODE joint.

        This method has to be called when the underlying ODE object
        was destroyed by someone else (e.g. by a joint group). The Python
        wrapper will then refrain from destroying it again.
        """
        self.jid = NULL

    # Add by Zhenhua Song
    @property
    def world(self) -> World:
        return self._world

    # Add by Zhenhua Song
    @property
    def instance_id(self) -> int:
        return self._instance_id

    # Add by Zhenhua Song
    @instance_id.setter
    def instance_id(self, int value):
        self._instance_id = value

    # Add by Zhenhua Song
    def get_jid(self):
        return <size_t>self.jid

    # Add by Zhenhua Song
    def getName(self) -> str:
        return self._name

    # Add by Zhenhua Song
    @property
    def name(self) -> str:
        return self._name

    # Add by Zhenhua Song
    @name.setter
    def name(self, str value):
        self._name = value

    # Add by Zhenhua Song
    def getNumBodies(self):
        cdef int cnt = 0
        if self._body1 is not None:
            cnt += 1
        if self._body2 is not None:
            cnt += 1
        return cnt

    def enable(self):
        """enable()

        Enable the joint. Disabled joints are completely ignored during the
        simulation. Disabled joints don't lose the already computed information
        like anchors and axes.
        """
        dJointEnable(self.jid)

    def disable(self):
        """disable()

        Disable the joint. Disabled joints are completely ignored during the
        simulation. Disabled joints don't lose the already computed information
        like anchors and axes.
        """
        dJointDisable(self.jid)

    def isEnabled(self):
        """isEnabled() -> bool

        Determine whether the joint is enabled. Disabled joints are completely
        ignored during the simulation. Disabled joints don't lose the already
        computed information like anchors and axes.
        """
        return dJointIsEnabled(self.jid)

    # Add by Zhenhua Song
    def detach(self):
        if self._body1 is not None and self._body1._bid_is_not_null():
            # self._body1._joints.remove(self)
            self._body1 = None

        if self._body2 is not None and self._body2._bid_is_not_null():
            # self._body2._joints.remove(self)
            self._body2 = None

        if self.jid != NULL:
            dJointAttach(self.jid, NULL, NULL)

    def attach(self, Body body1, Body body2):
        """attach(body1, body2)

        Attach the joint to some new bodies. A body can be attached
        to the environment by passing None as second body.

        @param body1: First body
        @param body2: Second body
        @type body1: Body
        @type body2: Body
        """
        cdef dBodyID id1, id2
        self.detach()

        if body1 is None:
            id1 = NULL
        else:
            id1 = body1.bid

        if body2 is None:
            id2 = NULL
        else:
            id2 = body2.bid

        # Add by Zhenhua Song
        if id1 != NULL and id2 != NULL and id1 == id2:
            print("Warning: body1.bid == body2.bid in joint attach")

        self._body1 = body1
        self._body2 = body2

        dJointAttach(self.jid, id1, id2)

    # Add by Zhenhua Song
    def attach_ext(self, Body body1, Body body2):
        self.attach(body1, body2)

    def getBody(self, int index):
        """getBody(index) -> Body

        Return the bodies that this joint connects. If index is 0 the
        "first" body will be returned, corresponding to the body1
        argument of the attach() method. If index is 1 the "second" body
        will be returned, corresponding to the body2 argument of the
        attach() method.

        @param index: Bodx index (0 or 1).
        @type index: int
        """

        if index == 0:
            return self._body1
        elif index == 1:
            return self._body2
        else:
            raise IndexError()

    # Add by Zhenhua Song
    def dJointGetBody(self, int index):
        return <size_t> dJointGetBody(self.jid, index)

    # Add by Zhenhua Song
    @property
    def body1(self):
        return self._body1

    # Add by Zhenhua Song
    @property
    def body2(self):
        return self._body2

    # Add by Zhenhua Song
    @property
    def euler_order(self) -> str:
       return self._euler_order

    # Add by Zhenhua Song
    @euler_order.setter
    def euler_order(self, str euler_order_):
       self._euler_order = euler_order_

    # @property
    # def euler_axis(self) -> np.ndarray:
    #    return self._euler_axis

    # @euler_axis.setter
    # def euler_axis(self, np.ndarray value):
    #    self._euler_axis = value

    def setFeedback(self, int flag=1):
        """setFeedback(flag=True)

        Create a feedback buffer. If flag is True then a buffer is
        allocated and the forces/torques applied by the joint can
        be read using the getFeedback() method. If flag is False the
        buffer is released.

        @param flag: Specifies whether a buffer should be created or released
        @type flag: bool
        """

        if flag:
            # Was there already a buffer allocated? then we're finished
            if self.feedback != NULL:
                return
            # Allocate a buffer and pass it to ODE
            self.feedback = <dJointFeedback*>malloc(sizeof(dJointFeedback))
            if self.feedback == NULL:  # this will not happen.
                raise MemoryError("can't allocate feedback buffer")
            dJointSetFeedback(self.jid, self.feedback)
        else:
            if self.feedback != NULL:
                # Free a previously allocated buffer
                dJointSetFeedback(self.jid, NULL)
                free(self.feedback)
                self.feedback = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def FeedBackForce(self) -> np.ndarray:
        cdef dJointFeedback* fb = dJointGetFeedback(self.jid)
        assert fb != NULL
        cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(3)
        res[0] = fb.f1[0]
        res[1] = fb.f1[1]
        res[2] = fb.f1[2]
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def FeedBackTorque(self) -> np.ndarray:
        cdef dJointFeedback* fb = dJointGetFeedback(self.jid)
        assert fb != NULL
        cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(3, np.float64)
        res[0] = fb.t1[0]
        res[1] = fb.t1[1]
        res[2] = fb.t1[2]
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getFeedback(self):
        """getFeedback() -> (force1, torque1, force2, torque2)

        Get the forces/torques applied by the joint. If feedback is
        activated (i.e. setFeedback(True) was called) then this method
        returns a tuple (force1, torque1, force2, torque2) with the
        forces and torques applied to body 1 and body 2.  The
        forces/torques are given as 3-tuples.

        If feedback is deactivated then the method always returns None.
        """
        cdef dJointFeedback* fb

        fb = dJointGetFeedback(self.jid)
        if fb == NULL:
            return None

        f1 = (fb.f1[0], fb.f1[1], fb.f1[2])
        t1 = (fb.t1[0], fb.t1[1], fb.t1[2])
        f2 = (fb.f2[0], fb.f2[1], fb.f2[2])
        t2 = (fb.t2[0], fb.t2[1], fb.t2[2])
        return f1, t1, f2, t2

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getKd(self):
        cdef const dReal * res = dJointGetKd(self.jid)
        return res[0], res[1], res[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getKdNumpy(self) -> np.ndarray:
        cdef const dReal * kd = dJointGetKd(self.jid)
        cdef np.ndarray[np.float64_t, ndim = 1] res = np.zeros(3)
        res[0] = kd[0]
        res[1] = kd[1]
        res[2] = kd[2]
        return res

    # Add by Zhenhua Song
    @property
    def joint_damping(self) -> np.ndarray:
        return self.getKdNumpy()

    # Add by Zhenhua Song
    @property
    def joint_erp(self):
        raise NotImplementedError

    @property
    def joint_cfm(self):
        raise NotImplementedError

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setKd(self, dReal kdx, dReal kdy, dReal kdz):
        dJointSetKd(self.jid, kdx, kdy, kdz)

    # Add by Zhenhua Song
    def setSameKd(self, dReal kd):
        dJointSetKd(self.jid, kd, kd, kd)

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setKd_arrNumpy(self, np.ndarray[np.float64_t, ndim = 1] kd):
        dJointSetKd_arr(self.jid, <const dReal *> kd.data)

    # Add by Zhenhua Song
    def getType(self) -> int:
        return dJointGetType(self.jid)

    # Add by Zhenhua Song
    @property
    def joint_dof(self):
        return self.get_joint_dof()

    # Add by Zhenhua Song
    def get_joint_dof(self) -> int:
        return 0

    # Add by Zhenhua Song
    @property
    def odeAnchor(self):
        return self.getAnchor()

    # Add by Zhenhua Song
    @property
    def odeAnchor2(self):
        return self.getAnchor2()

    # Add by Zhenhua Song
    # @property
    # def odeAnchorRaw(self):
    #    pass

    # Add by Zhenhua Song
    def _setData(self, value):
        cdef void * res
        res = <void *> value
        dJointSetData(self.jid, res)

    # Add by Zhenhua Song
    def _getData(self):
        return <object> dJointGetData(self.jid)

    # Add by Zhenhua Song
    def setAnchor(self, val):
        raise NotImplementedError

    # Add by Zhenhua Song
    def setAnchorNumpy(self, np.ndarray[np.float64_t, ndim = 1] val):
        raise NotImplementedError

    # Add by Zhenhua Song
    def getAnchor(self):
        raise NotImplementedError

    # Add by Zhenhua Song
    def getAnchor2(self):
        raise NotImplementedError

    # Add by Zhenhua Song
    def getAnchorNumpy(self):
        raise NotImplementedError

    # Add by Zhenhua Song
    def getAnchor2Numpy(self):
        raise NotImplementedError


######################################################################

cdef class EmptyBallJoint(Joint):

    def __cinit__(self, World world not None):
        self.jid = dJointCreateEmptyBall(world.wid, NULL)

    def __init__(self, World world not None):
        pass


cdef class BallJointBase(Joint):
    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        # TODO: This may memory leak...
        # self._euler_axis = np.eye(3)
        pass

    def __init__(self, World world not None, JointGroup jointgroup=None):
        pass

    @property
    def joint_erp(self):
        return dJointGetBallParam(self.jid, dParamERP)

    @property
    def joint_cfm(self):
        return dJointGetBallParam(self.jid, dParamCFM)


# Add by Zhenhua Song
# Ball Joint with amotor.
cdef class BallJointAmotor(BallJointBase):
    """Ball joint with AMotor.

    Constructor::BallJointAmotor(world, jointgroup=None)
    """

    cdef dJointID amotor_jid  # ball joint and amotor joint are both attached to bodies

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateBall(world.wid, jgid)
        self.amotor_jid = dJointCreateAMotor(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)  # DO NOT use weakref.proxy or weakref.ref

    # Modify by Zhenhua Song
    def __dealloc__(self):
        self._destroy_amotor()

    # Add by Zhenhua Song
    def _destroy_amotor(self):
        if self.amotor_jid != NULL:
            dJointDestroy(self.amotor_jid)
            self.amotor_jid = NULL

    # Add by Zhenhua Song
    def destroy_immediate(self):
        super(BallJointAmotor, self).destroy_immediate()
        self._destroy_amotor()

    def attach_ext(self, Body body1, Body body2):
        self.attach(body1, body2)
        dJointAttach(self.amotor_jid, self._body1.bid if self._body1 is not None else NULL, self._body2.bid if self._body2 is not None else NULL)

    def get_amotor_jid(self):
        return <size_t> self.amotor_jid

    def setAnchor(self, pos):
        dJointSetBallAnchor(self.jid, pos[0], pos[1], pos[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAnchorNumpy(self, np.ndarray[np.float64_t, ndim = 1] np_pos):
        cdef dReal * pos = <dReal *>np_pos.data
        dJointSetBallAnchor(self.jid, pos[0], pos[1], pos[2])

    def getAnchor(self):
        cdef dVector3 p
        dJointGetBallAnchor(self.jid, p)
        return p[0], p[1], p[2]

    def getAnchor2(self):
        cdef dVector3 p
        dJointGetBallAnchor2(self.jid, p)
        return p[0], p[1], p[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchorNumpy(self) -> np.ndarray:
        cdef dVector3 p
        dJointGetBallAnchor(self.jid, p)
        cdef np.ndarray[np.float64_t, ndim=1] np_pos = np.zeros(3)
        cdef dReal * pos_res = <dReal *> np_pos.data
        pos_res[0] = p[0]
        pos_res[1] = p[1]
        pos_res[2] = p[2]

        return np_pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2Numpy(self) -> np.ndarray:
        cdef dVector3 p
        dJointGetBallAnchor2(self.jid, p)
        cdef np.ndarray[np.float64_t, ndim=1] np_pos = np.zeros(3)
        cdef dReal * pos_res = <dReal *> np_pos.data
        pos_res[0] = p[0]
        pos_res[1] = p[1]
        pos_res[2] = p[2]

        return np_pos

    def getAnchor1Raw(self):
        cdef const dReal * res = dJointGetBallAnchor1Raw(self.jid)
        return res[0], res[1], res[2]

    def getAnchor2Raw(self):
        cdef const dReal * res = dJointGetBallAnchor2Raw(self.jid)
        return res[0], res[1], res[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor1RawNumpy(self) -> np.ndarray:
        cdef const dReal * p = dJointGetBallAnchor1Raw(self.jid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]
        return np_buff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2RawNumpy(self) -> np.ndarray:
        cdef const dReal * p = dJointGetBallAnchor2Raw(self.jid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]
        return np_buff

    def setAmotorMode(self, int mode):
        dJointSetAMotorMode(self.amotor_jid, mode)

    def getAmotorMode(self):
        return dJointGetAMotorMode(self.amotor_jid)

    @property
    def AMotorMode(self) -> int:
        return dJointGetAMotorMode(self.amotor_jid)

    def setAmotorNumAxes(self, int num):
        dJointSetAMotorNumAxes(self.amotor_jid, num)

    def getAmtorNumAxes(self):
        return dJointGetAMotorNumAxes(self.amotor_jid)

    @property
    def AMotorNumAxis(self):
        return dJointGetAMotorNumAxes(self.amotor_jid)

    def setAmotorAxis(self, int anum, int rel, axis):
        dJointSetAMotorAxis(self.amotor_jid, anum, rel, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAmotorAxisNumpy(self, int anum, int rel, np.ndarray[np.float64_t, ndim = 1] np_axis):
        cdef np.ndarray[np.float64_t, ndim = 1] np_buf = np.ascontiguousarray(np_axis)
        cdef dReal * axis = <dReal *>np_buf.data
        dJointSetAMotorAxis(self.amotor_jid, anum, rel, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAmotorAxis(self, int anum):
        cdef dVector3 a
        dJointGetAMotorAxis(self.amotor_jid, anum, a)
        return a[0], a[1], a[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAmotorAxisNumpy(self, int anum) -> np.ndarray:
        cdef dVector3 a
        dJointGetAMotorAxis(self.amotor_jid, anum, a)

        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(3)
        cdef dReal * res = <dReal *>np_res.data

        res[0] = a[0]
        res[1] = a[1]
        res[2] = a[2]
        return np_res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAmotorAllAxisNumpy(self) -> np.ndarray:
        cdef dVector3 a
        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(9)
        cdef dReal * res = <dReal *> np_res.data
        cdef int i = 0
        while i < 3:
            dJointGetAMotorAxis(self.amotor_jid, i, a)
            res[3 * i + 0] = a[0]
            res[3 * i + 1] = a[1]
            res[3 * i + 2] = a[2]
            i += 1
        return np_res

    @property
    def AllAxis(self) -> np.ndarray:
        return self.getAmotorAllAxisNumpy().reshape((3, 3))

    def getAmotorAxisRel(self, int anum):
        return dJointGetAMotorAxisRel(self.amotor_jid, anum)

    @property
    def AllAxisRel(self):
        return dJointGetAMotorAxisRel(self.amotor_jid, 0), dJointGetAMotorAxisRel(self.amotor_jid, 1), dJointGetAMotorAxisRel(self.amotor_jid, 2)

    def setAmotorAngle(self, int anum, dReal angle):
        dJointSetAMotorAngle(self.amotor_jid, anum, angle)

    def getAmotorAngle(self, int anum):
        return dJointGetAMotorAngle(self.amotor_jid, anum)

    def getAmotorAngleRate(self, int anum):
        raise NotImplementedError
        # return dJointGetAMotorAngleRate(self.amotor_jid, anum)

    def getAmotorAngleRateNumpy(self):
        raise NotImplementedError

    def addAmotorTorques(self, dReal torque0, dReal torque1, dReal torque2):
        dJointAddAMotorTorques(self.amotor_jid, torque0, torque1, torque2)

    def setAmotorParam(self, int param, dReal value):
        dJointSetAMotorParam(self.amotor_jid, param, value)

    def getAmotorParam(self, int param):
        return dJointGetAMotorParam(self.amotor_jid, param)

    def setAngleLim1(self, dReal lo, dReal hi):
        dJointSetAMotorParam(self.amotor_jid, dParamLoStop, lo)
        dJointSetAMotorParam(self.amotor_jid, dParamHiStop, hi)

    def setAngleLim2(self, dReal lo, dReal hi):
        dJointSetAMotorParam(self.amotor_jid, dParamLoStop2, lo)
        dJointSetAMotorParam(self.amotor_jid, dParamHiStop2, hi)

    def setAngleLim3(self, dReal lo, dReal hi):
        dJointSetAMotorParam(self.amotor_jid, dParamLoStop3, lo)
        dJointSetAMotorParam(self.amotor_jid, dParamHiStop3, hi)

    def getAngleLimit1(self):
        return dJointGetAMotorParam(self.amotor_jid, dParamLoStop), dJointGetAMotorParam(self.amotor_jid, dParamHiStop)

    def getAngleLimit2(self):
        return dJointGetAMotorParam(self.amotor_jid, dParamLoStop2), dJointGetAMotorParam(self.amotor_jid, dParamHiStop2)

    def getAngleLimit3(self):
        return dJointGetAMotorParam(self.amotor_jid, dParamLoStop3), dJointGetAMotorParam(self.amotor_jid, dParamHiStop3)

    @property
    def AngleLimit(self):
        return [self.getAngleLimit1(), self.getAngleLimit2(), self.getAngleLimit3()]

    @property
    def Angles(self):
        return dJointGetAMotorAngle(self.amotor_jid, 0), dJointGetAMotorAngle(self.amotor_jid, 1), dJointGetAMotorAngle(self.amotor_jid, 2)

    @property
    def ball_erp(self) -> dReal:
        return dJointGetBallParam(self.jid, dParamERP)

    @property
    def ball_cfm(self) -> dReal:
        return dJointGetBallParam(self.jid, dParamCFM)

    @property
    def amotor_erp(self) -> dReal:
        return dJointGetAMotorParam(self.amotor_jid, dParamERP)

    @property
    def amotor_cfm(self) -> dReal:
        return dJointGetAMotorParam(self.amotor_jid, dParamCFM)

    def get_joint_dof(self) -> int:
        return 3

    def _setData(self, value):
        cdef void * res
        res = <void *> value
        dJointSetData(self.jid, res)
        dJointSetData(self.amotor_jid, res)


cdef class BallJoint(BallJointBase):
    """Ball joint.

    Constructor::BallJoint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateBall(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)  # DO NOT use weakref.proxy or weakref.ref

    def setAnchor(self, pos):
        """setAnchor(pos)

        Set the joint anchor point which must be specified in world
        coordinates.

        @param pos: Anchor position
        @type pos: 3-sequence of floats
        """
        dJointSetBallAnchor(self.jid, pos[0], pos[1], pos[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAnchorNumpy(self, np.ndarray[np.float64_t, ndim = 1] np_pos):
        cdef dReal * pos = <dReal *>np_pos.data
        dJointSetBallAnchor(self.jid, pos[0], pos[1], pos[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor(self):
        """getAnchor() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates.  This
        returns the point on body 1.  If the joint is perfectly
        satisfied, this will be the same as the point on body 2.
        """

        cdef dVector3 p
        dJointGetBallAnchor(self.jid, p)
        return p[0], p[1], p[2]

    def getAnchor2(self):
        """getAnchor2() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates.  This
        returns the point on body 2. If the joint is perfectly
        satisfied, this will be the same as the point on body 1.
        """

        cdef dVector3 p
        dJointGetBallAnchor2(self.jid, p)
        return p[0], p[1], p[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchorNumpy(self) -> np.ndarray:
        cdef dVector3 p
        dJointGetBallAnchor(self.jid, p)
        cdef np.ndarray[np.float64_t, ndim=1] np_pos = np.zeros(3)
        cdef dReal * pos_res = <dReal *> np_pos.data
        pos_res[0] = p[0]
        pos_res[1] = p[1]
        pos_res[2] = p[2]

        return np_pos

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2Numpy(self) -> np.ndarray:
        cdef dVector3 p
        dJointGetBallAnchor2(self.jid, p)
        cdef np.ndarray[np.float64_t, ndim = 1] np_pos = np.zeros(3)
        cdef dReal * pos_res = <dReal *> np_pos.data
        pos_res[0] = p[0]
        pos_res[1] = p[1]
        pos_res[2] = p[2]

        return np_pos

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor1Raw(self):
        cdef const dReal * res = dJointGetBallAnchor1Raw(self.jid)
        return res[0], res[1], res[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2Raw(self):
        cdef const dReal * res = dJointGetBallAnchor2Raw(self.jid)
        return res[0], res[1], res[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor1RawNumpy(self) -> np.ndarray:
        cdef const dReal * p = dJointGetBallAnchor1Raw(self.jid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]
        return np_buff

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2RawNumpy(self) -> np.ndarray:
        cdef const dReal * p = dJointGetBallAnchor2Raw(self.jid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]
        return np_buff

    def setParam(self, int param, dReal value):
        # modify by Zhenhua Song
        raise NotImplementedError

    def getParam(self, int param):
        # modify by Zhenhua Song
        raise NotImplementedError

    # Add by Zhenhua Song
    def get_joint_dof(self) -> int:
        return 3

    # Add by Zhenhua Song
    @property
    def joint_cfm(self) -> dReal:
        return dJointGetBallParam(self.jid, dParamCFM)

    # Add by Zhenhua Song
    @property
    def joint_erp(self) -> dReal:
        return dJointGetBallParam(self.jid, dParamERP)


cdef class HingeJoint(Joint):
    """Hinge joint.

    Constructor::HingeJoint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        # self._euler_axis = np.eye(3)
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateHinge(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAnchor(self, pos):
        """setAnchor(pos)

        Set the hinge anchor which must be given in world coordinates.

        @param pos: Anchor position
        @type pos: 3-sequence of floats
        """
        dJointSetHingeAnchor(self.jid, pos[0], pos[1], pos[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAnchorNumpy(self, np.ndarray[np.float64_t, ndim = 1] np_pos):
        cdef dReal * pos = <dReal *>np_pos.data
        dJointSetHingeAnchor(self.jid, pos[0], pos[1], pos[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor(self):
        """getAnchor() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates. This returns
        the point on body 1. If the joint is perfectly satisfied, this
        will be the same as the point on body 2.
        """
        cdef dVector3 p
        dJointGetHingeAnchor(self.jid, p)
        return p[0], p[1], p[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchorNumpy(self) -> np.ndarray:
        cdef dVector3 p
        dJointGetHingeAnchor(self.jid, p)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]

        return np_buff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2(self):
        """getAnchor2() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates. This returns
        the point on body 2. If the joint is perfectly satisfied, this
        will be the same as the point on body 1.
        """
        cdef dVector3 p
        dJointGetHingeAnchor2(self.jid, p)
        return p[0], p[1], p[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2Numpy(self):
        cdef dVector3 p
        dJointGetHingeAnchor2(self.jid, p)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]

        return np_buff

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor1Raw(self):
        cdef const dReal * p = dJointGetHingeAnchor1Raw(self.jid)
        return p[0], p[1], p[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2Raw(self):
        cdef const dReal * p = dJointGetHingeAnchor2Raw(self.jid)
        return p[0], p[1], p[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor1RawNumpy(self):
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef const dReal * p = dJointGetHingeAnchor1Raw(self.jid)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]

        return np_buff

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2RawNumpy(self):
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef const dReal * p = dJointGetHingeAnchor2Raw(self.jid)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]

        return np_buff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAxis(self, axis):
        """setAxis(axis)

        Set the hinge axis.

        @param axis: Hinge axis
        @type axis: 3-sequence of floats
        """
        dJointSetHingeAxis(self.jid, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis(self):
        """getAxis() -> 3-tuple of floats

        Get the hinge axis.
        """
        cdef dVector3 a
        dJointGetHingeAxis(self.jid, a)
        return a[0], a[1], a[2]

    # Add by Zhenhua Song
    @property
    def HingeAxis(self):
        return self.getAxis()

    @property
    def HingeAxis1(self) -> np.ndarray:  # calc hinge axis by body1
        cdef dVector3 a
        cdef np.ndarray[np.float64_t, ndim = 1] res = np.zeros(3)
        dJointGetHingeAxis1(self.jid, a)
        memcpy(<dReal *> res.data, a, sizeof(dReal) * 3)
        return res

    @property
    def HingeAxis2(self) -> np.ndarray:  # calc hinge axis by body2
        cdef dVector3 a
        cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(3)
        dJointGetHingeAxis2(self.jid, a)
        memcpy(<dReal *> res.data, a, sizeof(dReal) * 3)
        return res

    # Add by Zhenhua Song
    @property
    def HingeAngle(self) -> dReal:
        """HingeAngle() -> float

        Get the hinge angle. The angle is measured between the two
        bodies, or between the body and the static environment. The
        angle will be between -pi..pi.

        When the hinge anchor or axis is set, the current position of
        the attached bodies is examined and that position will be the
        zero angle.
        """
        return dJointGetHingeAngle(self.jid)

    # Add by Zhenhua Song
    @property
    def HingeAngleRate(self) -> dReal:
        """
        Get the time derivative of the angle.
        """
        return dJointGetHingeAngleRate(self.jid)

    def addTorque(self, torque):
        """addTorque(torque)

        Applies the torque about the hinge axis.

        @param torque: Torque magnitude
        @type torque: float
        """
        dJointAddHingeTorque(self.jid, torque)

    def setParam(self, int param, dReal value):
        """setParam(param, value)

        Set limit/motor parameters for the joint.

        param is one of ParamLoStop, ParamHiStop, ParamVel, ParamFMax,
        ParamFudgeFactor, ParamBounce, ParamCFM, ParamStopERP, ParamStopCFM,
        ParamSuspensionERP, ParamSuspensionCFM.

        These parameter names can be optionally followed by a digit (2
        or 3) to indicate the second or third set of parameters.

        @param param: Selects the parameter to set
        @param value: Parameter value
        @type param: int
        @type value: float
        """

        dJointSetHingeParam(self.jid, param, value)

    def getParam(self, int param):
        """getParam(param) -> float

        Get limit/motor parameters for the joint.

        param is one of ParamLoStop, ParamHiStop, ParamVel, ParamFMax,
        ParamFudgeFactor, ParamBounce, ParamCFM, ParamStopERP, ParamStopCFM,
        ParamSuspensionERP, ParamSuspensionCFM.

        These parameter names can be optionally followed by a digit (2
        or 3) to indicate the second or third set of parameters.

        @param param: Selects the parameter to read
        @type param: int
        """
        return dJointGetHingeParam(self.jid, param)

    # Add by Zhenhua Song
    @property
    def HingeFlags(self) -> int:
        return dJointGetHingeFlags(self.jid)

    # Add by Zhenhua Song
    @property
    def Axis1RawNumpy(self) -> np.ndarray:
        cdef dVector3 a
        cdef np.ndarray[np.float64_t, ndim = 1] res = np.zeros(3)
        dJointGetHingeAxis1Raw(self.jid, a)
        memcpy(<dReal *> res.data, a, sizeof(dReal) * 3)
        return res

    # Add by Zhenhua Song
    @property
    def Axis2RawNumpy(self) -> np.ndarray:
        cdef dVector3 a
        cdef np.ndarray[np.float64_t, ndim = 1] res = np.zeros(3)
        dJointGetHingeAxis2Raw(self.jid, a)
        memcpy(<dReal *> res.data, a, sizeof(dReal) * 3)
        return res

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getQRelScipy(self) -> np.ndarray:
        cdef dQuaternion q
        dJointGetHingeQRel(self.jid, q)
        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(4)
        cdef dReal * data = <dReal *>np_res.data
        data[0] = q[1]
        data[1] = q[2]
        data[2] = q[3]
        data[3] = q[0]
        return np_res

    # Add by Zhenhua Song
    def get_joint_dof(self):
        return 1

    # Add by Zhenhua Song
    def setAngleLimit(self, dReal lo, dReal hi):
        dJointSetHingeParam(self.jid, dParamLoStop, lo)
        dJointSetHingeParam(self.jid, dParamHiStop, hi)

    # Add by Zhenhua Song
    @property
    def AngleLoStop(self) -> dReal:
        return dJointGetHingeParam(self.jid, dParamLoStop)

    # Add by Zhenhua Song
    @property
    def AngleHiStop(self) -> dReal:
        return dJointGetHingeParam(self.jid, dParamHiStop)

    # Add by Zhenhua Song
    @property
    def joint_erp(self) -> dReal:
        return dJointGetHingeParam(self.jid, dParamERP)

    @property
    def joint_cfm(self) -> dReal:
        return dJointGetHingeParam(self.jid, dParamCFM)

    # Add by Zhenhua Song
    @property
    def hinge_erp(self) -> dReal:
        return dJointGetHingeParam(self.jid, dParamERP)

    # Add by Zhenhua Song
    @property
    def hinge_cfm(self) -> dReal:
        return dJointGetHingeParam(self.jid, dParamCFM)

    # Add by Heyuan Yao
    @property
    def hinge_stop_cfm(self) -> dReal:
        return dJointGetHingeParam(self.jid, dParamStopCFM)


cdef class SliderJoint(Joint):
    """Slider joint.
    Constructor::SlideJoint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateSlider(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    def setAxis(self, axis):
        """setAxis(axis)

        Set the slider axis parameter.

        @param axis: Slider axis
        @type axis: 3-sequence of floats
        """
        dJointSetSliderAxis(self.jid, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis(self):
        """getAxis() -> 3-tuple of floats

        Get the slider axis parameter.
        """
        cdef dVector3 a
        dJointGetSliderAxis(self.jid, a)
        return a[0], a[1], a[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getPosition(self):
        """getPosition() -> float

        Get the slider linear position (i.e. the slider's "extension").

        When the axis is set, the current position of the attached
        bodies is examined and that position will be the zero
        position.
        """

        return dJointGetSliderPosition(self.jid)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getPositionRate(self):
        """getPositionRate() -> float

        Get the time derivative of the position.
        """
        return dJointGetSliderPositionRate(self.jid)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def addForce(self, force):
        """addForce(force)

        Applies the given force in the slider's direction.

        @param force: Force magnitude
        @type force: float
        """
        dJointAddSliderForce(self.jid, force)

    def setParam(self, param, value):
        dJointSetSliderParam(self.jid, param, value)

    def getParam(self, param):
        return dJointGetSliderParam(self.jid, param)


cdef class UniversalJoint(Joint):
    """Universal joint.

    Constructor::UniversalJoint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateUniversal(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    def setAnchor(self, pos):
        """setAnchor(pos)

        Set the universal anchor.

        @param pos: Anchor position
        @type pos: 3-sequence of floats
        """
        dJointSetUniversalAnchor(self.jid, pos[0], pos[1], pos[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor(self):
        """getAnchor() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates. This returns
        the point on body 1. If the joint is perfectly satisfied, this
        will be the same as the point on body 2.
        """

        cdef dVector3 p
        dJointGetUniversalAnchor(self.jid, p)
        return p[0], p[1], p[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2(self):
        """getAnchor2() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates. This returns
        the point on body 2. If the joint is perfectly satisfied, this
        will be the same as the point on body 1.
        """

        cdef dVector3 p
        dJointGetUniversalAnchor2(self.jid, p)
        return p[0], p[1], p[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAxis1(self, axis):
        """setAxis1(axis)

        Set the first universal axis. Axis 1 and axis 2 should be
        perpendicular to each other.

        @param axis: Joint axis
        @type axis: 3-sequence of floats
        """
        dJointSetUniversalAxis1(self.jid, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis1(self):
        """getAxis1() -> 3-tuple of floats

        Get the first univeral axis.
        """
        cdef dVector3 a
        dJointGetUniversalAxis1(self.jid, a)
        return a[0], a[1], a[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAxis2(self, axis):
        """setAxis2(axis)

        Set the second universal axis. Axis 1 and axis 2 should be
        perpendicular to each other.

        @param axis: Joint axis
        @type axis: 3-sequence of floats
        """
        dJointSetUniversalAxis2(self.jid, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis2(self):
        """getAxis2() -> 3-tuple of floats

        Get the second univeral axis.
        """
        cdef dVector3 a
        dJointGetUniversalAxis2(self.jid, a)
        return a[0], a[1], a[2]

    def addTorques(self, torque1, torque2):
        """addTorques(torque1, torque2)

        Applies torque1 about axis 1, and torque2 about axis 2.

        @param torque1: Torque 1 magnitude
        @param torque2: Torque 2 magnitude
        @type torque1: float
        @type torque2: float
        """
        dJointAddUniversalTorques(self.jid, torque1, torque2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngle1(self):
        return dJointGetUniversalAngle1(self.jid)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngle2(self):
        return dJointGetUniversalAngle2(self.jid)

    def getAngle1Rate(self):
        return dJointGetUniversalAngle1Rate(self.jid)

    def getAngle2Rate(self):
        return dJointGetUniversalAngle2Rate(self.jid)

    def setParam(self, int param, dReal value):
        dJointSetUniversalParam(self.jid, param, value)

    def getParam(self, int param):
        return dJointGetUniversalParam(self.jid, param)


cdef class Hinge2Joint(Joint):
    """Hinge2 joint.

    Constructor::Hinge2Joint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateHinge2(world.wid, jgid)

    def __init__(self, World world, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    def setAnchor(self, pos):
        """setAnchor(pos)

        Set the hinge-2 anchor.

        @param pos: Anchor position
        @type pos: 3-sequence of floats
        """
        dJointSetHinge2Anchor(self.jid, pos[0], pos[1], pos[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor(self):
        """getAnchor() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates. This returns
        the point on body 1. If the joint is perfectly satisfied, this
        will be the same as the point on body 2.
        """

        cdef dVector3 p
        dJointGetHinge2Anchor(self.jid, p)
        return p[0], p[1], p[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAnchor2(self):
        """getAnchor2() -> 3-tuple of floats

        Get the joint anchor point, in world coordinates. This returns
        the point on body 2. If the joint is perfectly satisfied, this
        will be the same as the point on body 1.
        """

        cdef dVector3 p
        dJointGetHinge2Anchor2(self.jid, p)
        return p[0], p[1], p[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAxis1(self, axis):
        """setAxis1(axis)

        Set the first hinge-2 axis. Axis 1 and axis 2 must not lie
        along the same line.

        @param axis: Joint axis
        @type axis: 3-sequence of floats
        """

        dJointSetHinge2Axis1(self.jid, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis1(self):
        """getAxis1() -> 3-tuple of floats

        Get the first hinge-2 axis.
        """
        cdef dVector3 a
        dJointGetHinge2Axis1(self.jid, a)
        return a[0], a[1], a[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAxis2(self, axis):
        """setAxis2(axis)

        Set the second hinge-2 axis. Axis 1 and axis 2 must not lie
        along the same line.

        @param axis: Joint axis
        @type axis: 3-sequence of floats
        """
        dJointSetHinge2Axis2(self.jid, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis2(self):
        """getAxis2() -> 3-tuple of floats

        Get the second hinge-2 axis.
        """
        cdef dVector3 a
        dJointGetHinge2Axis2(self.jid, a)
        return a[0], a[1], a[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngle1(self) -> dReal:
        """getAngle1() -> float

        Get the first hinge-2 angle (around axis 1).

        When the anchor or axis is set, the current position of the
        attached bodies is examined and that position will be the zero
        angle.
        """
        return dJointGetHinge2Angle1(self.jid)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngle1Rate(self) -> dReal:
        """getAngle1Rate() -> float

        Get the time derivative of the first hinge-2 angle.
        """
        return dJointGetHinge2Angle1Rate(self.jid)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngle2Rate(self) -> dReal:
        """getAngle2Rate() -> float

        Get the time derivative of the second hinge-2 angle.
        """
        return dJointGetHinge2Angle2Rate(self.jid)

    def addTorques(self, torque1, torque2):
        """addTorques(torque1, torque2)

        Applies torque1 about axis 1, and torque2 about axis 2.

        @param torque1: Torque 1 magnitude
        @param torque2: Torque 2 magnitude
        @type torque1: float
        @type torque2: float
        """
        dJointAddHinge2Torques(self.jid, torque1, torque2)

    def setParam(self, int param, dReal value):
        dJointSetHinge2Param(self.jid, param, value)

    def getParam(self, int param):
        return dJointGetHinge2Param(self.jid, param)


cdef class FixedJoint(Joint):
    """Fixed joint.

    Constructor::FixedJoint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateFixed(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    def setFixed(self):
        """setFixed()

        Call this on the fixed joint after it has been attached to
        remember the current desired relative offset and desired
        relative rotation between the bodies.
        """
        dJointSetFixed(self.jid)


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


cdef class ContactJoint(ContactJointBase):
    """Contact joint.

    Constructor::ContactJoint(world, jointgroup, contact)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup, Contact contact):
        cdef JointGroup jg
        cdef dJointGroupID jgid
        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid

        self._contact = contact
        self.jid = dJointCreateContact(world.wid, jgid, &contact._contact)

    def __init__(self, World world not None, JointGroup jointgroup, Contact contact):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)


# Add by Zhenhua Song
cdef class ContactJointMaxForce(ContactJointBase):
    """
    A simplified contact model.

    simple, the formula is simplified as follow:
    0 <= support force <= +infty
    friction 0 <= contact mu (or max friction)
    friction 1 <= contact mu (or max friction)
    """
    def __cinit__(self, World world not None, JointGroup jointgroup, Contact contact):
        cdef JointGroup jg
        cdef dJointGroupID jgid
        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid

        self._contact = contact
        self.jid = dJointCreateContactMaxForce(world.wid, jgid, &contact._contact)

    def __init__(self, World world not None, JointGroup jointgroup, Contact contact):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)


cdef class AMotor(Joint):
    """AMotor joint.

    Constructor::AMotor(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateAMotor(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        raise NotImplementedError("Please use class BallJointAmotor instead of class AMotor")
        #
        # self._world = world
        # if jointgroup != None:
        #     jointgroup._addjoint(self)
        #
        # self._setData(self)

    def setMode(self, int mode):
        """setMode(mode)

        Set the angular motor mode.  mode must be either AMotorUser or
        AMotorEuler.

        @param mode: Angular motor mode
        @type mode: int
        """
        dJointSetAMotorMode(self.jid, mode)

    def getMode(self):
        """getMode()

        Return the angular motor mode (AMotorUser or AMotorEuler).
        """
        return dJointGetAMotorMode(self.jid)

    def setNumAxes(self, int num):
        """setNumAxes(num)

        Set the number of angular axes that will be controlled by the AMotor.
        num may be in the range from 0 to 3.

        @param num: Number of axes (0-3)
        @type num: int
        """
        dJointSetAMotorNumAxes(self.jid, num)

    def getNumAxes(self):
        """getNumAxes() -> int

        Get the number of angular axes that are controlled by the AMotor.
        """
        return dJointGetAMotorNumAxes(self.jid)

    def setAxis(self, int anum, int rel, axis):
        """setAxis(anum, rel, axis)

        Set an AMotor axis.

        The anum argument selects the axis to change (0,1 or 2).
        Each axis can have one of three "relative orientation" modes,
        selected by rel:

        0: The axis is anchored to the global frame.
        1: The axis is anchored to the first body.
        2: The axis is anchored to the second body.

        The axis vector is always specified in global coordinates
        regardless of the setting of rel.

        @param anum: Axis number
        @param rel: Relative orientation mode
        @param axis: Axis
        @type anum: int
        @type rel: int
        @type axis: 3-sequence of floats
        """
        dJointSetAMotorAxis(self.jid, anum, rel, axis[0], axis[1], axis[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAxisNumpy(self, int anum, int rel, np.ndarray[np.float64_t, ndim = 1] np_axis):
        cdef dReal * axis = <dReal *>np_axis.data
        dJointSetAMotorAxis(self.jid, anum, rel, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis(self, int anum):
        """getAxis(anum)

        Get an AMotor axis.

        @param anum: Axis index (0-2)
        @type anum: int
        """
        cdef dVector3 a
        dJointGetAMotorAxis(self.jid, anum, a)
        return a[0], a[1], a[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxisNumpy(self, int anum):
        cdef dVector3 a
        dJointGetAMotorAxis(self.jid, anum, a)

        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(3)
        cdef dReal * res = <dReal *>np_res.data

        res[0] = a[0]
        res[1] = a[1]
        res[2] = a[2]
        return np_res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxisRel(self, int anum):
        """getAxisRel(anum) -> int

        Get the relative mode of an axis.

        @param anum: Axis index (0-2)
        @type anum: int
        """
        return dJointGetAMotorAxisRel(self.jid, anum)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setAngle(self, int anum, dReal angle):
        """setAngle(anum, angle)

        Tell the AMotor what the current angle is along axis anum.

        @param anum: Axis index
        @param angle: Angle
        @type anum: int
        @type angle: float
        """
        dJointSetAMotorAngle(self.jid, anum, angle)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngle(self, int anum):
        """getAngle(anum) -> float

        Return the current angle for axis anum.

        @param anum: Axis index
        @type anum: int
        """
        return dJointGetAMotorAngle(self.jid, anum)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAngleRate(self, int anum):
        """getAngleRate(anum) -> float

        Return the current angle rate for axis anum.

        @param anum: Axis index
        @type anum: int
        """
        return dJointGetAMotorAngleRate(self.jid, anum)

    def addTorques(self, dReal torque0, dReal torque1, dReal torque2):
        """addTorques(torque0, torque1, torque2)

        Applies torques about the AMotor's axes.

        @param torque0: Torque 0 magnitude
        @param torque1: Torque 1 magnitude
        @param torque2: Torque 2 magnitude
        @type torque0: float
        @type torque1: float
        @type torque2: float
        """
        dJointAddAMotorTorques(self.jid, torque0, torque1, torque2)

    def setParam(self, int param, dReal value):
        dJointSetAMotorParam(self.jid, param, value)

    def getParam(self, int param):
        return dJointGetAMotorParam(self.jid, param)


cdef class LMotor(Joint):
    """LMotor joint.

    Constructor::LMotor(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreateLMotor(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    def setNumAxes(self, int num):
        """setNumAxes(num)

        Set the number of angular axes that will be controlled by the LMotor.
        num may be in the range from 0 to 3.

        @param num: Number of axes (0-3)
        @type num: int
        """
        dJointSetLMotorNumAxes(self.jid, num)

    def getNumAxes(self):
        """getNumAxes() -> int

        Get the number of angular axes that are controlled by the LMotor.
        """
        return dJointGetLMotorNumAxes(self.jid)

    def setAxis(self, int anum, int rel, axis):
        """setAxis(anum, rel, axis)

        Set an LMotor axis.

        The anum argument selects the axis to change (0,1 or 2).
        Each axis can have one of three "relative orientation" modes,
        selected by rel:

        0: The axis is anchored to the global frame.
        1: The axis is anchored to the first body.
        2: The axis is anchored to the second body.

        @param anum: Axis number
        @param rel: Relative orientation mode
        @param axis: Axis
        @type anum: int
        @type rel: int
        @type axis: 3-sequence of floats
        """
        dJointSetLMotorAxis(self.jid, anum, rel, axis[0], axis[1], axis[2])

    # Add by Zhenhua Song
    def setAxisNumpy(self, int anum, int rel, np.ndarray[np.float64_t, ndim = 1] np_axis):
        cdef dReal * axis = <dReal *> np_axis.data
        dJointSetLMotorAxis(self.jid, anum, rel, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis(self, int anum):
        """getAxis(anum)

        Get an LMotor axis.

        @param anum: Axis index (0-2)
        @type anum: int
        """
        cdef dVector3 a
        dJointGetLMotorAxis(self.jid, anum, a)
        return a[0], a[1], a[2]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxisNumpy(self, int anum):
        cdef dVector3 a
        dJointGetLMotorAxis(self.jid, anum, a)

        cdef np.ndarray[np.float64_t, ndim=1] np_res = np.zeros(3)
        cdef dReal * res = <dReal *> np_res.data
        res[0] = a[0]
        res[1] = a[1]
        res[2] = a[2]

        return np_res

    def setParam(self, int param, dReal value):
        dJointSetLMotorParam(self.jid, param, value)

    def getParam(self, int param):
        return dJointGetLMotorParam(self.jid, param)


cdef class Plane2DJoint(Joint):
    """Plane-2D Joint.

    Constructor::Plane2DJoint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreatePlane2D(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    def setXParam(self, int param, dReal value):
        dJointSetPlane2DXParam(self.jid, param, value)

    def setYParam(self, int param, dReal value):
        dJointSetPlane2DYParam(self.jid, param, value)

    def setAngleParam(self, int param, dReal value):
        dJointSetPlane2DAngleParam(self.jid, param, value)


cdef class PRJoint(Joint):
    """Prismatic and Rotoide Joint.

    Constructor::PRJoint(world, jointgroup=None)
    """

    def __cinit__(self, World world not None, JointGroup jointgroup=None):
        cdef JointGroup jg
        cdef dJointGroupID jgid

        jgid = NULL
        if jointgroup != None:
            jg = jointgroup
            jgid = jg.gid
        self.jid = dJointCreatePR(world.wid, jgid)

    def __init__(self, World world not None, JointGroup jointgroup=None):
        self._world = world
        if jointgroup != None:
            jointgroup._addjoint(self)

        self._setData(self)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getPosition(self):
        """getPosition()

        Get a PRJoint's linear extension.  (i.e. the prismatic's extension)
        """
        return dJointGetPRPosition(self.jid)

    def setAnchor(self, pos):
        """setAnchor(pos)

        Set a PRJoint anchor.

        @param pos: Anchor position
        @type pos: 3-sequence of floats
        """
        dJointSetPRAnchor(self.jid, pos[0], pos[1], pos[2])

    def getAnchor(self):
        """getAnchor()

        Get a PRJoint anchor.
        """
        cdef dVector3 a
        dJointGetPRAnchor(self.jid, a)
        return a[0], a[1], a[2]

    def setAxis1(self, axis):
        """setAxis1(axis)

        Set a PRJoint's prismatic axis.

        @param axis: Axis
        @type axis: 3-sequence of floats
        """
        dJointSetPRAxis1(self.jid, axis[0], axis[1], axis[2])

    def getAxis1(self):
        """getAxis1()

        Get a PRJoint's prismatic axis.
        """
        cdef dVector3 a
        dJointGetPRAxis1(self.jid, a)
        return a[0], a[1], a[2]

    def setAxis2(self, axis):
        """setAxis2(axis)

        Set a PRJoint's rotoide axis.

        @param axis: Axis
        @type axis: 3-sequence of floats
        """
        dJointSetPRAxis2(self.jid, axis[0], axis[1], axis[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAxis2(self):
        """getAxis2()

        Get a PRJoint's rotoide axis.
        """
        cdef dVector3 a
        dJointGetPRAxis2(self.jid, a)
        return a[0], a[1], a[2]


# Add by Zhenhua Song
cdef class _GeomAttrs:
    cdef str name
    cdef dReal friction
    cdef dReal bounce

    cdef dReal max_friction

    cdef int clung_env
    cdef list ignore_geom_id
    cdef int collidable
    cdef object character

    cdef dGeomID ignore_geom_buf[64]
    cdef size_t ignore_geom_buf_len

    cdef int instance_id

    cdef int character_self_collide  # collision detection with same character

    def __cinit__(self):
        self.name = ""
        self.friction = 0.8
        self.bounce = 0

        self.max_friction = dInfinity  # default value is +infty

        self.clung_env = 0
        self.ignore_geom_id = list()
        self.collidable = 1
        self.character = None

        memset(self.ignore_geom_buf, 0, sizeof(dGeomID) * 64)
        self.ignore_geom_buf_len = 0

        self.instance_id = 0

        self.character_self_collide = 1

# Geom base class
cdef class GeomObject:
    """This is the abstract base class for all geom objects."""

    # The id of the geom object as returned by dCreateXxxx()
    cdef dGeomID gid
    # The space in which the geom was placed (or None). This reference
    # is kept so that the space won't be destroyed while there are still
    # geoms around that might use it.
    cdef SpaceBase _space

    # The body that the geom was attached to (or None).
    # cdef Body body  # Modify by Zhenhua Song
    cdef object _body

    # Add by Zhenhua Song
    cdef _GeomAttrs geom_attrs

    cdef object __weakref__

    def __cinit__(self, *a, **kw):
        self.gid = NULL
        self._space = None
        self._body = None
        self.geom_attrs = _GeomAttrs()

    def __init__(self, *a, **kw):
        raise NotImplementedError("GeomObject base class can't be used directly")

    def __hash__(self):
        return <size_t>self.gid

    # Modify by Zhenhua Song
    def __dealloc__(self):
        self.destroy_immediate()

    def copy_geom(self, Body body, SpaceBase space):
        raise NotImplementedError

    cpdef copy_pos_quat(self, GeomObject result):
        cdef const dReal* pos = dGeomGetPosition(self.gid)
        dGeomSetPosition(result.gid, pos[0], pos[1], pos[2])

        cdef dQuaternion q
        dGeomGetQuaternion(self.gid, q)
        dGeomSetQuaternion(result.gid, q)

        return result

    # Add by Yulong Zhang
    def set_draw_local_axis(self, x):
        dGeomSetDrawAxisFlag(self.gid, x)

    # Add by Zhenhua Song
    def destroy_immediate(self):
        if self.gid != NULL:
            # print("Destroy Geometry ", self.name)
            dGeomDestroy(self.gid)
            self.gid = NULL

    # Add by Zhenhua Song
    def __eq__(self, GeomObject other):
        return self.gid == other.gid

    # Add by Zhenhua Song
    def extend_ignore_geom_id(self, list res):
        self.geom_attrs.ignore_geom_id.extend(res)
        for i in res:
            self.geom_attrs.ignore_geom_buf[self.geom_attrs.ignore_geom_buf_len] = <dGeomID>(<size_t>i)
            self.geom_attrs.ignore_geom_buf_len += 1

    # Add by Zhenhua Song
    @property
    def bounce(self) -> dReal:
        return self.geom_attrs.bounce

    # Add by Zhenhua Song
    @bounce.setter
    def bounce(self, dReal value):
        self.geom_attrs.bounce = value

    # Add by Zhenhua Song
    @property
    def max_friction(self) -> dReal:
        return self.geom_attrs.max_friction

    # Add by Zhenhua Song
    @max_friction.setter
    def max_friction(self, dReal value):
        self.geom_attrs.max_friction = value

    # Add by Zhenhua Song
    @property
    def character_self_collide(self) -> int:
        return self.geom_attrs.character_self_collide

    # Add by Zhenhua Song
    @character_self_collide.setter
    def character_self_collide(self, int value):
        self.geom_attrs.character_self_collide = value

    # Add by Zhenhua Song
    @property
    def geom_index(self) -> int:
        return dGeomGetIndex(self.gid)

    # Add by Zhenhua Song
    @geom_index.setter
    def geom_index(self, int value):
        dGeomSetIndex(self.gid, value)

    # Add by Zhenhua Song
    @property
    def instance_id(self) -> int:
        return self.geom_attrs.instance_id

    @instance_id.setter
    def instance_id(self, int value):
        self.geom_attrs.instance_id = value

    # Add by Zhenhua Song
    @property
    def is_environment(self):
        return dGeomGetBody(self.gid) == NULL
        # return self._body is None or self._body() is None

    # Add by Zhenhua Song
    @property
    def ignore_geom_id(self) -> list:
        return self.geom_attrs.ignore_geom_id

    # Add by Zhenhua Song
    @property
    def ignore_geoms(self) -> list:
        cdef res = list()
        for i in self.geom_attrs.ignore_geom_id:
            res.append(<GeomObject>dGeomGetData(<dGeomID>i))
        return res

    # Add by Zhenhua Song
    @property
    def character_id(self) -> int:
        return dGeomGetCharacterID(self.gid)

    # Add by Zhenhua Song
    @character_id.setter
    def character_id(self, int value):
        dGeomSetCharacterID(self.gid, value)

    # Add by Zhenhua Song
    @property
    def clung_env(self):
        return self.geom_attrs.clung_env

    # Add by Zhenhua Song
    @clung_env.setter
    def clung_env(self, value):
        self.geom_attrs.clung_env = value

    # Add by Zhenhua Song
    @property
    def name(self) -> str:
        return self.geom_attrs.name

    # Add by Zhenhua Song
    @name.setter
    def name(self, str value):
        self.geom_attrs.name = value

    # Add by Zhenhua Song
    @property
    def friction(self) -> dReal:
        return self.geom_attrs.friction

    # Add by Zhenhua Song
    @friction.setter
    def friction(self, dReal value):
        self.geom_attrs.friction = value

    # Add by Zhenhua Song
    @property
    def collidable(self):
        return self.geom_attrs.collidable

    # Add by Zhenhua Song
    @collidable.setter
    def collidable(self, object value):
        self.geom_attrs.collidable = value

    # Add by Zhenhua Song
    @property
    def character(self):
        return self.geom_attrs.character

    # Add by Zhenhua Song
    @character.setter
    def character(self, object value):
        self.geom_attrs.character = weakref.proxy(value)

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def append_ignore_geom(self, GeomObject other):
        if self.geom_attrs.ignore_geom_buf_len >= 64:
            raise ValueError("only support 64 ignore geoms")

        self.geom_attrs.ignore_geom_buf[self.geom_attrs.ignore_geom_buf_len] = other.gid
        self.geom_attrs.ignore_geom_buf_len += 1

    # Add by Zhenhua Song
    def get_gid(self):
        return <size_t> self.gid

    # Add by Zhenhua Song
    @property
    def space(self) -> SpaceBase:
        return self._space

    # Add by Zhenhua Song
    @space.setter
    def space(self, SpaceBase space):
        if self._space is not None:
            self._space.remove(self)
        if space is not None:
            space.add(self)
        self._space = space

    # Add by Zhenhua Song
    @property
    def body(self):
        """getBody() -> Body

        Get the body associated with this geom.
        """
        cdef dBodyID c_body = dGeomGetBody(self.gid)
        if c_body == NULL:
            return None
        else:
            return self._body()

    # Modify by Zhenhua Song
    @body.setter
    def body(self, Body body):
        """setBody(body)

        Set the body associated with a placeable geom.

        @param body: The Body object or None.
        @type body: Body
        """

        if self._body is not None:
            self._body._geoms.remove(self)

        if body == None:
            dGeomSetBody(self.gid, NULL)
            self._body = None
        else:
            dGeomSetBody(self.gid, body.bid)
            body._geoms.append(self)
            self._body = weakref.ref(body)

    # Add By Zhenhua Song
    @property
    def PositionNumpy(self) -> np.ndarray:
        cdef const dReal* p = dGeomGetPosition(self.gid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]
        return np_buff

    # Add by Zhenhua Song
    @PositionNumpy.setter
    def PositionNumpy(self, np.ndarray[np.float64_t, ndim = 1] p):
        """setPosition(pos)

        Set the position of the geom. If the geom is attached to a body,
        the body's position will also be changed.

        @param pos: Position
        @type pos: 3-sequence of floats
        """

        cdef const dReal * pos = <const dReal *> (p.data)
        dGeomSetPosition(self.gid, pos[0], pos[1], pos[2])

    # Add by Zhenhua Song
    @property
    def RotationNumpy(self) -> np.ndarray:
        """getRotation() -> 9-tuple

        Get the current orientation of the geom. If the geom is attached to
        a body the returned value is the body's orientation.
        """

        cdef const dReal* m = dGeomGetRotation(self.gid)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(9)
        ODEMat3ToDenseMat3(m, <dReal *> (np_buff.data), 0)

        return np_buff

    # Add by Zhenhua Song
    @RotationNumpy.setter
    def RotationNumpy(self, np.ndarray[np.float64_t, ndim = 1] Rot):
        """setRotation(R)

        Set the orientation of the geom. If the geom is attached to a body,
        the body's orientation will also be changed.

        @param R: Rotation matrix
        @type R: 9-sequence of floats
        """
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.ascontiguousarray(Rot)
        cdef dMatrix3 m
        DenseMat3ToODEMat3(m, <const dReal *> np_buff.data, 0)
        dGeomSetRotation(self.gid, m)

    # Add by Zhenhua Song
    @property
    def QuaternionScipy(self):
        """getQuaternion() -> (x,y,z, w)

        Get the current orientation of the geom. If the geom is attached to
        a body the returned value is the body's orientation.
        """

        cdef dQuaternion q
        dGeomGetQuaternion(self.gid, q)
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(4)
        # scipy quat: (x, y, z, w)
        # ode quat: (w, x, y, z)
        cdef dReal * res = <dReal *> np_buff.data
        res[0] = q[1]
        res[1] = q[2]
        res[2] = q[3]
        res[3] = q[0]

        return np_buff

    # Add by Zhenhua Song
    @QuaternionScipy.setter
    def QuaternionScipy(self, np.ndarray[np.float64_t, ndim = 1] quat):
        """setQuaternionScipy(q)

        Set the orientation of the geom. If the geom is attached to a body,
        the body's orientation will also be changed.

        @param q: Quaternion (x,y,z,w)
        @type q: 4-sequence of floats
        """

        cdef const dReal * q = <const dReal *> quat.data
        # in scipy: (x, y, z, w)
        # in ode: (w, x, y, z)
        cdef dQuaternion cq
        cq[0] = q[3]
        cq[1] = q[0]
        cq[2] = q[1]
        cq[3] = q[2]
        dGeomSetQuaternion(self.gid, cq)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setOffsetPosition(self, pos):
        """setOffsetPosition(pos)

        Set the offset position of the geom. The geom must be attached to a
        body.  If the geom did not have an offset, it is automatically created.
        This sets up an additional (local) transformation for the geom, since
        geoms attached to a body share their global position and rotation.

        @param pos: Position
        @type pos: 3-sequence of floats
        """
        if self._body == None:
            raise ValueError("Cannot set an offset position on a geom before calling setBody")
        dGeomSetOffsetPosition(self.gid, pos[0], pos[1], pos[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setOffsetWorldPositionNumpy(self, np.ndarray[np.float64_t, ndim = 1] pos):
        cdef const dReal * res = <const dReal *> pos.data
        dGeomSetOffsetWorldPosition(self.gid, res[0], res[1], res[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getOffsetPosition(self):
        """getOffsetPosition() -> 3-tuple

        Get the offset position of the geom.
        """
        cdef dReal* p
        p = <dReal*>dGeomGetOffsetPosition(self.gid)
        return (p[0],p[1],p[2])

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getOffsetPositionNumpy(self):
        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(3)
        cdef dReal * res = <dReal *> np_buff.data
        cdef const dReal * p = dGeomGetOffsetPosition(self.gid)
        res[0] = p[0]
        res[1] = p[1]
        res[2] = p[2]

        return np_buff

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setOffsetWorldRotationNumpy(self, np.ndarray[np.float64_t, ndim = 1] rot):
        if self._body is None or self._body() is None:
            raise ValueError("Cannot set an offset rotation on a geom before calling setBody")

        cdef dMatrix3 m
        cdef const dReal * R = <const dReal *> rot.data
        DenseMat3ToODEMat3(m, R, 0)
        dGeomSetOffsetWorldRotation(self.gid, m)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setOffsetRotationNumpy(self, np.ndarray[np.float64_t, ndim = 1] rot):
        """setOffsetRotationNumpy(R)

        Set the offset rotation of the geom. The geom must be attached to a
        body.  If the geom did not have an offset, it is automatically created.
        This sets up an additional (local) transformation for the geom, since
        geoms attached to a body share their global position and rotation.

        @param R: Rotation matrix
        @type R: 9-sequence of floats
        """
        if self._body is None or self._body() is None:
            raise ValueError("Cannot set an offset rotation on a geom before calling setBody")

        cdef dMatrix3 m
        DenseMat3ToODEMat3(m, <const dReal *> rot.data, 0)
        dGeomSetOffsetRotation(self.gid, m)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getOffsetRotation(self):
        """getOffsetRotation() -> 9-tuple

        Get the offset rotation of the geom.
        """
        cdef const dReal* m = <const dReal*>dGeomGetOffsetRotation(self.gid)
        return [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getOffsetRotationNumpy(self) -> np.ndarray:
        cdef const dReal* m = dGeomGetOffsetRotation(self.gid)
        cdef np.ndarray[np.float64_t, ndim=1] np_buff = np.zeros(9)
        ODEMat3ToDenseMat3(m, <dReal *> (np_buff.data), 0)

        return np_buff

    def clearOffset(self):
        """clearOffset()

        Disable the offset transform of the geom.
        """
        dGeomClearOffset(self.gid)

    # Add by Zhenhua Song
    @property
    def AABBNumpy(self) -> np.ndarray:
        """getAABB() -> np.ndarray

        Return an axis aligned bounding box that surrounds the geom.
        The return value is a 6-tuple (minx, maxx, miny, maxy, minz, maxz).
        """

        cdef np.ndarray[np.float64_t, ndim = 1] np_buff = np.zeros(6)
        dGeomGetAABB(self.gid, <dReal *> np_buff.data)

        return np_buff

    @property
    def isSpace(self) -> bool:
        """isSpace() -> bool

        Return 1 if the given geom is a space, or 0 if not."""
        return bool(dGeomIsSpace(self.gid))

    @property
    def CollideBits(self) -> long:
        """getCollideBits() -> long

        Return the "collide" bitfields for this geom.
        """
        return dGeomGetCollideBits(self.gid)

    @property
    def CategoryBits(self) -> long:
        """getCategoryBits() -> long

        Return the "category" bitfields for this geom.
        """
        return dGeomGetCategoryBits(self.gid)

    @CollideBits.setter
    def CollideBits(self, long bits) -> int:
        """setCollideBits(bits)

        Set the "collide" bitfields for this geom.

        @param bits: Collide bit field
        @type bits: int/long
        """
        dGeomSetCollideBits(self.gid, long(bits))

    @CategoryBits.setter
    def CategoryBits(self, long bits):
        """setCategoryBits(bits)

        Set the "category" bitfields for this geom.

        @param bits: Category bit field
        @type bits: int/long
        """
        dGeomSetCategoryBits(self.gid, long(bits))

    def enable(self):
        """enable()

        Enable the geom."""
        dGeomEnable(self.gid)

    def disable(self):
        """disable()

        Disable the geom."""
        dGeomDisable(self.gid)

    @property
    def isEnabled(self) -> bool:
        """isEnabled() -> bool

        Return True if the geom is enabled."""
        return bool(dGeomIsEnabled(self.gid))

    # Add by Zhenhua Song
    def getClass(self) -> int:
        return dGeomGetClass(self.gid)

    # Add by Zhenhua Song
    def _setData(self, value):
        cdef void * res
        res = <void *> value
        dGeomSetData(self.gid, res)

    # Add by Zhenhua Song
    def _getData(self):
        return <object> dGeomGetData(self.gid)

    # Add by Zhenhua Song, for rendering with different color in Long Ge's Framework
    @property
    def render_user_color(self):
        cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(3)
        dGeomRenderGetUserColor(self.gid, <dReal *> result.data)
        return result

    # Add by Zhenhua Song, for visualize in Long Ge's draw stuff framework
    @render_user_color.setter
    def render_user_color(self, np.ndarray[np.float64_t, ndim=1] color_):
        cdef np.ndarray[np.float64_t, ndim=1] color = np.ascontiguousarray(color_)
        dGeomRenderInUserColor(self.gid, <dReal *> color.data)

    # Add by Zhenhua Song, for visualize in Long Ge's draw stuff framework
    @property
    def render_by_default_color(self):
        return dGeomIsRenderInDefaultColor(self.gid)

    # Add by Zhenhua Song, for visualize in Long Ge's draw stuff framework
    @render_by_default_color.setter
    def render_by_default_color(self, int value):
        dGeomRenderInDefaultColor(self.gid, value)


# Add by Zhenhua Song. Test OK
cdef class _SpaceIterator2:
    cdef dGeomID g
    cdef int num_geom
    cdef dSpaceID sid

    def __cinit__(self, size_t sid):
        self.sid = <dSpaceID> sid
        self.num_geom = dSpaceGetNumGeoms(self.sid)
        self.g = dSpaceGetFirstGeom(self.sid)

    def __iter__(self):
        return self

    def __next__(self):
        if self.g == NULL:
            raise StopIteration
        else:
            res = <GeomObject> dGeomGetData(<dGeomID> self.g)
            self.g = dSpaceGetNextGeom(self.g)
            return res


# SpaceBase
cdef class SpaceBase(GeomObject):
    """Space class (container for geometry objects).

    A Space object is a container for geometry objects which are used
    to do collision detection.
    The space does high level collision culling, which means that it
    can identify which pairs of geometry objects are potentially
    touching.

    This Space class can be used for both, a SimpleSpace and a HashSpace
    (see ODE documentation).

    >>> space = Space(type=0)   # Create a SimpleSpace
    >>> space = Space(type=1)   # Create a HashSpace
    """

    # The id of the space. Actually this is a copy of the value in self.gid
    # (as the Space is derived from GeomObject) which can be used without
    # casting whenever a *space* id is required.
    cdef dSpaceID sid

    def __cinit__(self, *a, **kw):
        self.sid = NULL

    def __init__(self, *a, **kw):
        raise NotImplementedError("The SpaceBase class can't be used directly")

    # Modify by Zhenhua Song
    def __dealloc__(self):
        self.destroy_immediate()

    # Add by Zhenhua Song
    def __eq__(self, SpaceBase other):
        return self.sid == other.sid

    # Add by Zhenhua Song
    def __len__(self):
        return dSpaceGetNumGeoms(self.sid)

    # Add by Zhenhua Song
    def destroy_immediate(self):
        if self.gid != NULL:
            dSpaceDestroy(self.sid)
            self.sid = NULL
            self.gid = NULL

    cdef dSpaceID get_sid(self):
        return self.sid

    cdef size_t _id(self):
        return <size_t>self.sid

    # Add by Zhenhua Song
    def __iter__(self):
        return _SpaceIterator2(<size_t> self.gid)

    # Add by Zhenhua Song
    def _setData(self, value):
        cdef void * res = <void*> value
        dSpaceSetData(self.sid, res)

    # Add by Zhenhua Song
    def _getData(self):
        cdef void * res = dSpaceGetData(self.sid)
        return <object> res

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getGeomIDs(self) -> np.ndarray:
        cdef int geom_num = dSpaceGetNumGeoms(self.sid)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_ids = np.zeros(geom_num, np.uint64)
        # on x64 System
        cdef size_t * ids_res = <size_t *> np_ids.data
        cdef dGeomID g = dSpaceGetFirstGeom(self.sid)
        cdef int idx = 0

        while g != NULL:
            ids_res[idx] = <size_t> g
            g = dSpaceGetNextGeom(g)
            idx += 1

        return np_ids

    # Add by Zhenhua Song
    def getPlaceableCount(self):
        return dSpaceGetPlaceableCount(self.sid)

    # Add by Zhenhua Song. resort geometries in space.
    def ResortGeoms(self):
        dSpaceResortGeoms(self.sid)

    # Add by Zhenhua Song
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAllGeomName(self):
        cdef list name_list = []
        cdef GeomObject geom_py
        cdef dGeomID g = dSpaceGetFirstGeom(self.sid)
        while g != NULL:
            geom_py = <GeomObject> dGeomGetData(<dGeomID>g)
            name_list.append(geom_py.name)
            g = dSpaceGetNextGeom(g)
        return name_list

    # Add by Zhenhua Song
    # return geom id, type(ode), pos, quat(scipy), create_scale
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getAllGeomInfos(self, with_scale_flag=False, with_name_flag=False):
        #  how to get position and rotation of plane..
        #  We can get parameter of plane, or (a, b, c, d)
        #  Position of Plane: Nearest Point to Origin on Plane
        #  Quaternion of Plane: quaternion between (0, 0, 1) and (a, b, c)

        cdef int geom_num = dSpaceGetNumGeoms(self.sid)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_ids = np.zeros(geom_num, np.uint64)
        cdef np.ndarray[np.int32_t, ndim = 1] np_types = np.zeros(geom_num, np.int32)
        cdef np.ndarray[np.float64_t, ndim = 1] np_pos = np.zeros(geom_num * 3)
        cdef np.ndarray[np.float64_t, ndim = 1] np_q_scipy = np.zeros(geom_num * 4)

        cdef int with_scale = with_scale_flag
        cdef np.ndarray[np.float64_t, ndim = 1] np_scale = None
        if with_scale:
            np_scale = np.zeros(geom_num * 3)

        cdef int with_name = with_name_flag
        cdef list name_list = []

        # on x64 System
        cdef size_t * ids_res = <size_t *> np_ids.data
        cdef int * types_res = <int *> np_types.data
        cdef dReal * pos_res = <dReal *> np_pos.data
        cdef dReal * q_res = <dReal * > np_q_scipy.data

        cdef dReal * scale_res
        if with_scale:
            scale_res = <dReal *> np_scale.data

        cdef dGeomID g = dSpaceGetFirstGeom(self.sid)
        cdef int idx = 0

        cdef const dReal * pos
        cdef dVector3 pos_plane
        cdef dQuaternion q
        cdef int geom_type

        cdef dReal radius, length
        cdef dVector3 box_len

        cdef GeomObject geom_py

        while g != NULL:
            geom_type = dGeomGetClass(g)
            if dGeomIsPlaceable(g):
                pos = dGeomGetPosition(g)
                pos_res[idx * 3 + 0] = pos[0]
                pos_res[idx * 3 + 1] = pos[1]
                pos_res[idx * 3 + 2] = pos[2]

                dGeomGetQuaternion(g, q)  # w, x, y, z
                q_res[idx * 4 + 0] = q[1]
                q_res[idx * 4 + 1] = q[2]
                q_res[idx * 4 + 2] = q[3]
                q_res[idx * 4 + 3] = q[0]

            elif geom_type == dPlaneClass:
                dGeomPlaneGetNearestPointToOrigin(g, pos_plane)
                pos_res[idx * 3 + 0] = pos_plane[0]
                pos_res[idx * 3 + 1] = pos_plane[1]
                pos_res[idx * 3 + 2] = pos_plane[2]

                dGeomPlaneGetQuatFromZAxis(g, q)  # w, x, y, z
                q_res[idx * 4 + 0] = q[1]
                q_res[idx * 4 + 1] = q[2]
                q_res[idx * 4 + 2] = q[3]
                q_res[idx * 4 + 3] = q[0]
            else:
                raise ValueError("Geom Type not support.")

            ids_res[idx] = <size_t> g
            types_res[idx] = geom_type

            if with_scale:
                if geom_type == dSphereClass:
                    radius = dGeomSphereGetRadius(g)
                    scale_res[3 * idx + 0] = radius
                    scale_res[3 * idx + 1] = radius
                    scale_res[3 * idx + 2] = radius
                elif geom_type == dBoxClass:
                    dGeomBoxGetLengths(g, box_len)
                    scale_res[3 * idx + 0] = box_len[0]
                    scale_res[3 * idx + 1] = box_len[1]
                    scale_res[3 * idx + 2] = box_len[2]
                elif geom_type == dCapsuleClass:
                    dGeomCapsuleGetParams(g, &radius, &length)
                    # Render code in unity should be modified
                    scale_res[3 * idx + 0] = radius
                    scale_res[3 * idx + 1] = length
                    scale_res[3 * idx + 2] = radius
                elif geom_type == dCylinderClass:
                    dGeomCylinderGetParams(g, &radius, &length)
                    scale_res[3 * idx + 0] = radius
                    scale_res[3 * idx + 1] = length
                    scale_res[3 * idx + 2] = radius
                elif geom_type == dPlaneClass:
                    # Assume Normal Vector is along z axis
                    scale_res[3 * idx + 0] = 20
                    scale_res[3 * idx + 1] = 20
                    scale_res[3 * idx + 2] = 0.01
                else:
                    raise NotImplementedError

            if with_name:
                geom_py = <GeomObject> dGeomGetData(<dGeomID>g)
                name_list.append(geom_py.name)

            g = dSpaceGetNextGeom(g)
            idx += 1

        return np_ids, np_types, np_pos, np_q_scipy, np_scale, name_list

    # Add by Zhenhua Song
    # return geom id, type(ode), pos, quat(scipy)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getPlaceableGeomInfos(self):
        cdef int geom_num = dSpaceGetPlaceableCount(self.sid)
        cdef np.ndarray[np.uint64_t, ndim = 1] np_ids = np.zeros(geom_num, np.uint64)
        cdef np.ndarray[np.int32_t, ndim = 1] np_types = np.zeros(geom_num, np.int32)
        cdef np.ndarray[np.float64_t, ndim=1] np_pos = np.zeros(geom_num * 3)
        cdef np.ndarray[np.float64_t, ndim=1] np_q_scipy = np.zeros(geom_num * 4)

        # on x64 System
        cdef size_t * ids_res = <size_t *> np_ids.data
        cdef int * types_res = <int *> np_types.data
        cdef dReal * pos_res = <dReal *> np_pos.data
        cdef dReal * q_res = <dReal * > np_q_scipy.data
        # cdef dReal * mat_res = <dReal *> np_rot_mat.data

        cdef dGeomID g = dSpaceGetFirstGeom(self.sid)
        cdef int idx = 0

        cdef const dReal * pos
        cdef dQuaternion q
        while g != NULL:
            if dGeomIsPlaceable(g):
                ids_res[idx] = <size_t> g
                types_res[idx] = dGeomGetClass(g)

                pos = dGeomGetPosition(g)
                pos_res[idx * 3 + 0] = pos[0]
                pos_res[idx * 3 + 1] = pos[1]
                pos_res[idx * 3 + 2] = pos[2]

                dGeomGetQuaternion(g, q)  # w, x, y, z
                q_res[idx * 4 + 0] = q[1]
                q_res[idx * 4 + 1] = q[2]
                q_res[idx * 4 + 2] = q[3]
                q_res[idx * 4 + 3] = q[0]

                idx += 1

            g = dSpaceGetNextGeom(g)

        return np_ids, np_types, np_pos, np_q_scipy

    # Add by Zhenhua Song
    # Get AABB bounding box of bodies
    # min_x, max_x, min_y, max_y, min_z, max_z
    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_bodies_aabb(np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef np.ndarray[np.uint64_t, ndim=1] np_id_buff = np.ascontiguousarray(np_id)
        cdef dBodyID * res_id = <dBodyID*> np_id_buff.data
        cdef dBodyID b = NULL
        cdef int idx = 0
        cdef int cnt = np_id_buff.size
        cdef np.ndarray[np.float64_t, ndim=1] np_aabb = np.zeros(6)
        cdef dReal * aabb_res = <dReal *> np_aabb.data
        _init_aabb_impl(aabb_res)

        while idx < cnt:
            b = res_id[idx]
            _get_body_aabb_impl(b, aabb_res)
            idx += 1

        return np_aabb

    # Add by Zhenhua Song
    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_batch_aabb(np.ndarray[np.uint64_t, ndim=1] np_id):
        cdef dBodyID * res_id = <dBodyID*>(np_id.data)
        cdef dBodyID b = NULL
        cdef int idx = 0
        cdef int cnt = np_id.size
        cdef np.ndarray[np.float64_t, ndim=1] np_aabb = np.zeros((cnt, 6))
        cdef dReal * aabb_res = <dReal *> np_aabb.data
        _init_aabb_impl(aabb_res)

        while idx < cnt:
            b = res_id[idx]
            _get_body_aabb_impl(b, aabb_res + 6 * idx)
            idx += 1

        return np_aabb

    def add(self, GeomObject geom):
        """add(geom)

        Add a geom to a space. This does nothing if the geom is
        already in the space.

        @param geom: Geom object to add
        @type geom: GeomObject
        """
        dSpaceAdd(self.sid, geom.gid)

    def remove(self, GeomObject geom):
        """remove(geom)

        Remove a geom from a space.

        @param geom: Geom object to remove
        @type geom: GeomObject
        """
        dSpaceRemove(self.sid, geom.gid)

    def query(self, GeomObject geom) -> bool:
        """query(geom) -> bool

        Return True if the given geom is in the space.

        @param geom: Geom object to check
        @type geom: GeomObject
        """
        return bool(dSpaceQuery(self.sid, geom.gid))

    @property
    def NumGeoms(self) -> int:  # O(1)
        """getNumGeoms() -> int

        Return the number of geoms contained within the space.
        """
        return dSpaceGetNumGeoms(self.sid)

    def getGeom(self, int idx) -> GeomObject:
        """getGeom(idx) -> GeomObject

        Return the geom with the given index contained within the space.

        @param idx: Geom index (0,1,...,getNumGeoms()-1)
        @type idx: int
        """
        # Check the index
        if idx < 0 or idx >= dSpaceGetNumGeoms(self.sid):
            raise IndexError("geom index out of range")

        cdef dGeomID gid = dSpaceGetGeom(self.sid, idx)

        return <GeomObject>dGeomGetData(gid)

    def collide(self, arg, callback):
        """collide(arg, callback)

        Call a callback function one or more times, for all
        potentially intersecting objects in the space. The callback
        function takes 3 arguments:

        def NearCallback(arg, geom1, geom2):

        The arg parameter is just passed on to the callback function.
        Its meaning is user defined. The geom1 and geom2 arguments are
        the geometry objects that may be near each other. The callback
        function can call the function collide() (not the Space
        method) on geom1 and geom2, perhaps first determining
        whether to collide them at all based on other information.

        @param arg: A user argument that is passed to the callback function
        @param callback: Callback function
        @type callback: callable
        """
        cdef object tup = (callback, arg)
        dSpaceCollide(self.sid, <void*>tup, collide_callback)

    # Add by Zhenhua Song
    cdef void fast_collide(self, dJointGroupWithdWorld * info):
        dSpaceCollide(self.sid, <void*> info, &fast_collide_callback)


# Callback function for the dSpaceCollide() call in the Space.collide() method
# The data parameter is a tuple (Python-Callback, Arguments).
# The function calls a Python callback function with 3 arguments:
# def callback(UserArg, Geom1, Geom2)
# Geom1 and Geom2 are instances of GeomXyz classes.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void collide_callback(void* data, dGeomID o1, dGeomID o2) noexcept:
    if (dGeomGetBody(o1)==dGeomGetBody(o2)):  # contains dGeomGetBody(o1) == NULL and dGeomGetBody(o2) == NULL
        return

    cdef GeomObject g1 = <GeomObject> dGeomGetData(o1)
    cdef GeomObject g2 = <GeomObject> dGeomGetData(o2)

    if not g1.geom_attrs.collidable or not g2.geom_attrs.collidable:
        return

    cdef int i = 0
    
    while i < g1.geom_attrs.ignore_geom_buf_len:
        if o2 == g1.geom_attrs.ignore_geom_buf[i]:
            return
        i += 1

    i = 0
    while i < g2.geom_attrs.ignore_geom_buf_len:
        if o1 == g2.geom_attrs.ignore_geom_buf[i]:
            return
        i += 1
    
    cdef object tup = <object>data
    callback, arg = tup
    callback(arg, g1, g2)
    

# Add by Zhenhua Song, collision detection in cython (not using python)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fast_collide_callback(void * data, dGeomID o1, dGeomID o2) noexcept:  # should not use nogil because GeomObject exists:
    cdef dBodyID b1 = dGeomGetBody(o1)
    cdef dBodyID b2 = dGeomGetBody(o2)

    if b1 == b2:  # contains dGeomGetBody(o1) == NULL and dGeomGetBody(o2) == NULL
        return

    if b1 != NULL and b2 != NULL and dAreConnected(b1, b2):
        return

    cdef GeomObject g1 = <GeomObject> dGeomGetData(o1)
    cdef GeomObject g2 = <GeomObject> dGeomGetData(o2)

    cdef int class_1 = dGeomGetClass(o1)
    cdef int class_2 = dGeomGetClass(o2)

    cdef dJointGroupWithdWorld * group_info = <dJointGroupWithdWorld *> data
    cdef dWorldID world = group_info.world
    cdef dJointGroupID contact_group = group_info.group
    cdef int max_contact_num = group_info.max_contact_num
    cdef int use_max_force = group_info.use_max_force_contact
    cdef int use_soft_contact = group_info.use_soft_contact
    cdef dReal soft_cfm = group_info.soft_cfm
    cdef dReal soft_erp = group_info.soft_erp

    # print(use_soft_contact, soft_cfm, soft_erp)

    if (dGeomGetCharacterID(o1) == dGeomGetCharacterID(o2)) and (not g1.geom_attrs.character_self_collide or not group_info.self_collision):
        return

    cdef size_t i = 0
    while i < g1.geom_attrs.ignore_geom_buf_len:
        if o2 == g1.geom_attrs.ignore_geom_buf[i]:
            return
        i += 1

    i = 0
    while i < g2.geom_attrs.ignore_geom_buf_len:
        if o1 == g2.geom_attrs.ignore_geom_buf[i]:
            return
        i += 1

    cdef dContactGeom c[256] # Zhenhua Song: I don't know why ode python binding uses 150..for fast, I use 4 instead
    cdef int n
    cdef dContact contact[256]
    cdef dJointID joint = NULL

    n = dCollide(o1, o2, max_contact_num, c, sizeof(dContactGeom))

    i = 0
    while i < n:
        contact[i].surface.mode = dContactApprox1  #
        if not use_max_force:
            if g1.geom_attrs.friction < g2.geom_attrs.friction:
                contact[i].surface.mu = g1.geom_attrs.friction
            else:
                contact[i].surface.mu = g2.geom_attrs.friction
        else:
            if g1.geom_attrs.max_friction < g2.geom_attrs.max_friction:
                contact[i].surface.mu = g1.geom_attrs.max_friction
            else:
                contact[i].surface.mu = g2.geom_attrs.max_friction

        # mu2 is ignored..
        if g1.geom_attrs.bounce < g2.geom_attrs.bounce:
            contact[i].surface.bounce = g1.geom_attrs.bounce
        else:
            contact[i].surface.bounce = g2.geom_attrs.bounce

        if use_soft_contact:
            contact[i].surface.soft_cfm = soft_cfm
            contact[i].surface.soft_erp = soft_erp
            contact[i].surface.mode = contact[i].surface.mode | dContactSoftCFM | dContactSoftERP

        contact[i].geom = c[i]
        # Note: here we should judge the contact type.
        if use_max_force:
            joint = dJointCreateContactMaxForce(world, contact_group, &contact[i])
        else:
            joint = dJointCreateContact(world, contact_group, &contact[i])
        dJointAttach(joint, b1, b2)
        i += 1

    # remove joint group after simulation

# SimpleSpace
cdef class SimpleSpace(SpaceBase):
    """Simple space.

    This does not do any collision culling - it simply checks every
    possible pair of geoms for intersection, and reports the pairs
    whose AABBs overlap. The time required to do intersection testing
    for n objects is O(n**2). This should not be used for large numbers
    of objects, but it can be the preferred algorithm for a small
    number of objects. This is also useful for debugging potential
    problems with the collision system.
    """

    def __cinit__(self, SpaceBase space=None):
        cdef SpaceBase sp
        cdef dSpaceID parentid = NULL

        if space != None:
            sp = space
            parentid = sp.sid

        self.sid = dSimpleSpaceCreate(parentid)

        # Copy the ID
        self.gid = <dGeomID>self.sid

        dSpaceSetCleanup(self.sid, 0)

    def __init__(self, SpaceBase space=None):
        self._setData(self)


cdef class HashSpace(SpaceBase):
    """Multi-resolution hash table space.

    This uses an internal data structure that records how each geom
    overlaps cells in one of several three dimensional grids. Each
    grid has cubical cells of side lengths 2**i, where i is an integer
    that ranges from a minimum to a maximum value. The time required
    to do intersection testing for n objects is O(n) (as long as those
    objects are not clustered together too closely), as each object
    can be quickly paired with the objects around it.
    """

    def __cinit__(self, SpaceBase space=None):
        cdef SpaceBase sp
        cdef dSpaceID parentid = NULL

        if space != None:
            sp = space
            parentid = sp.sid

        self.sid = dHashSpaceCreate(parentid)

        # Copy the ID
        self.gid = <dGeomID>self.sid

        dSpaceSetCleanup(self.sid, 0)

    def __init__(self, SpaceBase space=None):
        self._setData(self)

    def setLevels(self, int minlevel, int maxlevel):
        """setLevels(minlevel, maxlevel)

        Sets the size of the smallest and largest cell used in the
        hash table. The actual size will be 2^minlevel and 2^maxlevel
        respectively.
        """

        if minlevel > maxlevel:
            raise ValueError(
                "minlevel (%d) must be less than or equal to maxlevel (%d)" %
                (minlevel, maxlevel))

        dHashSpaceSetLevels(self.sid, minlevel, maxlevel)

    def getLevels(self):
        """getLevels() -> (minlevel, maxlevel)

        Gets the size of the smallest and largest cell used in the
        hash table. The actual size is 2^minlevel and 2^maxlevel
        respectively.
        """
        cdef int minlevel, maxlevel
        dHashSpaceGetLevels(self.sid, &minlevel, &maxlevel)
        return minlevel, maxlevel


# QuadTreeSpace
cdef class QuadTreeSpace(SpaceBase):
    """Quadtree space.

    This uses a pre-allocated hierarchical grid-based AABB tree to
    quickly cull collision checks. It's exceptionally quick for large
    amounts of objects in landscape-shaped worlds. The amount of
    memory used is 4**depth * 32 bytes.

    Currently getGeom() is not implemented for the quadtree space.
    """

    def __cinit__(self, center, extents, depth, SpaceBase space=None):
        cdef SpaceBase sp
        cdef dSpaceID parentid
        cdef dVector3 c
        cdef dVector3 e

        parentid = NULL
        if space != None:
            sp = space
            parentid = sp.sid

        c[0] = center[0]
        c[1] = center[1]
        c[2] = center[2]
        e[0] = extents[0]
        e[1] = extents[1]
        e[2] = extents[2]
        self.sid = dQuadTreeSpaceCreate(parentid, c, e, depth)

        # Copy the ID
        self.gid = <dGeomID>self.sid

        dSpaceSetCleanup(self.sid, 0)

    def __init__(self, center, extents, depth, SpaceBase space=None):
        self._setData(self)


def Space(int space_type=0) ->SpaceBase:
    """Space factory function.

    Depending on the type argument this function either returns a
    SimpleSpace (space_type=0) or a HashSpace (space_type=1).

    This function is provided to remain compatible with previous
    versions of PyODE where there was only one Space class.

    >>> space = Space(space_type=0)   # Create a SimpleSpace
    >>> space = Space(space_type=1)   # Create a HashSpace
    """
    if space_type == 0:
        return SimpleSpace()
    elif space_type == 1:
        return HashSpace()
    else:
        raise ValueError("Unknown space type (%d)" % space_type)


# GeomSphere
cdef class GeomSphere(GeomObject):
    """Sphere geometry.

    This class represents a sphere centered at the origin.

    Constructor::GeomSphere(space=None, radius=1.0)
    """

    def __cinit__(self, SpaceBase space=None, dReal radius=1.0):
        cdef SpaceBase sp
        cdef dSpaceID sid = NULL

        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreateSphere(sid, radius)

    def __init__(self, SpaceBase space=None, dReal radius=1.0):
        self._space = space
        self._body = None

        self._setData(self)  # DO NOT use weakref.proxy or weakref.ref

    # TODO: Test. maybe it will not work..
    def copy_geom(self, Body body, SpaceBase space):
        cdef dReal radius = dGeomSphereGetRadius(self.gid)
        cdef GeomSphere result = GeomSphere(space, radius)
        result.body = body
        self.copy_pos_quat(result)
        return result

    def setRadius(self, dReal radius):
        """setRadius(radius)

        Set the radius of the sphere.

        @param radius: New radius
        @type radius: float
        """
        dGeomSphereSetRadius(self.gid, radius)

    # Add by Zhenhua Song
    @property
    def geomRadius(self):
        return self.getRadius()

    def getRadius(self):
        """getRadius() -> float

        Return the radius of the sphere.
        """
        return dGeomSphereGetRadius(self.gid)

    def pointDepth(self, p):
        """pointDepth(p) -> float

        Return the depth of the point p in the sphere. Points inside
        the geom will have positive depth, points outside it will have
        negative depth, and points on the surface will have zero
        depth.

        @param p: Point
        @type p: 3-sequence of floats
        """
        return dGeomSpherePointDepth(self.gid, p[0], p[1], p[2])


# GeomBox
cdef class GeomBox(GeomObject):
    """Box geometry.

    This class represents a box centered at the origin.

    Constructor::GeomBox(space=None, lengths=(1.0, 1.0, 1.0))
    """

    def __cinit__(self, SpaceBase space=None, lengths=(1.0, 1.0, 1.0)):
        cdef SpaceBase sp
        cdef dSpaceID sid = NULL

        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreateBox(sid, lengths[0], lengths[1], lengths[2])

    def __init__(self, SpaceBase space=None, lengths=(1.0, 1.0, 1.0)):
        self._space = space
        self._body = None

        self._setData(self)  # DO NOT use weakref.ref or weakref.proxy

    def copy_geom(self, Body body, SpaceBase space):
        cdef dVector3 dxyz
        dGeomBoxGetLengths(self.gid, dxyz)
        cdef GeomBox result = GeomBox(space, (dxyz[0], dxyz[1], dxyz[2]))
        result.body = body
        self.copy_pos_quat(result)
        return result

    def setLengths(self, lengths):
        dGeomBoxSetLengths(self.gid, lengths[0], lengths[1], lengths[2])

    # Add by Zhenhua Song
    @property
    def geomLength(self):
        return self.getLengths()

    def getLengths(self):
        cdef dVector3 res
        dGeomBoxGetLengths(self.gid, res)
        return res[0], res[1], res[2]

    @property
    def LengthNumpy(self) -> np.ndarray:
        cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(3)
        dGeomBoxGetLengths(self.gid, <double * > result.data)
        return result

    @LengthNumpy.setter
    def LengthNumpy(self, np.ndarray[np.float64_t, ndim = 1] lengths) -> np.ndarray:
        dGeomBoxSetLengths(self.gid, lengths[0], lengths[1], lengths[2])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getLengthNumpy(self) -> np.ndarray:
        cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(3)
        dGeomBoxGetLengths(self.gid, <double * > result.data)
        return result

    def pointDepth(self, p):
        """pointDepth(p) -> float

        Return the depth of the point p in the box. Points inside the
        geom will have positive depth, points outside it will have
        negative depth, and points on the surface will have zero
        depth.

        @param p: Point
        @type p: 3-sequence of floats
        """
        return dGeomBoxPointDepth(self.gid, p[0], p[1], p[2])


# GeomPlane
cdef class GeomPlane(GeomObject):
    """Plane geometry.

    This class represents an infinite plane. The plane equation is:
    n.x*x + n.y*y + n.z*z = dist

    This object can't be attached to a body.
    If you call getBody() on this object it always returns ode.environment.

    Constructor::GeomPlane(space=None, normal=(0,0,1), dist=0)

    """

    def __cinit__(self, SpaceBase space=None, normal=(0, 0, 1), dReal dist=0):
        # (a, b, c) is normalized.
        cdef SpaceBase sp
        cdef dSpaceID sid = NULL

        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreatePlane(sid, normal[0], normal[1], normal[2], dist)

    def __init__(self, SpaceBase space=None, normal=(0, 0, 1), dist=0):
        self._space = space

        self._setData(self)  # DO NOT use weakref.proxy or weakref.ref

    def copy_geom(self, Body body, SpaceBase space):
        cdef dVector4 res
        dGeomPlaneGetParams(self.gid, res)
        cdef GeomPlane result = GeomPlane(space, (res[0], res[1], res[2]), res[3])
        result.body = body
        self.copy_pos_quat(result)

        return result

    def setParams(self, normal, dist):
        dGeomPlaneSetParams(self.gid, normal[0], normal[1], normal[2], dist)

    def getParams(self):
        cdef dVector4 res
        dGeomPlaneGetParams(self.gid, res)
        return ((res[0], res[1], res[2]), res[3])

    # Add by Zhenhua Song
    @property
    def odePlaneParam(self):
        return self.getParams()

    def pointDepth(self, p):
        """pointDepth(p) -> float

        Return the depth of the point p in the plane. Points inside the
        geom will have positive depth, points outside it will have
        negative depth, and points on the surface will have zero
        depth.

        @param p: Point
        @type p: 3-sequence of floats
        """
        return dGeomPlanePointDepth(self.gid, p[0], p[1], p[2])

    # Add by Zhenhua Song
    @property
    def odePosition(self):  # nearest point to (0, 0, 0) on plane
        return 0.0, 0.0, 0.0


# GeomCapsule
cdef class GeomCapsule(GeomObject):
    """Capped cylinder geometry.

    This class represents a capped cylinder aligned along the local Z axis
    and centered at the origin.

    Constructor::
    GeomCapsule(space=None, radius=0.5, length=1.0)

    The length parameter does not include the caps.
    """

    def __cinit__(self, SpaceBase space=None, dReal radius=0.5, dReal length=1.0):
        cdef SpaceBase sp
        cdef dSpaceID sid = NULL

        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreateCapsule(sid, radius, length)

    def __init__(self, SpaceBase space=None, dReal radius=0.5, dReal length=1.0):
        self._space = space
        self._body = None

        self._setData(self)  # DO NOT use weakref.ref or weakref.proxy

    def copy_geom(self, Body body, SpaceBase space):
        cdef dReal radius, length
        dGeomCapsuleGetParams(self.gid, &radius, &length)
        cdef GeomCapsule result = GeomCapsule(space, radius)
        result.body = body
        self.copy_pos_quat(result)
        return result

    def setParams(self, dReal radius, dReal length):
        dGeomCapsuleSetParams(self.gid, radius, length)

    def getParams(self):
        cdef dReal radius, length
        dGeomCapsuleGetParams(self.gid, &radius, &length)
        return radius, length

    # Add by Zhenhua Song
    @property
    def radius(self):
        cdef dReal radius, length
        dGeomCapsuleGetParams(self.gid, &radius, &length)
        return radius

    # Add by Zhenhua Song
    @property
    def geomLength(self):
        cdef dReal radius, length
        dGeomCapsuleGetParams(self.gid, &radius, &length)
        return length

    # Add by Zhenhua Song
    @property
    def geomRadiusAndLength(self):
        return self.getParams()

    def pointDepth(self, p):
        """pointDepth(p) -> float

        Return the depth of the point p in the cylinder. Points inside the
        geom will have positive depth, points outside it will have
        negative depth, and points on the surface will have zero
        depth.

        @param p: Point
        @type p: 3-sequence of floats
        """
        return dGeomCapsulePointDepth(self.gid, p[0], p[1], p[2])
    
    # Add by Zhenhua Song
    def capsule_axis(self):
        """
        """
        pass

GeomCCylinder = GeomCapsule # backwards compatibility


# GeomCylinder
cdef class GeomCylinder(GeomObject):
    """Plain cylinder geometry.

    Note: ou should compile ode with libccd for supporting collision detection between cylinder and other geoms.

    This class represents an uncapped cylinder aligned along the local Z axis
    and centered at the origin.

    Constructor:: GeomCylinder(space=None, radius=0.5, length=1.0)
    """

    def __cinit__(self, SpaceBase space=None, dReal radius=0.5, dReal length=1.0):
        cdef SpaceBase sp
        cdef dSpaceID sid = NULL

        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreateCylinder(sid, radius, length)

    def __init__(self, SpaceBase space=None, dReal radius=0.5, dReal length=1.0):
        self._space = space
        self._body = None

        self._setData(self)  # DO NOT use weakref.proxy or weakref.ref

    def copy_geom(self, Body body, SpaceBase space):
        pass

    def setParams(self, dReal radius, dReal length):
        dGeomCylinderSetParams(self.gid, radius, length)

    def getParams(self):
        cdef dReal radius, length
        dGeomCylinderGetParams(self.gid, &radius, &length)
        return radius, length

    ## dGeomCylinderPointDepth not implemented upstream in ODE 0.7


# GeomRay
cdef class GeomRay(GeomObject):
    """Ray object.

    A ray is different from all the other geom classes in that it does
    not represent a solid object. It is an infinitely thin line that
    starts from the geom's position and extends in the direction of
    the geom's local Z-axis.

    Constructor:: GeomRay(space=None, rlen=1.0)
    """

    def __cinit__(self, SpaceBase space=None, dReal rlen=1.0):
        cdef SpaceBase sp
        cdef dSpaceID sid = NULL

        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreateRay(sid, rlen)

    def __init__(self, SpaceBase space=None, dReal rlen=1.0):
        self._space = space
        self._body = None

        self._setData(self)  # DO NOT use weakref.proxy or weakref.ref

    def setLength(self, dReal rlen):
        '''setLength(rlen)

        Set length of the ray.

        @param rlen: length of the ray
        @type rlen: float'''
        dGeomRaySetLength(self.gid, rlen)

    def getLength(self):
        '''getLength() -> length

        Get the length of the ray.

        @returns: length of the ray (float)'''
        return dGeomRayGetLength(self.gid)

    def set(self, p, u):
        '''set(p, u)

        Set the position and rotation of a ray.

        @param p: position
        @type p: 3-sequence of floats
        @param u: rotation
        @type u: 3-sequence of floats'''
        dGeomRaySet(self.gid, p[0], p[1], p[2], u[0], u[1], u[2])

    def get(self):
        '''get() -> ((p[0], p[1], p[2]), (u[0], u[1], u[2]))

        Return the position and rotation as a pair of
        tuples.

        @returns: position and rotation'''
        cdef dVector3 start
        cdef dVector3 dir
        dGeomRayGet(self.gid, start, dir)
        return (start[0], start[1], start[2]), (dir[0], dir[1], dir[2])


# GeomTransform
cdef class GeomTransform(GeomObject):
    """GeomTransform.

    A geometry transform "T" is a geom that encapsulates another geom
    "E", allowing E to be positioned and rotated arbitrarily with
    respect to its point of reference.

    Constructor::GeomTransform(space=None)
    """

    cdef object geom

    def __cinit__(self, SpaceBase space=None):
        cdef SpaceBase sp
        cdef dSpaceID sid = NULL

        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreateGeomTransform(sid)
        # Set cleanup mode to 0 as a contained geom will be deleted
        # by its Python wrapper class
        dGeomTransformSetCleanup(self.gid, 0)

    def __init__(self, SpaceBase space=None):
        self._space = space
        self._body = None
        self.geom = None

        self._setData(self)  # DO NOT use weakref.proxy or weakref.ref

    def setGeom(self, GeomObject geom not None):
        """setGeom(geom)

        Set the geom that the geometry transform encapsulates.
        A ValueError exception is thrown if a) the geom is not placeable,
        b) the geom was already inserted into a space or c) the geom is
        already associated with a body.

        @param geom: Geom object to encapsulate
        @type geom: GeomObject
        """
        cdef size_t id

        if dGeomGetSpace(geom.gid) != <dSpaceID>0:
            raise ValueError("The encapsulated geom was already inserted into a space")
        if dGeomGetBody(geom.gid) != <dBodyID>0:
            raise ValueError("The encapsulated geom is already associated with a body")

        dGeomTransformSetGeom(self.gid, geom.gid)
        self.geom = geom

    def getGeom(self):
        """getGeom() -> GeomObject

        Get the geom that the geometry transform encapsulates.
        """
        return self.geom

    def setInfo(self, int mode):
        """setInfo(mode)

        Set the "information" mode of the geometry transform.

        With mode 0, when a transform object is collided with another
        object, the geom field of the ContactGeom structure is set to the
        geom that is encapsulated by the transform object.

        With mode 1, the geom field of the ContactGeom structure is set
        to the transform object itself.

        @param mode: Information mode (0 or 1)
        @type mode: int
        """
        if mode < 0 or mode > 1:
            raise ValueError(
                "Invalid information mode (%d). Must be either 0 or 1." % mode)
        dGeomTransformSetInfo(self.gid, mode)

    def getInfo(self):
        """getInfo() -> int

        Get the "information" mode of the geometry transform (0 or 1).

        With mode 0, when a transform object is collided with another
        object, the geom field of the ContactGeom structure is set to the
        geom that is encapsulated by the transform object.

        With mode 1, the geom field of the ContactGeom structure is set
        to the transform object itself.
        """
        return dGeomTransformGetInfo(self.gid)

######################################################################



######################################################################

# TODO: the implement has bug...
def collide_unused(GeomObject geom1_, GeomObject geom2_, int contact_count=4) -> list:
    cdef GeomObject geom1 = geom1_
    cdef GeomObject geom2 = geom2_
    cdef int iMaxContact = contact_count
    if iMaxContact > 4:
        iMaxCollide = 4
    cdef dGeomID o1 = geom1.gid, o2 = geom2.gid
    cdef dContact contact[16]
    memset(contact, 0, sizeof(contact))
    cdef int n = ModifiedCollisionDetection(o1, o2, contact, iMaxContact), i = 0
    if n <= 0:
        return None

    cdef list res = list()
    cdef Contact cont
    while i < n:
        cont = Contact()
        cont._contact = contact[i]
        res.append(cont)
        i = i + 1

    return res

# Modified by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def collide(GeomObject geom1, GeomObject geom2, int contact_count=200) -> list:
    """collide(geom1, geom2) -> contacts

    Generate contact information for two objects.

    Given two geometry objects that potentially touch (geom1 and geom2),
    generate contact information for them. Internally, this just calls
    the correct class-specific collision functions for geom1 and geom2.

    [flags specifies how contacts should be generated if the objects
    touch. Currently the lower 16 bits of flags specifies the maximum
    number of contact points to generate. If this number is zero, this
    function just pretends that it is one - in other words you can not
    ask for zero contacts. All other bits in flags must be zero. In
    the future the other bits may be used to select other contact
    generation strategies.]

    If the objects touch, this returns a list of Contact objects,
    otherwise it returns an empty list.

    @param geom1: First Geom
    @type geom1: GeomObject
    @param geom2: Second Geom
    @type geom2: GeomObject
    @returns: Returns a list of Contact objects.
    """
    # Zhen Wu: Take the mesh in consideration, 200 may be not enough.
    if contact_count >= 200:
        contact_count = 200
    cdef dContactGeom c[200]  # Zhenhua Song: 150 is too large...perhaps 1 is enough?
    cdef Contact cont

    cdef int n = dCollide(geom1.gid, geom2.gid, contact_count, c, sizeof(dContactGeom))
    cdef list res = list()
    cdef int i = 0
    while i < n:
        cont = Contact()
        cont._contact.geom = c[i]
        res.append(cont)
        i = i + 1

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def collide2(GeomObject geom1, GeomObject geom2, arg, callback):
    """collide2(geom1, geom2, arg, callback)

    Calls the callback for all potentially intersecting pairs that contain
    one geom from geom1 and one geom from geom2.

    @param geom1: First Geom
    @type geom1: GeomObject
    @param geom2: Second Geom
    @type geom2: GeomObject
    @param arg: A user argument that is passed to the callback function
    @param callback: Callback function
    @type callback: callable
    """
    cdef object tup = (callback, arg)
    # collide_callback is defined in space.pyx
    dSpaceCollide2(geom1.gid, geom2.gid, <void*>tup, collide_callback)


def areConnected(Body body1, Body body2) -> bool:
    """areConnected(body1, body2) -> bool

    Return True if the two bodies are connected together by a joint,
    otherwise return False.

    @param body1: First body
    @type body1: Body
    @param body2: Second body
    @type body2: Body
    @returns: True if the bodies are connected
    """

    if body1 is environment:
        return False
    if body2 is environment:
        return False

    return bool(dAreConnected(<dBodyID> body1.bid, <dBodyID> body2.bid))

# Add by Zhenhua Song
# wrapper of ODE dSolveLCP function
# void dSolveLCP (dxWorldProcessMemArena *memarena, int n, dReal *A, dReal *x, dReal *b,
#                dReal *outer_w, int nub, dReal *lo, dReal *hi, int *findex)
# return x, outer_w as np.ndarray
def solve_lcp(
    int m,
    np.ndarray[np.float64_t, ndim = 1] a,
    np.ndarray[np.float64_t, ndim = 1] b,
    int nub,
    np.ndarray[np.float64_t, ndim = 1] lo,
    np.ndarray[np.float64_t, ndim = 1] hi,
    np.ndarray[np.int32_t, ndim = 1] findex
):
    """

    Solve LCP problem.
    given (A,b,lo,hi), solve the LCP problem: A*x = b+w, where each x(i),w(i) satisfies one of
	(1) x = lo, w >= 0
	(2) x = hi, w <= 0
	(3) lo < x < hi, w = 0
    A is a matrix of dimension n*n, everything else is a vector of size n*1.
    lo and hi can be +/- dInfinity as needed. the first `nub' variables are
    unbounded, i.e. hi and lo are assumed to be +/- dInfinity.

    we restrict lo(i) <= 0 and hi(i) >= 0.


    @param a:
    @type a: np.ndarray
    @param b:
    @type b: np.ndarray
    @param nub:
    @type nub: int
    @param lo:
    @type lo: np.ndarray
    @param hi:
    @type hi: np.ndarray
    @param findex:
    @type findex: np.ndarray
    @returns: x, outer_w
    """
    cdef int m_ = m
    cdef int dpad_m = dPADFunction(m)

    # convert data to ODE format. +16: avoid index out of boundary
    cdef dReal * ode_a = <dReal *> malloc((m_ * dpad_m + 16) * sizeof(dReal))
    cdef dReal * ode_b = <dReal *> malloc((m_ + 16) * sizeof(dReal))
    cdef dReal * ode_lo = <dReal *> malloc((m_ + 16) * sizeof(dReal))
    cdef dReal * ode_hi = <dReal *> malloc((m_ + 16) * sizeof(dReal))
    cdef int * ode_findex = <int *> malloc((m_ + 16) * sizeof(int))

    cdef dReal * a_ptr = <dReal *> a.data
    memset(ode_a, 0, sizeof(dReal) * m_ * dpad_m)
    cdef int i = 0
    while i < m_:
        memcpy(ode_a + dpad_m * i, a_ptr + m_ * i, sizeof(dReal) * m_)
        i += 1

    memcpy(ode_b, <dReal *> b.data, m_ * sizeof(dReal))
    memcpy(ode_lo, <dReal *> lo.data, m_ * sizeof(dReal))
    memcpy(ode_hi, <dReal *> hi.data, m_ * sizeof(dReal))
    memcpy(ode_findex, <int *> findex.data, m_ * sizeof(int))

    cdef np.ndarray[np.float64_t, ndim=1] x = np.zeros(m)
    cdef np.ndarray[np.float64_t, ndim=1] w = np.zeros(m)
    cdef dReal * x_buf = <dReal *> x.data
    cdef dReal * w_buf = <dReal *> w.data
    cdef int nub_ = nub
    # print("Before LCP wrapper", m_, dpad_m, nub_)
    dSolveLCPWrapper(m_, ode_a, x_buf, ode_b, w_buf, nub_, ode_lo, ode_hi, ode_findex)
    # print("end LCP wrapper")

    # free memory
    free(ode_a)
    free(ode_b)
    free(ode_lo)
    free(ode_hi)
    free(ode_findex)

    # print("after free memory")
    return x, w


def CloseODE():
    """CloseODE()

    Deallocate some extra memory used by ODE that can not be deallocated
    using the normal destroy functions.
    """
    dCloseODE()


def InitODE():
    '''InitODE()

    Initialize some ODE internals. This will be called for you when you
    "import ode", but you should call this again if you CloseODE().'''
    dInitODE()
    dRandSetSeed(0)


def SetInitSeed(int value):
    dRandSetSeed(value)


# Add by Yulong Zhang
cimport DrawStuffWorld
import platform
import atexit
import os

# from DrawStuffWorld cimport *
cdef class RenderWorld:

    def __init__(self, myworld, SimpleSpace space=None, vis_geoms=None):
        assert platform.system() == 'Windows'
        if hasattr(myworld, "world"):
            myworld = myworld.world
        cdef World dsWorld = <World>myworld
        cdef dWorldID wid = dsWorld.wid
        cdef dSpaceID sid = NULL
        if space is not None:
            sid = <dSpaceID> space.sid
        cdef std_vector[dGeomID] vis_geom_arr
        cdef dGeomID gid
        if vis_geoms is not None:
            for node_ in vis_geoms:
                gid = (<GeomObject> node_).gid
                vis_geom_arr.push_back(gid)

        DrawStuffWorld.dsWorldSetter(wid, sid, vis_geom_arr)
        atexit.register(self.kill)

    def track_body(self, Body dsBody, int sync_y):
        DrawStuffWorld.dsTrackBodyWrapper(<dBodyID>(dsBody.bid), 0, sync_y)

    def look_at(self, pos, target, up):
        DrawStuffWorld.dsCameraLookAtWrapper(pos[0], pos[1], pos[2], target[0], target[1], target[2], up[0], up[1], up[2])

    def set_color(self, col):
        DrawStuffWorld.dsAssignColor(col[0], col[1], col[2])
    
    def set_joint_radius(self, dReal x):
        DrawStuffWorld.dsAssignJointRadius(x)

    def set_axis_length(self, dReal x):
        DrawStuffWorld.dsAssignAxisLength(x)

    def draw_background(self, int x):
        DrawStuffWorld.dsAssignBackground(x)

    def draw_hingeaxis(self, int x):
        DrawStuffWorld.dsWhetherHingeAxis(x)

    def draw_localaxis(self, int x):
        DrawStuffWorld.dsWhetherLocalAxis(x)

    def start(self):
        DrawStuffWorld.dsDrawWorldinThread()
    
    def kill(self):
        print("Killing renderer!", "taskkill /f /pid " + str(os.getpid()))
        os.system("taskkill /f /pid " + str(os.getpid()))
        DrawStuffWorld.dsKillThread()
        print("Killed renderer!")
        os.system("taskkill /f /pid " + str(os.getpid()))
    
    def pause(self, int time):
        DrawStuffWorld.dsAssignPauseTime(<int>time)
        DrawStuffWorld.dsSlowforRender()

    @property
    def world(self):
        cdef dWorldID value = DrawStuffWorld.dsWorldGetter()
        if value == NULL:
            return None
        return 
        
    @property
    def space(self):  # get the debug space..
        return None # cdef space_id = None
    
    def start_record_video(self):
        DrawStuffWorld.dsStartRecordVideo()

    def get_video_size(self):
        return DrawStuffWorld.dsGetWindowWidth(), DrawStuffWorld.dsGetWindowHeight()

    def get_record_buffer(self) -> np.ndarray:
        DrawStuffWorld.dsPauseRecordVideo()
        cdef size_t frame = DrawStuffWorld.dsGetVideoFrame()
        cdef int width = DrawStuffWorld.dsGetWindowWidth()
        cdef int height = DrawStuffWorld.dsGetWindowHeight()
        cdef np.ndarray[np.uint8_t, ndim=4] result = np.empty((frame, height, width, 3), np.uint8)
        DrawStuffWorld.dsEndRecordVideo(<unsigned char *> (result.data), frame)
        return result

    def pause_record_video(self):
        DrawStuffWorld.dsPauseRecordVideo()


cdef class TriMeshData:
    """This class stores the mesh data.
    """

    cdef dTriMeshDataID tmdid
    cdef dReal* vertex_buffer
    cdef unsigned int* face_buffer

    def __cinit__(self):
        self.tmdid = dGeomTriMeshDataCreate()
        self.vertex_buffer = NULL
        self.face_buffer = NULL

    def __dealloc__(self):
        if self.tmdid != NULL:
            dGeomTriMeshDataDestroy(self.tmdid)
        if self.vertex_buffer != NULL:
            free(self.vertex_buffer)
        if self.face_buffer != NULL:
            free(self.face_buffer)
    
    def build(self, verts, faces):
        """build(verts, faces)

        @param verts: Vertices
        @type verts: Sequence of 3-sequences of floats
        @param faces: Face definitions (three indices per face)
        @type faces: Sequence of 3-sequences of ints
        """
        cdef size_t numverts
        cdef size_t numfaces
        cdef dReal* vp
        cdef unsigned int* fp
        cdef int a, b, c
        
        numverts = len(verts)
        numfaces = len(faces)
        # Allocate the vertex and face buffer
        self.vertex_buffer = <dReal*>malloc(numverts * 4 * sizeof(dReal))
        self.face_buffer = <unsigned int*>malloc(numfaces * 3 * sizeof(unsigned int))

        # Fill the vertex buffer
        vp = self.vertex_buffer
        for v in verts:
            vp[0] = v[0]
            vp[1] = v[1]
            vp[2] = v[2]
            vp[3] = 0
            vp = vp + 4

        # Fill the face buffer
        fp = self.face_buffer
        for f in faces:
            a = f[0]
            b = f[1]
            c = f[2]
            if (a < 0 or b < 0 or c < 0 or a >= numverts or b >= numverts or c >= numverts):
                raise ValueError("Vertex index out of range")
            fp[0] = a
            fp[1] = b
            fp[2] = c
            fp = fp + 3

        # Pass the data to ODE
        dGeomTriMeshDataBuildSimple(self.tmdid, self.vertex_buffer, numverts,
                                    self.face_buffer, numfaces * 3)


# GeomTriMesh
cdef class GeomTriMesh(GeomObject):
    """TriMesh object.

    To construct the trimesh geom you need a TriMeshData object that
    stores the actual mesh. This object has to be passed as first
    argument to the constructor.

    Constructor::
    
      GeomTriMesh(data, space=None)
    """

    # Keep a reference to the data
    cdef TriMeshData data

    def __cinit__(self, TriMeshData data not None, space=None):
        cdef SpaceBase sp
        cdef dSpaceID sid

        self.data = data

        sid = NULL
        if space != None:
            sp = space
            sid = sp.sid
        self.gid = dCreateTriMesh(sid, data.tmdid, NULL, NULL, NULL)

    def __init__(self, TriMeshData data not None, space=None):
        self._space = space
        self._body = None

        self._setData(self)

    def _id(self):
        return <size_t>self.gid

    def clearTCCache(self):
        """clearTCCache()

        Clears the internal temporal coherence caches.
        """
        dGeomTriMeshClearTCCache(self.gid)

    def getTriangle(self, int idx):
        """getTriangle(idx) -> (v0, v1, v2)

        @param idx: Triangle index
        @type idx: int
        """

        cdef dVector3 v0, v1, v2

        dGeomTriMeshGetTriangle(self.gid, idx, &v0, &v1, &v2)
        return ((v0[0], v0[1], v0[2]),
                (v1[0], v1[1], v1[2]),
                (v2[0], v2[1], v2[2]))

    def getTriangleCount(self):
        """getTriangleCount() -> n

        Returns the number of triangles in the TriMesh."""

        return dGeomTriMeshGetTriangleCount(self.gid)

######################################################################
environment = None
InitODE()

######################################################################
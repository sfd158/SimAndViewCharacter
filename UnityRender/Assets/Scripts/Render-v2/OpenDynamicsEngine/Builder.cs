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
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using static RenderV2.ODE.CommonFunc;

namespace RenderV2.ODE
{
    public class CharacterBuilder
    {
        ODEWorld world;
        SpaceBase space;
        DCharacter character;
        ODECharacterHandle handle;
        bool use_angle_limit = true;
        static readonly double[][] raw_axis = EyeArray3DImpl2();

        public CharacterBuilder(
            ODEWorld world_, SpaceBase space_, DCharacter character_, 
            bool use_angle_limit_ = true)
        {
            world = world_;
            space = space_;
            character = character_;
            use_angle_limit = use_angle_limit_;
            // raw_axis = EyeArray3DImpl2();
        }

        public CharacterBuilder(ODEWorldHandle handle, DCharacter character_, bool use_angle_limit_)
        {
            world = handle.world;
            space = handle.space;
            character = character_;
            use_angle_limit = use_angle_limit_;
        }

        public void AddIgnore() // TODO: add parent ignore.
        {
            int nbody = character.BodyCount;
            var ode_bodies = handle.GetBodyList();
            for (int i = 0; i < nbody; i++)
            {
                var body1 = character.BodyList[i];
                var ode_body1 = ode_bodies[body1.IDNum];
                foreach(var ignore in body1.IgnoreCollision)
                {
                    var body2 = ignore.GetComponent<DRigidBody>();
                    ode_body1.BodyIgnoreAdd(ode_bodies[body2.IDNum]);
                }
            }
        }

        public void CreateJoint()
        {
            int nj = character.JointCount;
            var blist = handle.GetBodyList();
            var jlist = handle.GetJointList();
            var swapEuler = new string[3]{"XZY", "YXZ", "ZYX"};
            for (int j = 0; j < nj; j++)
            {
                var joint = character.JointList[j];
                var parent = blist[joint.parentBody.IDNum]; 
                var child = blist[joint.childBody.IDNum];
                
                if (joint is DHingeJoint)
                {
                    var hinge = new ODEHingeJoint(world);
                    var dhinge = joint as DHingeJoint;
                    hinge.SetAxis(raw_axis[dhinge.EulerOrder[0] - 'X']);
                    // TODO: check deg or rad
                    hinge.SetAngleLimit(dhinge.AngleLimit[0], dhinge.AngleLimit[1]);

                    hinge.SetSameKd(dhinge.Damping);
                    hinge.Attach(parent, child);
                    hinge.SetAnchor(dhinge.transform.position);
                }
                else if (joint is DBallJoint)
                {
                    var ode_ball = new BallJointAmotor(world);
                    // ball.setAnchor();
                    
                    var dball = joint as DBallJoint;
                    ode_ball.SetSameKd(dball.Damping);
                    var euler = dball.EulerOrder;
                    var low_lim = CopyArr(dball.AngleLoLimit);
                    var hi_lim = CopyArr(dball.AngleHiLimit);
                    if (swapEuler.Contains(euler))
                    {
                        int eidx = euler[2] - 'X';
                        // angle_limits[idx] = np.array([-angle_limits[idx][1], -angle_limits[idx][0]])
                        var tmp = -low_lim[eidx];
                        low_lim[eidx] = -hi_lim[eidx];
                        hi_lim[eidx] = tmp;
                    }
                    ode_ball.setAmotorMode((int)MotorType.dAMotorEuler);
                    ode_ball.setAmotorNumAxes(3);
                    
                    ode_ball.setAmotorAxis(0, 1, raw_axis[euler[0] - 'X']);
                    ode_ball.setAmotorAxis(2, 2, raw_axis[euler[2] - 'X']);  // Axis 2, body 2
                    ode_ball.setAngleLim1(low_lim[0], hi_lim[0]);
                    ode_ball.setAngleLim2(low_lim[1], hi_lim[1]);
                    ode_ball.setAngleLim3(low_lim[2], hi_lim[2]);

                    ode_ball.attach_ext(parent, child);
                    ode_ball.SetAnchor(dball.transform.position);
                }
                else
                {
                    throw new NotSupportedException("Only support ball joint and hinge joint here.");
                }
            }
        }

        public ODEBody BuildBody(DRigidBody dBody)
        {
            var odeBody = new ODEBody(world);
            odeBody.BodySetIntValue(dBody.IDNum);
            odeBody.SetPosition(dBody.transform.position);
            
            bool calc_mass = dBody.MassMode == DRigidBodyMassMode.Density;
            int NumGeom = dBody.vGeomList.Count;
            
            float[] mass_list = new float[NumGeom];
            Geom[] geom_list = new Geom[NumGeom];
            Vector3 geom_com = Vector3.zero;
            ODEMass[] mgeom_list = new ODEMass[NumGeom];
            for (int gidx = 0; gidx < NumGeom; gidx++)
            {
                var vgeom = dBody.vGeomList[gidx];
                Geom odeGeom = null;
                ODEMass geom_mass = calc_mass ? new ODEMass() : null;
                var dgeom = vgeom.childGeom;
                if (dgeom is DBallGeom)
                {
                    var ball = dgeom as DBallGeom;
                    var ballGeom = new GeomSphere(space, ball.Radius);
                    odeGeom = ballGeom;
                    if (calc_mass)
                    {
                        geom_mass.SetSphere(dBody.Density, ball.Radius);
                    }
                }
                else if (dgeom is DBoxGeom)
                {
                    var box = dgeom as DBoxGeom;
                    odeGeom = new GeomBox(space, box.Length);
                    if (calc_mass)
                    {
                        geom_mass.SetBox(dBody.Density, box.Length);
                    }
                }
                else if (dgeom is DCapsuleGeom)
                {
                    var capsule = dgeom as DCapsuleGeom;
                    odeGeom = new GeomCapsule(space, capsule.Radius, capsule.Length);
                    if (calc_mass)
                    {
                        geom_mass.SetCapsule(dBody.Density, 3, capsule.Radius, capsule.Length);
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }
                odeGeom.SetFriction(dgeom.Friction);
                odeGeom.SetPosition(dgeom.transform.position);
                odeGeom.SetQuaternion(dgeom.transform.rotation);
                if (calc_mass)
                {
                    mass_list[gidx] = (float)geom_mass.MassValue;
                    geom_com += mass_list[gidx] * dgeom.transform.position;
                    mgeom_list[gidx] = geom_mass;
                }
                geom_list[gidx] = odeGeom;
            }
            
            if (calc_mass) geom_com /= mass_list.Sum();
            ODEMass mass = new ODEMass();
            for (int gidx = 0; gidx < NumGeom; gidx++)
            {
                var dgeom = dBody.vGeomList[gidx].childGeom;
                var geom = geom_list[gidx];
                geom.body = odeBody;
                geom.SetOffsetWorldPosition(dgeom.transform.position);
                geom.SetOffsetWorldQuaternion(dgeom.transform.rotation);
                var inertia = new Inertia(mgeom_list[gidx].dmass);
                inertia.RotInertia(dgeom.transform.rotation);
                inertia.TransInertia(geom_com - dgeom.transform.position);
                inertia.AddToMass(mass.dmass);
            }

            if (dBody.MassMode == DRigidBodyMassMode.MassValue)
                mass.MassValue = dBody.Mass;
            if (dBody.InertiaMode == DRigidBodyInertiaMode.InertiaValue)
                mass.dmass.I.SetValueDense(dBody.BodyInertia);
            odeBody.SetMass(mass);
            odeBody.SetQuaternion(dBody.transform.rotation);
            odeBody.SetLinearVel(dBody.LinearVelocity);
            odeBody.SetAngularVel(dBody.AngularVelocity);
            return odeBody;
        }

        public ODECharacterHandle build()
        {
            foreach(var body in character.BodyList)
            {
                var res = BuildBody(body);
                handle.BodyList.Add(res);
            }
            CreateJoint();
            AddIgnore();
            return handle;
        }
    }
}
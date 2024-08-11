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

using RenderV2;
using RenderV2.ODE;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestFunc : MonoBehaviour
{

    void TestWorld()
    {
        ODEWorld world = new ODEWorld();
        world.SetGravityY();
        ODEBody body = new ODEBody(world);
        ODEMass mass = new ODEMass();
        mass.SetBox(1000, 1, 1, 1);
        Debug.Log(mass.ToString());
        body.SetMass(mass);
        SimpleSpace space = new SimpleSpace();
        GeomPlane plane = GeomPlane.CreateDefault(space);
        GeomBox box = new GeomBox(space, 1, 1, 1);
        box.body = body;
        body.SetPosition(0, 2, 0);
        JointGroup group = new JointGroup();
        for(int i = 0; i < 100; i++)
        {
            space.Collide(world, group);
            world.Step();
        }
    }

    void TestBVH()
    {
        var fname = "C:\\Users\\24357\\Downloads\\ode-develop-szh-develop\\Tests\\CharacterData\\WalkF-mocap-100.bvh";
        var loader = new RenderV2.BVHLoader();
        var mocap = loader.load_bvh(fname);
        mocap = mocap.resample(10);
        SaveBVH.SaveTo(mocap, "fuck.bvh");
    }

    private void Start()
    {
        TestWorld();
    }
    // Update is called once per frame
    void Update()
    {
        
    }
}

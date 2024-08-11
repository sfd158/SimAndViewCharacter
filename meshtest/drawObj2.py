import ModifyODE as ode
import os
import atexit, time
import numpy as np
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.PDControler import DampedPDControler
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.Render import Renderer
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter

import trimesh

################################################
# 在这里更改测试物体

mesh_path = os.path.dirname(__file__)
pymesh = trimesh.load_mesh(os.path.join(mesh_path, "ObjectModel/Primitive/sphere.obj"))

pymesh.density *= 10
pymesh.apply_scale(0.5)

world = ode.World()
world.setGravity([0, -9.8, 0])
space = ode.Space()

meshbody1 = ode.Body(world)

meshData = ode.TriMeshData()
meshData.build(pymesh.vertices-pymesh.center_mass, pymesh.faces)


meshGeom = ode.GeomTriMesh(meshData, space)
meshGeom.body = meshbody1 
meshbody1.PositionNumpy = np.array([0.5, 5.0, 0.0])
mass = ode.Mass()
mass.setParameters(pymesh.mass_properties['mass'], 0.0, 0.0, 0.0, 
    pymesh.moment_inertia[0, 0], pymesh.moment_inertia[1, 1], pymesh.moment_inertia[2, 2], 
    pymesh.moment_inertia[0, 1], pymesh.moment_inertia[0, 2], pymesh.moment_inertia[1, 2])
meshbody1.setMass(mass)
meshbody1.setRotationNumpy(np.array([0., 30., 50., 10., 1., 0., 40., 0., 0.]))

pymesh2 = trimesh.load_mesh("ObjectModel/Dining/cup1.obj")

pymesh2.density *= 10
pymesh2.apply_scale(0.5)

meshbody2 = ode.Body(world)

meshData2 = ode.TriMeshData()
meshData2.build(pymesh2.vertices-pymesh2.center_mass, pymesh2.faces)


meshGeom2 = ode.GeomTriMesh(meshData2, space)
meshGeom2.body = meshbody2
meshbody2.PositionNumpy = np.array([-0.2, 8.0, 0.0])
mass2 = ode.Mass()
mass2.setParameters(pymesh2.mass_properties['mass'], 0.0, 0.0, 0.0, 
    pymesh2.moment_inertia[0, 0], pymesh2.moment_inertia[1, 1], pymesh2.moment_inertia[2, 2], 
    pymesh2.moment_inertia[0, 1], pymesh2.moment_inertia[0, 2], pymesh2.moment_inertia[1, 2])
meshbody2.setMass(mass2)
meshbody2.setRotationNumpy(np.array([0., 30., 50., 10., 1., 0., 40., 0., 0.]))


################################################
# 用不同的物体来测试碰撞

bodybox = ode.Body(world)
box = ode.GeomBox(space)
box.body = bodybox
massbox = ode.Mass()
massbox.setBox(1.0, 1.0, 1.0, 1.0)
bodybox.setMass(massbox)
bodybox.PositionNumpy = np.array([0.0, 2.0, 0.0])

# bodycapsule = ode.Body(world)
# capsule = ode.GeomCapsule(space)
# capsule.body = bodycapsule
# masscapsule = ode.Mass()
# masscapsule.setCapsule(1.0, 3, 0.5, 1.0)
# bodycapsule.setMass(masscapsule)
# bodycapsule.PositionNumpy = np.array([0.2, 11.0, 0.3])

################################################

floor = ode.GeomPlane(space, (0, 1, 0), 0)
group = ode.JointGroup()

renderObj = Renderer.RenderWorld(world)
renderObj.draw_background(1)
# renderObj.set_joint_radius(0.005)
renderObj.start()

def exit_func():
    renderObj.kill()
atexit.register(exit_func)

################################################
# 如果contact数量太大会崩溃
# 目前设置最大值100

def callback(args, geom1, geom2):
    world, contactgroup = args
    contacts = ode.collide(geom1, geom2, 2000)
    if (len(contacts) >= 2000):
        print("too much contacts!!!\n")
    # print(len(contacts))
    for c in contacts:
        c.mu = 3.0
        c.mu2 = 3.0
        c.mode = 0x004
        c.bounce = 0.2
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.body, geom2.body)


while(1):
    space.collide((world, group), callback)
    world.step(0.01)
    group.empty()
    renderObj.pause(1)
    # time.sleep(0.05)
    # print(body.PositionNumpy)

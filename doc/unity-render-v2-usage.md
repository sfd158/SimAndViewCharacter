# Unity Rigid Body Render Document

## Run

### With Unity Rendering

- Start Python server: 
- Start Unity playing.

### Only Simulate in Python

You can export world as `json` format in Unity, and load this `json` format file in Python.

## Load And Save

### Load

- Load Character in `xml` format:

  Click `Assets/Load/XML Character` in menu bar, and select character configuration file in `xml` format.

- Load PD Control Parameter in `xml` format:

  In Unity Hierarchy Window, select target `GameObject` with `DCharacter` or `PDController` Component.

   Click `Assets/Load/XML PD Param` in menu bar, and select PD Controller  configuration file in `xml` format.

### Export

- Export world to `json`  format

  Click `Assets/Export/World` in menu bar.

## World

There are some attributes in inspector for `DWorld` Component

- Capsule Axis: Default capsule axis. 
- Gravity: gravity in world
- Step Count: simulation step executed at each frame. If set to 0, `SimulateFPS // RenderFPS` times of simulation step will be executed
- Render FPS: Render FPS in Unity
- Simulate FPS: Simulation FPS used in physics engine. In ODE physics engine, `dWorldDampedStep(1.0 /SimulateFPS)`  or `dWorldStep(1.0 /SimulateFPS)` will be called
- Use Hinge: If selected, `Hinge Joint` will be used normally. If not selected, Physical simulation module in background will use `Ball Joint` instead of `Hinge Joint`.
- Use Angle Limit: If selected, angle limit will be treated in simulation. If not selected,  angle limit will be set to $(-\infty, +\infty)$
- Self Collision: If selected, character will do collision detection with itself.

## Environment

## Character List

All characters are stored under `GameObject` with `CharacterList` Component.

There are some buttons in inspector for `CharacterList` Component.

- Re Compute All Characters:
- Sphere, Box, Capsule: 

## Ext Joint List

All external joints are stored under `GameObject` with `ExtJointList` Component.

## Character

There are some attributes in inspector for `DCharacter` Component.

### Geometry

There are some attributes in inspector for `DGeomObject` Component

- Initial Position: Geometry's initial global position
- Initial Quaternion: Geometry's initial global quaternion
- ID Num: Geometry instance ID 
- Collidable: enable collision detection with other geometry.
- Friction: Friction coefficient of geometry.
- Restitution: Restitution coefficient of geometry.
- Clung Env: 

#### Ball Geometry

`DBallGeom` is subclass of `DGeomObject` . There are some attributes in inspector for `DBallGeom` Component.

#### Plane Geometry



### Rigid Body

Parent of `GameObject` with `DRigidBody` Component should have one of these Component: `DCharacter`, `DJoint`.

Each child of `GameObject` with `DRigidBody` Component should have one of these Component: `VirtualGeom`.

There are some attributes in inspector for `DRigidBody` Component:



### Joint

Parent of `GameObject` with `DJoint` Component should have one of these Component: `DCharacter`, `DJoint`, `DExtJointList`. 

Each child of `GameObject` with `DJoint` Component should have one of  these Component: `DRigidBody`, `DJoint`, `DEndJoint`, `DEulerAxis`

#### Joint Orientation

#### Set Joint Angle Limit

#### View and set joint Euler angle under joint orientation

#### Ball Joint

#### Hinge Joint


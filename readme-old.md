## Compile Modify ODE

1 . Compile Modify ODE:

Go to `ModifyODE` directory, and compile with `CMakeLists.txt` in Release or RelWithDebInfo Mode.

2 . Compile and install python bindings:

Copy `ModifyODE.lib` to the folder `ModifyODE` 

Go to `ModifyODE/bindings/python`, and run `python3 setup.py install`, or `pip install -e .`

To use python bindings, please execute `import ModifyODE as ode` in python.

## Run Unity Render

### Windows 10

#### Version 1(Original Version)

Python Server: Go to folder  `entrance/TestStablePDControler`, and run `python3 main-unity.py`

Unity Client: Go to `unity-ode/Assets/Scenes/`, and run `SampleScene.unity` in Unity.

#### Version 2

Python Server: Go to folder `Server/v2`, and run `python3 ServerForUnityV2.py`

Unity Client: Go to folder `unity-ode/Assets/Scenes/`, and run `Render-v2.unity` in Unity.

### Linux or MacOS

If you use Linux or MacOS system, please compile `unity-ode/Assets/Plugins/src/CMakeLists.txt` . Other operations are the same as in windows.


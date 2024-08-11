import os
import atexit
import time
import numpy as np
from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.PDControler import DampedPDControler
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.Render import Renderer
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter

fdir = os.path.dirname(__file__)
scene_fname = os.path.join(fdir, 'left-with-root.json')
bvh_fname = os.path.join(fdir, 'check_left.bvh')
SceneLoader = JsonSceneLoader()
scene = SceneLoader.load_from_file(scene_fname)
bvh2target = BVHToTargetBase(bvh_fname, 120, scene.character0)
target = bvh2target.init_target()
set_target = SetTargetToCharacter(scene.character0, target)
stable_pd = DampedPDControler(scene.character0)

'''
i = 0
bvh_cnt = bvh2target.frame_cnt
'''
set_target.set_character_byframe(0)
renderObj = Renderer.RenderWorld(scene.world)
renderObj.draw_background(1)
renderObj.set_joint_radius(0.005)
renderObj.start()

def exit_func():
    renderObj.kill()
atexit.register(exit_func)

# body_pos = scene.character0.body_info.get_body_pos() - np.array([0.0, 2.0, 0.0])
# body_name = scene.character0.body_info.get_name_list()
# body_cnt = len(body_name)
# for i in range(body_cnt):
#     print('-----')
#     print(body_name[i])
#     print(body_pos[i])

# set_target.set_character_byframe(1)
# while True:
#     pass

i = 0
while True:
    set_target.set_character_byframe(i)
    time.sleep(0.1)
    i = (i + 1) % bvh2target.frame_cnt
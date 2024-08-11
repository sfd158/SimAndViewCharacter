import numpy as np
import os
import pickle
import torch
from VclSimuBackend.SimpleCIO.Optimize2d import Optimize2d
from VclSimuBackend.Utils.Camera.CameraNumpy import CameraParamNumpy
from VclSimuBackend.Utils.Camera.CameraPyTorch import CameraParamTorch
from VclSimuBackend.Utils.Camera.Human36CameraBuild import pre_build_camera
from VclSimuBackend.pymotionlib import BVHLoader


fdir = os.path.dirname(__file__)

def test_torch_camera():  # check OK
    rand_pos = np.random.randn(30).reshape((-1, 3))
    cam = pre_build_camera["S1"][0]
    torch_cam = CameraParamTorch.build_from_numpy(cam, torch.float64)
    np_cam_pos = cam.world_to_camera(rand_pos)
    torch_cam_pos = torch_cam.world_to_camera(torch.from_numpy(rand_pos))
    torch_rebuild_pos = torch_cam.camera_to_world(torch_cam_pos)
    print(np.max(np.abs(np_cam_pos - torch_cam_pos.numpy())))
    print(np.max(np.abs(torch_rebuild_pos.numpy() - rand_pos)))


def main():
    fname = r"D:\song\documents\GitHub\ode-scene\Tests\CharacterData\Samcon\0\network-output.bin"
    with open(fname, "rb") as fin:
        result = pickle.load(fin)
    pred_motion = BVHLoader.load(result["pred_motion"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    opt = Optimize2d(pred_motion, result["pos2d"], result["camera_param"], device)
    opt.train()


if __name__ == "__main__":
    main()

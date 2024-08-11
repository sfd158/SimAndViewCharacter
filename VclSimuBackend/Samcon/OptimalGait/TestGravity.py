'''
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
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import ModifyODE as ode
import time
import torch
from torch import nn
from torch.nn import functional as F
from VclSimuBackend.DiffODE.DiffFrameInfo import diff_frame_import_from_tensor

from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.DiffODE.Build import BuildFromODEScene
from VclSimuBackend.Render.Renderer import RenderWorld


fdir = os.path.dirname(__file__)
def main():

    def test_diffode2():
        scene = JsonSceneLoader().load_from_file(os.path.join(fdir, "../../../Tests/CharacterData/lamp.json"))
        scene.character0.joint_info.kps = np.ones(2)
        scene.character0.joint_info.kds = np.ones(2)
        scene.character0.joint_info.torque_limit = np.ones(2)
        character = scene.character0
        builder = BuildFromODEScene(scene)
        diff_world = builder.build()
        num_frame = 100

    def test_diffode_body1():
        scene = ODEScene(sim_fps=100)
        character = ODECharacter(scene.world, scene.space)
        body = ode.Body(scene.world)
        body.PositionNumpy = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        character.bodies.append(body)
        character.body_info.mass_val = np.array([body.mass_val])
        character.body_info.body_c_id = np.array([body.get_bid()], dtype=np.uint64)
        character.joint_info.joint_c_id = np.array([], dtype=np.uint64)
        character.init_body_state = character.save()
        scene.characters.append(character)
        builder = BuildFromODEScene(scene)
        diff_world = builder.build()
        num_frame = 50
        pos_param = nn.Parameter(torch.zeros((num_frame, 3), dtype=torch.float64))
        vel_param = nn.Parameter(torch.zeros((num_frame, 3), dtype=torch.float64))
        opt = torch.optim.AdamW([pos_param, vel_param], lr=1e-2)
        for epoch in range(5000):
            new_pos = []
            new_velo = []
            opt.zero_grad(True)
            for frame in range(num_frame - 1):
                diff_frame_import_from_tensor(
                    diff_world.curr_frame,
                    pos_param[frame],
                    vel_param[frame],
                    torch.eye(3, dtype=torch.float64),
                    torch.as_tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
                    torch.zeros((1, 3, 1), dtype=torch.float64),
                    scene.world,
                    character.body_info.body_c_id
                )
                diff_world.step(do_contact=False)
                new_pos.append(diff_world.curr_frame.body_pos.view(1, 3))
                new_velo.append(diff_world.curr_frame.body_velo.view(1, 3))
            loss = 1e4 * F.mse_loss(torch.cat(new_pos, dim=0), pos_param[1:].detach()) \
                + 1e4 * F.mse_loss(torch.cat(new_velo, dim=0), vel_param[1:].detach())
            loss.backward()
            pos_param.grad[0] = 0
            vel_param.grad[0] = 0
            # pos_param.grad[0] = 0
            if epoch % 50 == 0:
                print(epoch, loss.item(), pos_param.grad.abs().mean())
            opt.step()
            if epoch % 50 ==0:
                plt.figure()
                plt.plot(pos_param[..., 1].detach().numpy())
                plt.show()

    def test_diffode3():
        multi_frame = 10
        scene = JsonSceneLoader().load_from_file(os.path.join(fdir, "../../../Tests/CharacterData/lamp.json"))
        scene.set_sim_fps(50)
        num_body = len(scene.character0.bodies)
        scene.character0.joint_info.kps = np.ones(2)
        scene.character0.joint_info.kds = np.ones(0)
        scene.character0.joint_info.torque_limit = np.ones(2)
        character = scene.character0
        builder = BuildFromODEScene(scene)
        diff_world = builder.build()
        num_frame = 50
        pos_off = torch.as_tensor(character.get_body_pos())
        pos_off[1:] -= pos_off[:1]
        pos_opt = torch.zeros((num_frame, 3), dtype=torch.float64)
        pos_param = nn.Parameter(pos_opt)
        velo_param = nn.Parameter(torch.zeros_like(pos_opt))
        opt = torch.optim.AdamW([pos_param, velo_param], lr=5e-1)
        # render = RenderWorld(scene.world)
        # render.start()
        for epoch in range(10000):
            opt.zero_grad(True)
            tot_loss = torch.as_tensor(0.0, dtype=torch.float64)
            for frame in range(num_frame - multi_frame):
                new_pos = []
                new_velo = []
                import_pos = torch.tile(pos_param[frame], (num_body, 1,))[..., None]
                import_pos = import_pos + pos_off[..., None]
                import_velo = torch.tile(velo_param[frame], (num_body, 1))[..., None]
                diff_frame_import_from_tensor(
                    diff_world.curr_frame,
                    import_pos,
                    import_velo,
                    torch.tile(torch.eye(3, dtype=torch.float64), (3, 1, 1)),
                    torch.tile(torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64), (3, 1)),
                    torch.zeros((3, 3), dtype=torch.float64),
                    scene.world,
                    character.body_info.body_c_id
                )
                num_forward = min(multi_frame, num_frame - frame)
                for sub_frame in range(num_forward):
                    diff_world.step(do_contact=False, damping_step=False)
                    new_pos.append(diff_world.curr_frame.body_pos.view(1, 3, 3))
                    new_velo.append(diff_world.curr_frame.body_velo.view(1, 3, 3))

                new_pos = torch.cat(new_pos, dim=0)
                new_velo = torch.cat(new_velo, dim=0)
                # loss = 40000 * F.mse_loss(new_pos, torch.tile(pos_param[1:, None], (1, 3, 1)))
                loss = 1e3 * F.mse_loss(new_pos[:, 0], pos_param[frame + 1:frame + 1 + num_forward]) +\
                     1e3 * F.mse_loss(new_velo[:, 0], velo_param[frame + 1:frame + 1 + num_forward])
                tot_loss += loss
            tot_loss.backward()
            pos_param.grad[0] = 0
            velo_param.grad[0] = 0
            print(epoch, loss.item(), pos_param.grad.abs().mean(), pos_param[-1])
            opt.step()
            if epoch % 20 ==0:
                plt.figure()
                plt.plot(pos_param[..., 1].detach().numpy())
                plt.show()
                plt.figure()
                plt.plot(velo_param[..., 1].detach().numpy())
                plt.show()
            # print(pos_param.tolist())

        plt.figure()
        plt.plot(pos_param[..., 1].detach().numpy())
        plt.show()

    # def test_diffode3():
    #    pass

    def test_direct():
        num_frame = 100
        pos_param = nn.Parameter(torch.zeros(num_frame, dtype=torch.float64))
        velo_param = nn.Parameter(torch.zeros(num_frame, dtype=torch.float64))
        opt = torch.optim.AdamW([pos_param, velo_param], lr=1e-2)
        for epoch in range(20000):
            opt.zero_grad(True)
            new_pos = []
            new_vel = []
            for frame in range(num_frame - 1):
                next_vel = velo_param[frame] - 9.8 * 0.01
                next_pos = pos_param[frame] + next_vel * 0.01
                new_pos.append(next_pos)
                new_vel.append(next_vel)

            loss = 1e2 * (F.mse_loss(torch.as_tensor(new_pos), pos_param[1:]) + F.mse_loss(torch.as_tensor(new_vel), velo_param[1:]))
            loss.backward()
            opt.step()
            if epoch % 100 == 0 and False:
                opt.step()
                plt.figure()
                plt.plot(pos_param.detach().numpy())
                plt.show()

    test_diffode3()


if __name__ == "__main__":
    main()

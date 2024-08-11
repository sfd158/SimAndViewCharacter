import torch
import torch.nn as nn
from .TrajOptDirectBVH import DirectTrajOptBVH
from ...DiffODE import DiffQuat


def test_trajopt_gradient(traj_opt: DirectTrajOptBVH):
    """
    Test with single frame, to check backward gradient
    """

    def test(action_input: torch.Tensor):
        action_param = nn.Parameter(action_input)
        # opt = torch.optim.SGD([action_param], 1e-4)
        character = traj_opt.worker.character
        traj_opt.worker.tar_set.set_character_byframe(0)
        traj_opt.worker.diff_world.import_curr_frame()
        q1: torch.Tensor = DiffQuat.quat_from_rotvec(action_param)
        q0: torch.Tensor = traj_opt.worker.torch_target_local_quat[1, :, :]
        q_tot: torch.Tensor = DiffQuat.quat_multiply(q1, q0)
        tau, power = traj_opt.worker.curr_frame.stable_pd_control_wrapper(traj_opt.worker.stable_pd, q_tot)
        traj_opt.worker.diff_world.step()
        loss = traj_opt.worker.torch_loss.loss(traj_opt.worker.curr_frame, traj_opt.worker.torch_target, 1)
        loss.backward()
        return loss, action_param.grad.clone()

    zero_input = torch.zeros((traj_opt.num_joints, 3), dtype=torch.float64)
    loss_a, grad_a = test(zero_input)
    print(loss_a)
    step = 1e-6
    modify_input = -grad_a
    # modify_input = zero_input.detach().clone()
    # modify_input[0, 0] = grad_a[0, 0]
    loss_b, grad_b = test(step * modify_input)
    delta_loss = loss_b.item() - loss_a.item()
    print(f"loss_b = {loss_b.item():.10f}, delta loss = {delta_loss:.10f}, grad_a norm = {torch.linalg.norm(grad_a) ** 2}, ratio = {delta_loss/step}")

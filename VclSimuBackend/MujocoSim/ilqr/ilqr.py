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
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from mpi4py import MPI
from argparse import ArgumentParser, Namespace
import numpy as np
import time
from typing import List, Optional, Union

from env import HumanoidStand

fdir = os.path.dirname(__file__)
mpi_comm = MPI.COMM_WORLD
mpi_world_size: int = mpi_comm.Get_size()
mpi_rank: int = mpi_comm.Get_rank()

class ILQRSolver:
    """
    Maybe we need to find the initial solution...
    """
    def __init__(self, env):
        self.env = env
        self.finite_diff_eps = 1e-5
        self.reg = 1.0
        self.alpha_array = 1.4 ** (-np.arange(0, 10)**2)  #10 line steps
        # self.alpha_array = 1.05 ** (-np.arange(20)**2)
        # print(self.alpha_array)
        self.reg_max = 1e2 # 1000
        self.reg_min = 1e-9
        self.reg_factor = 1e3 # 10

        self.x_dim: int = env.model.nq + env.model.nv
        self.u_dim: int = env.data.ctrl.shape[0]

        self.diag = None

    def evaluate_trajectory_cost(self, x_array, u_array) -> float:
        return sum([self.env.cost(x_array[t][0], u, t) for t, u in enumerate(u_array)])

    def ilqr_iterate(self, u_init, n_itrs=20, tol=1e-6):
        self.reg = 1.0
        x_array = self.forward_propagation(u_init)
        u_array = np.copy(u_init)
        J_opt = self.evaluate_trajectory_cost(x_array, u_init)
        print(f"Init cost = {J_opt}", flush=True)
        J_hist = [J_opt]
        converged = False
        save_alpha = None
        for i in range(n_itrs):
            k_array, K_array = self.back_propagation(x_array, u_array)
            norm_k = np.mean(np.linalg.norm(k_array, axis=1))
            accept = False
            for alpha in self.alpha_array:  # apply the control to update the trajectory by trying different alpha
                x_array_new, u_array_new = self.apply_control(x_array, u_array, k_array, K_array, alpha)
                J_new = self.evaluate_trajectory_cost(x_array_new, u_array_new)

                if J_new < J_opt:
                    save_alpha = alpha
                    if np.abs((J_opt - J_new )/J_opt) < tol:
                        J_opt, x_array, u_array = J_new, x_array_new, u_array_new
                        converged = True
                        break
                    else:
                        J_opt, x_array, u_array = J_new, x_array_new, u_array_new
                        # successful step, decrease the regularization term
                        # momentum like adaptive regularization
                        self.reg = np.max([self.reg_min, self.reg / self.reg_factor])
                        accept = True
                        print('Iteration {0}:\tJ = {1};\tnorm_k = {2};\treg = {3}'.format(i+1, J_opt, norm_k, self.reg), flush=True)
                        break
                else:
                    accept = False

            J_hist.append(J_opt)
            if converged:
                print('Converged at iteration {0}; J = {1}; reg = {2}'.format(i+1, J_opt, self.reg), flush=True)
                break

            if not accept:
                if self.reg > self.reg_max:
                    print('Exceeds regularization limit at iteration {0}; terminate the iterations'.format(i+1), flush=True)
                    break

                self.reg = self.reg * self.reg_factor
                print(f'Reject the control perturbation. loss = {J_new}. best_loss = {J_opt}. Increase the regularization term to {self.reg}.', flush=True)
                # TODO: maybe we can test add noise on the control policy..

        res_dict = {
            'x_array': x_array,
            'u_array': np.array(u_array),
            'k_array': np.array(k_array),
            'K_array': np.array(K_array),
            "alpha": save_alpha
        }

        return res_dict

    def play_result(self, res_dict):
        while True:
            self.apply_control(**res_dict, is_render=True)

    def apply_control(self, x_array, u_array, k_array, K_array, alpha=None, is_render=False):
        self.env.reset_init()
        x_new_array = [None] * len(x_array)
        u_new_array = [None] * len(u_array)

        x_new_array[0] = x_array[0]
        if alpha is None:
            alpha = 1.0
        for t in range(self.env.T):
            u_new_array[t] = u_array[t] + alpha * (k_array[t] + K_array[t].dot(x_new_array[t][0] - x_array[t][0]))
            x_new_array[t+1] = self.env.plant_dyn(None, u_new_array[t], t, is_render=is_render)

        return x_new_array, np.array(u_new_array)  # Note: here qacc_warmstart is also stored.

    def forward_propagation(self, u_array, is_render=False):
        self.env.reset_init()
        traj_array = [self.env.get_state()]
        for t, u in enumerate(u_array):
            traj_array.append(self.env.plant_dyn(None, u, t, is_render=is_render))
        return traj_array

    def back_propagation(self, x_array, u_array):
        if mpi_world_size > 1:
            lqr_sys = self.build_lqr_system_parallel(x_array, u_array)
        else:
            lqr_sys = self.build_lqr_system(x_array, u_array)

        fdfwd = [None] * self.env.T  # k
        fdbck_gain = [None] * self.env.T  # K

        # initialize with the terminal cost parameters to prepare the backpropagation
        Vxx = lqr_sys['dldxx'][-1]
        Vx = lqr_sys['dldx'][-1]

        for t in range(self.env.T-1, -1, -1):
            dfdu_T = np.ascontiguousarray(lqr_sys['dfdu'][t].T)
            dfdx_T = np.ascontiguousarray(lqr_sys['dfdx'][t].T)
            #note to double check if we need the transpose lqr_sys['dfdu'] or not
            Qx: np.ndarray = lqr_sys['dldx'][t] + lqr_sys['dfdx'][t].T.dot(Vx)
            Qu: np.ndarray = dfdu_T.dot(Vx)  # lqr_sys['dldu'][t] == 0
            # Qxx = lqr_sys['dldxx'][t] + lqr_sys['dfdx'][t].T.dot(Vxx).dot(lqr_sys['dfdx'][t])
            Qxx: np.ndarray = dfdx_T.dot(Vxx).dot(lqr_sys['dfdx'][t]) + lqr_sys["dldxx"][t]

            # diag_dldxx
            Qux: np.ndarray = dfdu_T.dot(Vxx).dot(lqr_sys['dfdx'][t])  # lqr_sys['dldux'][t]
            # Quu = lqr_sys['dlduu'][t] + lqr_sys['dfdu'][t].T.dot(Vxx).dot(lqr_sys['dfdu'][t])
            Quu: np.ndarray = dfdu_T.dot(Vxx).dot(lqr_sys['dfdu'][t])  # here dlduu == 0

            #use regularized inverse for numerical stability
            inv_Quu = self.regularized_persudo_inverse_(Quu, self.reg)

            fdfwd[t] = -inv_Quu.dot(Qu)  # k
            fdbck_gain[t] = -inv_Quu.dot(Qux)  # K

            #update value function for the previous time step
            Vxx = Qxx - fdbck_gain[t].T.dot(Quu).dot(fdbck_gain[t])
            Vx = Qx - fdbck_gain[t].T.dot(Quu).dot(fdfwd[t])

        return fdfwd, fdbck_gain

    def build_lqr_system_base(self, x, u, t, warm_start):
        plant_dyn = self.env.plant_dyn
        x1 = np.tile(x, (self.x_dim, 1)) + np.eye(self.x_dim) * self.finite_diff_eps
        x2 = np.tile(x, (self.x_dim, 1)) - np.eye(self.x_dim) * self.finite_diff_eps
        u1 = np.tile(u, (self.u_dim, 1)) + np.eye(self.u_dim) * self.finite_diff_eps
        u2 = np.tile(u, (self.u_dim, 1)) - np.eye(self.u_dim) * self.finite_diff_eps
        fx1 = np.array([plant_dyn(x1_dim, u, t, warm_start)[0] for x1_dim in x1])
        fx2 = np.array([plant_dyn(x2_dim, u, t, warm_start)[0] for x2_dim in x2])
        dfdx = np.ascontiguousarray((fx1-fx2).T/2./self.finite_diff_eps)
        fu1 = np.array([plant_dyn(x, u1_dim, t, warm_start)[0] for u1_dim in u1])
        fu2 = np.array([plant_dyn(x, u2_dim, t, warm_start)[0] for u2_dim in u2])
        dfdu = np.ascontiguousarray((fu1-fu2).T/2./self.finite_diff_eps)
        dldx = self.env.dl_dx(x, u, t)
        dldxx = self.env.dl_dxx(x, u, t)
        return np.clip(dfdx, -2, 2), np.clip(dfdu, -10, 10), np.clip(dldx, -50, 50), np.clip(dldxx, -50, 50)

    def list_to_2d(self, info: Union[List, np.ndarray]):
        """
        reshape 1d list to shape (worker_size, len / worker_size)
        """
        w, n = mpi_world_size, len(info)
        if isinstance(info, list):
            return [info[w_idx * n // w: (w_idx + 1) * n // w] for w_idx in range(w)]
        elif isinstance(info, np.ndarray):
            return [info[w_idx * n // w: (w_idx + 1) * n // w]
                    if (w_idx+1)*n//w > w_idx*n//w else [] for w_idx in range(w)]
        else:
            raise NotImplementedError

    def build_lqr_system_parallel(self, x_array, u_array):  # here we can run parallel..
        # 1. divide the x_array and u_array  # 2. scatter to sub workers.
        x_array = x_array[:-1]
        state = [node[0] for node in x_array]
        warm_start = [node[1] for node in x_array]
        t_array = np.arange(0, len(u_array))
        info_2d = [self.list_to_2d(info) for info in [state, u_array, t_array, warm_start]]
        send = [[info[w_idx] for info in info_2d] for w_idx in range(mpi_world_size)]
        all_result = self.build_lqr_system_compute(send)
        all_result = sum(all_result, [])
        all_result.sort(key=lambda x: x["t"])
        keys = ["dfdx", "dfdu", "dldx", "dldxx"]
        result = {}
        for key in keys:
            result[key] = [node[key] for node in all_result]
        return result

    def build_lqr_system_compute(self, divide_state=None):
        sub_task = mpi_comm.scatter(divide_state)
        sub_x_arr, sub_u_arr, sub_t_arr, sub_warm_arr = sub_task
        sub_result = []
        for i in range(len(sub_x_arr)):
            x_, u_, t_, warm_start_ = sub_x_arr[i], sub_u_arr[i], sub_t_arr[i], sub_warm_arr[i]
            dfdx, dfdu, dldx, dldxx = self.build_lqr_system_base(x_, u_, t_, warm_start_)
            sub_result.append({"t": t_, "dfdx": dfdx, "dfdu": dfdu, "dldx": dldx, "dldxx": dldxx})
        # 3. gather results.
        all_result = mpi_comm.gather(sub_result)
        return all_result

    def build_lqr_system(self, x_array, u_array):
        u_array = np.vstack([u_array, np.zeros(len(u_array[0]))])
        dfdx_array, dfdu_array, dldx_array, dldxx_array = [], [], [], []
        for t in range(len(u_array)):
            x, warm_start = x_array[t]
            u = u_array[t]
            dfdx, dfdu, dldx, dldxx = self.build_lqr_system_base(x, u, t, warm_start)
            dfdx_array.append(dfdx)
            dfdu_array.append(dfdu)
            dldx_array.append(dldx)
            dldxx_array.append(dldxx)

        return {"dfdx": dfdx_array, "dfdu": dfdu_array, "dldx": dldx_array, "dldxx": dldxx_array}

    def regularized_persudo_inverse_(self, mat, reg=1e-5):
        if self.diag is None:
            self.diag = np.arange(0, len(mat))
        mat[self.diag, self.diag] += reg
        return np.linalg.inv(mat)

    def run_sliding_window(self):
        start_time_time = time.time()
        total_time = 60
        move_time = 40
        solution = np.zeros((total_time, self.env.nu))
        # we can pre-compute with bvh as start frame..
        env_t_old = self.env.T
        self.env.T = 4
        if False:
            for i in range(10):
                curr_iter = 5
                result = self.ilqr_iterate(solution[self.env.start_t:self.env.start_t + self.env.T].copy(), curr_iter)
                u_new_array = result["u_array"]
                x_new_array = self.forward_propagation(u_new_array)
                J = self.evaluate_trajectory_cost(x_new_array, u_new_array)
                print(f"start_t = {self.env.start_t}, J = {J}\n\n", flush=True)
                solution[self.env.start_t:self.env.start_t+self.env.T] = u_new_array[:]
                self.env.start_t += 3
                self.env.reset_by_index(self.env.start_t)
                self.env.init_state[self.env.start_t] = self.env.get_state()

        self.env.T = env_t_old
        self.env.start_t = 0
        self.env.reset_init_state()
        for i in range(move_time):
            curr_iter = 40 if i == 0 else 20
            result = self.ilqr_iterate(solution[self.env.start_t:self.env.start_t + self.env.T].copy(), curr_iter)
            u_new_array = result["u_array"]
            x_new_array = self.forward_propagation(u_new_array)
            J = self.evaluate_trajectory_cost(x_new_array, u_new_array)
            print(f"start_t = {self.env.start_t}, J = {J}\n\n", flush=True)
            self.env.init_state[self.env.start_t: self.env.start_t + self.env.T] = x_new_array
            solution[self.env.start_t:self.env.start_t+self.env.T] = u_new_array[:]
            self.env.start_t += 1

        # here we need to play the total result
        # maybe a feedback controller is required..
        # or we can run iLQR on the total path..
        self.env.start_t = 0
        self.env.T += move_time - 5
        print(f"Begin Playing")
        print("time", time.time() - start_time_time)
        self.env.change_camera()
        while True:
            self.forward_propagation(solution[:self.env.T], True)

    def run_sub_worker(self):
        # TODO: write stop
        while True:
            self.build_lqr_system_compute()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mocap_fname", type=str, default=r"D:\song\documents\GitHub\ode-develop\Tests\CharacterData\WalkF-mocap-100.bvh")
    parser.add_argument("--bvh_start", type=int, default=0, help="start frame of reference motion")
    parser.add_argument("--bvh_end", type=int, default=100, help="end frame of reference motion")
    parser.add_argument("--start_core", type=int, default=0, help="for parallel")
    parser.add_argument("--w_qpos", type=float, default=2, help="loss weight of qpos")
    parser.add_argument("--w_qvel", type=float, default=1e-3, help="loss weight of qvel")
    parser.add_argument("--w_up", type=float, default=1, help="loss weight of up vector")
    parser.add_argument("--control_fps", type=int, default=10, help="control freq")
    parser.add_argument("--time_count", type=int, default=8)
    # parser.add_argument("--")
    return parser.parse_args()


def main():
    args = parse_args()
    args.mocap_fname = r"D:\song\Documents\GitHub\ode-scene\Tests\CharacterData\sfu\0005_Jogging001-mocap-100.bvh"
    env = HumanoidStand(args)
    solver = ILQRSolver(env)
    # env.play_bvh()
    if mpi_rank == 0:
        start_time = time.time()
        # result = solver.ilqr_iterate(np.zeros((env.T, env.nu)), 40)
        result = solver.run_sliding_window()
        # pickle.dump(result, "result.pickle")
        print("time", time.time() - start_time, flush=True)
        solver.play_result(result)
    else:
        solver.run_sub_worker()


if __name__ == '__main__':
    main()

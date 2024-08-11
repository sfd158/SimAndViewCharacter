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
import copy
import numpy as np
import logging
from typing import Optional


class CMAUpdate:
    __slots__ = ("n", "lambda_value", "mu", "mu_raw", "sigma", "g", "eigeneval", "forward_g", "x_mean",
                 "weights", "mueff", "c_sigma", "d_sigma", "c_c", "c_1", "c_mu", "chi_n",
                 "b", "c", "d", "inv_sqrt_c", "p_sigma", "p_c", "sorted_argx", "sorted_cost",
                 "cost_good", "early_converge_eps", "init_sigma", "init_cov", "_max_cov", "learning_rate",
                 "history_best_cost")

    def __init__(self, x_mean: np.ndarray, sigma: float = 0.5, cov_mat: Optional[np.ndarray] = None,
                 lambda_val: Optional[int] = None, mu: Optional[float] = None,
                 cost_good: float = 0.0, early_converge_eps: float = 1e-5):
        """
        param:
        x_mean: initial mean
        sigma:
        lambda_vel: CMA algorithm gives a method to calc initial lambda.
        mu: CMA algorithm gives a method to calc initial mu.
        cost_good:
        stopfitness:
        """
        self.n, self.lambda_value, self.mu = (0,) * 3
        self.mu_raw, self.sigma = (0.0,) * 2

        self.g, self.eigeneval = 0, 0  # generation iteration, eigen eval
        self.forward_g = 0
        self.x_mean, self.weights = (np.array([]),) * 2
        self.mueff, self.c_sigma, self.d_sigma, self.c_c, self.c_1, self.c_mu, self.chi_n = (0.0,) * 7
        self.b, self.c, self.d, self.inv_sqrt_c, self.p_sigma, self.p_c = (np.array([]),) * 6
        self.sorted_argx, self.sorted_cost = (np.array([]),) * 2

        self.cost_good = cost_good
        self.early_converge_eps = early_converge_eps

        self.reset(x_mean, sigma, cov_mat, lambda_val, mu)
        self.init_sigma = self.sigma
        self.init_cov = copy.deepcopy(self.c)

        self._max_cov: Optional[float] = None  # maximal component in cov matrix

        self.learning_rate: float = 1.0  # Add by Zhenhua Song

        self.history_best_cost: float = float("inf")

    @property
    def max_cov(self):
        if self._max_cov is None:
            self._max_cov = np.max(np.abs(self.c))
        return self._max_cov

    def reset_lambda_mu(self, lambda_val: Optional[int] = None, mu: Optional[float] = None):
        """
        param:
        lambda_vel: CMA algorithm gives a method to calc initial lambda.
        mu: CMA algorithm gives a method to calc initial mu.
        """
        self.lambda_value = 4 + int(3 * np.log(self.n)) if lambda_val is None else lambda_val
        self.mu_raw = self.lambda_value / 2 if mu is None else mu  # Number of Elite Samples
        self.mu = int(self.mu_raw)

        self.weights: np.ndarray = np.log(self.mu_raw + 0.5) - np.log(np.arange(1, self.mu + 1, dtype=np.float))
        sumw = np.sum(self.weights)
        sumww = np.sum(self.weights ** 2)
        self.weights /= sumw

        # variance effective selection mass for the mean
        self.mueff = sumw ** 2 / sumww

        # coefficient for step size control
        self.c_sigma = (self.mueff + 2.0) / (self.n + self.mueff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (self.n + 1)) - 1.0) + self.c_sigma

        # coefficient for Covariance matrix adaptation
        self.c_c = (4.0 + self.mueff / self.n) / (self.n + 4.0 + 2.0 * self.mueff / self.n)
        self.c_1 = 2.0 / ((self.n + 1.3) ** 2 + self.mueff)
        self.c_mu = min(1 - self.c_1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) /
                        ((self.n + 2.0) * (self.n + 2.0) + self.mueff))

        # E || N(0, I) ||
        self.chi_n = np.sqrt(self.n) * (1.0 - 1.0 / (4.0 * self.n) + 1.0 / (21.0 * self.n ** 2))

        self.sorted_argx: Optional[np.ndarray] = np.zeros((self.mu, self.n))
        self.sorted_cost: Optional[np.ndarray] = np.zeros(self.mu)

    def reset_update_param(self, cov: Optional[np.ndarray] = None):
        self.set_cov_mat(cov)
        self.p_sigma: Optional[np.ndarray] = np.zeros(self.n)
        self.p_c: Optional[np.ndarray] = np.zeros(self.n)

    def calc_cov_attrs(self):
        cov_eps = 1e-8
        self.c[:] = np.triu(self.c) + np.triu(self.c, 1).T
        self.c[np.diag_indices(self.c.shape[0], 2)] += cov_eps * np.random.rand(self.c.shape[0])
        # eigen decomposition
        d, self.b = np.linalg.eig(self.c)
        if d.dtype != np.float64:
            for i in range(self.c.shape[0]):
                logging.info(self.c[i].tolist())
            raise ValueError(f"dtype of d in CMA is {d.dtype}")

        d: np.ndarray = np.sqrt(d)
        self.d = np.diag(d)
        # print(self.g, "Eigen Value", d)
        self.inv_sqrt_c = self.b @ np.diag(1.0 / d) @ self.b.T

        self._max_cov = np.max(np.abs(self.c))

    def set_cov_mat(self, cov: Optional[np.ndarray]):
        if cov is None:
            self.b: Optional[np.ndarray] = np.eye(self.n)  # [B,D] = eig(C)
            self.c: Optional[np.ndarray] = np.eye(self.n)  # Cov matrix
            self.d: Optional[np.ndarray] = np.eye(self.n)  # [B,D] = eig(C)
            self.inv_sqrt_c: Optional[np.ndarray] = np.eye(self.n)  # C^{-1/2}
            self._max_cov = 1.0
        elif cov.ndim == 1:
            cov_tmp, abs_cov = cov.copy(), np.abs(cov)
            eps = 1e-8
            cov_tmp[abs_cov < eps] += eps * np.random.rand(*cov_tmp[abs_cov < eps].shape)
            self.c = np.diag(cov_tmp)
            self.calc_cov_attrs()
        elif cov.ndim == 2:
            assert cov.shape[0] == cov.shape[1]
            self.c = cov.copy()
            self.calc_cov_attrs()
        else:
            raise ValueError

    def reset(self, x_mean: Optional[np.ndarray] = None, sigma: Optional[float] = None,
              cov_mat: Optional[np.ndarray] = None,
              lambda_val: Optional[float] = None, mu: Optional[float] = None, reset_generation: bool = True):
        """
        param:
        x_mean: mean value
        sigma:
        """
        if x_mean is not None:
            self.n = x_mean.shape[0]  # dim
            self.x_mean = x_mean.copy()
        if sigma is not None:
            self.sigma = sigma

        self.reset_lambda_mu(lambda_val, mu)
        self.reset_update_param(cov_mat)
        if reset_generation:
            self.g = 0

    def counteval(self) -> int:
        return self.lambda_value * (self.g + 1)

    @staticmethod
    def cost_func(argx: np.ndarray) -> np.ndarray:
        return np.sum(argx ** 2, axis=-1)

    def sample(self, mask: Optional[np.ndarray] = None, sample_cnt: Optional[int] = None) -> np.ndarray:
        """
        param:
        mask:
        sample_cnt: batch size of samples

        returns: gaussian sample result with mean, sigma, cov
        in shape (sample_cnt, dim(x))
        """
        logging.info(f"sample, {self.lambda_value}, {sample_cnt}, {self.n}, {self.sigma}")
        return self.x_mean + self.sigma * (np.random.normal(
            0, 1 if mask is None else mask,
            (self.lambda_value if sample_cnt is None else sample_cnt, self.n)
        ) @ self.d.T) @ self.b.T

    def reset_to_init(self, reset_generation=False):
        self.reset(sigma=self.init_sigma, cov_mat=self.init_cov, reset_generation=reset_generation)

    def eliminate_early_converge(self):
        if self.sorted_cost[0] > self.cost_good and self.sigma * self.max_cov < self.early_converge_eps:
            logging.info(f"Early Converge. Reset sigma to {self.init_sigma}. Reset C to Identity.")
            self.sigma = self.init_sigma
            self.reset_update_param(self.init_cov.copy())

    def update(self, argx_in: Optional[np.ndarray] = None, cost_in: Optional[np.ndarray] = None):
        self._max_cov: Optional[float] = None

        if argx_in is None:
            argx: np.ndarray = self.sample()
            cost: np.ndarray = self.cost_func(argx)
        else:
            argx: np.ndarray = argx_in.copy()
            cost: np.ndarray = cost_in.copy()
        cost_order: np.ndarray = np.argsort(cost)
        self.sorted_argx[:] = argx[cost_order[:self.mu]]
        self.sorted_cost[:] = cost[cost_order[:self.mu]]
        if self.sorted_cost[0] < self.history_best_cost:
            self.history_best_cost = float(self.sorted_cost[0])
        # self.eliminate_early_converge()

        x_mean_old: np.ndarray = self.x_mean.copy()
        # Modify by Zhenhua Song: Maybe we should add learning rate here...
        self.x_mean[:] = self.x_mean + self.learning_rate * (self.sorted_argx.T @ self.weights - self.x_mean)  # (n,)

        # Step-size control
        self.p_sigma[:] = (1.0 - self.c_sigma) * self.p_sigma + \
                          np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mueff) * \
                          self.inv_sqrt_c @ ((self.x_mean - x_mean_old) / self.sigma)

        hsig = np.sum(self.p_sigma ** 2) / \
               np.sqrt(1.0 - np.power(1 - self.c_sigma, 2.0 * self.counteval() / self.lambda_value)) / self.n < \
               2.0 + 4.0 / (self.n + 1)
        self.p_c *= (1.0 - self.c_c)
        if hsig:
            self.p_c += np.sqrt(self.c_c * (2.0 - self.c_c) * self.mueff) * ((self.x_mean - x_mean_old) / self.sigma)

        y_old: np.ndarray = (1.0 / self.sigma) * (self.sorted_argx - x_mean_old)
        c_old: np.ndarray = self.c.copy()
        self.c[:] = (1.0 - self.c_1 - self.c_mu) * self.c + self.c_1 * (
                self.p_c[:, np.newaxis] @ self.p_c[np.newaxis, :] + self.c_mu * y_old.T @ np.diag(self.weights) @ y_old)

        if not hsig:
            self.c += self.c_1 * self.c_c * (2.0 - self.c_c) * c_old

        new_sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        if not np.isnan(new_sigma) and not np.isinf(new_sigma):
            self.sigma = new_sigma

        if self.counteval() - self.eigeneval > self.lambda_value / (self.c_1+self.c_mu) / self.n / 10:
            self.eigeneval: Optional[int] = self.counteval()
            self.calc_cov_attrs()

        self.g += 1
        # self.forward_g += 1
        # assert self.c.dtype == np.float64 and self.x_mean.dtype == np.float64

    def enlarge_cov(self):
        # if max(cov) is too small (e.g. < 1e-3), adjust sigma and cov matrix..
        if self.max_cov < 5e-4:
            ratio: float = 10
            self._max_cov = None
            self.c *= ratio
            self.sigma /= ratio
            self.calc_cov_attrs()


def simple_test():
    import matplotlib.pyplot as plt

    def draw_hmap():
        n = 256
        x_ = np.linspace(-10, 10, n)
        y_ = np.linspace(-10, 10, n)
        x, y = np.meshgrid(x_, y_)
        plt.contourf(x, y, x ** 2 + y ** 2, 8, alpha=0.75, cmap='cool')

    cma = CMAUpdate(np.array([5, 3]), lambda_val=100, cost_good=1e-3)  #lambda_val=50, mu=25)
    for i in range(1000):
        cma.update()
        draw_hmap()
        plt.scatter(cma.sorted_argx[:, 0], cma.sorted_argx[:, 1])
        plt.show()
        print("=========", i)
        print(cma.x_mean)
        print(cma.c)
        print(cma.sigma)


if __name__ == "__main__":
    simple_test()

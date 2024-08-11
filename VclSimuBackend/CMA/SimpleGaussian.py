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
import matplotlib.pyplot as plt
import numpy as np


# It can not converge...
def rastrigin(x, y):
    return 20 + x + y - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))


def circle(x, y):
    return x**2 + y ** 2


def schaffer2(x, y):
    return 0.5 + (np.sin(x**2 - y**2) ** 2 - 0.5) / (1 + 0.001 * (x**2 + y ** 2))


def draw_hmap(func):
    n = 256
    x = np.linspace(-10, 10, n)
    y = np.linspace(-10, 10, n)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, func(X, Y), 8, alpha=0.75, cmap='cool')
    # C = plt.contour(X, Y, height(X, Y), 8)
    # plt.clabel(C, inline=True, fontsize=10)


# Min Object Function
def simple_gaussian(func):
    max_iter = 100
    num_sample = 1000
    lam = 400
    mu = np.array([5, 4])
    dim = mu.size
    sigma = 0.5
    sigma_max = 10
    for t in range(1, max_iter):
        d = mu + sigma * np.random.multivariate_normal(np.zeros_like(mu), np.eye(dim), num_sample)
        val = func(d[:, 0], d[:, 1])
        bound = np.sort(val)[lam]
        elite = np.argwhere(val < bound).flatten()
        sigma = np.minimum(np.mean(np.linalg.norm(d[elite] - mu, axis=1)), sigma_max)
        mu = np.mean(d[elite], axis=0)

        draw_hmap(func)
        plt.scatter(d[:, 0], d[:, 1])
        plt.show()


# simple_gaussian(circle)

import numpy as np


class SingleBall:
    """
    Add 1 dim force to ball to reach target position at time T.
    """
    def __init__(self, mass: float = 1.0, x0: float = 0.0, xt: float = 2.0,
                 total_t:int = 20, dt: float = 0.05, gravity: float = -9.8,
                 alpha1: float = 2000.0, alpha2: float = 0.01, max_iter: int = 200, lr: float = 0.01):
        self.mass = mass
        self.x0 = x0
        self.xt = xt
        self.total_t = total_t
        self.dt = dt
        self.gravity = gravity
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.lr = lr

        self.acc = np.zeros(total_t+1)
        self.velo = np.zeros(total_t+1)
        self.pos = np.zeros(total_t+1)
        self.pos[0] = self.x0
        self.force = np.zeros(total_t+1)

        self.partial_l_acc = np.zeros_like(self.acc)
        self.partial_l_velo = np.zeros_like(self.velo)
        self.partial_l_pos = np.zeros_like(self.pos)
        self.partial_l_force = np.zeros_like(self.force)

        self.loss = 0.0

    def clear_grad(self):
        self.partial_l_acc[:] = 0.0
        self.partial_l_velo[:] = 0.0
        self.partial_l_pos[:] = 0.0
        self.partial_l_force[:] = 0.0

    def forward_no_grad(self):
        self.acc[:] = self.force[:] / self.mass + self.gravity
        for i in range(self.total_t):
            self.velo[i+1] = self.velo[i] + self.acc[i] * self.dt
            self.pos[i+1] = self.pos[i] + self.velo[i+1] * self.dt

        self.loss = self.alpha1 * (self.pos[-1] - self.xt) ** 2 + self.alpha2 * np.sum(self.force ** 2)

    def forward_with_grad(self):
        self.clear_grad()
        self.forward_no_grad()

        self.partial_l_pos[-1] += 2 * self.alpha1 * (self.pos[-1] - self.xt)
        self.partial_l_force[:] += 2 * self.alpha2 * self.force

        for i in range(self.total_t-1, -1, -1):
            if i > 0:
                self.partial_l_pos[i] += self.partial_l_pos[i+1]
            self.partial_l_velo[i+1] += self.partial_l_pos[i+1] * self.dt

            self.partial_l_velo[i] += self.partial_l_velo[i+1]
            self.partial_l_acc[i] += self.partial_l_velo[i] * self.dt

            self.partial_l_force[i] += self.partial_l_acc[i] / self.mass

    def optimize(self):
        for i in range(1, self.max_iter + 1):
            self.forward_with_grad()
            self.force[:] -= self.lr * self.partial_l_force[:]  # only control force is optimized.
            self.forward_no_grad()
            if i % 40 == 0:
                self.lr *= 0.5
            print(f"iter {i}, loss = {self.loss}, xt = {self.pos[-1]}\n"
                  f"pos = {self.pos}, \n"
                  f"force = {self.force}\n"
                  f"velo = {self.velo}, \n"
                  f"acc = {self.acc}\n"
                  f"partial_pos = {self.partial_l_pos},\n"
                  f"partial_velo = {self.partial_l_velo}\n"
                  f"partial_acc = {self.partial_l_velo},\n "
                  f"partial_force = {self.partial_l_force}\n\n")


if __name__ == "__main__":
    np.set_printoptions(threshold=1000000, linewidth=100000, precision=3)
    ball = SingleBall()
    ball.optimize()

import numpy as np
import copy
from scipy.integrate import odeint


class MaxLikelihood:
    def __init__(self, Q, R, H,
                 x_0, P, t, a,
                 F, Gamma, nu, u, N):
        self._Q = Q
        self._R = R
        self.x_0 = x_0
        self._P = P
        self.t = t
        self.F = F
        self._H = H
        self.N = N
        self.nu = nu
        self.Gamma = Gamma
        self.u = u
        self.a = a

    @property
    def x0(self):
        return self.x_0

    @property
    def p0(self):
        return self._P

    @property
    def R(self):
        return self._R

    @property
    def Q(self):
        return self._Q

    @property
    def H(self):
        return self._H

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, _y):
        if _y:
            self._y = copy.copy(_y)

    def calc_P(self, P, t):
        return self.F * P + P * self.F + self.Gamma * self.Q * self.Gamma

    def eq_for_p(self, t, x0):
        "TODO сделать t именнованным кортежем"
        t = np.linspace(t[0], t[1], 10)
        x = odeint(self.calc_P, x0, t)
        x = np.array(x).flatten()
        return x[len(x) - 1]

    def _dxdt(self, x, t, tetha_1, tetha_2, u):
        return tetha_1 * x[0] + u * tetha_2

    def eq_for_x(self, t, x, u, estimated_values):
        t_a = np.linspace(t[0], t[1], 10)
        x0 = x
        x = odeint(self._dxdt, [x0], t_a, args=(estimated_values[0], estimated_values[1], u))
        x = np.array(x).flatten()
        return x[len(x) - 1]

    def estimate(self, params):
        x_tk = self.x_0
        p_tk = self._P
        result = self.N * np.log(2 * np.pi)
        for i in range(self.N):
            step = self.t[i:i+2]
            x = self.eq_for_x(step, x_tk, self.u[i], params)
            p = self.eq_for_p(step, p_tk)
            e_tk = self._y[i] - self.H * x
            B = self.H * p * self.H + self.R
            K = p * self.H / B
            x_tk = x + K * e_tk
            p_tk = (1 - K - self.H) * p
            result += e_tk * (1 / B) * e_tk + 0.5 * self.nu * B
        result /= 2
        return result




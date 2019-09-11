import numpy as np
import copy
from scipy.integrate import odeint
from collections import namedtuple


class MaxLikelihood:
    def __init__(self, Q, R, H,
                 x_0, P, t,
                 F, Gamma, nu, u, N, phi):
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
        self.phi = phi

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
        p = np.diagflat(P)
        return np.diag(self.F @ p + p @ self.F.T + self.Gamma @ self.Q @ self.Gamma)

    def eq_for_p(self, time_step, x0):
        t = np.linspace(time_step.start, time_step.end, 10)
        x = odeint(self.calc_P, np.diag(x0), t)
        return np.diagflat([x[:, i][-1] for i in range(len(x0))])

    def _dxdt(self, x, t, tetha_1, tetha_2, u):
        return tetha_1 * x + self.phi * u * tetha_2

    def _dxdt_2(self, x, t, tetha_1, tetha_2, u):
        dxdt_1 = x[1]
        dxdt_2 = -tetha_1 * x[0] + tetha_2 * x[1] + u * tetha_1
        return [dxdt_1, dxdt_2]

    def eq_for_x(self, time_step, x, u, estimated_values):
        t_tk = np.linspace(time_step.start, time_step.end, 10)
        x0 = x
        x = odeint(self._dxdt_2, x0, t_tk, args=(estimated_values[0], estimated_values[1], u))
        return [x[:, i][-1] for i in range(len(x0))]

    def estimate(self, params):
        x_tk = self.x_0
        p_tk = self._P
        TimeStep = namedtuple('TimeStep', ['start', 'end'])
        result = self.N * np.log(2 * np.pi)
        for i in range(self.N):
            step = TimeStep(self.t[i], self.t[i + 1])
            x = self.eq_for_x(step, x_tk, self.u[i], params)
            p = self.eq_for_p(step, p_tk)
            e_tk = self._y[i].T - self.H @ x
            B = self.H @ p @ self.H + self.R
            K = p @ self.H @ np.linalg.inv(B)
            x_tk = x + K @ e_tk
            p_tk = (np.eye(self.F.shape[0]) - K @ self.H) * p
            result += e_tk @ np.linalg.inv(B) @ e_tk.T + 0.5 * self.nu * np.log(np.linalg.det(B))
        result /= 2
        return result




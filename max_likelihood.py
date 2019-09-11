import numpy as np
import copy
from scipy.integrate import odeint
from collections import namedtuple


class MaxLikelihood:
    def __init__(self, Q, R, H,
                 x_0, P, t,
                 F, Gamma, nu,
                 u, N, phi, update_dict):
        self._Q = Q
        self._R = R
        self.x_0 = x_0
        self._P = P
        self.t = t
        self._F = F
        self._H = H
        self.N = N
        self.nu = nu
        self.Gamma = Gamma
        self.u = u
        self.phi = phi
        self.update_dict = update_dict

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

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, _F):
        if _F:
            self._F = copy.copy(_F)

    def update(self, params):
        for var in self.update_dict:
            if self.__dict__.get(var) is not None:
                self.__dict__[var] = self.update_dict[var](params)

    def dxdt(self, x, t, u):
        return self._F @ x + np.array([u]) @ self.phi

    def calc_P(self, P, t):
        p = np.diagflat(P)
        return np.diag(self._F @ p + p @ self._F.T + self.Gamma @ self.Q @ self.Gamma.T)

    def eq_for_p(self, time_step, x0):
        t = np.linspace(time_step.start, time_step.end, 10)
        x = odeint(self.calc_P, np.diag(x0), t)
        return np.diagflat([x[:, i][-1] - (x[:, i][0]) for i in range(len(x0))])

    def eq_for_x(self, time_step, x, u, estimated_values):
        t_tk = np.linspace(time_step.start, time_step.end, 31)
        x0 = x
        x = odeint(self.dxdt, x0, t_tk, args=(u,))
        return [x[:, i][-1] for i in range(len(x0))]

    def estimate(self, params):
        if self.dxdt is None:
            raise Exception("Before start estimate, please init methods.")

        self.update(params)
        x_tk = self.x_0
        p_tk = self._P
        TimeStep = namedtuple('TimeStep', ['start', 'end'])
        result = self.N / 2 * np.log(2 * np.pi)
        for i in range(self.N):
            step = TimeStep(self.t[i], self.t[i + 1])
            x = self.eq_for_x(step, x_tk, self.u[i], params)
            p = self.eq_for_p(step, p_tk)
            e_tk = self._y[i] - self.H @ x
            B = self.H @ p @ self.H + self.R
            K = p @ self.H / B
            x_tk = x + K @ e_tk
            p_tk = (np.eye(self._F.shape[0]) - K @ self.H) * p
            result += e_tk @ (1 / B) @ e_tk.T + 0.5 * self.nu * np.log(np.linalg.det(B))
        result /= 2
        return result




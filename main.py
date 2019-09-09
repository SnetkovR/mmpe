import numpy as np
import copy
from max_likelihood import MaxLikelihood
from scipy.optimize import minimize
from collections import namedtuple


def noise(sigma, N):
    return [np.random.normal(0, np.sqrt(sigma)) for i in range(N)]


TimeStep = namedtuple('TimeStep', ['start', 'end'])
tetha = [-1, 1]
t = list(range(11))
u = copy.copy(t)
estimator = MaxLikelihood(Q=0.1, R=0.3, P=0.1,
                          x_0=0, F=tetha[0], H=1, Gamma=1,
                          t=t, nu=1, u=u, N=10, a=tetha[1])

x_array = []
x1 = np.random.normal(estimator.x0, np.sqrt(estimator.p0))
for i in range(len(t) - 1):
    step = TimeStep(t[i], t[i + 1])
    x_array.append(estimator.eq_for_x(step, x1, u[i], tetha))
    x1 = x_array[i]
print(x_array)
x_noisy_array = x_array + noise(estimator.Q,  len(x_array))

y_array = estimator.H * x_noisy_array + noise(estimator.R, len(x_array))
estimator.y = y_array

new_array = minimize(estimator.estimate, [-2, 2], method='SLSQP', bounds=[[-3, -0.5], [0.5, 3]])

print(new_array)

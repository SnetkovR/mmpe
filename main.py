import numpy as np
import copy
from max_likelihood import MaxLikelihood
from scipy.optimize import minimize
from collections import namedtuple


def noise(sigma, N):
    mean = np.zeros(sigma.shape[0])
    return [np.random.multivariate_normal(mean, np.sqrt(sigma)) for i in range(N)]





def _dxdt_two_dim(x, t, tetha_1, tetha_2, u):
    dxdt_1 = x[1]
    dxdt_2 = -tetha_1 * x[0] + tetha_2 * x[1] + u * tetha_1
    return [dxdt_1, dxdt_2]


def one_dim(size):
    TimeStep = namedtuple('TimeStep', ['start', 'end'])
    tetha = [-1, 1]
    t = list(range(size + 1))
    u = [1] * size
    update = {"_F": lambda params: params[0] * np.eye(1),
              "phi": lambda params: params[1] * np.eye(1)}

    estimator = MaxLikelihood(Q=0.1*np.eye(1),
                              R=0.3*np.eye(1),
                              P=0.1*np.eye(1),
                              x_0=np.zeros(1),
                              F=tetha[0] * np.eye(1),
                              H=np.eye(1),
                              Gamma=np.eye(1),
                              t=t,
                              nu=1,
                              u=u,
                              N=size,
                              phi=tetha[1] * np.eye(1),
                              update_dict=update)

    x_array = []
    x1 = np.random.normal(estimator.x0, estimator.p0[0])
    for i in range(len(t) - 1):
        step = TimeStep(t[i], t[i + 1])
        x_array.append(estimator.eq_for_x(step, x1, u[i], tetha))
        x1 = x_array[i]
    print(x_array)
    x_noisy_array = x_array + noise(estimator.Q,  len(x_array))

    y_array = [estimator.H @ x for x in x_noisy_array]
    y_array += noise(estimator.R, len(x_array))
    print(y_array)
    estimator.y = y_array
    new_array = minimize(estimator.estimate, [-1.5, 1], method='SLSQP', bounds=[[-3, 0.05], [0.05, 3]])
    return new_array


def two_dim(size):
    TimeStep = namedtuple('TimeStep', ['start', 'end'])
    tetha = [1, -1]
    t = list(range(size + 1))
    u = [1] * size
    update = {"_F": lambda params: np.array([[0, 1], [-params[0], params[1]]]),
              "phi": lambda params: np.array([[0], [tetha[1]]])}
    estimator = MaxLikelihood(Q=0.1*np.eye(2),
                              R=0.3*np.eye(2),
                              P=np.array([[0.05, 0], [0, 0.05]]),
                              x_0=np.array([-5, 0]),
                              F=np.array([[0, 1], [-tetha[0], tetha[1]]]),
                              H=np.eye(2),
                              Gamma=np.eye(2),
                              t=t,
                              nu=1,
                              u=u,
                              N=size,
                              phi=np.array([[0], [tetha[1]]]),
                              update_dict=update)
    estimator._init_methods(_dxdt_two_dim)
    x_array = []
    x1 = np.random.multivariate_normal(estimator.x0, np.sqrt(estimator.p0))
    for i in range(len(t) - 1):
        step = TimeStep(t[i], t[i + 1])
        x_array.append(estimator.eq_for_x(step, x1, u[i], tetha))
        x1 = x_array[i]
    print(x_array)
    x_noisy_array = x_array + noise(estimator.Q,  len(x_array))

    y_array = [estimator.H @ x for x in x_noisy_array]
    y_array += noise(estimator.R, len(x_array))
    estimator.y = y_array

    new_array = minimize(estimator.estimate, [2, -2], method='SLSQP', bounds=[[0.5, 3], [-3, -0.5]])
    print(new_array)


res = one_dim(40)
print(res)
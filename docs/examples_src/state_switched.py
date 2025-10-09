"""
Optimal Transfer With State Trigger
===================================
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from _sgm_test_util import LTI_plot

import condor as co


class DblInt(co.ODESystem):
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])

    x = state(shape=A.shape[0])
    mode = state()

    p1 = parameter()
    p2 = parameter()

    u = modal()

    dot[x] = A @ x + B * u


class Accel(DblInt.Mode):
    condition = mode == 0.0
    action[u] = 1.0


class Switch1(DblInt.Event):
    function = x[0] - p1
    update[mode] = 1.0


class Decel(DblInt.Mode):
    condition = mode == 1.0
    action[u] = -1.0


class Switch2(DblInt.Event):
    function = x[0] - p2
    # update[mode] = 2.
    terminate = True


from scipy.optimize import bisect


class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9.0, 0.0]
    xd = [1.0, 2.0]
    Q = np.eye(2)
    cost = trajectory_output(((x - xd).T @ (x - xd)) / 2)
    tf = 20.0

    class Options:
        rootfinder = bisect
        rootfinder_xtol = 1e-15


p0 = -4.0, -1.0
sim = Transfer(*p0)
# sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])

# %%


class MinimumTime(co.OptimizationProblem):
    p1 = variable()
    p2 = variable()
    sim = Transfer(p1, p2)
    objective = sim.cost

    class Options:
        # exact_hessian = False
        __implementation__ = co.implementations.ScipyCG


MinimumTime.set_initial(p1=p0[0], p2=p0[1])

t_start = perf_counter()
opt = MinimumTime()
t_stop = perf_counter()

print("time to run:", t_stop - t_start)
print(opt.p1, opt.p2)
print(opt._stats)

# %%

LTI_plot(opt.sim)

# %%

plt.show()

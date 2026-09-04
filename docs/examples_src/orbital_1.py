"""
Linear Covariance Analysis
==========================
"""

# %%
# This example is a simple implementation of linear covariance analysis [#geller2006]_
# to evaluate closed-loop guidance, navigation, and control performance by propagating
# navigation errors and trajectory dispersions through a single simulation pass rather
# than simulating many realizations as in a Monte Carlo analysis.
#
# We'll use the Clohessy-Wiltshire equations to simulate the relative motion between a
# target vehicle in circular orbit and a chaser vehicle.

import casadi as ca
import numpy as np

import condor as co
from condor.backend import operators as ops

I6 = ops.eye(6)
Z6 = ops.zeros((6, 6))
W = ops.concat([I6, Z6], axis=0)

I3 = ops.eye(3)
Z3 = ops.zeros((3, 3))
V = ops.concat([Z3, I3, ops.zeros((6, 3))], axis=0)

# %%
# The core of the analysis is an ODE system that represents the true state the follows
# the CW equations.


class LinCovCW(co.ODESystem):
    omega = parameter()
    scal_w = parameter(shape=6)

    x = state(shape=6)
    C = state(shape=(12, 12), symmetric=True)

    initial[x] = parameter(shape=x.shape, name="initial_x")
    initial[C] = parameter(shape=C.shape, name="initial_C")

    # [0, 0, 0,            1, 0, 0],
    # [0, 0, 0,            0, 1, 0],
    # [0, 0, 0,            0, 0, 1],
    # [0, 0, 0,            0, 0, 2*omega],
    # [0, -omega**2, 0,    0, 0, 0],
    # [0, 0, 3*omega**2,   -2*omega, 0, 0]
    Acw = ops.zeros((6, 6))
    Acw[:3, 3:] = ops.eye(3)
    Acw[3, 5] = 2 * omega
    Acw[4, 1] = -(omega**2)
    Acw[5, 2] = 3 * omega**2
    Acw[5, 3] = -2 * omega

    Scal_w = ops.diag(scal_w)
    Cov_prop_offset = W @ Scal_w @ W.T

    Fcal = ops.zeros((12, 12))
    Fcal[:6, :6] = Acw
    Fcal[6:, 6:] = Acw

    dot[x] = Acw @ x
    dot[C] = Fcal @ C + C @ Fcal.T + Cov_prop_offset


# %%
# A discrete targeting maneuver can be implemented as an event on the ODE system that
# instantaneously updates the relative velocity as well as the augmented covariance.


class MajorBurn(LinCovCW.Event):
    #: Target position
    rd = parameter(shape=3)
    #: Time ignition
    tig = parameter()
    #: Time end maneuver
    tem = parameter()

    scal_v = parameter(shape=3)

    Delta_v_mag = state()
    Delta_v_disp = state()

    at_time = tig
    td = tem - tig

    stm = ops.zeros((6, 6))
    stm[0, 0] = 1
    stm[0, 2] = 6 * omega * td - 6 * ops.sin(omega * td)
    stm[0, 3] = -3 * td + 4 * ops.sin(omega * td) / omega
    stm[0, 5] = 2 * (1 - ops.cos(omega * td)) / omega
    stm[1, 1] = ops.cos(omega * td)
    stm[1, 4] = ops.sin(omega * td) / omega
    stm[2, 2] = 4 - 3 * ops.cos(omega * td)
    stm[2, 3] = 2 * (ops.cos(omega * td) - 1) / omega
    stm[2, 5] = ops.sin(omega * td) / omega
    stm[3, 2] = 6 * omega * (1 - ops.cos(omega * td))
    stm[3, 3] = 4 * ops.cos(omega * td) - 3
    stm[3, 5] = 2 * ops.sin(omega * td)
    stm[4, 1] = -omega * ops.sin(omega * td)
    stm[4, 4] = ops.cos(omega * td)
    stm[5, 2] = 3 * omega * ops.sin(omega * td)
    stm[5, 3] = -2 * ops.sin(omega * td)
    stm[5, 5] = ops.cos(omega * td)

    T_pp = stm[:3, :3]
    T_pv = stm[:3, 3:]
    T_pv_inv = ca.solve(T_pv, ops.eye(3))

    Delta_v = (T_pv_inv @ rd - T_pv_inv @ T_pp @ x[:3, [0]]) - x[3:, [0]]

    update[Delta_v_mag] = Delta_v_mag + ca.norm_2(Delta_v)
    update[x] = x + ops.concat([ops.zeros((3, 1)), Delta_v])

    DG = ca.vertcat(ops.zeros((3, 6)), ca.horzcat(-(T_pv_inv @ T_pp), -I3))
    Dcal = ca.vertcat(
        ca.horzcat(I6, DG),
        ca.horzcat(Z6, I6 + DG),
    )

    Scal_v = ops.diag(scal_v)

    update[C] = Dcal @ C @ Dcal.T + V @ Scal_v @ V.T

    Mc = DG @ ops.concat([Z6, I6], axis=1)
    sigma_Dv__2 = ca.trace(Mc @ C @ Mc.T)

    update[Delta_v_disp] = Delta_v_disp + ca.sqrt(sigma_Dv__2)


# %%
# A Terminating event at the end of the maneuver


class Terminate(LinCovCW.Event):
    terminate = True
    at_time = MajorBurn.tem


# %%
# Trajectory analysis to simulate the system


class Sim(LinCovCW.TrajectoryAnalysis):
    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
    # dispersion)
    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
    tot_Delta_v_disp = trajectory_output(Delta_v_disp)

    # tf = parameter()

    Mr = ca.horzcat(I3, ca.MX(3, 9))
    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))


# %%
# Simulate


def make_C0(sigma_p, sigma_v, rho, sigma_p_nav=None, sigma_v_nav=None, rho_nav=None):
    D0 = np.zeros((6, 6))
    D0[:3, :3] = np.eye(3) * (sigma_p**2)
    D0[2, 3] = rho * sigma_p * sigma_v
    D0[3, 2] = rho * sigma_p * sigma_v
    D0[3:, 3:] = np.eye(3) * (sigma_v**2)

    sigma_p_nav = sigma_p_nav or sigma_p
    sigma_v_nav = sigma_v_nav or sigma_v
    rho_nav = rho_nav or rho

    P0 = np.zeros((6, 6))
    P0[:3, :3] = np.eye(3) * (sigma_p_nav**2)
    P0[2, 3] = rho_nav * sigma_p_nav * sigma_v_nav
    P0[3, 2] = rho_nav * sigma_p_nav * sigma_v_nav
    P0[3:, 3:] = np.eye(3) * (sigma_v_nav**2)

    C = np.concat(
        (np.concat((D0, D0), axis=1), np.concat((D0, D0 + P0), axis=1)), axis=0
    )
    return C


initial_C = make_C0(100 / 3, 0.11 / 3, 0.9, 10 / 3, 0.011 / 3, 0.9)

sim_kwargs = dict(
    omega=0.00114,
    scal_w=[0.0] * 3 + [4.8e-10] * 3,
    scal_v=[2.5e-7] * 3,
    initial_x=[-2000.0, 0.0, 1000.0, 1.71, 0.0, 0.0],
    initial_C=make_C0(100 / 3, 0.11 / 3, 0.9, 10 / 3, 0.011 / 3, 0.9),
    rd=[500.0, 0.0, 0.0],
)

out = Sim(**sim_kwargs, tig=156.7, tem=156.7 + 15 * 60)

print(out.final_pos_disp)

# %%

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_traj(sim, x_idx=0, y_idx=2):
    """create relative motion plot in the vertical plane"""
    fig, ax = plt.subplots(constrained_layout=True)
    ax.invert_xaxis()
    ax.set_xlabel("V-bar (m)")
    ax.invert_yaxis()
    ax.set_ylabel("R-bar (m)")
    ax.grid(True)
    ax.set_aspect("equal")
    plt.xlim(1000, -2500)
    plt.ylim(1500, -500)
    fig.set_size_inches(7.4, 4.2)

    sim_color = "C0"
    plt.plot([0.0], [0.0], "ok")
    xs = sim.x[x_idx].squeeze()
    ys = sim.x[y_idx].squeeze()
    ax.plot(xs, ys, "o-", color=sim_color)
    ax.plot(xs[0], ys[0], "og")

    n_std = 3

    for pos_vel, cov in zip(sim.x.T.squeeze(), sim.C.T):
        pos = pos_vel[[x_idx, y_idx]]
        C = cov[[x_idx, y_idx]][:, [x_idx, y_idx]]
        lambda_, v = np.linalg.eigh(C)
        lambda_ = np.sqrt(lambda_)

        ellipse = Ellipse(
            pos,
            width=lambda_[0] * n_std * 2,
            height=lambda_[1] * n_std * 2,
            angle=np.degrees(np.arctan2(*v[:, 0][::-1])),
            edgecolor=sim_color,
            facecolor="none",
        )
        ax.add_patch(ellipse)
    return ax


plot_traj(out)

# TODO
# DV_idx = Sim.trajectory_output.flat_index(Sim.tot_Delta_v_mag)
# tig_idx = Sim.parameter.flat_index(Sim.tig)
# tem_idx = Sim.parameter.flat_index(Sim.tem)
# init_jac = Sim.implementation.callback.jac_callback(Sim.implementation.callback.p, [])
# print("init grad  wrt tig", init_jac[DV_idx, tig_idx])
# print("init grad  wrt tem", init_jac[DV_idx, tem_idx])
# """
# init grad  wrt tig 0.0209833
# init grad  wrt tem -0.0260249
# """


# %%


class Hohmann(co.OptimizationProblem):
    tig = variable(initializer=200.0)
    tem = variable(initializer=500.0)

    sim = Sim(tig=tig, tem=tem, **sim_kwargs)

    objective = sim.tot_Delta_v_mag

    constraint(tem > tig + 30)
    constraint(tig > 0.1)

    class Options:
        # TODO
        # __implementation__ = co.implementations.ScipyTrustConstr
        exact_hessian = False


from time import perf_counter

hoh_start = perf_counter()
hohmann = Hohmann()
hoh_stop = perf_counter()

print(hohmann._stats)
print((hohmann.tem - hohmann.tig) * hohmann.sim.omega * 180 / np.pi)

# TODO
# opt_jac = Sim.implementation.callback.jac_callback(Sim.implementation.callback.p, [])
# print("opt grad  wrt tig", opt_jac[DV_idx, tig_idx])
# print("opt grad  wrt tem", opt_jac[DV_idx, tem_idx])
# """
# opt grad  wrt tig -4.48258e-09
# opt grad  wrt tem -1.47125e-09
# """

# %%

hohmann_sim = Sim(**sim_kwargs, tig=hohmann.tig, tem=hohmann.tem)
print(hohmann_sim.tot_Delta_v_disp)
print(hohmann_sim.final_pos_disp)
print(hohmann.tig, hohmann.tem)
print("time:", hoh_stop - hoh_start)

# %%

plot_traj(hohmann_sim)

# %%
#


class TotalDeltaV(co.OptimizationProblem):
    tig = variable(initializer=200.0)
    tem = variable(initializer=500.0)
    constraint(tem - tig, lower_bound=30.0)
    constraint(tig, lower_bound=0.0)
    sim = Sim(tig=tig, tem=tem, **sim_kwargs)

    # TODO: adding a parameter and constraint to existing problem SHOULD be done by
    # inheritance... I suppose the originally Hohmann model could easily be written to
    # include more parameters to solve all permutations of this problem... weights for
    # each output, upper bounds for each output (and combinations?)
    # what about including a default for a paremter at a model level? no, just make a
    # dict like unbounded_kwargs to fill in with a large number/inf
    pos_disp_max = parameter()
    constraint(sim.final_pos_disp - pos_disp_max, upper_bound=0.0)

    objective = sim.tot_Delta_v_mag + 3 * sim.tot_Delta_v_disp

    class Options:
        exact_hessian = False


total_delta_v = TotalDeltaV(pos_disp_max=1000)

# %%

tot_delta_v_sim = Sim(**sim_kwargs, tig=total_delta_v.tig, tem=total_delta_v.tem)
plot_traj(tot_delta_v_sim)

# %%

total_delta_v_constrained = TotalDeltaV(pos_disp_max=10.0)

# %%

tot_delta_v_constrained_sim = Sim(
    **sim_kwargs, tig=total_delta_v_constrained.tig, tem=total_delta_v_constrained.tem
)
plot_traj(tot_delta_v_constrained_sim)

# %%


# %%

print("\n" * 2, "unconstrained Delta v")
print(total_delta_v._stats)
print((total_delta_v.tem - total_delta_v.tig) * total_delta_v.sim.omega * 180 / np.pi)
print(tot_delta_v_sim.final_pos_disp)

print("\n" * 2, "constrained Delta v")
print(total_delta_v_constrained._stats)
print(
    (total_delta_v_constrained.tem - total_delta_v_constrained.tig)
    * total_delta_v_constrained.sim.omega
    * 180
    / np.pi
)
print(tot_delta_v_constrained_sim.final_pos_disp)


plt.show()

# %%
# .. rubric:: References
#
# .. [#geller2006] Geller, D. K., "Linear Covariance Techniques for Orbital Rendezvous
#    Analysis and Autonomous Onboard Mission Planning," Journal of Guidance, Control,
#    and Dynamics, Vol. 29, No. 6, 2006, pp. 1404–1414. https://doi.org/10.2514/1.19447

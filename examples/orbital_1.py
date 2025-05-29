import numpy as np
import condor as co
import casadi as ca

from condor.backends.casadi.implementations import OptimizationProblem

I6 = ca.MX.eye(6)
Z6 = ca.MX(6, 6)
W = ca.vertcat(I6, Z6)

#I3 = np.eye(3)
#Z3 = np.zeros((3,3))
#V = np.vstack((Z3, I3, np.zeros((6,3))))
#V = ca.sparsify(V)

I3 = ca.MX.eye(3)
Z3 = ca.MX(3,3)
V = ca.vertcat(Z3, I3, ca.MX(6,3))


numeric_constants = False
class LinCovCW(co.ODESystem):
    if numeric_constants:
        #omega = 0.0011
        omega = 0.00114
        scal_w = ca.MX.ones(6)
        scal_v = ca.MX.ones(3)
    else:
        omega = parameter()
        scal_w = parameter(shape=6)
        scal_v = parameter(shape=3)

    initial_x = parameter(shape=6)
    initial_C = parameter(shape=(12,12))

    Acw = ca.MX(6,6)

    """[

    [0, 0, 0,            1, 0, 0],
    [0, 0, 0,            0, 1, 0],
    [0, 0, 0,            0, 0, 1],

    [0, 0, 0,            0, 0, 2*omega],
    [0, -omega**2, 0,    0, 0, 0],
    [0, 0, 3*omega**2,   -2*omega, 0, 0]

    ] """

    Acw[0,3] = 1
    Acw[1,4] = 1
    Acw[2,5] = 1

    Acw[3,5] = 2*omega
    Acw[4,1] = -omega**2
    Acw[5,2] = 3*omega**2
    Acw[5,3] = -2*omega

    x = state(shape=6)
    C = state(shape=(12,12))
    Delta_v_mag = state()
    Delta_v_disp = state()

    Scal_w = ca.diag(scal_w)
    Cov_prop_offset = W @ Scal_w @ W.T

    Scal_v = ca.diag(scal_v)
    Cov_ctrl_offset = V @ Scal_v @ V.T

    Fcal = ca.MX(12,12)
    Fcal[:6, :6] = Acw
    Fcal[6:, 6:] = Acw

    initial[x] = initial_x
    initial[C] = initial_C

    dot[x] = Acw @ x
    dot[C] = Fcal @ C + C @ Fcal.T + Cov_prop_offset


sin = ca.sin
cos = ca.cos


class MajorBurn(LinCovCW.Event):
    rd = parameter(shape=3) # desired position
    tig = parameter() # time ignition
    tem = parameter() # time end maneuver

    #function = t - tig
    at_time = [tig]

    t_d = tem - tig

    stm = ca.MX(6,6)
    stm[0,0] = 1
    stm[0,2] = 6*omega*t_d - 6*sin(omega*t_d)
    stm[0,3] = -3*t_d + 4*sin(omega*t_d)/omega
    stm[0,5] = 2*(1 - cos(omega*t_d))/omega
    stm[1,1] = cos(omega*t_d)
    stm[1,4] = sin(omega*t_d)/omega
    stm[2,2] = 4 - 3*cos(omega*t_d)
    stm[2,3] = 2*(cos(omega*t_d) - 1)/omega
    stm[2,5] = sin(omega*t_d)/omega
    stm[3,2] = 6*omega*(1 - cos(omega*t_d))
    stm[3,3] = 4*cos(omega*t_d) - 3
    stm[3,5] = 2*sin(omega*t_d)
    stm[4,1] = -omega*sin(omega*t_d)
    stm[4,4] = cos(omega*t_d)
    stm[5,2] = 3*omega*sin(omega*t_d)
    stm[5,3] = -2*sin(omega*t_d)
    stm[5,5] = cos(omega*t_d)
    T_pp = stm[:3, :3]
    T_pv = stm[:3, 3:]
    T_pv_inv = ca.solve(T_pv, ca.MX.eye(3))

    Delta_v = (T_pv_inv @ rd - T_pv_inv@T_pp @ x[:3, 0]) - x[3:, 0]

    update[Delta_v_mag] = Delta_v_mag + ca.norm_2(Delta_v)
    update[x]  = x + ca.vertcat(Z3, I3) @ (Delta_v)

    DG = ca.vertcat(
        ca.MX(3,6),
        ca.horzcat(-(T_pv_inv@T_pp), -I3)
    )
    Dcal = ca.vertcat(
        ca.horzcat(I6, DG),
        ca.horzcat(Z6, I6 + DG),
    )

    update[C] = Dcal @ C @ Dcal.T + Cov_ctrl_offset

    Mc = DG @ ca.horzcat(Z6, I6)
    sigma_Dv__2 = ca.trace( Mc @ C @ Mc.T)

    update[Delta_v_disp] = Delta_v_disp + ca.sqrt(sigma_Dv__2)


class Terminate(LinCovCW.Event):
    terminate = True
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    #function = t - MajorBurn.tem.backend_repr
    at_time = [MajorBurn.tem.backend_repr]


class Sim(LinCovCW.TrajectoryAnalysis):
    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
    # dispersion)
    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
    tot_Delta_v_disp = trajectory_output(Delta_v_disp)

    #tf = parameter()

    Mr = ca.horzcat(I3, ca.MX(3,9))
    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))

    class Casadi(co.Options):
        #state_rtol = 1E-9
        #state_atol = 1E-15
        #adjoint_rtol = 1E-9
        #adjoint_atol = 1E-15
        #state_max_step_size = 30.

        state_adaptive_max_step_size = 4#16
        adjoint_adaptive_max_step_size = 4


from scipy.io import loadmat
Cov_0_matlab = loadmat('P_aug_0.mat')['P_aug_0'][0]

sim_kwargs = dict(
    omega = 0.00114,
    scal_w=[0.]*3 + [4.8E-10]*3,
    scal_v=[2.5E-7]*3,
    initial_x=[-2000., 0., 1000., 1.71, 0., 0.,],
    initial_C=Cov_0_matlab,
    rd=[500., 0., 0.],
)

class Hohmann(co.OptimizationProblem):
    tig = variable(initializer=200.)
    tf = variable(initializer=500.)
    constraint(tf - tig, lower_bound=30.)
    constraint(tig, lower_bound=0.1)
    sim = Sim(
        tig=tig,
        tem=tf,
        **sim_kwargs
    )

    objective = sim.tot_Delta_v_mag

    class Casadi(co.Options):
        exact_hessian=False
        method = OptimizationProblem.Method.scipy_trust_constr


class TotalDeltaV(co.OptimizationProblem):
    tig = variable(initializer=200.)
    tf = variable(initializer=500.)
    constraint(tf - tig, lower_bound=30.)
    constraint(tig, lower_bound=0.)
    sim = Sim(
        tig=tig,
        tem=tf,
        **sim_kwargs
    )

    # TODO: adding a parameter and constraint to existing problem SHOULD be done by
    # inheritance... I suppose the originally Hohmann model could easily be written to
    # include more parameters to solve all permutations of this problem... weights for
    # each output, upper bounds for each output (and combinations?)
    # what about including a default for a parameter at a model level? no, just make a
    # dict like unbounded_kwargs to fill in with a large number/inf
    pos_disp_max = parameter()
    constraint(sim.final_pos_disp-pos_disp_max, upper_bound=0.)

    objective = sim.tot_Delta_v_mag + 3*sim.tot_Delta_v_disp

    class Casadi(co.Options):
        exact_hessian=False
        method = OptimizationProblem.Method.scipy_trust_constr


##############
from time import perf_counter

DV_idx = Sim.trajectory_output.flat_index(Sim.tot_Delta_v_mag)
tig_idx = Sim.parameter.flat_index(Sim.tig)
tem_idx = Sim.parameter.flat_index(Sim.tem)

init_sim = Sim(**sim_kwargs, tig=200., tem=500.)
init_jac = Sim.implementation.callback.jac_callback(Sim.implementation.callback.p, [])
print("init grad  wrt tig", init_jac[DV_idx, tig_idx])
print("init grad  wrt tem", init_jac[DV_idx, tem_idx])
"""
init grad  wrt tig 0.0209833
init grad  wrt tem -0.0260249
"""

hoh_start = perf_counter()
hohmann = Hohmann()
hoh_stop = perf_counter()

hohmann_sim = Sim(**sim_kwargs, tig=hohmann.tig, tem=hohmann.tf)
opt_jac = Sim.implementation.callback.jac_callback(Sim.implementation.callback.p, [])


total_delta_v  = TotalDeltaV(pos_disp_max=1000)
tot_delta_v_sim = Sim(**sim_kwargs, tig=total_delta_v.tig, tem=total_delta_v.tf)


total_delta_v_constrained  = TotalDeltaV(pos_disp_max=10.)
tot_delta_v_constrained_sim = Sim(
    **sim_kwargs, tig=total_delta_v_constrained.tig, tem=total_delta_v_constrained.tf
)

print("\n"*2,"hohmann")
print(hohmann._stats)
print((hohmann.tf - hohmann.tig)*hohmann.sim.omega*180/np.pi)
print(hohmann_sim.tot_Delta_v_disp)
print(hohmann_sim.final_pos_disp)
print(hohmann.tig, hohmann.tf)
print("time:", hoh_stop- hoh_start)

print("opt grad  wrt tig", opt_jac[DV_idx, tig_idx])
print("opt grad  wrt tem", opt_jac[DV_idx, tem_idx])
"""
opt grad  wrt tig -4.48258e-09
opt grad  wrt tem -1.47125e-09
"""

print("\n"*2,"unconstrained Delta v")
print(total_delta_v._stats)
print((total_delta_v.tf - total_delta_v.tig)*total_delta_v.sim.omega*180/np.pi)
print(tot_delta_v_sim.final_pos_disp)

print("\n"*2,"constrained Delta v")
print(total_delta_v_constrained._stats)
print((total_delta_v_constrained.tf - total_delta_v_constrained.tig)*total_delta_v_constrained.sim.omega*180/np.pi)
print(tot_delta_v_constrained_sim.final_pos_disp)


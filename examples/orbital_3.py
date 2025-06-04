import condor as co
import casadi as ca

from condor.backends.casadi.implementations import OptimizationProblem
from LinCovCW import LinCovCW, make_burn, sim_kwargs, I3, Z3, I6, Z6

W = ca.vertcat(I6, Z6)
V = ca.vertcat(Z3, I3, ca.MX(6,3))
Wonb = ca.MX.eye(6)
Vonb = ca.vertcat(Z3, I3)

class LinCovCW(co.ODESystem):
    omega = parameter()
    scal_w = parameter(shape=6) # covariance elements for propagation
    scal_v = parameter(shape=3) # covariance elements for control update
    # estimated covariance elements for navigation covariance propagation and control
    scalhat_w = parameter(shape=6)
    scalhat_v = parameter(shape=3)

    initial_x = parameter(shape=6)
    initial_C = parameter(shape=(12,12))
    initial_P = parameter(shape=(6,6))

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

    x = state(shape=6) # true state position and velocity
    C = state(shape=(12,12)) # augmented covariance
    P =  state(shape=(6,6)) # onboard covariance for navigation system (Kalman filter)
    Delta_v_mag = state()
    Delta_v_disp = state()

    tt = state()



    Scal_w = ca.diag(scal_w)
    Cov_prop_offset = W @ Scal_w @ W.T

    Scal_v = ca.diag(scal_v)
    Cov_ctrl_offset = V @ Scal_v @ V.T

    Scalhat_w = ca.diag(scalhat_w)
    P_prop_offset = Wonb @ Scalhat_w @ Wonb.T

    Scalhat_v = ca.diag(scalhat_v)
    P_ctrl_offset = Vonb @ Scalhat_v @ Vonb.T

    Fcal = ca.MX(12,12)
    Fcal[:6, :6] = Acw
    Fcal[6:, 6:] = Acw

    initial[x] = initial_x
    initial[C] = initial_C
    initial[P] = initial_P

    dot[x] = Acw @ x
    dot[C] = Fcal @ C + C @ Fcal.T + Cov_prop_offset
    dot[tt] = 1.

    # TODO: in general case, this should be a dfhat/dx(hat) instead of exact Acw
    # and should be reflected in bottom right corner of Fcal as well
    dot[P] = Acw @ P + P @ Acw.T + P_prop_offset


sin = ca.sin
cos = ca.cos

def make_burn(rd, tig, tem):
    burn_name = "Burn%d" % (1 + sum([
        event.__name__.startswith("Burn") for event in LinCovCW.Event._meta.subclasses
    ]))
    attrs = co.InnerModelType.__prepare__(burn_name, (LinCovCW.Event,))
    update = attrs["update"]
    x = attrs["x"]
    C = attrs["C"]
    t = attrs["t"]
    omega = attrs["omega"]
    Cov_ctrl_offset = attrs["Cov_ctrl_offset"]
    Delta_v_mag = attrs["Delta_v_mag"]
    Delta_v_disp = attrs["Delta_v_disp"]

    if not LinCovCW.parameter.get(backend_repr=rd).name:
        attrs["rd_%d" % (1 + sum(
            [name.startswith("rd_") for name in LinCovCW.parameter.list_of('name')]
        ))] = rd

    if not LinCovCW.parameter.get(backend_repr=tig).name:
        attrs["tig_%d" % (1 + sum(
            [name.startswith("tig_") for name in LinCovCW.parameter.list_of('name')]
        ))] = tig

    if not LinCovCW.parameter.get(backend_repr=tem).name:
        attrs["tem_%d" % (1 + sum(
            [name.startswith("tem_") for name in LinCovCW.parameter.list_of('name')]
        ))] = tem

    #attrs["function"] = t - tig
    attrs["at_time"] = [tig]

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
    Burn = co.InnerModelType(burn_name, (LinCovCW.Event,), attrs=attrs)

    Burn.rd = rd
    Burn.tig = tig
    Burn.tem = tem

    Burn.DG = DG
    Burn.Dcal = Dcal
    Burn.sigma_Dv__2 = sigma_Dv__2
    Burn.Mc = Mc
    return Burn


MajorBurn = make_burn(
    rd = LinCovCW.parameter(shape=3), # desired position
    tig = LinCovCW.parameter(), # time ignition
    tem = LinCovCW.parameter(), # time end maneuver
)


class Terminate(LinCovCW.Event):
    terminate = True
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    #function = t - MajorBurn.tem
    at_time = [MajorBurn.tem]

class Measurements(LinCovCW.Event):
    rcal = parameter(shape=3)
    rcalhat = parameter(shape=3)

    Rcal = ca.diag(rcal)
    Rcalhat = ca.diag(rcalhat)

    Hcal = ca.horzcat(I3, Z3)
    Hcalhat = ca.horzcat(I3, Z3)

    Khat = (P @ Hcalhat.T) @ ca.solve(Hcalhat @ P @ Hcalhat.T + Rcalhat, ca.MX.eye(3))
    Acalhat = I6 - Khat @ Hcalhat
    update[P] = Acalhat @ P @ Acalhat.T + Khat @ Rcalhat @ Khat.T

    M01 = Khat @ Hcal
    M = ca.vertcat( ca.horzcat(I6, Z6), ca.horzcat(M01, Acalhat) )
    N = ca.vertcat(Z3, Z3, Khat)
    update[C] = M @ C @ M.T + N @ Rcal @ N.T

    #update[Delta_v_disp] = Delta_v_disp# + ca.sqrt(sigma_Dv__2)
    #update[Delta_v_mag] = Delta_v_mag# + ca.sqrt(sigma_Dv__2)
    #update[x] = x

    meas_dt = parameter()
    meas_t_offset = parameter()

    #function = ca.sin(np.pi*(t-meas_t_offset)/meas_dt)
    #function = t - meas_t_offset
    at_time = [meas_t_offset]

# 1-burn sim
class Sim(LinCovCW.TrajectoryAnalysis):
    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
    # dispersion)
    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
    tot_Delta_v_disp = trajectory_output(Delta_v_disp)


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

sim_kwargs.update(dict(
    tem_1 = 2300.,
    meas_dt = 2300.,
))


class Burn1(co.OptimizationProblem):
    t1 = variable(initializer=1900.)
    sigma_r_weight = parameter()
    sigma_Dv_weight = parameter()
    mag_Dv_weight = parameter()
    sim = Sim(
        meas_t_offset = 851.,
        tig_1=t1,
        **sim_kwargs

    )

    constraint(t1, lower_bound=30., upper_bound=sim_kwargs['tem_1']-30.)
    constraint(sim.final_pos_disp, upper_bound=10.)
    objective = (
        sigma_Dv_weight*sim.tot_Delta_v_disp
        + sigma_r_weight*sim.final_pos_disp
        + mag_Dv_weight*sim.tot_Delta_v_mag
    )
    class Casadi(co.Options):
        exact_hessian=False
        method = OptimizationProblem.Method.scipy_trust_constr
        gtol = 1E-3
        xtol = 1E-3

class Meas1(co.OptimizationProblem):
    t1 = variable(initializer=0.)
    sigma_r_weight = parameter()
    sigma_Dv_weight = parameter()
    mag_Dv_weight = parameter()
    sim = Sim(
        meas_t_offset = t1,
        tig_1=851.,
        **sim_kwargs

    )

    constraint(t1, lower_bound=00., upper_bound=sim_kwargs['tem_1']-30.)
    constraint(sim.final_pos_disp, upper_bound=10.)
    objective = (
        sigma_Dv_weight*sim.tot_Delta_v_disp
        + sigma_r_weight*sim.final_pos_disp
        + mag_Dv_weight*sim.tot_Delta_v_mag
    )
    class Casadi(co.Options):
        exact_hessian=False
        method = OptimizationProblem.Method.scipy_trust_constr
        gtol = 1E-3
        xtol = 1E-3

opt = Meas1(sigma_Dv_weight=0, mag_Dv_weight=0, sigma_r_weight=1)
opt_sim = Sim(
        tig_1=851.,
        meas_t_offset = opt.t1,
        **sim_kwargs
)

print("\n"*3,"measurement time minimization")
print(opt._stats)

opt = Burn1(sigma_Dv_weight=3, mag_Dv_weight=1, sigma_r_weight=0)
opt_sim = Sim(
        tig_1=opt.t1,
        meas_t_offset = 851.,
        **sim_kwargs
)

print("\n"*3,"burn time minimization")
print(opt._stats)

"""
burn time minimization
           message: `xtol` termination condition is satisfied.
           success: True
            status: 2
               fun: 1.1974569637169519
                 x: [ 8.510e+02]
               nit: 149
              nfev: 251
              njev: 251
              nhev: 0
          cg_niter: 137
      cg_stop_cond: 2
              grad: [ 1.643e-03]
   lagrangian_grad: [ 1.643e-03]
            constr: [array([ 8.510e+02]), array([ 7.261e+00])]
               jac: [<1x1 sparse matrix of type '<class 'numpy.float64'>'
                    	with 1 stored elements in Compressed Sparse Row format>, <1x1 sparse matrix of type '<class 'numpy.float64'>'
                    	with 1 stored elements in Compressed Sparse Row format>]
       constr_nfev: [0, 251]
       constr_njev: [0, 251]
       constr_nhev: [0, 0]
                 v: [array([-3.255e-09]), array([ 7.800e-08])]
            method: tr_interior_point
        optimality: 0.0016431173214682697
  constr_violation: 0.0
    execution_time: 1320.708018064499
         tr_radius: 1.0000000000000005e-09
    constr_penalty: 1.0
 barrier_parameter: 2.048000000000001e-09
 barrier_tolerance: 2.048000000000001e-09
             niter: 149


with scaling & changing settings for trajectory analysis
           message: `xtol` termination condition is satisfied.
           success: True
            status: 2
               fun: 1.1974569662175378
                 x: [ 8.510e-01]
               nit: 173
              nfev: 304
              njev: 304
              nhev: 0
          cg_niter: 161
      cg_stop_cond: 2
              grad: [ 1.643e+00]
   lagrangian_grad: [ 5.483e-01]
            constr: [array([ 8.510e-01]), array([ 7.261e+00])]
               jac: [<1x1 sparse matrix of type '<class 'numpy.float64'>'
                    	with 1 stored elements in Compressed Sparse Row format>, <1x1 sparse matrix of type '<class 'numpy.float64'>'
                    	with 1 stored elements in Compressed Sparse Row format>]
       constr_nfev: [0, 304]
       constr_njev: [0, 304]
       constr_nhev: [0, 0]
                 v: [array([-1.086e+00]), array([ 2.566e-02])]
            method: tr_interior_point
        optimality: 0.5483186407562503
  constr_violation: 0.0
    execution_time: 562.644110918045
         tr_radius: 1.0000000000000005e-09
    constr_penalty: 1.0
 barrier_parameter: 2.048000000000001e-09
 barrier_tolerance: 2.048000000000001e-09
             niter: 173

changing xtol to 1E-3
burn time minimization
           message: `xtol` termination condition is satisfied.
           success: True
            status: 2
               fun: 1.1974578234547488
                 x: [ 8.510e+02]
               nit: 84
              nfev: 132
              njev: 132
              nhev: 0
          cg_niter: 72
      cg_stop_cond: 2
              grad: [ 1.643e-03]
   lagrangian_grad: [ 1.643e-03]
            constr: [array([ 8.510e+02]), array([ 7.261e+00])]
               jac: [<1x1 sparse matrix of type '<class 'numpy.float64'>'
                    	with 1 stored elements in Compressed Sparse Row format>, <1x1 sparse matrix of type '<class 'numpy.float64'>'
                    	with 1 stored elements in Compressed Sparse Row format>]
       constr_nfev: [0, 132]
       constr_njev: [0, 132]
       constr_nhev: [0, 0]
                 v: [array([-3.255e-09]), array([ 7.765e-08])]
            method: tr_interior_point
        optimality: 0.0016431186701373075
  constr_violation: 0.0
    execution_time: 239.6840078830719
         tr_radius: 0.00010000000000000003
    constr_penalty: 1.0
 barrier_parameter: 2.048000000000001e-09
 barrier_tolerance: 2.048000000000001e-09
             niter: 84

In [2]: opt.t1
Out[2]: 851.000523726278

In [3]: opt.t1 - 851
Out[3]: 0.0005237262779473895

"""

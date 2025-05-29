import numpy as np
import condor as co
import casadi as ca
import matplotlib.pyplot as plt

from condor.backends.casadi.implementations import OptimizationProblem

I6 = ca.MX.eye(6)
Z6 = ca.MX(6, 6)
W = ca.vertcat(I6, Z6)

I3 = ca.MX.eye(3)
Z3 = ca.MX(3,3)
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

burns = []
def make_burn(rd, tig, tem,):
    #burn_num = (1 + sum([
    #    event.__name__.startswith("Burn") for event in LinCovCW.Event.subclasses
    #]))
    burn_num = 1 + len(burns)
    burn_name = "Burn%d" % burn_num
    attrs = co.InnerModelType.__prepare__(burn_name, (LinCovCW.Event,))
    update = attrs["update"]
    x = attrs["x"]
    C = attrs["C"]
    P = attrs["P"]
    t = attrs["t"]
    omega = attrs["omega"]
    Cov_ctrl_offset = attrs["Cov_ctrl_offset"]
    P_ctrl_offset = attrs["P_ctrl_offset"]

    Delta_v_disp = attrs["Delta_v_disp_%d" % burn_num] = attrs["state"]()
    Delta_v_mag = attrs["Delta_v_mag_%d" % burn_num] = attrs["state"]()

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
    #update[Delta_v_mag] = ca.norm_2(Delta_v)
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
    # TODO: in general case, this requires something like the bottom right corner of
    # Dcal which should use onboard models of control instead of exact control
    update[P] = P + P_ctrl_offset

    Mc = DG @ ca.horzcat(Z6, I6)
    sigma_Dv__2 = ca.trace( Mc @ C @ Mc.T)

    update[Delta_v_disp] = Delta_v_disp + ca.sqrt(sigma_Dv__2)
    #update[Delta_v_disp] = ca.sqrt(sigma_Dv__2)
    Burn = co.InnerModelType(burn_name, (LinCovCW.Event,), attrs=attrs)
    burns.append(Burn)

    Burn.rd = rd
    Burn.tig = tig
    Burn.tem = tem
    Burn.Delta_v_mag = Delta_v_mag
    Burn.Delta_v_disp = Delta_v_disp

    Burn.DG = DG
    Burn.Dcal = Dcal
    Burn.sigma_Dv__2 = sigma_Dv__2
    Burn.Mc = Mc
    return Burn
make_burn.burns = burns

def make_sim(sim_name="Sim"):
    attrs = co.InnerModelType.__prepare__(sim_name, (LinCovCW.TrajectoryAnalysis,))
    C = attrs["C"]
    trajectory_output = attrs["trajectory_output"]

    Mr = ca.horzcat(I3, ca.MX(3,9))
    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
    attrs["final_pos_disp"] = trajectory_output(ca.sqrt(sigma_r__2))

    Mv = ca.horzcat(Z3, I3, ca.MX(3,6))
    sigma_vf__2 = ca.trace(Mv @ C @ Mv.T)
    attrs["final_vel_disp"] = trajectory_output(ca.sqrt(sigma_vf__2))

    attrs["final_vel_mag"] = trajectory_output(ca.norm_2(attrs["x"][3:, 0]))

    Delta_v_mags = []
    Delta_v_disps = []
    for burn in make_burn.burns:
        Dv_mag_state = LinCovCW.state.get(backend_repr=burn.Delta_v_mag)
        attrs[Dv_mag_state.name] = trajectory_output(burn.Delta_v_mag)
        Delta_v_mags.append(attrs[Dv_mag_state.name])

        Dv_disp_state = LinCovCW.state.get(backend_repr=burn.Delta_v_disp)
        attrs[Dv_disp_state.name] = trajectory_output(burn.Delta_v_disp)
        Delta_v_disps.append(attrs[Dv_disp_state.name])

    #attrs["tot_Delta_v_mag"] = trajectory_output(
    #    sum([burn.Delta_v_mag for burn in make_burn.burns])
    #)
    #attrs["tot_Delta_v_disp"] = trajectory_output(
    #    sum([burn.Delta_v_disp for burn in make_burn.burns])
    #)

    class Casadi(co.Options):
        #state_rtol = 1E-9
        #state_atol = 1E-15
        #adjoint_rtol = 1E-9
        #adjoint_atol = 1E-15
        #state_max_step_size = 30.

        state_adaptive_max_step_size = 4#16
        adjoint_adaptive_max_step_size = 4

        #integrator_options = dict(
        #    max_step = 1.,
        #    #atol = 1E-15,
        #    #rtol = 1E-12,
        #    nsteps = 10000,
        #)
    attrs["Casadi"] = Casadi

    Sim = co.InnerModelType(sim_name, (LinCovCW.TrajectoryAnalysis,), attrs=attrs)
    Sim.Delta_v_mags = tuple(Delta_v_mags)
    Sim.Delta_v_disps = tuple(Delta_v_disps)
    Sim.implementation.callback.get_jacobian(f'jac_{Sim.implementation.callback.name}', None, None, {})

    return Sim


from scipy.io import loadmat
Cov_0_matlab = loadmat('P_aug_0.mat')['P_aug_0'][0]

sim_kwargs = dict(
    omega = 0.00114,
    # env.translation_process_noise_disp from IC_Traj_demo_000c.m
    scal_w=[0.]*3 + [4.8E-10]*3,
    #env.translation_maneuvers.var_noise_disp
    scal_v=[2.5E-7]*3,
    # env.translation_process_noise_err from IC_Traj_demo_000c.m
    scalhat_w=[0.]*3 + [4.8E-10]*3,
    #scalhat_w=[0.]*6,
    # env.translation_maneuvers.var_noise_err
    scalhat_v=[2.5E-7]*3,
    # io.R = env.sensors.rel_pos.measurement_var; from rel_pos.m which is set by
    # sig = 1e-3*(40/3); pos_var = [(sig)^2    (sig)^2    (sig)^2]; in rel_pos_ic.m
    rcal=[(1e-3*(40/3))**2]*3,
    # io.R_onb = io.onboard.sensors.rel_pos.measurement_var from rel_pos.m
    # in runRelMotionSetup.m, io.onboard = io.environment
    rcalhat=[(1e-3*(40/3))**2]*3,

    rd_1=[500., 0., 0.],

)
sim_kwargs.update(dict(
    initial_x=[-2000., 0., 1000., sim_kwargs['omega']*1000.*3/2, 0., 0.,],
    initial_C=Cov_0_matlab,
    initial_P=Cov_0_matlab[-6:, -6:] - Cov_0_matlab[:6, :6],
))




from scipy.interpolate import make_interp_spline
def deriv_check_plots(indep_var, output_vars, sims, title_prefix='', interp_k =2):
    sims1 = [simout[0] for simout in sims]
    jac = np.stack([simout[1] for simout in sims])
    #Dv_mags = [sim.tot_Delta_v_mag for sim in sims1]
    #Dv_disps = [sim.tot_Delta_v_disp for sim in sims1]

    xgrid = [getattr(sim, indep_var.name) for sim in sims1]
    xidx = indep_var.field_type.flat_index(indep_var)

    ordinate_names = [output_var.name.replace('_', ' ') for output_var in output_vars]
    field = output_vars[0].field_type
    Sim = field._model
    ordinates = [
        [getattr(sim, output_var.name) for sim in sims1]
        for output_var in output_vars
    ]
    ord_idxs = [field.flat_index(output_var) for output_var in output_vars]


    #Dv_mag_idx = Sim.trajectory_output.flat_index(Sim.Delta_v_mag_1)
    #Dv_disp_idx = Sim.trajectory_output.flat_index(Sim.Delta_v_disp_1)
    #pos_disp_idx = Sim.trajectory_output.flat_index(Sim.final_pos_disp)

    #Dv_mags = [sim.Delta_v_mag_1[-1] for sim in sims1]
    #Dv_disps = [sim.Delta_v_disp_1[-1] for sim in sims1]
    #pos_disps = [sim.final_pos_disp for sim in sims1]
    #breakpoint()
    #ord_idxs = [Dv_mag_idx, Dv_disp_idx, pos_disp_idx]
    for ord_idx, ord_name, ord_val in zip(ord_idxs, ordinate_names, ordinates):
        interp = make_interp_spline(xgrid, ord_val, k=interp_k)
        derinterp = interp.derivative()
        fig, axes = plt.subplots(2, constrained_layout=True, sharex=True)
        plt.suptitle(' '.join([title_prefix, ord_name]))
        axes[0].plot(xgrid, ord_val)
        axes[0].grid(True)
        axes[1].plot(xgrid, derinterp(xgrid), label='numerical')
        axes[1].plot(xgrid, jac[:, ord_idx, xidx], '--', label='SGM')
        axes[1].grid(True)
        axes[1].legend()
    return xgrid, ordinates




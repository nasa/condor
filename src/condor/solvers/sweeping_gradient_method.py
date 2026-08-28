from dataclasses import dataclass, field

import numpy as np

# from condor import backend
from scipy.interpolate import make_interp_spline

try:
    from sundials4py.core import (
        SUN_COMM_NULL,
        SUN_SUCCESS,
        N_VGetArrayPointer,
        N_VNew_Serial,
        SUNContext_Create,
        SUNLinSol_SPGMR,
    )
    from sundials4py.cvodes import (
        CV_ADAMS,
        CV_BDF,
        CV_HERMITE,
        CV_NORMAL,
        CV_ONE_STEP,
        CV_ROOT_RETURN,
        CV_SUCCESS,
        CV_TSTOP_RETURN,
        CVode,
        CVodeAdjInit,
        CVodeB,
        CVodeCreate,
        CVodeCreateB,
        CVodeF,
        CVodeGetB,
        CVodeGetQuad,
        CVodeGetQuadB,
        CVodeInit,
        CVodeInitB,
        CVodeQuadInit,
        CVodeQuadInitB,
        CVodeQuadSStolerancesB,
        CVodeSetInterpolateStopTime,
        CVodeSetJacFn,
        CVodeSetJacFnB,
        CVodeSetLinearSolver,
        CVodeSetLinearSolverB,
        CVodeSetMaxNumSteps,
        CVodeSetQuadErrConB,
        CVodeSetStopTime,
        CVodeSStolerances,
        CVodeSStolerancesB,
    )
except ModuleNotFoundError:
    has_cvode = False
else:
    has_cvode = True

if has_cvode:
    status, sunctx = SUNContext_Create(SUN_COMM_NULL)
    if status != SUN_SUCCESS:
        raise ValueError

from typing import NamedTuple, Optional

from scipy.integrate import ode as scipy_ode
from scipy.optimize import brentq


class SolverMixin:
    def store_result(self, store_t, store_x, store_y=False):
        system = self.system
        results = system.result
        results.t.append(store_t)
        results.x.append(store_x)
        if system.dynamic_output and store_y:
            results.y.append(
                np.array(system.dynamic_output(results.p, store_t, store_x)).reshape(-1)
            )


class SolverSciPyBase(SolverMixin):
    """Baseclass for using :class:`scipy.integrate.ode` with advanced features.

    Wraps standard scipy integrator class with functionality for  event handling,
    multiple-segment time generators, and adaptive minumum number of steps.
    Options are generally a pass through to the underlying integrator (e.g.,
    :class:`scipy.integrate.ode`, ``dopri5`` or ``dop853`` currently). This
    implementation performs an event function zero-crossing check at each time-step;
    when an event function zero-crossing is detected, a rootfinding and event update
    operation is performed. Special options on the ``sweeping_gradient_method``
    solver which are described below.

    Options
    --------

    adaptive_min_steps : int
        minimum number of steps per time-defined segment
    max_step_size : float
        maximum step size for the forward evaluation, normalized name to scipy's
        max_step
    separate_events : bool
        Coincident events get stored separately in the ODE result object
    reset_step_after_event : bool
        After an event occurs, reset the step size to the first_step
    rootfinder : callable
        Interval root-finder function. Defaults to :func:`scipy.optimize.brentq`, and
        must take the equivalent positional arguments, ``f``, ``a``, and ``b``, and
        return ``x0``, where ``a <= x0 <= b`` and ``f(x0)`` is the zero.

        Keyword arguments with a ``rootfinder_`` prefix are passed to the rootfinder
        method during the rootfinding step.
    """

    SOLVER_NAME = None

    def __init__(
        self,
        system,
        sen_system=None,
        adj_system=None,
        atol=1e-12,
        rtol=1e-6,
        adaptive_min_steps=0,
        reset_step_after_event=True,
        rootfinder=brentq,
        max_step_size=0.0,
        separate_events=False,
        nsteps=10_000,
        **kwargs,
    ):
        self.original_system = system
        self.sen_system = sen_system
        self.adj_system = adj_system

        # set the primary in precedence: adjoint, then sensitivity, then original
        system = system
        if sen_system is not None:
            system = sen_system
        if adj_system is not None:
            system = adj_system

        self.system = system

        self.adaptive_min_steps = adaptive_min_steps
        self.separate_events = separate_events
        self.reset_step_after_event = reset_step_after_event
        self.max_step_size = max_step_size
        self.solver = scipy_ode(
            system.dots,
        )
        self.int_options = dict(
            name=self.SOLVER_NAME,
            atol=atol,
            rtol=rtol,
            max_step=max_step_size,
            nsteps=nsteps,
        )
        self.rootfinder = rootfinder
        self.root_options = {}
        for k, v in kwargs.items():
            if k.startswith("rootfinder_"):
                self.root_options[k.replace("rootfinder_", "")] = v
            else:
                self.int_options[k] = v

        self.do_update_integrator_options = (
            adaptive_min_steps or not reset_step_after_event or max_step_size == 0.0
        )

        self.solver.set_integrator(**self.int_options)
        self.solver.set_solout(self.solout)

    def solout(self, t, x):
        system = self.system
        results = system.result
        # had to do some debugging and figured out a decent pattern for conditional
        # breakpoints... TODO move to notes or something?
        # if (
        #     results.e
        #     and results.e[-1].index == 19
        #     and np.abs(results.t[-1] - 56.81341256) < 1.5e-9
        # ):
        #     breakpoint()

        # if any sign change, return -1. maybe could determine rootinfo here?
        new_gs = system.events(t, x)
        gs_sign = np.sign(new_gs) - np.sign(self.gs)
        immediately_after_event = (
            results.e and results.e[-1].index >= len(results.t) - 1
        )
        if not immediately_after_event and np.any(
            np.abs(gs_sign) > 1
            # & (np.abs(new_gs) > self.int_options["atol"]/1E3)
        ):
            self.rootinfo = gs_sign
            store_t, store_x = self.find_root(t, x, new_gs, gs_sign)
        else:
            self.rootinfo = None
            store_t, store_x = t, np.copy(x)
            if (
                results.e
                and results.e[-1].index == len(results.t)
                and t != results.t[-1]
            ):
                store_t = results.t[-1]

        self.gs = new_gs
        self.store_result(store_t, store_x)
        if np.any(self.rootinfo):
            return -1

    def find_root(self, t, x, new_gs, gs_sign):
        system = self.system
        results = system.result
        num_points_since_last_event = len(results.t) - results.e[-1].index + 1
        k = min(num_points_since_last_event - 1, 3)
        spline_ts = results.t[-k:] + [t]
        spline_xs = np.array(results.x[-k:] + [x])
        spline_gs = np.array(
            [system.events(tt, xx) for tt, xx in zip(spline_ts[:-2], spline_xs[:-2])]
            + [self.gs, new_gs]
        )

        if self.integration_direction > 0:
            time_sort = slice(None)
        else:
            time_sort = slice(None, None, -1)

        g_spl = make_interp_spline(spline_ts[time_sort], spline_gs[time_sort], k=k)
        x_spl = make_interp_spline(spline_ts[time_sort], spline_xs[time_sort], k=k)
        t_events = np.empty(system.num_events)

        # ts_for_g = np.linspace(*spline_ts[-2:], 4)
        # spline_gs = np.array([system.events(tt, x_spl(tt)) for tt in ts_for_g])
        # g_spl = make_interp_spline(ts_for_g[time_sort], spline_gs[time_sort], k=k)

        for g_idx, g_sign in enumerate(gs_sign):
            if g_sign:

                def find_function(t):
                    return g_spl(t)[g_idx]  # noqa: B023 false positive

                if np.diff(np.sign(g_spl(spline_ts[-2:])[:, g_idx])) == 0:
                    self.rootinfo[g_idx] = 0
                    return t, x

                set_t = self.rootfinder(
                    find_function, spline_ts[-2], spline_ts[-1], **self.root_options
                )
            else:
                set_t = spline_ts[-1]
            t_events[g_idx] = set_t

        min_t = np.min(t_events)
        self.rootinfo[t_events > min_t] = 0

        return min_t, x_spl(min_t)

    def simulate(self, do_compute_output=True):
        system = self.system
        results = system.result

        if self.system is self.original_system:
            dots = self.system.dots
            last_x = system.initial_state()

        if self.system is not self.original_system:
            state_result = self.original_system(system.result.p, compute_output=False)
            state_interp = ResultInterpolant(state_result)
            results.state_result = state_result
            results.state_interp = state_interp

        if self.system is self.adj_system:
            last_x = self.adj_system.initial_adjoint(
                state_result.t[-1], state_result.x[-1]
            )

        time_generator = system.time_generator()
        last_t = next(time_generator)

        # self.gs  will be used to monitor the event function
        self.gs = system.events(last_t, last_x)
        self.solout(last_t, last_x)

        rootsfound = (self.gs == 0.0).astype(int)
        if np.any(rootsfound):
            # subsequent events use length of time for index, so root index is the index
            # of the updated state to the right of event. -> root coinciding with
            # initialization has index 1
            last_x = system.update(rootsfound, last_t, last_x)
        results.e.append(Root(1, rootsfound))

        terminate = np.any(rootsfound[system.terminating] != 0)
        if terminate:
            self.store_result(last_t, last_x)
            return

        solver = self.solver

        # each iteration of this loop simulates until next generated time
        while True:
            next_t = next(time_generator)
            self.gs = system.events(last_t, last_x)
            if np.isinf(next_t):
                break
            # if next_t < 0:
            #    breakpoint()

            if self.adaptive_min_steps:
                self.int_options.update(
                    dict(max_step=np.abs(next_t - last_t) / self.adaptive_min_steps)
                )

            if self.max_step_size == 0.0:
                self.int_options["max_step"] = np.abs(next_t - last_t)

            if self.do_update_integrator_options:
                self.solver.set_integrator(**self.int_options)
                self.solver.set_solout(self.solout)

            solver.set_initial_value(
                last_x,
                last_t,
            )
            self.integration_direction = np.sign(next_t - last_t)

            # each iteration of this loop is one step until next event or time stop
            while True:
                solver.integrate(next_t)
                if not solver.successful():
                    results.e.append(Root(len(results.t), np.zeros(system.num_events)))
                    breakpoint()
                    return

                # assume we have an event, either a true event or at next_t

                solver_flag = solver.get_return_code()
                # == 1: #equivalent to tstop
                # == 2: #equivalent to rootfound

                # if (
                #     len(results.t) > 10
                #     and np.abs(solver.t - next_t) < 1e-13
                #     and solver.t != next_t
                # ):
                #     breakpoint()

                if np.any(self.rootinfo):
                    rootsfound = self.rootinfo

                else:
                    # assume this is associated with an event
                    gs = system.events(results.t[-1], results.x[-1])
                    min_e = np.abs(gs).min()
                    rootsfound = (gs == min_e).astype(int)

                idx = len(results.t)

                if not self.reset_step_after_event:
                    self.int_options["first_step"] = float(
                        np.abs(results.t[-1] - results.t[-2])
                    )

                if self.separate_events:
                    for num_processed_events, g_idx in enumerate(
                        np.where(rootsfound)[0]
                    ):
                        send_rootsfound = np.zeros_like(rootsfound)
                        send_rootsfound[g_idx] = rootsfound[g_idx]
                        next_x = system.update(
                            send_rootsfound,
                            results.t[-1],
                            results.x[-1],
                        )
                        if num_processed_events:
                            self.store_result(results.t[-1], next_x)

                    results.e.append(Root(len(results.t), rootsfound))
                else:
                    results.e.append(Root(idx, rootsfound))
                    next_x = system.update(
                        rootsfound,
                        results.t[-1],
                        results.x[-1],
                    )

                terminate = np.any(rootsfound[system.terminating] != 0)

                last_t = results.t[-1]
                self.gs = system.events(last_t, next_x)

                if terminate:
                    self.store_result(last_t, next_x)
                    self.compute_output(do_compute_output)
                    return

                last_x = next_x
                solver.set_initial_value(
                    last_x,
                    last_t,
                )

                if (
                    (self.integration_direction * last_t)
                    >= (self.integration_direction * next_t)
                ) or np.nextafter(next_t, last_t) == results.t[-1]:
                    # ideally this one gets triggered by an at_time event
                    # subsequent case is really only if integrator terminated (and
                    # thinks successful) but closest event

                    break

                if solver_flag == 1 and np.any(
                    (system.events(next_t, results.x[-1]) == 0.0)
                    & rootsfound.astype(bool)
                ):
                    break

                elif solver_flag == 2 and isinstance(system, AdjointSystem):
                    breakpoint()

            last_t = next_t
        self.compute_output(do_compute_output)

    def compute_output(self, doit=True):
        if not doit:
            return result
        system = self.system
        result = system.result
        integral = 0.0
        integrand_interpolant = ResultInterpolant(
            result=result, function=lambda p, t, x: system.integrand_terms(t, x)
        )
        for segment in integrand_interpolant:
            integrand_antideriv = segment.interpolant.antiderivative()
            integral += integrand_antideriv(segment.t1) - integrand_antideriv(
                segment.t0
            )
        result.o = (
            system.terminal_terms(
                # result.p,
                result.t[-1],
                result.x[-1],
            )
            + integral
        )

        return result


class SolverSciPyDopri5(SolverSciPyBase):
    """:class:`SolverSciPyBase` for the ``dopri5`` method"""

    SOLVER_NAME = "dopri5"


class SolverSciPyDop853(SolverSciPyBase):
    """:class:`SolverSciPyBase` for the ``dop853`` method"""

    SOLVER_NAME = "dop853"


class SolverCVODE(SolverMixin):
    def __init__(
        self,
        system,
        sen_system=None,
        adj_system=None,
        atol=1e-12,
        rtol=1e-6,
        adaptive_min_steps=0.0,
        max_step_size=0.0,
    ):
        self.system = system
        self.sen_system = sen_system
        self.adj_system = adj_system
        self.adaptive_min_steps = adaptive_min_steps
        self.atol = atol
        self.rtol = rtol

        self.y = N_VNew_Serial(self.system.dim_state, sunctx)
        assert self.y is not None

        self.Q = N_VNew_Serial(self.system.dim_output, sunctx)
        assert self.Q is not None

        # should take options for linear solver, CVOde method (CV_BDF, CV_*)
        # adjoint interp, steps, etc

        # Set linear solver
        self.ls = SUNLinSol_SPGMR(self.y, 0, 3, sunctx)
        # 0 maybe correspodns to no preconditioner and krylov basis vectors 3?
        assert self.ls is not None

        if adj_system is not None:
            self.uB = N_VNew_Serial(adj_system.dim_state, sunctx)
            self.qB = N_VNew_Serial(adj_system.dim_output, sunctx)
            self.lsb = SUNLinSol_SPGMR(self.uB, 0, 3, sunctx)
            assert self.lsb is not None

    def adjoint_quad(self, t, x_SD, lamda_SD, qBd_SD, _):
        x = N_VGetArrayPointer(x_SD)
        lamda = N_VGetArrayPointer(lamda_SD)
        qBd = N_VGetArrayPointer(qBd_SD)
        qBd[:] = self.adj_system.integrand_terms(t, lamda, x)
        return 0

    def adjoint_dots(self, t, x_SD, lamda_SD, lamda_dot_SD, _):
        x = N_VGetArrayPointer(x_SD)
        lamda = N_VGetArrayPointer(lamda_SD)
        lamda_dot = N_VGetArrayPointer(lamda_dot_SD)
        lamda_dot[:] = self.adj_system.adjoint_dots(t, lamda[:], x[:])
        return 0

    def dots(
        self,
        t,
        x_sd,
        xdot_sd,
        _,
    ):  # userdata=None,):
        x = N_VGetArrayPointer(x_sd)
        xdot = N_VGetArrayPointer(xdot_sd)
        xdot[:] = self.system.dots(t, x[:])
        return 0

    def adjoint_jac(self, t, x_sd, lamda_sd, lamda_dot_sd, jac_sd, _):
        x = N_VGetArrayPointer(x_sd)
        jac = SUNDenseMatrix_Data(jac_sd)
        jac[...] = self.system.jac(t, x).T
        return 0

    def jac(self, t, x_sd, xdot_sd, jac_sd, _):
        x = N_VGetArrayPointer(x_sd)
        xdot = N_VGetArrayPointer(xdot_sd)
        jac = SUNDenseMatrix_Data(jac_sd)
        jac[...] = self.system.jac(t, x)
        return 0

    def integrand_terms(self, t, x_sd, Qd_sd, _):
        x = N_VGetArrayPointer(x_sd)
        Qd = N_VGetArrayPointer(Qd_sd)
        Qd[:] = self.system.integrand_terms(t, x)
        return 0

    def events(self, t, x, g):
        g[:] = self.system.events(
            t,
            x,
        )

    def simulate(self, one_step=0, test_out_vectors=None):
        """
        expects:
        self.solver is an object with CVODE-like interface to parameterized
        dots, [jac,] and rootfns with init_step, set_options(tstop=...), and step()
        returns StatusEnum with TSTOP_RETURN and ROOT_RETURN members.

        self.result is a namespace with t, x, and e attributes that can be appended with
        time value, state at time, and possibly events e
        initializer
        update

        events have index corresponding to start of each segment. except perhaps the
        first. -- can always assume the 0 index corresponds to initialization
        initial and final segments may be singular if an event causes an update that
        coincides with
        """
        system = self.system
        results = system.result
        last_x = system.initial_state()

        time_generator = system.time_generator()
        last_t = next(time_generator)
        # TODO: add dynamic_output feature

        gs = system.events(last_t, last_x)
        self.store_result(last_t, last_x)

        rootsfound = (gs == 0.0).astype(int)
        if np.any(rootsfound):
            # subsequent events use length of time for index, so root index is the index
            # of the updated state to the right of event. -> root coinciding with
            # initialization has index 1
            last_x = system.update(rootsfound, last_t, np.copy(last_x))
        results.e.append(Root(1, rootsfound))
        self.store_result(last_t, last_x)
        terminate = np.any(rootsfound[system.terminating] != 0)
        if terminate:
            return

        # each iteration of this loop simulates until next generated time

        # solver = self.solver
        y = self.y
        ls = self.ls
        steps = 1

        yarr = N_VGetArrayPointer(y)
        yarr[:] = last_x

        Q = self.Q
        Qarr = N_VGetArrayPointer(Q)
        Qarr[:] = 0.0

        if self.adj_system is not None and test_out_vectors is not None:
            test_out_vectors = np.array(test_out_vectors)

        while True:
            # contents of loop could be a funciton call?
            next_t = next(time_generator)
            if np.isinf(next_t):
                break
            if next_t < 0:
                # breakpoint()
                pass

            if self.adaptive_min_steps:
                solver.set_options(
                    max_step_size=np.abs(next_t - last_t) / self.adaptive_min_steps
                )

            # Create CVODE solver and set up problem
            cvode = CVodeCreate(CV_ADAMS, sunctx)
            # cvode = CVodeCreate(CV_BDF, sunctx)
            results.cvode_mems.append(cvode)
            assert cvode is not None

            # Initialize CVODE with ODE RHS
            status = CVodeInit(cvode.get(), self.dots, last_t, y)
            assert status == CV_SUCCESS

            # Set tolerances
            status = CVodeSStolerances(cvode.get(), self.rtol, self.atol)
            assert status == CV_SUCCESS

            # Set max steps
            status = CVodeSetMaxNumSteps(cvode.get(), 100000)
            assert status == CV_SUCCESS

            status = CVodeSetLinearSolver(cvode.get(), ls, None)
            assert status == CV_SUCCESS

            # status = CVodeSetJacFn(cvode.get(), self.jac)

            if (
                self.system.dim_output
                and self.sen_system is None
                and self.adj_system is None
            ):
                status = CVodeQuadInit(cvode.get(), self.integrand_terms, Q)
                assert status == CV_SUCCESS

                # options

            if self.sen_system is not None and test_in is not None:
                status = CVodeSensInit()

                # options

            status = CVodeSetStopTime(cvode.get(), next_t)
            status = CVodeSetInterpolateStopTime(cvode.get(), 1)

            # Enable adjoint sensitivity analysis if desired
            # need forward sensitivity option as well

            # Ns - number of sensitivities
            # Na - number of adjoints
            # both specified, then implementation must parameterize by weighting vector
            # "test vectors"
            # of size Ns/Na, if for jacobian, must be eye
            # or 1 for directional hessian
            # implementation makes function dependent on test vector,
            # but also wraps in an if-else to bypass 0'd out channels

            if self.adj_system is not None:  # and test_out_vectors is not None:
                status = CVodeAdjInit(cvode.get(), steps, CV_HERMITE)
            assert status == CV_SUCCESS

            # need to setup CVodeRootInit()

            # if one step mode, will need to do single steps to copy all data --
            # if not one step mode, full step to next_t
            if one_step and 1:
                while True:
                    status, tret = CVode(cvode.get(), next_t, y, CV_ONE_STEP)
                    if status >= 0:
                        self.store_result(tret, np.copy(yarr[:]))
                    else:
                        print(status)
                        breakpoint()
                    if status != CV_SUCCESS:
                        break
            else:
                if self.adj_system is not None:
                    status, tret, ncheck = CVodeF(cvode.get(), next_t, y, CV_NORMAL)
                else:
                    status, tret = CVode(cvode.get(), next_t, y, CV_NORMAL)
                if status not in (CV_TSTOP_RETURN, CV_SUCCESS, CV_ROOT_RETURN):
                    breakpoint()
                self.store_result(tret, np.copy(yarr[:]))

            if (
                self.system.dim_output
                and self.sen_system is None
                and self.adj_system is None
            ):
                # original system owns output if no sensitivity or adjoint system
                # sensitivity owns output if no adjoint
                # adjoint takes presedence

                quad_status, tret = CVodeGetQuad(cvode.get(), Q)
                assert quad_status == CV_SUCCESS
                results.o = Qarr[:] + system.terminal_terms(tret, yarr[:])
            # probably can put one-step loop here and it would behave correctly -- break
            # loop on root found, then handle end-of-segment logic (find root, update,
            # or terminate)

            if status == CV_ROOT_RETURN:
                rootsfound = solver.rootinfo()

            if status in (CV_TSTOP_RETURN, CV_SUCCESS):
                # assume this is associated with an event
                # does occur on time_switch but not sp_lqr
                gs = system.events(results.t[-1], results.x[-1])
                min_e = np.abs(gs).min()
                rootsfound = (gs == min_e).astype(int)
                assert next_t == tret

            if status in (CV_TSTOP_RETURN, CV_ROOT_RETURN, CV_SUCCESS):
                idx = len(results.t)
                results.e.append(Root(idx, rootsfound))
                next_x = system.update(
                    rootsfound,
                    results.t[-1],
                    results.x[-1],
                )
                try:
                    terminate = np.any(rootsfound[system.terminating] != 0)
                except Exception as e:
                    print("Hit exemption:")
                    print(e)
                    print("You may try to continue through or exit")
                    breakpoint()
                self.store_result(tret, next_x)

                if terminate:
                    break

            last_t = tret

        if self.adj_system is None:  # or test_out_vectors is None:
            return

        if one_step:
            raise ValueError

        uB = self.uB
        lsb = self.lsb
        qB = self.qB
        uBarr = N_VGetArrayPointer(uB)
        last_lamda = self.adj_system.initial_adjoint(tret, yarr[:])
        uBarr[:] = last_lamda

        qBarr = N_VGetArrayPointer(qB)
        qBarr[:] = self.adj_system.terminal_terms(tret, yarr[:])

        for cvode, e1, e0 in zip(
            results.cvode_mems,
            results.e[-1::-1],
            results.e[-2::-1],
        ):
            status, which = CVodeCreateB(cvode.get(), CV_BDF)
            results.adjoint_mems.append(which)

            status = CVodeInitB(
                cvode.get(),
                which,
                self.adjoint_dots,
                results.t[e1.index],  # Tf
                uB,
            )

            # Set the tolerances for the backward problem
            status = CVodeSStolerancesB(cvode.get(), which, self.rtol, self.atol)
            assert status == CV_SUCCESS

            # Create the linear solver for the backward problem
            status = CVodeSetLinearSolverB(cvode.get(), which, lsb, None)
            assert status == CV_SUCCESS

            # status = CVodeSetJacFnB(cvode.get(), self.adjoint_jac)

            if self.adj_system.dim_output:
                status = CVodeQuadInitB(cvode.get(), which, self.adjoint_quad, qB)
                assert status == CV_SUCCESS
                status = CVodeSetQuadErrConB(cvode.get(), which, True)
                assert status == CV_SUCCESS
                status = CVodeQuadSStolerancesB(
                    cvode.get(), which, self.rtol, self.atol
                )
                assert status == CV_SUCCESS

            status = CVodeB(cvode.get(), results.t[e0.index], CV_NORMAL)
            if status < 0:
                breakpoint()

            # Get the final adjoint solution
            status, t = CVodeGetB(cvode.get(), which, uB)
            assert status == CV_SUCCESS

            if self.adj_system.dim_output:
                quad_status, t = CVodeGetQuadB(cvode.get(), which, qB)
                assert quad_status == CV_SUCCESS

            last_lamda = uBarr[:]

        results.o = qBarr[:]
        print(results.o)
        # breakpoint()

        # need to establish datastructure for holding each adjoint system
        # so turn it into one system!
        pass


class Root(NamedTuple):
    index: int
    rootsfound: list[int]


class NextTimeFromSlice:
    def __init__(self, at_time_func):
        self.at_time_func = at_time_func

    def set_p(self, p):
        start, stop, step = np.array(self.at_time_func(p)).squeeze()
        self.start = start
        self.step = step
        self.stop = stop
        self.direction = np.sign(step)

    def before_start(self, t):
        if self.direction < 0:
            return t > self.start
        return t < self.start

    def after_stop(self, t):
        # if t is exactly stop time, this has already occured
        if self.direction < 0:
            return t <= self.stop
        return t >= self.stop

    def __call__(self, t):
        if self.before_start(t):
            return self.start
        if self.after_stop(t):
            return self.direction * np.inf
        # TODO handle negative step -- may need to adjust a few of these
        return (1 + (t - self.start) // self.step) * self.step + self.start


class TimeGeneratorFromSlices:
    def __init__(self, time_slices, direction=1):
        self.time_slices = time_slices
        self.direction = direction

    def __call__(self, p):
        # TODO handle negative step?
        for time_slice in self.time_slices:
            time_slice.set_p(p)

        t = -self.direction * np.inf
        get_time = min if self.direction > 0 else max
        while True:
            next_times = [time_slice(t) for time_slice in self.time_slices]
            t = get_time(next_times)
            yield t
            if np.isinf(t):
                breakpoint()


class System:
    def __init__(
        self,
        dim_state,
        initial_state,
        dot,
        time_generator,
        # these are related Event objects...
        events,
        # processed from Event objects
        updates,
        terminating,
        num_events,
        dim_output,
        integrand_terms,
        terminal_terms,
        jac=None,
        dynamic_output=None,
        **solver_options,
    ):
        """
        if adaptive_min_steps, treat max_step_size as the fraction of the next
        simulation span. Otherwise, use as absolute value.
        """

        # simulation must be terminated with event so must provide everything

        # result is a temporary instance attribute so system model can pass information
        # to functions the sundials solvers call -- could be passed via the userdata
        # option but these functions need wrappers to handle returned values anyway
        self.result = None

        # these instance attributes encapsolate the business data of a system

        #   functions, see method wrapper for expected signature

        #     for CVODE interface once wrapped
        self._dot = dot
        self._jac = jac

        # these all go together to define events
        self._events = events
        self._updates = updates
        self.terminating = terminating

        # these go together to to define trajectory output
        self._integrand_terms = integrand_terms
        self._terminal_terms = terminal_terms

        self.dynamic_output = dynamic_output

        #     define initial conditions
        self._initial_state = initial_state
        # who owns t0? time generator? for adjoint system, very easy to own all of them.
        # I guess can just handle single point as a special case instead of assuming all
        # take the form of an interval? Does this make it easier to allow events that
        # occur at t0? Then

        # data for time_generator method -- should this just be a generator class
        # itself? yes, basically just a sub-name space to the System which should own
        # the (parameterized) callables. Use wrapper method to define interface, then
        # AdjointSystem can re-implement wrapper method to change interface
        self._time_generator = time_generator

        # list of root indices that are terminating events...
        # any(rootsfound[terminating]) --> terminates simulation

        self.dim_state = dim_state
        self.dim_output = dim_output
        self.num_events = len(updates)
        self.make_solver(
            **solver_options,
        )

    def make_solver(self, solver_class, **solver_options):
        self.system_solver = solver_class(  # SolverSciPy( #SolverCVODE(
            system=self,
            **solver_options,
        )

    def initial_state(self):
        return np.array(self._initial_state(self.result.p)).reshape(-1)

    def terminal_terms(self, t, x):
        return np.array(self._terminal_terms(self.result.p, t, x)).reshape(-1)

    def integrand_terms(self, t, x):
        return np.array(self._integrand_terms(self.result.p, t, x)).reshape(-1)

    def dots(self, *args):
        return np.array(self._dot(self.result.p, *args)).reshape(-1)

    def jac(self, *args):
        return np.array(self._jac(self.result.p, *args)).squeeze()

    def events(self, *args):
        return np.array(self._events(self.result.p, *args)).reshape(-1)

    def update(self, rootsfound, t, x, *args):
        next_x = x  # np.copy(x) # who is responsible for copying? I suppose simulate
        for root_sign, update in zip(rootsfound, self._updates):
            if root_sign != 0:
                next_x = update(self.result.p, t, next_x, *args)
        return np.array(next_x).squeeze()

    def time_generator(self):
        for t in self._time_generator(self.result.p):
            yield np.array(t).reshape(-1)[0]

    def attach_result(self, p):
        if isinstance(self.system_solver, SolverCVODE):
            self.result = CVodeResult(p=p, system=self)
        else:
            self.result = Result(p=p, system=self)

    def run_sim(self, p, from_implementation=False):
        self.attach_result(p)
        if isinstance(self.system_solver, SolverCVODE):
            self.system_solver.simulate(one_step=from_implementation)
        else:
            self.system_solver.simulate()
        result = self.result
        result.t = np.array(result.t)
        if self.dim_state == 1:
            result.x = [np.atleast_1d(x) for x in result.x]
        result.x = np.array(result.x)
        result.y = np.array(result.y)

    def __call__(self, p, from_implementation=False, compute_output=True):
        self.run_sim(p, from_implementation)
        result = self.result
        self.result = None
        return result


@dataclass
class ResultMixin:
    p: list[float]


@dataclass
class ResultBase:
    system: System
    o: list[float] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    x: list[list] = field(default_factory=list)
    y: list[list] = field(default_factory=list)
    e: list[Root] = field(default_factory=list)

    def __getitem__(self, key):
        return self.__class__(
            *tuple(getattr(self, field.name) for field in fields(self)[:-3]),
            t=self.t[key],
            x=self.x[key],
            e=self.e[key],
        )

    def save(self, filename):
        e_idxs = [e.index for e in self.e]
        e_roots = [e.rootsfound for e in self.e]
        np.savez_compressed(
            filename,
            e_idxs=e_idxs,
            e_roots=e_roots,
            t=self.t,
            x=self.x,
            y=self.y,
            p=self.p,
            o=self.o,
        )

    @classmethod
    def load(cls, filename):
        data = dict(np.load(filename))
        data["e"] = [
            Root(index=int(ei), rootsfound=er)
            for ei, er in zip(data.pop("e_idxs"), data.pop("e_roots"))
        ]
        return cls(system=None, **data)


@dataclass
class Result(ResultBase, ResultMixin):
    pass


@dataclass
class CVodeResult(ResultBase, ResultMixin):
    cvode_mems: list[CVodeCreate] = field(default_factory=list)
    adjoint_mems: list[CVodeCreateB] = field(default_factory=list)


class ResultSegmentInterpolant(NamedTuple):
    interpolant: callable
    idx0: int
    idx1: int
    t0: float
    t1: float
    x0: list[float]
    x1: list[float]

    def __call__(self, t):
        return self.interpolant(t)


@dataclass
class ResultInterpolant:
    result: Result
    function: callable = lambda p, t, x: x
    # don't pass interpolants to init?
    # should state_Result be saved or just be an initvar? I back-references are OK so
    # we can keep it...
    interpolants: Optional[list[callable]] = None  # field(init=False)
    time_bounds: Optional[list[float]] = None  # field(init=False)
    time_comparison: callable = field(init=False)  # method
    interval_select: int = field(init=False)
    event_idxs: Optional[list] = None
    max_deg: int = 3

    def __post_init__(self):
        # make interpolants
        result = self.result
        function = self.function
        if self.event_idxs is None:
            # currently expect each root.index to be to the right of each event so is
            # the start of each segment, so zero belongs with this set by adding the
            # element len(result.t), the slice of pairs slice(idx[i], idx[i+1])
            # captures the whole segment including the last one.

            self.event_idxs = np.array(
                [root.index for root in result.e]
            )  # + [len(result.t)])
            if np.any(np.diff(self.event_idxs) < 1):
                breakpoint()

        event_idxs = self.event_idxs
        if self.time_bounds is None:
            self.time_bounds = np.empty(self.event_idxs.shape)
            self.time_bounds[:-1] = np.array(result.t)[event_idxs[:-1]]
            self.time_bounds[-1] = result.t[-1]
        if result.t[-1] < result.t[0]:
            self.time_comparison = self.time_bounds.__le__
            self.interval_select = -1
            self.time_sort = slice(None, None, -1)
        else:
            self.time_comparison = self.time_bounds.__ge__
            self.interval_select = 0
            self.time_sort = slice(None)

        if self.interpolants is None:
            # TODO figure out how to combine (and possibly reverse direction) state and
            # parameter jacobian of state equation to reduce number of calls (and
            # distance at each call), since this is potentially most expensive call -- I
            # guess only if the ODE depends on inner-loop solver? then split
            # coefficients
            # --> then also combine e.g., adjoint forcing function and jacobian
            # interpolant

            # expect integrand-term related functions to be cheap functions of state
            # point-wise along trajectory
            try:
                all_coeff_data = [
                    [
                        np.array(function(result.p, t, x)).squeeze()
                        for t, x in zip(result.t[idx0:idx1], result.x[idx0:idx1])
                    ]
                    for idx0, idx1 in zip(event_idxs[:-1], event_idxs[1:])
                ]
            except Exception as my_e:
                print(my_e)
                breakpoint()
                pass

            self.interpolants = []
            for idx0, idx1, coeff_data in zip(
                event_idxs[:-1],
                event_idxs[1:],
                all_coeff_data,
            ):
                if np.all(np.diff(coeff_data, axis=0) == 0.0):
                    try:
                        interp = ResultSegmentInterpolant(
                            make_interp_spline(
                                result.t[idx0],
                                [coeff_data[0]],
                                k=0,
                            ),
                            idx0,
                            idx1,
                            result.t[idx0],
                            result.t[idx1],
                            result.x[idx0],
                            result.x[idx1],
                        )
                    except Exception as my_e:
                        print(my_e)
                        breakpoint()

                else:
                    ts = result.t[idx0:idx1][self.time_sort]
                    coefs = coeff_data[self.time_sort]

                    for count, orig_idx in enumerate(np.where(np.diff(ts) <= 0)[0]):
                        idx = count + orig_idx
                        ts = ts[:idx] + ts[idx + 1 :]
                        coefs = coefs[:idx] + coefs[idx + 1 :]
                        print(
                            f"stripping non-decreasing {ts[idx]} -- creating result "
                            "interpolant"
                        )

                    try:
                        interp = ResultSegmentInterpolant(
                            make_interp_spline(
                                ts,
                                coefs,
                                k=min(
                                    self.max_deg, idx1 - idx0 - 1
                                ),  # not needed with adaptive step size!
                                # bc_type=["natural", "natural"],
                            ),
                            idx0,
                            idx1,
                            result.t[idx0],
                            result.t[idx1],
                            result.x[idx0],
                            result.x[idx1],
                        )
                    except Exception as my_e:
                        print(my_e)
                        breakpoint()
                        pass
                self.interpolants.append(interp)
                # if result.t[idx1] != result.t[idx0]

    def __call__(self, t):
        interval_idx = np.where(self.time_comparison(t))[0][self.interval_select] - 1
        return self.interpolants[interval_idx](t)

    def __iter__(self):
        yield from self.interpolants


@dataclass
class AdjointResultMixin:
    state_jacobian: ResultInterpolant
    forcing_function: ResultInterpolant
    state_result: Result = field(init=False)
    p: list[float] = field(init=False)

    def __post_init__(self):
        # TODO: validate forcing_function.result is the same?
        self.state_result = self.state_jacobian.result
        self.p = self.state_result.p


@dataclass
class AdjointResult(
    ResultBase,
    AdjointResultMixin,
):
    pass


class AdjointSystem(System):
    def __init__(
        self,
        adjoint_to,
        autonomous_dot,
        forcing_dot,
        initial_state,
        integrand_terms,
        terminal_terms,
        dim_state=0,
        dim_output=0,
        sen_system=None,
        dynamic_output=None,
        **solver_options,
    ):
        self.sen_system = sen_system
        self.adjoint_to = adjoint_to
        self.autonomous_dot = autonomous_dot
        self.forcing_dot = forcing_dot
        self._initial_state = initial_state
        self.dim_state = dim_state
        self.dim_output = dim_output
        self._integrand_terms = integrand_terms
        self._terminal_terms = terminal_terms

        self.dynamic_output = dynamic_output
        self.num_events = 1

        self.make_solver(**solver_options)

    def events(self, t, lamda):
        return np.array(
            t
            - self.result.state_result.t[
                self.result.state_result.e[self.segment_idx].index
            ]
        ).reshape(-1)

    def update(self, rootsfound, t, lamda):
        return lamda

    def time_generator(self):
        """ """
        result = self.result
        for (
            segment_idx,
            event,
            # jac_segment, forcing_segment,
        ) in zip(
            range(len(result.state_result.e) - 1, -1, -1),
            result.state_result.e[::-1],
            # result.state_jacobian[::-1], result.forcing_function[::-1],
        ):
            # the  event corresponds to the t0+ of the segment index which can be used
            # for selecting the jacobian and forcing segments
            self.segment_idx = segment_idx
            if segment_idx == 0:
                self.terminating = [0]
            # if event.index == 1:
            #    breakpoint()
            yield result.state_result.t[event.index]
            # nothing about segment_idx will get used the first time (terminal event)
            # and would be out-of-bounds if if tried -- simulate will use the yielded
            # time to determine inital condition, then calls next to set endpoint of
            # next segment then propoagate it.

            # so on first iteration, will not propoagate, will hit first iteration of
            # simulate's loop and call next-yield. so do terminal update (can optimize
            # code to reduce computations if needed) then loop.
            # self.update(event)

            # so update for terminal event is right, but could optimize performance for
            # special cases/maybe need distinct expression for update (to implement
            # update from from true terminal condition to effect of an immediate update)
            # then when this yields initial event (i.e., index=1) will THEN propoagate
            # backwards to initial condition.

        # then exits above loop, will have just simulated to t0 and handled any possible
        # update
        breakpoint()
        yield np.inf

    def initial_adjoint(self, t, x, *sens):
        p = self.result.p
        return np.array(self._initial_state(p, t, x, *sens)).reshape(-1)

    def attach_result(self, p):
        super().attach_result(p)
        self.adjoint_to.result = self.result

    def adjoint_terminal_terms(self, t, x, *sens):
        p = self.result.p
        return -np.array(self._terminal_terms(p, t, x, *sens)).reshape(-1)

    def terminal_terms(self, t, lamda):
        x = self.result.state_interp(t)
        return self.adjoint_terminal_terms(t, x)

    def integrand_terms(self, t, lamda):
        x = self.result.state_interp(t)
        return self.adjoint_integrand_terms(t, lamda, x)

    def adjoint_integrand_terms(self, t, lamda, x, *sens):
        p = self.result.p
        return -np.array(self._integrand_terms(p, t, lamda, x, *sens)).reshape(-1)

    def dots(self, t, lamda):
        x = self.result.state_interp(t)
        return self.adjoint_dots(t, lamda, x)

    def adjoint_dots(self, t, lamda, x, *sens):
        p = self.result.p
        return np.array(
            self.autonomous_dot(p, t, lamda, x, *sens)
            + self.forcing_dot(p, t, lamda, x, *sens)
        ).reshape(-1)

    def make_solver(self, solver_class, **solver_options):
        self.system_solver = solver_class(  # SolverSciPy( #SolverCVODE(
            system=self.adjoint_to,
            adj_system=self,
            sen_system=self.sen_system,
            **solver_options,
        )

    def adjoint_update(self, roots_found, t, lamda, x, *sens):
        breakpoint()

    def __call__(self, p, from_implementation=False):
        self.terminating = slice(0, 0)
        return super().__call__(p, from_implementation)


class SciPyAdjointSystem(System):
    def __init__(
        self,
        state_jac,
        dte_dxs,
        dh_dxs,
        **solver_options,
    ):
        """
        to support caching jacobians,

        adjoint solver will use parent simulate but without using events machinery
        time generator will call update method
        """
        self.state_jac = state_jac
        self.dte_dxs, self.dh_dxs = (
            dte_dxs,
            dh_dxs,
        )
        self.num_events = 1
        self.dynamic_output = None
        # self.adjoint_updates = adjoint_updates

        self.make_solver(**solver_options)

    def events(self, t, lamda):
        return np.array(
            t
            - self.result.state_result.t[
                self.result.state_result.e[self.segment_idx].index
            ]
        ).reshape(-1)

    def update(
        self,
        ignore_rootsfound,
        t,
        lamda,
    ):
        """
        for adjoint system, update will always get called for t1 of each segment,
        """
        lamda_res = self.result
        state_res = lamda_res.state_result
        event = state_res.e[self.segment_idx]
        # might be able to some magic to skip if nothing is done? but might only be for
        # terminal event anyway? maybe initial?
        active_update_idxs = np.where(event.rootsfound != 0)[0]
        last_lamda = lamda
        p = state_res.p
        state_idxp = event.index  # positive side of event
        te = state_res.t[state_idxp]

        if len(active_update_idxs) > 1:
            # not sure how to handle actual computation for this case, may need to
            # re-compute each update to get x at each update for correct partials and
            # time derivatives??
            # breakpoint()
            pass

        if len(active_update_idxs):
            xtep = state_res.x[state_idxp]
            ftep = state_res.system._dot(p, te, xtep)

            idxm = state_idxp - 1
            if state_res.t[idxm] != te:
                breakpoint()
            xtem = state_res.x[idxm]
            ftem = state_res.system._dot(p, te, xtem)

            if event is state_res.e[-1]:
                lamda_tem = last_lamda
                # TODO: add full transversality condition
                for event_channel in active_update_idxs[::-1]:
                    dh_dx = self.dh_dxs[event_channel](
                        p,
                        te,
                        xtem,
                    )
                    dte_dx_tr = self.dte_dxs[event_channel](p, te, xtem).T
                    lamda_tem = (dh_dx.T - dte_dx_tr @ (dh_dx @ ftem).T) @ last_lamda
                    last_lamda = lamda_tem
            else:
                for event_channel in active_update_idxs[::-1]:
                    dh_dx = self.dh_dxs[event_channel](
                        p,
                        te,
                        xtem,
                    )
                    dte_dx_tr = self.dte_dxs[event_channel](p, te, xtem).T
                    lamda_tem = (
                        dh_dx.T - dte_dx_tr @ (ftep - dh_dx @ ftem).T
                    ) @ last_lamda
                    last_lamda = lamda_tem
        else:
            lamda_tem = last_lamda

        lamda_tem = np.array(lamda_tem).squeeze()
        return lamda_tem

    def initial_state(self):
        return self.final_lamda

    def __call__(self, final_lamda, forcing_function, state_jacobian):
        self.terminating = slice(0, 0)
        self.result = AdjointResult(
            system=self,
            state_jacobian=state_jacobian,
            forcing_function=forcing_function,
        )
        self.final_lamda = final_lamda
        # I guess this could be put in result.x? then initial_state can pop and return?
        self.system_solver.simulate()
        result = self.result
        # self.result = None
        return result

    def dots(self, t, lamda):
        # TODO adjoint system takes jacobian and integrand terms, do matrix multiply and
        # vector add here. then can ust call jacobian term for jac method
        # then before simulate, could actually construct
        # who would own the data for interpolating ? maybe just a new (data) class that
        # also stores the interpolant? then adjointsystem simulate can create it if
        # it'snot provided
        return np.array(
            -self.result.state_jacobian.interpolants[self.segment_idx](t).T @ lamda
            - self.result.forcing_function.interpolants[self.segment_idx](t)
        ).reshape(-1)
        np.matmul(
            self.result.state_jacobian.interpolants[self.segment_idx](t).T,
            -lamda,
            out=lamdadot,
        )
        np.add(
            lamdadot,
            -self.result.forcing_function.interpolants[self.segment_idx](t),
            out=lamdadot,
        )

    def jac(
        self,
        t,
        lamda,
    ):
        return -self.result.state_jacobian.interpolants[self.segment_idx](t).T


@dataclass
class TrajectoryAnalysis:
    state_system: System

    # for constructing the output of the trajectory analysis
    integrand_terms: callable = None
    terminal_terms: callable = None

    sen_system: System = None
    adj_system: AdjointSystem = None

    cache_size: int = 1

    def __post_init__(self):
        self.cached_p = None

    def __call__(self, p):
        if self.cached_p is not None and np.all(self.cached_p == p):
            return self.cached_output
        self.cached_p = p
        result = self.res = self.state_system(p, self.from_implementation)
        self.cached_output = result.o
        return result.o


@dataclass
class SweepingGradientMethod:
    # per system
    adjoint_system: AdjointSystem
    p_x0_p_params: callable
    p_dots_p_params: callable  # could be cached, or just combined with integrand terms
    # per system's events (length number of events)
    dh_dps: list[callable]
    dte_dps: list[callable]

    # of length number of (trajectory) outputs
    p_integrand_terms_p_params: list[callable]
    p_terminal_terms_p_params: list[callable]
    p_integrand_terms_p_state: list[callable]
    p_terminal_terms_p_state: list[callable]

    def __call__(self, state_result):
        # iterate over each output, generate forcing functions, etc

        # TODO figure out combining and splitting to make this more efficient
        state_jacobian = ResultInterpolant(state_result, self.adjoint_system.state_jac)
        param_jacobian = ResultInterpolant(state_result, self.p_dots_p_params)

        p = state_result.p

        jac_rows = []

        for (
            p_integrand_term_p_params,
            p_terminal_term_p_params,
            p_integrand_term_p_state,
            p_terminal_term_p_state,
        ) in zip(
            self.p_integrand_terms_p_params,
            self.p_terminal_terms_p_params,
            self.p_integrand_terms_p_state,
            self.p_terminal_terms_p_state,
        ):
            # simulate adjoint for each one
            adjoint_forcing = ResultInterpolant(state_result, p_integrand_term_p_state)
            final_lamda = np.array(
                p_terminal_term_p_state(p, state_result.t[-1], state_result.x[-1])
            ).squeeze()
            adjoint_result = self.adjoint_system(
                final_lamda, adjoint_forcing, state_jacobian
            )

            p_integrand_p_param_interp = ResultInterpolant(
                state_result, p_integrand_term_p_params
            )

            adjoint_interp = ResultInterpolant(adjoint_result)
            jac_row = np.array(
                p_terminal_term_p_params(p, state_result.t[-1], state_result.x[-1])
            ).squeeze()

            # iterate over each segment/event for both  state + adjoint
            for (
                adjoint_segment,
                param_jacobian_segment,
                p_integrand_p_param_segment,
            ) in zip(
                adjoint_interp.interpolants[::-1],
                param_jacobian.interpolants,
                p_integrand_p_param_interp.interpolants,
            ):
                # compute discontinuous portion of gradient associated with each event
                # integrate continuous portion of gradient corresponding to
                # pre/suc-ceding segment
                adjoint_time_data = adjoint_result.t[
                    adjoint_segment.idx0 : adjoint_segment.idx1
                ][::-1]
                state_time_data = state_result.t[
                    param_jacobian_segment.idx0 : param_jacobian_segment.idx1
                ]

                # thought it would be faster to use the denser one, but don't achieve
                # required precision
                if len(adjoint_time_data) > len(state_time_data):
                    time_data = adjoint_time_data
                else:
                    time_data = state_time_data
                # time_data = state_time_data

                integrand_data = [
                    np.array(
                        adjoint_segment(t).T @ param_jacobian_segment(t)
                        + p_integrand_p_param_segment(t)
                    ).squeeze()
                    for t in time_data
                ]
                # idxs = np.where(np.diff(time_data) < 0)[0]
                for count, orig_idx in enumerate(np.where(np.diff(time_data) <= 0)[0]):
                    idx = count + orig_idx
                    time_data = time_data[:idx] + time_data[idx + 1 :]
                    integrand_data = integrand_data[:idx] + integrand_data[idx + 1 :]
                    print(f"stripping non-decreasing {time_data[idx]} -- calling SGM")
                # if idxs:
                #    breakpoint()
                #    pass
                integrand_interp = make_interp_spline(
                    time_data,
                    integrand_data,
                    # bc_type=["natural", "natural"],
                )
                integrand_antider = integrand_interp.antiderivative()
                jac_row += integrand_antider(time_data[-1]) - integrand_antider(
                    time_data[0]
                )

            for lamda_event, state_event in zip(adjoint_result.e, state_result.e[::-1]):
                # had to copy and paste this setup code from adjointsystem.update, not
                # sure if that's acceptable given the nature of these variables (mostly
                # selecting indices, a few hopefully cheap calls to time derivative,
                # etc)
                idxp = state_event.index  # positive side of event
                te = state_result.t[idxp]
                xtep = state_result.x[idxp]
                ftep = state_result.system._dot(p, te, xtep)

                idxm = idxp - 1
                if state_result.t[idxm] != te:
                    breakpoint()
                xtem = state_result.x[idxm]
                ftem = state_result.system._dot(p, te, xtem)

                active_update_idxs = np.where(state_event.rootsfound != 0)[0]

                idxm = lamda_event.index
                idxp = idxm - 1
                if adjoint_result.t[idxm] != adjoint_result.t[idxp]:
                    breakpoint()
                if not np.isclose(adjoint_result.t[idxm], te):
                    breakpoint()

                lamda_tep = adjoint_result.x[idxp]

                if state_event.index == 1:
                    # breakpoint()
                    jac_row += np.array(
                        lamda_tep[None, :] @ self.p_x0_p_params(p)
                    ).squeeze()
                if state_event is state_result.e[-1]:
                    ftep = np.zeros_like(ftep)
                for event_channel in active_update_idxs[::-1]:
                    dh_dp = self.dh_dps[event_channel](p, te, xtem)
                    dh_dx = adjoint_result.system.dh_dxs[event_channel](
                        p,
                        te,
                        xtem,
                    )
                    dte_dp = self.dte_dps[event_channel](p, te, xtem)

                    jac_row += np.array(
                        lamda_tep[None, :] @ (dh_dp - (ftep - dh_dx @ ftem) @ dte_dp)
                    ).squeeze()

            jac_rows.append(jac_row)

        jac_out = np.stack(jac_rows, axis=0)
        # breakpoint()
        print(jac_out)

        return np.stack(jac_rows, axis=0)


class TrajectoryAnalysisSGM:
    def __init__(
        self,
        trajectory_analysis,  # TrajectoryAnalysis
        # args for adjoint system
        dte_dxs=None,
        dh_dxs=None,
        state_jac=None,
        # for adjoint system, can provide here if state_system doesn't have
        # adjoint system solver options
        cache_size=1,
        # to construct SweepingGradientMethod
        p_x0_p_params=None,
        p_dots_p_params=None,
        dh_dps=None,
        dte_dps=None,
        # for constructing the output of the gradient
        p_integrand_terms_p_params=None,
        p_terminal_terms_p_params=None,
        p_integrand_terms_p_state=None,
        p_terminal_terms_p_state=None,
        **adjoint_options,
    ):
        if cache_size > 1:
            raise NotImplementedError
        self.cache_size = cache_size

        self.cached_p = None
        self.cached_output = None

        self.state_system = trajectory_analysis.state_system
        self.trajectory_analysis = trajectory_analysis

        if state_jac is None:
            if self.trajectory_analysis.state_system._jac is None:
                msg = "must provide state jacobian via state_system.jac or state_jac"
                raise ValueError(msg)
            state_jac = state_system._jac

        self.adjoint_system = SciPyAdjointSystem(
            state_jac=state_jac, dte_dxs=dte_dxs, dh_dxs=dh_dxs, **adjoint_options
        )

        self.sweeping_gradient_method = SweepingGradientMethod(
            adjoint_system=self.adjoint_system,
            p_x0_p_params=p_x0_p_params,
            p_dots_p_params=p_dots_p_params,
            dh_dps=dh_dps,
            dte_dps=dte_dps,
            p_terminal_terms_p_params=p_terminal_terms_p_params,
            p_integrand_terms_p_params=p_integrand_terms_p_params,
            p_terminal_terms_p_state=p_terminal_terms_p_state,
            p_integrand_terms_p_state=p_integrand_terms_p_state,
        )

    def __call__(self, p):
        _ = self.trajectory_analysis(p)
        return self.sweeping_gradient_method(self.trajectory_analysis.res)

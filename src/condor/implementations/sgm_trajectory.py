from enum import Enum

import numpy as np

import condor as co
import condor.solvers.sweeping_gradient_method as sgm
from condor import backend
from condor.backend import (
    callables_to_operator,
    expression_to_operator,
    symbol_class,
)
from condor.backend.operators import (
    concat,
    if_else,
    inf,
    jacobian,
    mod,
    pi,
    sin,
    substitute,
)

from .utils import options_to_kwargs


def get_state_setter(field, signature, on_field=None, subs=None):
    expr = field.flatten(on_field)
    if subs is not None:
        expr = substitute(expr, subs)
    func = expression_to_operator(
        signature,
        expr,
        f"{field._model_name}_{field._matched_to._name}_{field._name}",
    )
    func.expr = expr
    return func


def isnan(x):
    return isinstance(x, float) and np.isnan(x)


class TrajectoryAnalysis:
    """Implementation for :class:`~condor.contrib.TrajectoryAnalysis` model.

    All Options may be prefixed with ``state_`` or ``adjoint_`` to apply only to the
    forward or reverse solvers, respectively. Without either prefix, the option will be
    passed to both solvers. For additional details on the Solver options, see the
    :class:`~condor.solvers.sweeping_gradient_method` solvers.
    """

    class Solver(Enum):
        CVODE = sgm.SolverCVODE  #: currently unsupported
        dopri5 = sgm.SolverSciPyDopri5
        dop853 = sgm.SolverSciPyDop853

    def __init__(self, model_instance):
        model = model_instance.__class__
        self.model_instance = model_instance
        model_instance.options_dict = options_to_kwargs(model)
        self.construct(model, **model_instance.options_dict)
        self(model_instance)

    def construct(
        self,
        model,
        state_solver=None,
        adjoint_solver=None,
        solver=Solver.dopri5,
        **kwargs,
    ):
        state_options = {}
        adjoint_options = {}
        for k, v in kwargs.items():
            if k.startswith("state_"):
                state_options[k.replace("state_", "")] = v
            elif k.startswith("adjoint_"):
                adjoint_options[k.replace("adjoint_", "")] = v
            else:
                state_options[k] = v
                adjoint_options[k] = v

        self.model = model
        self.ode_model = ode_model = model._meta.primary

        self.x = model.state.flatten()
        self.lamda = backend.symbol_generator("lambda", model.state._count)

        self.p = model.parameter.flatten()

        self.simulation_signature = [
            self.p,
            self.model.t,
            self.x,
        ]

        self.traj_out_expr = model.trajectory_output.flatten()
        self.can_sgm = isinstance(self.p, symbol_class) and isinstance(
            self.traj_out_expr, symbol_class
        )

        traj_out_names = model.trajectory_output.list_of("name")

        integrand_terms = [
            elem.flatten_value(elem.integrand) for elem in model.trajectory_output
        ]
        self.traj_out_integrand = model.trajectory_output.flatten("integrand")
        traj_out_integrand_func = expression_to_operator(
            self.simulation_signature,
            self.traj_out_integrand,
            f"{model.__name__}_trajectory_output_integrand",
        )

        terminal_terms = [
            elem.flatten_value(elem.terminal_term) for elem in model.trajectory_output
        ]
        self.traj_out_terminal_term = model.trajectory_output.flatten("terminal_term")
        traj_out_terminal_term_func = expression_to_operator(
            self.simulation_signature,
            self.traj_out_terminal_term,
            f"{model.__name__}_trajectory_output_terminal_term",
        )

        self.state0 = get_state_setter(model.initial, [self.p])

        control_subs_pairs = {
            control.backend_repr: [control.default] for control in ode_model.modal
        }
        for mode in model._meta.modes:
            for act in mode.action:
                control_subs_pairs[act.match.backend_repr].insert(
                    -1, (mode.condition, act.backend_repr)
                )
        control_sub_expression = {}
        for k, v in control_subs_pairs.items():
            control_sub_expression[k] = substitute(if_else(*v), control_sub_expression)

        state_equation_func = get_state_setter(
            model.dot, self.simulation_signature, subs=control_sub_expression
        )

        lamda_jac = jacobian(state_equation_func.expr, self.x).T
        state_dot_jac_func = expression_to_operator(
            self.simulation_signature,
            lamda_jac.T,
            f"{ode_model.__name__}_state_jacobian",
        )

        self.e_exprs = []
        self.h_exprs = []
        at_time_slices = [
            sgm.NextTimeFromSlice(
                expression_to_operator(
                    [self.p],
                    # TODO in future allow t0 to occur at arbitrary times
                    [0.0, 0.0, inf],
                    f"{ode_model.__name__}_at_times_t0",
                )
            )
        ]

        terminating = []

        events = [e for e in model._meta.events]
        if (
            not isinstance(model.tf, (np.ndarray, float))
            or not np.isinf(model.tf).any()
        ):

            class Terminate(ode_model.Event):
                at_time = (model.tf,)
                terminate = True

            events += [Terminate]
            ode_model.Event._meta.subclasses = ode_model.Event._meta.subclasses[:-1]

        num_events = len(events)

        for event_idx, event in enumerate(events):
            if isnan(event.function) == isnan(event.at_time):
                msg = f"Event class `{event}` has set both `function` and `at_time`"
                raise ValueError(msg)
            if not isnan(getattr(event, "function", np.nan)):
                e_expr = event.function
            else:
                at_time = event.at_time
                if hasattr(at_time, "__len__"):
                    if len(at_time) in [2, 3]:
                        at_time = slice(*tuple(at_time))
                    else:
                        at_time = at_time[0]

                if isinstance(at_time, slice):
                    if at_time.step is None:
                        raise ValueError

                    at_time_start = 0 if at_time.start is None else at_time.start

                    e_expr = (
                        at_time.step
                        * sin(pi * (model.t - at_time_start) / at_time.step)
                        / (pi * 100)
                    )
                    # self.events(solver_res.values.t, solver_res.values.y, gs)
                    e_expr = mod(model.t - at_time_start, at_time.step)

                    # TODO: verify start and stop for at_time slice
                    if isinstance(at_time_start, symbol_class) or at_time_start != 0.0:
                        e_expr = e_expr * (model.t >= at_time_start)
                        # if there is a start offset, add a linear term to provide a
                        # zero-crossing at first occurance
                        pre_term = (at_time_start - model.t) * (
                            model.t <= at_time_start
                        )
                    else:
                        pre_term = 0

                    if at_time.stop is not None:
                        e_expr = e_expr * (model.t <= at_time.stop)
                        # if there is an end-time, hold constant to prevent additional
                        # zero crossings -- hopefully works even if stop is on an event
                        # post_term = (
                        #     (ode_model.t >= at_time.stop)
                        #     * at_time.step
                        #     * casadi.sin(
                        #         casadi.pi
                        #         * (at_time.stop - at_time_start)
                        #         / at_time.step
                        #     )
                        #     / casasadi.pi
                        # )
                        post_term = (model.t >= at_time.stop) * mod(
                            at_time.stop - at_time_start, at_time.step
                        )
                        at_time_stop = at_time.stop
                    else:
                        post_term = 0
                        at_time_stop = inf

                    e_expr = e_expr + pre_term + post_term

                    at_time_slices.append(
                        sgm.NextTimeFromSlice(
                            expression_to_operator(
                                [self.p],
                                concat([at_time_start, at_time_stop, at_time.step]),
                                f"{ode_model.__name__}_at_times_{event_idx}",
                            )
                        )
                    )
                else:
                    if isinstance(at_time, co.BaseElement):
                        at_time0 = at_time.backend_repr
                    else:
                        at_time0 = at_time
                    e_expr = at_time0 - model.t
                    at_time_slices.append(
                        sgm.NextTimeFromSlice(
                            expression_to_operator(
                                [self.p],
                                concat([at_time0, at_time0, inf]),
                                f"{ode_model.__name__}_at_times_{event_idx}",
                            )
                        )
                    )

            self.e_exprs.append(e_expr)

            if event.terminate:
                # For simupy, use nans to trigger termination; do we ever want to allow
                # an update to a terminating event?
                # self.h_exprs.append(
                #    casadi.MX.nan(ode_model.state._count)
                # )
                terminating.append(event_idx)

            h_expr = get_state_setter(
                event.update,
                self.simulation_signature,
                on_field=model.state,
                subs=control_sub_expression,
            )
            self.h_exprs.append(h_expr)

        if state_solver is None:
            state_solver = solver
        if adjoint_solver is None:
            adjoint_solver = solver
        state_options["solver_class"] = state_solver.value
        adjoint_options["solver_class"] = adjoint_solver.value

        if len(model.dynamic_output):
            self.y_expr = model.dynamic_output.flatten()
            self.y_expr = substitute(self.y_expr, control_sub_expression)
            self.dynamic_output_func = expression_to_operator(
                self.simulation_signature,
                self.y_expr,
                f"{ode_model.__name__}_dynamic_output",
            )
        else:
            self.dynamic_output_func = None

        self.state_system = sgm.System(
            dim_state=model.state._count,
            initial_state=self.state0,
            dot=state_equation_func,
            jac=state_dot_jac_func,
            time_generator=sgm.TimeGeneratorFromSlices(at_time_slices),
            events=expression_to_operator(
                self.simulation_signature,
                substitute(concat(self.e_exprs), control_sub_expression),
                f"{ode_model.__name__}_event",
            ),
            updates=self.h_exprs,
            num_events=num_events,
            terminating=terminating,
            dynamic_output=self.dynamic_output_func,
            **state_options,
        )
        self.state_system.model_instance = self.model_instance
        self.at_time_slices = at_time_slices

        self.p_state0_p_p_expr = jacobian(self.state0.expr, self.p)

        p_state0_p_p = expression_to_operator(
            [self.p],
            self.p_state0_p_p_expr,
            f"{model.__name__}_x0_jacobian",
        )
        p_state0_p_p.expr = self.p_state0_p_p_expr

        self.adjoint_signature = [
            self.p,
            self.x,
            self.model.t,
            self.lamda,
        ]
        # TODO: is there something more pythonic than repeated, very similar list
        # comprehensions?
        self.lamdaFs = [
            jacobian(terminal_term, self.x) for terminal_term in terminal_terms
        ]
        self.lamdaF_funcs = [
            expression_to_operator(
                self.simulation_signature,
                lamdaF,
                f"{model.__name__}_{traj_name}_lamdaF",
            )
            for lamdaF, traj_name in zip(self.lamdaFs, traj_out_names)
        ]
        self.gradFs = [
            jacobian(terminal_term, self.p) for terminal_term in terminal_terms
        ]
        self.gradF_funcs = [
            expression_to_operator(
                self.simulation_signature,
                gradF,
                f"{model.__name__}_{traj_name}_gradF",
            )
            for gradF, traj_name in zip(self.gradFs, traj_out_names)
        ]

        grad_jac = jacobian(state_equation_func.expr, self.p)

        param_dot_jac_func = expression_to_operator(
            self.simulation_signature,
            grad_jac,
            f"{model.__name__}_param_jacobian",
        )

        state_integrand_jacs = [
            jacobian(integrand_term, self.x).T for integrand_term in integrand_terms
        ]
        state_integrand_jac_funcs = [
            expression_to_operator(
                self.simulation_signature,
                ijac_expr,
                f"{model.__name__}_state_integrand_jac_{ijac_expr_idx}",
            )
            for ijac_expr_idx, ijac_expr in enumerate(state_integrand_jacs)
        ]

        param_integrand_jacs = [
            jacobian(integrand_term, self.p).T for integrand_term in integrand_terms
        ]
        param_integrand_jac_funcs = [
            expression_to_operator(
                self.simulation_signature,
                ijac_expr,
                f"{model.__name__}_param_integrand_jac_{ijac_expr_idx}",
            )
            for ijac_expr_idx, ijac_expr in enumerate(param_integrand_jacs)
        ]
        # TODO figure out how to combine (and possibly reverse direction) to reduce
        # number of calls, since this is potentially most expensive call with inner
        # loop solvers,

        # lamda updates
        # grad updates
        self.dte_dxs = []
        self.dh_dxs = []

        self.dte_dps = []
        self.dh_dps = []

        for event, e_expr, h_expr in zip(events, self.e_exprs, self.h_exprs):
            dg_dx = jacobian(e_expr, self.x)
            dg_dt = jacobian(e_expr, model.t)
            dg_dp = jacobian(e_expr, self.p)

            dte_dx = dg_dx / (dg_dx @ state_equation_func.expr)
            dte_dp = -dg_dp / (dg_dx @ state_equation_func.expr + dg_dt)

            dh_dx = jacobian(h_expr.expr, self.x)
            dh_dp = jacobian(h_expr.expr, self.p)

            # te = ode_model.t

            # xtem = self.x
            # xtep = h_expr(self.p, te, xtem)

            # ftem = state_equation_func(self.p, te, xtem)
            # ftep = state_equation_func(self.p, te, xtep)
            # delta_fs = ftep - ftem
            # delta_xs = xtep - xtem

            # lamda_tep = self.lamda
            # eyen = casadi.MX.eye(lamda_tep.shape[0])
            # lamda_tem = (
            #     (eyen + dte_dx.T @ ftem.T) @ dh_dx.T - dte_dx.T @ ftep.T
            # ) @ lamda_tep

            # delta_lamdas = lamda_tem - lamda_tep

            # # TODO update for forcing function
            # lamda_dot_tem = -state_dot_jac_func(self.p, te, xtem).T @ lamda_tem
            # lamda_dot_tep = -state_dot_jac_func(self.p, te, xtep).T @ lamda_tep

            # delta_lamda_dots = lamda_dot_tem - lamda_dot_tep

            # jac_update = (
            #     lamda_tep.T @ dh_dp - lamda_tep.T @ (ftep - dh_dx @ ftem) @ dte_dp
            # )

            # jac_update = substitute(jac_update, control_sub_expression)

            # self.jac_updates.append(
            #     casadi.Function(
            #         f"{event.__name__}_jac_update",
            #         self.adjoint_signature,
            #         [jac_update],
            #     )
            # )

            dte_dx = substitute(dte_dx, control_sub_expression)
            self.dte_dxs.append(
                expression_to_operator(
                    self.simulation_signature,
                    dte_dx,
                    f"{event.__name__}_dte_dx",
                )
            )
            self.dte_dxs[-1].expr = dte_dx

            dte_dp = substitute(dte_dp, control_sub_expression)
            self.dte_dps.append(
                expression_to_operator(
                    self.simulation_signature, dte_dp, f"{event.__name__}_dte_dp"
                )
            )
            self.dte_dps[-1].expr = dte_dp

            dh_dx = substitute(dh_dx, control_sub_expression)
            self.dh_dxs.append(
                expression_to_operator(
                    self.simulation_signature, dh_dx, f"{event.__name__}_dh_dx"
                )
            )
            self.dh_dxs[-1].expr = dh_dx

            dh_dp = substitute(dh_dp, control_sub_expression)
            self.dh_dps.append(
                expression_to_operator(
                    self.simulation_signature, dh_dp, f"{event.__name__}_dh_dp"
                )
            )
            self.dh_dps[-1].expr = dh_dp

        self.trajectory_analysis_sgm = sgm.TrajectoryAnalysisSGM(
            state_system=self.state_system,
            integrand_terms=traj_out_integrand_func,
            terminal_terms=traj_out_terminal_term_func,
            dte_dxs=self.dte_dxs,
            dh_dxs=self.dh_dxs,
            state_jac=state_dot_jac_func,
            p_x0_p_params=p_state0_p_p,
            p_dots_p_params=param_dot_jac_func,
            dh_dps=self.dh_dps,
            dte_dps=self.dte_dps,
            p_terminal_terms_p_params=self.gradF_funcs,
            p_integrand_terms_p_params=param_integrand_jac_funcs,
            p_terminal_terms_p_state=self.lamdaF_funcs,
            p_integrand_terms_p_state=state_integrand_jac_funcs,
            **adjoint_options,
        )

        wrapper_funcs = [
            self.trajectory_analysis_sgm.function,
            self.trajectory_analysis_sgm.jacobian,
        ]

        self.callback = callables_to_operator(
            wrapper_funcs,
            self,
            jacobian_of=None,
            input_symbol=self.p,
            output_symbol=self.traj_out_expr,
        )
        self.callback.construct()

        if not self.can_sgm:
            return

    def __call__(self, model_instance):
        self.callback.from_implementation = True
        self.args = model_instance.parameter.flatten()
        self.out = self.callback(self.args)
        self.callback.from_implementation = False

        if hasattr(self.trajectory_analysis_sgm, "res"):
            res = self.trajectory_analysis_sgm.res
            model_instance._res = res
            model_instance.t = np.array(res.t)
            model_instance.bind_field(
                self.model.state.wrap(
                    np.array(res.x).T,
                )
            )
            if self.dynamic_output_func:
                yy = np.empty((model_instance.t.size, self.model.dynamic_output._count))
                for idx, (t, x) in enumerate(zip(res.t, res.x)):
                    yy[idx, None] = self.dynamic_output_func(res.p, t, x).T
                model_instance.bind_field(
                    self.model.dynamic_output.wrap(
                        yy.T,
                    )
                )

        model_instance.bind_field(
            self.model.trajectory_output.wrap(
                self.out,
            )
        )

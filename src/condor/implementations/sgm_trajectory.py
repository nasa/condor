from enum import Enum
from functools import cached_property

import numpy as np

import condor as co
import condor.solvers.sweeping_gradient_method as sgm
from condor import backend
from condor.backend import (
    FunctionOperator,
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
    unstack,
)
from condor.fields import BaseElement

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
        self.state_options = state_options = {}
        self.adjoint_options = adjoint_options = {}
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

        self.traj_out_integrand = model.trajectory_output.flatten("integrand")
        self.traj_out_integrand_func = expression_to_operator(
            self.simulation_signature,
            self.traj_out_integrand,
            f"{model.__name__}_trajectory_output_integrand",
        )

        self.traj_out_terminal_term = model.trajectory_output.flatten("terminal_term")
        self.traj_out_terminal_term_func = expression_to_operator(
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
        self.control_sub_expression = control_sub_expression = {}
        for k, v in control_subs_pairs.items():
            control_sub_expression[k] = substitute(if_else(*v), control_sub_expression)

        self.state_equation_func = state_equation_func = get_state_setter(
            model.dot, self.simulation_signature, subs=control_sub_expression
        )

        self.e_exprs = []
        self.h_exprs = []

        if isinstance(model.t0, BaseElement):
            t0 = model.t0.backend_repr
        elif isinstance(model.t0, (backend.symbol_class, float, int, np.ndarray)):
            t0 = model.t0
        else:
            unexpcted_t0 = "unexpected value for t0"
            raise ValueError(unexpcted_t0)
        at_time_slices = [
            sgm.NextTimeFromSlice(
                expression_to_operator(
                    [self.p],
                    # TODO in future allow t0 to occur at arbitrary times
                    concat([t0, t0, inf]),
                    f"{ode_model.__name__}_at_times_t0",
                )
            )
        ]

        terminating = []

        self.events = events = [e for e in model._meta.events]
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
            jac=None if solver is not self.Solver.CVODE else self.state_jac_func,
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
            dim_output=model.trajectory_output._count,
            integrand_terms=self.traj_out_integrand_func,
            terminal_terms=self.traj_out_terminal_term_func,
            **state_options,
        )
        self.state_system.model_instance = self.model_instance
        self.at_time_slices = at_time_slices

        self.trajectory_analysis_nom = sgm.TrajectoryAnalysis(
            state_system=self.state_system,
            integrand_terms=self.traj_out_integrand_func,
            terminal_terms=self.traj_out_terminal_term_func,
        )

        self.callback = FunctionOperator(
            function=self.trajectory_analysis_nom,
            get_jacobian_func=self.generate_sgm_jacobian if self.can_sgm else None,
            input_symbol=self.p,
            output_symbol=self.traj_out_expr,
            implementation=self,
            jacobian_of=None,
        )

    @cached_property
    def state_jacobian_expr(self):
        return jacobian(self.state_equation_func.expr, self.x)

    @cached_property
    def state_jac_func(self):
        state_dot_jac_func = expression_to_operator(
            self.simulation_signature,
            self.state_jacobian_expr,
            f"{self.ode_model.__name__}_state_jacobian",
        )
        return state_dot_jac_func

    def generate_sgm_jacobian(self, jacobian_of):
        state_equation_func = self.state_equation_func
        # lamda_jac = self.state_jacobian_expr.T
        model = self.model
        control_sub_expression = self.control_sub_expression

        state_param_jac = jacobian(state_equation_func.expr, self.p)

        param_dot_jac_func = expression_to_operator(
            self.simulation_signature,
            state_param_jac,
            f"{model.__name__}_param_jacobian",
        )

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

        self.lamdaFs = []
        self.lamdaF_funcs = []
        self.gradFs = []
        self.gradF_funcs = []

        traj_out_names = model.trajectory_output.list_of("name")
        terminal_terms = [
            elem.flatten_value(elem.terminal_term) for elem in model.trajectory_output
        ]
        integrand_terms = [
            elem.flatten_value(elem.integrand) for elem in model.trajectory_output
        ]

        for terminal_term_stacked, out_name in zip(terminal_terms, traj_out_names):
            for idx, terminal_term in enumerate(unstack(terminal_term_stacked)):
                self.lamdaFs.append(jacobian(terminal_term, self.x))
                self.lamdaF_funcs.append(
                    expression_to_operator(
                        self.simulation_signature,
                        self.lamdaFs[-1],
                        f"{model.__name__}_{out_name}_{idx}_lamdaF",
                    )
                )
                self.gradFs.append(jacobian(terminal_term, self.p))
                self.gradF_funcs.append(
                    expression_to_operator(
                        self.simulation_signature,
                        self.gradFs[-1],
                        f"{model.__name__}_{out_name}_{idx}_gradF",
                    )
                )

        state_integrand_jacs = []
        state_integrand_jac_funcs = []
        param_integrand_jacs = []
        param_integrand_jac_funcs = []
        for integrand_term_stacked, out_name in zip(integrand_terms, traj_out_names):
            for idx, integrand_term in enumerate(unstack(integrand_term_stacked)):
                term_label = f"{out_name}_{idx}"
                state_integrand_jacs.append(jacobian(integrand_term, self.x).T)
                state_integrand_jac_funcs.append(
                    expression_to_operator(
                        self.simulation_signature,
                        state_integrand_jacs[-1],
                        f"{model.__name__}_state_integrand_jac_{term_label}",
                    )
                )
                param_integrand_jacs.append(jacobian(integrand_term, self.p).T)
                param_integrand_jac_funcs.append(
                    expression_to_operator(
                        self.simulation_signature,
                        param_integrand_jacs[-1],
                        f"{model.__name__}_param_integrand_jac_{term_label}",
                    )
                )

        # TODO figure out how to combine (and possibly reverse direction) to reduce
        # number of calls, since this is potentially most expensive call with inner
        # loop solvers,

        # lamda updates
        # grad updates
        self.dte_dxs = []
        self.dh_dxs = []

        self.dte_dps = []
        self.dh_dps = []

        for event, e_expr, h_expr in zip(self.events, self.e_exprs, self.h_exprs):
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

        self.adjoint_initial_condition_expr = jacobian(
            self.traj_out_terminal_term, self.x
        )
        self.adjoint_initial_condition_func = expression_to_operator(
            self.simulation_signature,
            self.adjoint_initial_condition_expr,
            f"{model.__name__}_adjoint_initial_expr",
        )

        self.lamda_full = backend.symbol_generator(
            "lambda", (model.state._count, model.trajectory_output._count)
        )

        adjoint_full_signature = (
            self.p,
            self.model.t,
            self.lamda_full,
            self.x,
        )
        autonomous_adjoint_expr = -self.state_jacobian_expr.T @ self.lamda_full
        forcing_adjoint_expr = -concat(state_integrand_jacs, axis=1)

        adjoint_deriv_terminal = jacobian(self.traj_out_terminal_term, self.p)
        adjoint_deriv_integrand = (
            concat(param_integrand_jacs, axis=1).T + self.lamda_full.T @ state_param_jac
        )

        self.adjoint_system = sgm.AdjointSystem(
            dim_state=model.trajectory_output._count * model.state._count,
            dim_output=adjoint_deriv_terminal.reshape((-1, 1)).shape[0],
            initial_state=self.adjoint_initial_condition_func,
            adjoint_to=self.state_system,
            autonomous_dot=expression_to_operator(
                adjoint_full_signature,
                autonomous_adjoint_expr,
                f"{model.__name__}_adjoint_autonomous_dots",
            ),
            forcing_dot=expression_to_operator(
                adjoint_full_signature,
                forcing_adjoint_expr,
                f"{model.__name__}_adjoint_forcing_dots",
            ),
            integrand_terms=expression_to_operator(
                adjoint_full_signature,
                adjoint_deriv_integrand,
                f"{model.__name__}_adjoint_integrand_terms",
            ),
            terminal_terms=expression_to_operator(
                self.simulation_signature,
                adjoint_deriv_terminal,
                f"{model.__name__}_adjoint_terminal_terms",
            ),
            **self.adjoint_options,
        )

        self.adjoint_analysis = sgm.TrajectoryAnalysis(
            state_system=self.adjoint_system,
            integrand_terms=self.traj_out_integrand_func,
            terminal_terms=self.traj_out_terminal_term_func,
        )
        self.adjoint_analysis.from_implementation = False
        return FunctionOperator(
            function=self.adjoint_analysis,
            get_jacobian_func=None,
            # model_name=model.__name__+"Jacobian",
            implementation=self,
            input_symbol=self.p,
            output_symbol=self.traj_out_expr,
            jacobian_of=jacobian_of,  # same as self.callback, currently
        )

        # keeping this for now until porting is finished
        self.trajectory_analysis_sgm = sgm.TrajectoryAnalysisSGM(
            trajectory_analysis=self.trajectory_analysis_nom,
            dte_dxs=self.dte_dxs,
            dh_dxs=self.dh_dxs,
            state_jac=self.state_jac_func,
            p_x0_p_params=p_state0_p_p,
            p_dots_p_params=param_dot_jac_func,
            dh_dps=self.dh_dps,
            dte_dps=self.dte_dps,
            p_terminal_terms_p_params=self.gradF_funcs,
            p_integrand_terms_p_params=param_integrand_jac_funcs,
            p_terminal_terms_p_state=self.lamdaF_funcs,
            p_integrand_terms_p_state=state_integrand_jac_funcs,
            **self.adjoint_options,
        )

        return FunctionOperator(
            function=self.trajectory_analysis_sgm,
            get_jacobian_func=None,
            # model_name=model.__name__+"Jacobian",
            implementation=self,
            input_symbol=self.p,
            output_symbol=self.traj_out_expr,
            jacobian_of=jacobian_of,  # same as self.callback, currently
        )

    @staticmethod
    def bind_result(model_instance, res):
        model_instance._res = res
        model_instance.t = np.array(res.t)

        model_instance.bind_field(
            model_instance.__class__.state.wrap(
                res.x.T,
            )
        )
        model_instance.bind_field(
            model_instance.__class__.dynamic_output.wrap(
                res.y.T,
            )
        )

    @staticmethod
    def load(model, filename):
        model_instance = model.__new__(model)
        TrajectoryAnalysis.bind_result(model_instance, sgm.Result.load(filename))
        model_instance.bind_field(model.parameter.wrap(model_instance._res.p))
        model_instance.input_kwargs = model_instance.parameter.asdict()
        return model_instance

    def save(self=None, model_instance=None, filename=None):
        if self is None:
            self = TrajectoryAnalysis  # noqa: PLW0642
        if model_instance._res is not None:
            model_instance._res.save(filename)

    def __call__(self, model_instance):
        self.callback.function.from_implementation = True
        # self.callback.function.from_implementation = False
        self.args = model_instance.parameter.flatten()
        self.out = self.callback(self.args)
        self.callback.function.from_implementation = False

        if hasattr(self.trajectory_analysis_nom, "res"):
            res = self.trajectory_analysis_nom.res
            yy = np.empty((res.t.size, self.model.dynamic_output._count))
            if self.dynamic_output_func:
                for idx, (t, x) in enumerate(zip(res.t, res.x)):
                    yy[idx, None] = self.dynamic_output_func(res.p, t, x).T
                # sweepyng solver doesn't compute outputs at every time step anymore to
                # reduce run-time of solver. Could tell res the size of the output to
                # move this processing there, but it makes sense to only do the
                # calculation if it's at the user level
            res.y = yy
            self.bind_result(model_instance, res)
        else:
            model_instance._res = None

        model_instance.bind_field(
            self.model.trajectory_output.wrap(
                self.out,
            )
        )

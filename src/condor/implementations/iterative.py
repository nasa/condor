import inspect
from enum import Enum, auto

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

import condor as co
from condor.backend import (
    expression_to_operator,
    symbol_class,
)
from condor.backend.operators import (
    concat,
    jacobian,
    unstack,
)
from condor.solvers.casadi_warmstart_wrapper import (
    CasadiIterationCallback,
    CasadiNlpsolWarmstart,
    CasadiRootfinderWarmstart,
)

from .utils import options_to_kwargs


class InitializerMixin:
    def __init__(self, model_instance):
        model = model_instance.__class__
        model_instance.options_dict = options_to_kwargs(model)
        self.construct(model, **model_instance.options_dict)
        self(model_instance)

    def construct(self, model):
        # could be shared between optimization and algebraic
        self.warm_start = model.variable.flatten("warm_start")
        self.initializer = model.variable.flatten("initializer")
        self.initializer_func = expression_to_operator(
            [self.p],
            self.initializer,
            f"{model.__name__}_initializer",
        )

    def write_initializer(self, model_instance):
        for k, v in model_instance.variable.asdict().items():
            model_var = getattr(self.model, k)
            if (
                np.array(model_var.warm_start).any()
                and not isinstance(model_var.initializer, symbol_class)
                and not isinstance(v, symbol_class)
            ):
                model_var.initializer = v


class AlgebraicSystem(InitializerMixin):
    """Implementation for :class:`AlgebraicSystem` model.

    Options
    --------
    atol : float
       absolute tolerance (default 1e-12)
    rtol : float
       relative tolerance (default 1e-12)
    warm_start : bool
       flag indicating whether subsequent calls should initialize the variable values to
       the last-run solution (default True)
    max_iter : int
       maximum number of iterations before terminating solver (default 100)
    error_on_fail : bool
       flag indicating whether to raise an error if the solver fails to converge within
       specified tolerance before reaching max_iter (default False)
    """

    """
    exact_hessian : bool
       flag indicating whether to use second-order gradient information or broyden
    re_initialize : ???
       ???
    default_initializer : ???
       ???
    """

    def construct(
        self,
        model,
        # standard options
        atol=1e-12,
        rtol=1e-12,
        warm_start=True,
        exact_hessian=True,
        max_iter=100,
        # new options
        # slice for re-initializing at every call. will support indexing
        re_initialize=None,
        default_initializer=0.0,
        error_on_fail=False,
    ):
        if re_initialize is None:
            re_initialize = slice(None)

        rootfinder_options = dict(
            error_on_fail=error_on_fail, abstol=atol, abstolStep=rtol, max_iter=max_iter
        )
        self.x = model.variable.flatten()
        self.g0 = model.residual.flatten()
        self.g1 = model.output.flatten()
        self.p = model.parameter.flatten()

        self.residual_func = expression_to_operator(
            [self.x, self.p], self.g0, f"{model.__name__}_output"
        )

        self.output_func = expression_to_operator(
            [self.x, self.p], self.g1, f"{model.__name__}_output"
        )

        self.model = model

        InitializerMixin.construct(self, model)

        self.callback = CasadiRootfinderWarmstart(
            primary_function=self.residual_func,
            output_function=self.output_func,
            n_variable=model.variable._count,
            n_parameter=model.parameter._count,
            init_var=self.initializer_func,
            warm_start=self.warm_start,
            method_string="newton",
            options=rootfinder_options,
            model_name=model.__name__,
        )

    def __call__(self, model_instance):
        self.call_args = model_instance.parameter.flatten()
        out = self.callback(self.call_args)
        self.var_out = self.model.variable.wrap(out[: self.model.variable._count])
        model_instance.bind_field(self.var_out)

        self.out_out = self.model.output.wrap(out[self.model.variable._count :])
        model_instance.bind_field(self.out_out)

        if hasattr(self.callback, "last_r"):
            # TODO: verify this will never become stale
            # TODO: it seems a bit WET to wrap each field and then, eventually, bind
            # each field.
            resid = self.callback.last_r
            model_instance.bind_field(
                self.model.residual.wrap(resid), symbols_to_instance=False
            )

        for k, v in model_instance.variable.asdict().items():
            model_var = getattr(self.model, k)
            if (
                model_var.warm_start
                and not isinstance(model_var.initializer, symbol_class)
                and not isinstance(v, symbol_class)
            ):
                model_var.initializer = v


class OptimizationProblem(InitializerMixin):
    """Implementation base class for :class:`OptimizationProblem` model.

    Options
    --------
    init_callback : callable
       callback with signature , called when an optimization problem is evalauted. Once
       when embedded or every time as a standalone.
    iter_callback : callable
       callback with signature , called at each iteration of the
       :class:`CasadiNlpsolImplementation` (only IPOPT) and :class:`SciPyBase`
       subclass optimization implementations.
    """

    # take an OptimizationProblem model with or without iteration spec and other Options
    # process options, create appropriate callback hooks
    # create appropriate operator --
    # "direct" nlpsol, nlpsol with warmstartwrapper, operator for scipy
    # technically even direct nlpsol should go through a set of functions so that it can
    # be used with a different backend
    # use expression_to_operator to create f, g, and if needed init_x0 and update_x0
    # hook to call other non-casadi optimizer using scipy infrastructure? cvxopt, cvxpy
    # was there a similar library to cvxopt?
    # maybe 3 different scipy implementations with a common base class
    # single casadi nlpsol implementation
    #
    def make_warm_start(self, x0=None, lam_g0=None, lam_x0=None):
        if x0 is not None:
            self.x0 = x0
        if lam_g0 is not None:
            self.lam_g0 = lam_g0
        if lam_x0 is not None:
            self.lam_x0 = lam_x0

    default_options = {}

    def construct(
        self,
        model,
        iter_callback=None,
        init_callback=None,
        **options,
    ):
        self.model = model
        self.options = self.default_options.copy()
        self.options.update(options)

        self.iter_callback = iter_callback
        self.init_callback = init_callback

        self.f = getattr(model, "objective", 0)
        self.has_p = bool(len(model.parameter))
        self.p = model.parameter.flatten()
        self.x = model.variable.flatten()
        self.g = model.constraint.flatten()

        InitializerMixin.construct(self, model)

        self.objective_func = expression_to_operator(
            [self.x, self.p],
            self.f,
            f"{model.__name__}_objective",
        )
        self.constraint_func = expression_to_operator(
            [self.x, self.p],
            self.g,
            f"{model.__name__}_constraint",
        )

        self.lbx = model.variable.flatten("lower_bound")
        self.ubx = model.variable.flatten("upper_bound")
        self.lbg = model.constraint.flatten("lower_bound")
        self.ubg = model.constraint.flatten("upper_bound")

    def load_initializer(self, model_instance):
        if self.has_p:
            self.eval_p = p = model_instance.parameter.flatten()
            initializer_args = [self.x0, p]
        else:
            initializer_args = [self.x0]
        if not self.has_p or not isinstance(p, symbol_class):
            self.x0 = self.initializer_func(*initializer_args)
            self.x0 = np.array(self.x0).reshape(-1)

    def __call__(self, model_instance):
        self.load_initializer(model_instance)
        self.run_optimizer(model_instance)
        self.write_initializer(model_instance)


class CasadiNlpsolImplementation(OptimizationProblem):
    """Implementation layer for casadi nlpsol for :class:`OptimizationProblem` models.

    Options
    --------
    method : CasadiNlpsolImplementation.Method
        value from method enum to specify supported methods
    exact_hessian : bool
        flag to use second order gradient information; use limited Broyden update
        otherwise
    calc_lam_x : bool
        flag to calculate the lagrange multipliers solution, used for IPOPT to perform a
        true warm start

    **options
        remaining keyword arguments are passed to casadi's nlpsol's constructor's
        solver-specific options argument. See :attr:`method_default_options` for the
        defaults


    """

    class Method(Enum):
        ipopt = auto()
        snopt = auto()  #: currently unsupported
        qrsqp = auto()
        fatrop = auto()

    method_strings = {
        Method.ipopt: "ipopt",
        Method.snopt: "snopt",
        Method.qrsqp: "sqpmethod",
        Method.fatrop: "fatrop",
    }

    method_default_options = {
        Method.ipopt: dict(
            # warm_start_init_point="no",
            warm_start_init_point="yes",
            sb="yes",  # suppress banner
        ),
        Method.snopt: dict(
            warm_start_init_point=False,
        ),
    }

    @property
    def default_options(self):
        """derived property to get a copy of :attr:`method_default_options` or an empty
        dict"""
        return self.method_default_options.get(self.method, dict())

    def construct(
        self,
        model,
        exact_hessian=True,  # False -> ipopt alias for limited memory
        method=Method.ipopt,
        calc_lam_x=False,
        # ipopt specific, default = False may help with computing sensitivity. To do
        # proper warm start, need to provide lam_x and lam_g so I assume need calc_lam_x
        # = True
        **options,
    ):
        self.method = method
        super().construct(model, **options)

        self.nlp_args = dict(f=self.f, x=self.x, g=self.g)
        if co.backend.get_symbol_data(self.p).size > 1:
            self.nlp_args["p"] = self.p

        self.nlp_opts = dict()
        if self.iter_callback is not None:
            self.nlp_opts["iteration_callback"] = CasadiIterationCallback(
                "iter", self.nlp_args, model, self.iter_callback
            )

        self.method = method
        if self.method is CasadiNlpsolImplementation.Method.ipopt:
            self.nlp_opts.update(
                print_time=False,
                ipopt=self.options,
                # print_level = 0-2: nothing, 3-4: summary, 5: iter table (default)
                # tol=1E-14, # tighter tol for sensitivty
                # accept_every_trial_step="yes",
                # max_iter=1000,
                # constr_viol_tol=10.,
                bound_consistency=True,
                clip_inactive_lam=True,
                calc_lam_x=calc_lam_x,
                calc_lam_p=False,
            )
            # additional options from https://groups.google.com/g/casadi-users/c/OdRQKR13R50/m/bIbNoEHVBAAJ
            # to try to get sensitivity from ipopt. so far no...
            if not exact_hessian:
                self.nlp_opts["ipopt"].update(
                    hessian_approximation="limited-memory",
                )

        elif self.method is CasadiNlpsolImplementation.Method.snopt:
            pass
        elif self.method is CasadiNlpsolImplementation.Method.qrsqp:
            self.nlp_opts.update(
                qpsol="qrqp",
                qpsol_options=dict(
                    print_iter=False,
                    error_on_fail=False,
                ),
                # qpsol='osqp',
                verbose=False,
                tol_pr=1e-16,
                tol_du=1e-16,
                # print_iteration=False,
                print_time=False,
                # hessian_approximation= "limited-memory",
                print_status=False,
            )
            if self.options["print_level"] == 0:
                self.nlp_opts["qpsol_options"] = dict(
                    print_iter=False,
                    print_header=False,
                    print_info=False,
                )

        self.callback = CasadiNlpsolWarmstart(
            primary_function=self.objective_func,
            constraint_function=self.constraint_func,
            n_variable=model.variable._count,
            n_parameter=model.parameter._count,
            init_var=self.initializer_func,
            warm_start=self.warm_start,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            method_string=CasadiNlpsolImplementation.method_strings[method],
            options=self.nlp_opts,
            model_name=model.__name__,
        )

        self.optimizer = self.callback.optimizer
        self.nlp_args = self.callback.nlp_args
        self.lam_g0 = None
        self.lam_x0 = None

    def __call__(self, model_instance):
        self.run_optimizer(model_instance)
        self.write_initializer(model_instance)

    def run_optimizer(self, model_instance):
        if self.init_callback is not None:
            self.init_callback(model_instance.parameter, self.nlp_opts)

        run_p = model_instance.parameter.flatten()
        if "p" not in self.nlp_args:
            var_out = self.callback(np.array([]))
        else:
            var_out = self.callback(run_p)

        if isinstance(var_out, symbol_class):
            out = dict(
                x=var_out,
                p=run_p,
                g=self.constraint_func(var_out, run_p),
                f=self.objective_func(var_out, run_p),
                lam_g=None,
                lam_x=None,
            )
        else:
            out = self.callback.out
            model_instance._stats = self.callback._stats

        model_instance.bind_field(self.model.variable.wrap(out["x"]))
        model_instance.bind_field(self.model.constraint.wrap(out["g"]))
        model_instance.objective = np.array(out["f"]).squeeze()
        self.out = out
        self.lam_g0 = out["lam_g"]
        self.lam_x0 = out["lam_x"]


class ScipyMinimizeBase(OptimizationProblem):
    """Base implementation class for SciPy minimize for :class:`OptimizationProblem`
    models.

    Options
    --------
    **options
        keyword options are passed directly to scipy.minimize's options keyword argument
    """

    def construct(
        self,
        model,
        **options,
    ):
        super().construct(model, **options)
        self.f_func = self.objective_func
        self.f_jac_func = expression_to_operator(
            [self.x, self.p],
            jacobian(self.f, self.x),
            f"{model.__name__}_objective_jac",
        )
        self.g_split = unstack(self.g)
        initializer_args = [self.x]
        if self.has_p:
            initializer_args += [self.p]

        self.initial_at_construction = model.variable.flatten("initializer")

        # setattr(self, f"{field._name}_initializer_func",
        self.initializer_func = expression_to_operator(
            initializer_args,
            self.initial_at_construction,
            f"{model.__class__.__name__}_var_initializer",
        )

        self.x0 = self.initial_at_construction.copy()

    def prepare_constraints(self, extra_args):
        return []

    def run_optimizer(self, model_instance):
        extra_args = (self.eval_p,) if self.has_p else ([],)

        scipy_constraints = self.prepare_constraints(extra_args)

        if self.init_callback is not None:
            self.init_callback(
                model_instance.parameter,
                {**self.options, "lbx": self.lbx, "ubx": self.ubx},
            )

        min_out = minimize(
            lambda *args: self.f_func(*args).toarray().squeeze(),
            self.x0,
            jac=lambda *args: self.f_jac_func(*args).toarray().squeeze(),
            method=self.method_string,
            args=extra_args,
            constraints=scipy_constraints,
            bounds=np.vstack([self.lbx, self.ubx]).T,
            # tol = 1E-9,
            # options=dict(disp=True),
            options=self.options,
            callback=SciPyIterCallbackWrapper.create_or_none(
                self.model, model_instance.parameter, self.iter_callback
            ),
        )

        model_instance.bind_field(self.model.variable.wrap(min_out.x))
        model_instance.objective = min_out.fun
        self.x0 = min_out.x
        self.stats = model_instance._stats = min_out


class ScipyCG(ScipyMinimizeBase):
    method_string = "CG"


class ScipySLSQP(ScipyMinimizeBase):
    method_string = "SLSQP"

    def construct(self, model, *args, **kwargs):
        super().construct(model, *args, **kwargs)
        self.equality_con_exprs = []
        self.inequality_con_exprs = []
        self.con = []
        for g, lbg, ubg in zip(self.g_split, self.lbg, self.ubg):
            if lbg == ubg:
                self.equality_con_exprs.append(g - lbg)
            else:
                if lbg > -np.inf:
                    self.inequality_con_exprs.append(g - lbg)
                if ubg < np.inf:
                    self.inequality_con_exprs.append(ubg - g)

        if self.equality_con_exprs:
            self.equality_con_expr = concat(self.equality_con_exprs)
            self.eq_g_func = expression_to_operator(
                [self.x, self.p],
                self.equality_con_expr,
                f"{model.__name__}_equality_constraint",
            )
            self.eq_g_jac_func = expression_to_operator(
                [self.x, self.p],
                jacobian(self.equality_con_expr, self.x),
                f"{model.__name__}_equality_constraint_jac",
            )
            self.con.append(
                dict(
                    type="eq",
                    fun=lambda x, p: self.eq_g_func(x, p).toarray().squeeze().T,
                    jac=self.eq_g_jac_func,
                )
            )

        if self.inequality_con_exprs:
            self.inequality_con_expr = concat(self.inequality_con_exprs)
            self.ineq_g_func = expression_to_operator(
                [self.x, self.p],
                self.inequality_con_expr,
                f"{model.__name__}_inequality_constraint",
            )
            self.ineq_g_jac_func = expression_to_operator(
                [self.x, self.p],
                jacobian(self.inequality_con_expr, self.x),
                f"{model.__name__}_inequality_constraint_jac",
            )
            self.con.append(
                dict(
                    type="ineq",
                    fun=lambda x, p: self.ineq_g_func(x, p).toarray().squeeze().T,
                    jac=self.ineq_g_jac_func,
                )
            )

    def prepare_constraints(self, extra_args):
        scipy_constraints = self.con
        for con in scipy_constraints:
            con["args"] = extra_args
        return scipy_constraints


class ScipyTrustConstr(ScipyMinimizeBase):
    method_string = "trust-constr"

    def construct(
        self,
        model,
        exact_hessian=True,  # only trust-constr can use hessian at all
        keep_feasible=True,  # flag that goes to scipy trust constr linear constraints
        **options,
    ):
        super().construct(model, **options)
        self.keep_feasible = keep_feasible
        g_split = self.g_split
        g_jacs = [jacobian(g, self.x) for g in self.g_split]

        nonlinear_flags = [
            not jacobian(g_jac, self.x).nnz() or not jacobian(g_expr, self.p).nnz()
            for g_jac, g_expr in zip(g_jacs, g_split)
            # could ignore dependence on parameters, but to be consistent with
            # ipopt specification require parameters within function not in
            # bounds
        ]

        # process nonlinear constraints
        nonlinear_g_fun_exprs = [
            g_expr
            for g_expr, is_nonlinear in zip(g_split, nonlinear_flags)
            if is_nonlinear
        ]
        self.num_nonlinear_g = len(nonlinear_g_fun_exprs)
        if self.num_nonlinear_g:
            nonlinear_g_jac_exprs = [
                g_jac
                for g_jac, is_nonlinear in zip(g_jacs, nonlinear_flags)
                if is_nonlinear
            ]

            self.g_func = expression_to_operator(
                [self.x, self.p],
                concat(nonlinear_g_fun_exprs),
                f"{model.__name__}_nonlinear_constraint",
            )

            self.g_jac_func = expression_to_operator(
                [self.x, self.p],
                concat(nonlinear_g_jac_exprs),
                f"{model.__name__}_nonlinear_constraint_jac",
            )

            self.nonlinear_ub = np.array(
                [
                    ub
                    for ub, is_nonlinear in zip(self.ubg, nonlinear_flags)
                    if is_nonlinear
                ]
            )

            self.nonlinear_lb = np.array(
                [
                    lb
                    for lb, is_nonlinear in zip(self.lbg, nonlinear_flags)
                    if is_nonlinear
                ]
            )

        # process linear constraints
        linear_a_exprs = [
            g_jac
            for g_jac, is_nonlinear in zip(g_jacs, nonlinear_flags)
            if not is_nonlinear
        ]
        self.num_linear_g = len(linear_a_exprs)
        if self.num_linear_g:
            self.linear_jac_func = expression_to_operator(
                [self.p],
                concat(linear_a_exprs),
                f"{model.__name__}_A",
            )

            self.linear_ub = np.array(
                [
                    ub
                    for ub, is_nonlinear in zip(self.ubg, nonlinear_flags)
                    if not is_nonlinear
                ]
            )

            self.linear_lb = np.array(
                [
                    lb
                    for lb, is_nonlinear in zip(self.lbg, nonlinear_flags)
                    if not is_nonlinear
                ]
            )

    def prepare_constraints(self, extra_args):
        scipy_constraints = []
        if self.num_linear_g:
            scipy_constraints.append(
                LinearConstraint(
                    A=self.linear_jac_func(*extra_args).sparse(),
                    ub=self.linear_ub,
                    lb=self.linear_lb,
                    keep_feasible=self.keep_feasible,
                )
            )
        if self.num_nonlinear_g:
            scipy_constraints.append(
                NonlinearConstraint(
                    fun=(
                        lambda *args: self.g_func(*args, *extra_args)
                        .toarray()
                        .squeeze()
                    ),
                    jac=(lambda *args: self.g_jac_func(*args, *extra_args).sparse()),
                    lb=self.nonlinear_lb,
                    ub=self.nonlinear_ub,
                )
            )
        return scipy_constraints


class SciPyIterCallbackWrapper:
    """Wrapper for iter_callback function to use with SciPy minimize, used internally"""

    @classmethod
    def create_or_none(cls, model, parameters, callback=None):
        if callback is None:
            return None
        return cls(model, parameters, callback)

    def __init__(self, model, parameters, callback):
        self.iter = 0
        self.callback = callback
        self.model = model
        self.parameters = parameters
        self.pass_instance = len(inspect.signature(self.callback).parameters) > 4

    def __call__(self, xk, res=None):
        variable = self.model.variable.wrap(xk)
        instance = self.model.from_values(
            **self.parameters.asdict(), **variable.asdict()
        )
        callback_args = (
            self.iter,
            variable,
            instance.objective,
            instance.constraint,
        )
        if self.pass_instance:
            callback_args += (instance,)

        self.callback(*callback_args)
        self.iter += 1

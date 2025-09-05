from dataclasses import dataclass

import casadi
import numpy as np

from condor.backends.casadi import (
    CasadiFunctionCallback,
    callables_to_operator,
    symbol_class,
)


@dataclass
class CasadiWarmstartWrapperBase:
    """
    baseclass to wrap Casadi nlpsol and rootfinder with a warmstart, without requiring
    casadi in the condor implementations module
    """

    """

    Future work:
    if we wanted to release a casadi-only library that supplied these solvers in way
    that was independent of condor, we would need to pass in enough information to make
    a casadi Callback -- jacobians, hessians, etc.

    since we do not intend to release a casadi-only library that supplies these solvers,
    we can assume we are working within condor and have at least an "Operator" and we
    can use condor to perform calculus on any backend. Would still require transforming
    generic Operator into casadi-specific (casadi.Callback)

    Similalry, the callback construction is going straight to a casadi.Callback, and
    in the future actually use the condor backend shims to build an operator for
    whichever backend is being used

    for now, since condor only has casadi as a backend, we will assume it is a casadi
    Callback, and don't need to perform any modification for casadi to know how do
    autodiff on it. And we will just become a casadi.Callback, rather than using the
    shim to construct an operator.

    addiitonal notes:

    I like having all the direct casadi stuff outside of implementations.iterative,
    so not having casadi-speicifc constructions, but maybe it's OK for this case since
    the solver is casadi itself?
    basically because of the API requirements of casadi, either condor needs to better
    define its MIMO (or maybe still MISO?) calculus API so we can go from
    {condor.backend} operator to casadi for NLPSOL or we allow a bit of tighter
    coupling.

    for now, assume I'm getting casadi MISO operator (which for now means
    casadi.Function of x and p, but in the future should be backend-agnostic Operator)
    and we'll just take for granted we can construct a full casadi expression for
    objective and constraints.

    and even more confusing/inconsistent, casadi rootfinder takes MIMO function that
    x,p input and residual,explicit outptus.

    and accctually, casadinlpsolwarmstart should not directly be a
    CasadiFunctionCallback, it should be generic


    """
    #: callable f(x, p). nlpsol's objective or rootfinder's residual
    primary_function: callable

    n_variable: int
    n_parameter: int

    #: initial guess function of p,  x0(p)
    init_var: list
    warm_start: list
    method_string: str

    model_name: str = ""
    options: dict = None

    def __post_init__(self):
        self.update_warmstart(self.warm_start)
        if self.options is None:
            self.options = {}

        self.last_x = None
        self.last_p = None

    def update_warmstart(self, warm_start):
        self.warm_start = warm_start
        self.any_warm_start = np.any(self.warm_start)
        self.all_warm_start = np.all(self.warm_start)

    def update_guess(self, run_p):
        x0 = self.init_var(run_p)
        if self.any_warm_start and self.last_x is not None:
            for idx, warm_start in enumerate(self.warm_start):
                if warm_start:
                    x0[idx] = self.last_x[idx]
        return x0

    def construct_jacobian_and_callback(self):
        # put in try block because if exact_hessian is False, won't be able to compute
        # jacobian
        try:
            self.placeholder_jacobian = self.placeholder_func_xp.jacobian()
            # self.placeholder_jacobian = casadi.Function(
            #    f"{self.model_name}_func_xp__jac_p",
            #    [x,p],
            #    [casadi.jacobian(self.sym_solved_inp_x0["x"], p)],
            # )
        except RuntimeError:
            self.placeholder_jacobian = None

        # stuff attributes expected by CasadiFunctionCallback, including jacobian
        # relationship
        self.wrapper_func = self.function
        self.jacobian_of = None
        self.opts = {}

        if self.placeholder_jacobian is not None:
            try:
                self.placeholder_hessian = self.placeholder_jacobian.jacobian()
            except RuntimeError:
                self.placeholder_hessian = None

            wrap_funcs = [self.eval_jacobian]
            if self.placeholder_hessian is not None:
                wrap_funcs += [self.eval_hessian]

            self.jacobian = callables_to_operator(
                wrapper_funcs=wrap_funcs,
                model_name=self.model_name,
                jacobian_of=self,
                opts=self.opts,
            )
        else:
            self.jacobian = None

        casadi.Callback.__init__(self)
        self.construct()

    def eval_jacobian(self, run_p):
        if np.all(self.last_p == run_p):
            # defensive check that last solution was right, then just use last_x
            jac_x, jac_p = self.placeholder_jacobian(self.last_x, run_p, [])
        else:
            # if not, try to update guess?
            jac_x, jac_p = self.placeholder_jacobian(
                self.update_guess(run_p), run_p, []
            )
        return jac_p

    def eval_hessian(self, run_p):
        if np.all(self.last_p == run_p):
            # defensive check that last solution was right, then just use last_x
            *_jac_nz, jac_pp, jac_px = self.placeholder_hessian(
                self.last_x, run_p, [], [], []
            )
        else:
            # if not, try to update guess?
            *_jac_nz, jac_pp, jac_px = self.placeholder_hessian(
                self.update_guess(run_p), run_p, [], [], []
            )
        return jac_pp, jac_px


@dataclass
class CasadiNlpsolWarmstart(CasadiWarmstartWrapperBase, CasadiFunctionCallback):
    """
    nlpsol
    """

    lbx: list = None  # lower bounds for vari
    ubx: list = None  # upper bounds for variable

    constraint_function: ... = None  # optional constraint function g(x, p)
    lbg: ... = None
    # bound for constraint function so lbg <= g(x, p); required if constraint function
    # is provided, not used if not
    ubg: ... = None
    # bound for constraint function so g(x, p) <= ubg; required if constraint function
    # is provided, not used if not

    def __post_init__(self):
        super().__post_init__()
        self.objective_function = self.primary_function

        self.output_symbol = self.x = x = symbol_class.sym(
            f"{self.model_name}_variable", (self.n_variable, 1)
        )
        self.input_symbol = self.p = p = symbol_class.sym(
            f"{self.model_name}_parameter", (self.n_parameter, 1)
        )

        f = self.objective_function(x, p)
        g = self.constraint_function(x, p)

        self.nlp_args = dict(f=f, x=x, g=g)
        if self.n_parameter:
            self.nlp_args["p"] = self.p

        self.lam_g0 = None
        self.lam_x0 = None

        # self.variable_at_construction.copy()
        self.apply_lamda_initial = (
            self.options.get("ipopt", {}).get("warm_start_init_point", "no") == "yes"
        )

        self.optimizer = casadi.nlpsol(
            f"{self.model_name}_optimizer",
            self.method_string,
            self.nlp_args,
            self.options,
        )
        if not self.init_var.jacobian()(self.p, []).nnz():
            sym_x0 = self.init_var(np.zeros(self.p.shape))

        # create a symbolic representation of optimization result that assumes init_var
        # is actually independent of parameters (is a constant)
        # create corresponding function (so CasadiFunctionCallback can get sparsity,
        # etc.

        self.sym_solved_const_x0 = self.optimizer(
            x0=sym_x0,
            p=p,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
        )

        self.placeholder_func = casadi.Function(
            f"{self.model_name}_func",
            [p],
            [self.sym_solved_const_x0["x"]],
        )

        # create a symbolic representation of optimizaiton result that has a provided x0
        self.sym_solved_inp_x0 = self.optimizer(
            x0=x,
            p=p,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        # create corresponding function to make a callable jacobian which will be used
        # by eval_jacobian. Didnt work to create a jacobian of the expression, but does
        # seem to work take jacobian of function
        self.placeholder_func_xp = casadi.Function(
            f"{self.model_name}_func_xp",
            [x, p],
            [self.sym_solved_inp_x0["x"]],
        )

        self.construct_jacobian_and_callback()

    def function(self, run_p):
        run_kwargs = dict(
            ubx=self.ubx,
            lbx=self.lbx,
            ubg=self.ubg,
            lbg=self.lbg,
        )

        if self.n_parameter:
            run_kwargs.update(p=run_p)

        x0 = self.update_guess(run_p)
        if self.any_warm_start and self.last_x is not None:
            # breakpoint()
            pass
        run_kwargs.update(x0=x0)
        if self.all_warm_start and self.apply_lamda_initial:
            if self.lam_x0 is not None:
                run_kwargs.update(
                    lam_x0=self.lam_x0,
                )
            if self.lam_g0 is not None:
                run_kwargs.update(
                    lam_g0=self.lam_g0,
                )

        if np.any(run_kwargs["ubx"] <= run_kwargs["lbx"]):
            breakpoint()
        if np.all(self.last_p == run_p):
            # print("repeat call", run_p)
            out = self.out
        else:
            # print("new call running with", run_p)
            self.run_kwargs = run_kwargs
            out = self.out = self.optimizer(**run_kwargs)
            self._stats = self.optimizer.stats()

        self.last_p = run_p
        self.last_x = out["x"]
        self.lam_g0 = out["lam_g"]
        self.lam_x0 = out["lam_x"]

        return out["x"]


@dataclass
class CasadiRootfinderWarmstart(CasadiWarmstartWrapperBase, CasadiFunctionCallback):
    output_function: callable = None

    def __post_init__(self):
        super().__post_init__()
        self.residual_function = self.primary_function

        self.output_symbol = self.x = x = symbol_class.sym(
            f"{self.model_name}_variable", (self.n_variable, 1)
        )
        self.input_symbol = self.p = p = symbol_class.sym(
            f"{self.model_name}_parameter", (self.n_parameter, 1)
        )

        g0 = self.residual_function(x, p)
        g1 = self.output_function(x, p)

        name = self.model_name

        self.rootfinder_f_arg = casadi.Function(
            f"{name}_rootfinder_f_arg", [x, p], [g0, g1]
        )
        self.rootfinder = casadi.rootfinder(
            f"{name}_rootfinder",
            "newton",
            self.rootfinder_f_arg,
            self.options,
        )

        if not self.init_var.jacobian()(self.p, []).nnz():
            sym_x0 = self.init_var(np.zeros(self.p.shape))

        self.sym_solved_const_x0 = [casadi.vertcat(*self.rootfinder(sym_x0, self.p))]
        # self.sym_solved_const_x0 = self.rootfinder(sym_x0, self.p)
        self.placeholder_func = casadi.Function(
            f"{name}_func",
            [p],
            self.sym_solved_const_x0,
        )

        self.sym_solved_inp_x0 = [casadi.vertcat(*self.rootfinder(self.x, self.p))]
        # self.sym_solved_inp_x0 = self.rootfinder(self.x, self.p)
        self.placeholder_func_xp = casadi.Function(
            f"{name}_func",
            [x, p],
            self.sym_solved_const_x0,
        )

        self.total_iters = 0
        self.construct_jacobian_and_callback()

    def function(self, run_p):
        x0 = self.update_guess(run_p)
        if np.all(self.last_p == run_p):
            # print("repeat call", run_p)
            out = self.out
        else:
            # print("new call running with", run_p)
            out = self.out = self.rootfinder(x0, run_p)
            self._stats = self.rootfinder.stats()
            self.total_iters += self._stats["iter_count"]
        self.last_p = run_p
        self.last_x = out[0]
        self.last_r = self.residual_function(self.last_x, self.last_p)
        self.last_o = out[1]
        return casadi.vertcat(*out)


class CasadiIterationCallback(casadi.Callback):
    def __init__(self, name, nlpdict, model, iteration_callback, opts=None):
        casadi.Callback.__init__(self)
        self.iteration_callback = iteration_callback
        self.nlpdict = nlpdict
        self.model = model
        self.iter = 0
        if opts is None:
            opts = {}
        self.construct(name, opts)

    def get_n_in(self):
        return casadi.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        n = casadi.nlpsol_out(i)
        return n

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = casadi.nlpsol_out(i)
        if n == "f":
            return casadi.Sparsity.dense(1)
        elif n in ("x", "lam_x"):
            return self.nlpdict["x"].sparsity()
        elif n in ("g", "lam_g"):
            g = self.nlpdict["g"]
            if not hasattr(g, "sparsity"):
                return casadi.Sparsity.dense(np.atleast_2d(g).shape)
            return g.sparsity()
        return casadi.Sparsity(0, 0)

    def eval(self, args):
        # x, f, g, lam_x, lam_g, lam_p
        x, f, g, *_ = args
        var_data = self.model.variable.wrap(x)
        constraint_data = self.model.constraint.wrap(g)
        self.iteration_callback(self.iter, var_data, f, constraint_data)
        self.iter += 1
        return [0]

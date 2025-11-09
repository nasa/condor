# -*- coding: utf-8 -*-
"""Utility wrappers around CasADi primitives with explanatory comments."""

# Standard library imports
import contextlib  # Used to suppress evaluation errors when attempting to simplify expressions

# Third-party imports
import numpy as np  # NumPy provides array helpers for non-symbolic inputs

import casadi  # CasADi supplies symbolic math primitives used throughout this module
import condor.backends.casadi as backend  # Backend-specific helpers (symbol class, etc.)

# useful but not sure if all backends would have:
# symvar -- list all symbols present in expression
# depends_on
#

# Re-export common mathematical constants from CasADi/NumPy for convenience
pi = casadi.pi  # π constant defined by CasADi for symbolic expressions
inf = casadi.inf  # Positive infinity compatible with CasADi expressions
nan = np.nan  # NaN is only available through NumPy, so we reuse it here

# Aliases that mirror NumPy API naming for consistency across backends
mod = casadi.fmod  # Symbolic modulus operator
trace = casadi.trace  # Matrix trace
cross = casadi.cross  # Vector cross product

# Trigonometric and exponential helpers (CasADi implementations)
atan = casadi.atan
atan2 = casadi.atan2
tan = casadi.tan
sin = casadi.sin
cos = casadi.cos
asin = casadi.asin
acos = casadi.acos
exp = casadi.exp
log = casadi.log
log10 = casadi.log10
sqrt = casadi.sqrt

# Rounding helpers
floor = casadi.floor
ceil = casadi.ceil

# Matrix factory helpers
eye = casadi.MX.eye
ones = casadi.MX.ones

# Scalar helpers
fabs = casadi.fabs
sign = casadi.sign


def diag(v, k=0):
    """Construct a diagonal matrix from ``v``; ``k`` other than zero unsupported."""
    if k != 0:
        msg = "Not supported for this backend"
        raise ValueError(msg)
    if not hasattr(v, "shape"):
        # try to concat list/tuple of elements
        v = concat(v)
    return casadi.diag(v)


def vector_norm(x, ord=2):
    """Compute vector norms while delegating to CasADi implementations."""
    if ord == 2:
        return casadi.norm_2(x)
    if ord == 1:
        return casadi.norm_1(x)
    if ord == inf:
        return casadi.norm_inf(x)


def sum(x, axis=None):
    """Aggregate ``x`` along ``axis`` while matching NumPy's signature."""
    if axis is None:
        return casadi.sum(x)
    return casadi.sum(x, axis)


def clip(val, amax, amin):
    """Clip ``val`` between ``amin`` and ``amax`` using symbolic conditionals."""
    val = casadi.if_else(val > amax, amax, val)
    val = casadi.if_else(val < amin, amin, val)
    return val


# Directly expose CasADi's linear solver helper
solve = casadi.solve


def concat(arrs, axis=0):
    """Implement ``concat`` from the array API for CasADi/NumPy inputs."""
    if not arrs:
        return arrs
    if np.any([isinstance(arr, backend.symbol_class) for arr in arrs]):
        if axis == 0:
            return casadi.vcat(arrs)
        elif axis in (1, -1):
            return casadi.hcat(arrs)
        else:
            msg = "Casadi only supports matrices"
            raise ValueError(msg)
    else:
        return np.concatenate([np.atleast_2d(arr) for arr in arrs], axis=axis)


def unstack(arr, axis=0):
    """Split ``arr`` along ``axis`` following CasADi semantics."""
    if axis == 0:
        return casadi.vertsplit(arr)
    elif axis in (1, -1):
        return casadi.horzsplit(arr)


def zeros(shape=(1, 1)):
    """Return a symbolic zero matrix matching the requested ``shape``."""
    return backend.symbol_class(*shape)


def min(x, axis=None):
    """Symbolic minimum supporting only the global reduction ``axis=None``."""
    if not isinstance(x, backend.symbol_class):
        x = concat(x)
    if axis is not None:
        msg = "Only axis=None supported"
        raise ValueError(msg)
    return casadi.mmin(x)


def max(x, axis=None):
    """Symbolic maximum supporting only the global reduction ``axis=None``."""
    if not isinstance(x, backend.symbol_class):
        x = concat(x)
    if axis is not None:
        msg = "Only axis=None supported"
        raise ValueError(msg)
    return casadi.mmax(x)


# Friendly message for unsupported Jacobian computations
unsupported_jacobian_message = (
    "jacobian of matrix expression wrt matrix variable not yet supported"
)


def jacobian(of, wrt):
    """Jacobian of expression ``of`` with respect to symbols ``wrt``."""
    """
    we can apply jacobian to ExternalSolverWrapper but it's a bit clunky because need
    symbol_class expressions for IO, and to evalaute need to create a Function. Not sure
    how to create a backend-generic interface for this. When do we want an expression vs
    a callable? Maybe the overall process is right (e.g., within an optimization
    problem, will have a variable flat input, and might just want the jac_expr)

    Example to extend from docs/howto_src/table_basics.py

       flat_inp = SinTable.input.flatten()
       wrap_inp = SinTable.input.wrap(flat_inp)
       instance = SinTable(**wrap_inp.asdict()) # needed so callback obj isn't destroyed
       wrap_out = instance.output
       flat_out = wrap_out.flatten()
       jac_expr = ops.jacobian(flat_out, flat_inp)
       from condor import backend
       jac = backend.expression_to_operator(flat_inp, jac_expr, "my_jac")
       #jac = casadi.Function("my_jac", [flat_inp], [jac_expr])
       jac(0.)
    """
    if of.size and wrt.size:
        transpose_in = False
        if isinstance(wrt, backend.symbol_class) and wrt.op() == casadi.OP_TRANSPOSE:
            transpose_in = True
            wrt = wrt.dep()

        if transpose_in and np.all(np.array(of.shape + wrt.shape) > 1):
            raise NotImplementedError(unsupported_jacobian_message)

        jac = casadi.jacobian(of, wrt)

        return jac

    else:
        return backend.symbol_class(0, np.prod(wrt.shape))


def jac_prod(of, wrt, rev=True):
    """Create directional derivative via CasADi's ``jtimes`` helper."""
    return casadi.jtimes(of, wrt, not rev)


def substitute(expr, subs):
    """Apply substitution map ``subs`` onto expression ``expr`` safely."""
    in_subs = subs
    subs = {}
    for k, v in in_subs.items():
        use_k = k.T if k.op() in (casadi.OP_RESHAPE, casadi.OP_TRANSPOSE) else k
        use_v = np.atleast_2d(v) if isinstance(v, np.ndarray) else v
        if use_k.shape != (1, 1) and use_k.shape == v.shape[::-1]:
            use_v = use_v.T
        subs[use_k] = use_v
        if (
            getattr(use_k, "shape", (1, 1)) != getattr(subs[use_k], "shape", (1, 1))
            and use_v.shape
        ):
            msg = f"did not find compatible shape, currently have {use_k} --> {use_v}"
            raise ValueError(msg)

    expr = casadi.graph_substitute(expr, subs.keys(), subs.values())

    if isinstance(expr, backend.symbol_class) and expr.is_constant():
        expr = expr.to_DM().toarray()

    # if expr is the output of a single call, try to to eval it
    if isinstance(expr, backend.symbol_class) and (
        (
            expr.op() == casadi.OP_GETNONZEROS
            and expr.dep().op() == -1
            and expr.dep().dep().is_call()
        )
        or (expr.op() == -1 and expr.dep().is_call())
    ):
        with contextlib.suppress(RuntimeError):
            expr = casadi.evalf(expr)

    return expr


def if_else(*conditions_actions, short_circuit=False):
    """Symbolic representation of an ``if``/``elif``/``else`` control flow."""
    """
    symbolic representation of a if/else control flow

    Parameters
    ---------
    *conditions_actions : list of (condition, value) pairs, ending with else_value

    Example
    --------

    The expression::

        value = if_else(
            (condition0, value0),
            (codnition1, value1),
            ...
            else_value
        )


    is equivalent to the numerical code::

        if condition0:
            value = value0
        elif condition1:
            value = value1
        ...
        else:
            value = else_value

    """
    if len(conditions_actions) == 1:
        else_action = conditions_actions[0]
        if isinstance(else_action, tuple):
            msg = "if_else requires an else_action to be provided"
            raise ValueError(msg)
        return else_action
    condition, action = conditions_actions[0]
    if hasattr(condition, "shape") and np.prod(condition.shape) > 1:
        msg = "if_else conditions should be a scalar"
        raise ValueError(msg)
    remainder = if_else(*conditions_actions[1:], short_circuit=short_circuit)
    return casadi.if_else(condition, action, remainder, short_circuit)

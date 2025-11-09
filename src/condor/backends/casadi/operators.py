# -*- coding: utf-8 -*-
"""Utility wrappers around CasADi primitives with clear documentation and optimized API consistency."""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import contextlib  # Used to suppress runtime evaluation errors during symbolic simplifications

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np  # Provides array utilities for non-symbolic or mixed inputs
import casadi  # Symbolic math library for optimization and control applications
import condor.backends.casadi as backend  # Backend-specific helpers (e.g., symbol class)

# ---------------------------------------------------------------------------
# Mathematical constants and aliases (CasADi + NumPy)
# ---------------------------------------------------------------------------
pi = casadi.pi      # Symbolic π constant
inf = casadi.inf    # Symbolic infinity (CasADi-compatible)
nan = np.nan        # NaN (NumPy-defined, since CasADi doesn’t provide one)

# ---------------------------------------------------------------------------
# Common operation aliases to mirror NumPy API naming
# ---------------------------------------------------------------------------
mod = casadi.fmod
trace = casadi.trace
cross = casadi.cross

# ---------------------------------------------------------------------------
# Trigonometric and exponential functions (CasADi implementations)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Rounding and scalar functions
# ---------------------------------------------------------------------------
floor = casadi.floor
ceil = casadi.ceil
fabs = casadi.fabs
sign = casadi.sign

# ---------------------------------------------------------------------------
# Matrix constructors (symbolic)
# ---------------------------------------------------------------------------
eye = casadi.MX.eye
ones = casadi.MX.ones

# ---------------------------------------------------------------------------
# Public exports for controlled namespace and faster imports
# ---------------------------------------------------------------------------
__all__ = [
    "pi","inf","nan","mod","trace","cross","atan","atan2","tan","sin","cos","asin","acos",
    "exp","log","log10","sqrt","floor","ceil","eye","ones","fabs","sign","diag",
    "vector_norm","sum","clip","solve","concat","unstack","zeros","min","max",
    "jacobian","jac_prod","substitute","if_else"
]

# ---------------------------------------------------------------------------
# Matrix and vector helpers
# ---------------------------------------------------------------------------
def diag(v, k=0):
    """Construct a diagonal matrix from `v`. Offsets (`k != 0`) are not supported for this backend."""
    if k != 0:
        raise ValueError("Not supported for this backend.")
    if not hasattr(v, "shape"):
        v = concat(v)
    return casadi.diag(v)


def vector_norm(x, ord=2):
    """Compute vector norms using CasADi’s symbolic norm functions."""
    if ord == 2:
        return casadi.norm_2(x)
    if ord == 1:
        return casadi.norm_1(x)
    if ord == inf:
        return casadi.norm_inf(x)
    raise NotImplementedError(f"Norm order {ord} not supported for this backend.")


def sum(x, axis=None):
    """Aggregate `x` along `axis`, mimicking NumPy’s behavior with a CasADi fallback."""
    try:
        if axis is None:
            return casadi.sum(x)
        return casadi.sum(x, axis)
    except TypeError:
        # Some CasADi builds don’t support the `axis` argument
        if axis is None:
            return casadi.sum1(casadi.sum2(x))
        raise NotImplementedError("Axis-specific sum not supported by this CasADi version.")


def clip(val, amin, amax):
    """Clip `val` between `amin` and `amax` using symbolic conditional expressions (NumPy-compatible order)."""
    val = casadi.if_else(val < amin, amin, val)
    val = casadi.if_else(val > amax, amax, val)
    return val


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------
solve = casadi.solve


# ---------------------------------------------------------------------------
# Concatenation and unstacking utilities
# ---------------------------------------------------------------------------
def concat(arrs, axis=0):
    """Concatenate arrays or symbols along the given `axis`, supporting both CasADi and NumPy inputs."""
    if not arrs:
        return arrs
    if any(isinstance(arr, backend.symbol_class) for arr in arrs):
        if axis == 0:
            return casadi.vcat(arrs)
        elif axis in (1, -1):
            return casadi.hcat(arrs)
        else:
            raise ValueError("CasADi only supports 2D matrices.")
    return np.concatenate([np.atleast_2d(arr) for arr in arrs], axis=axis)


def unstack(arr, axis=0):
    """Split `arr` along the specified `axis`, following CasADi’s semantics."""
    if axis == 0:
        return casadi.vertsplit(arr)
    elif axis in (1, -1):
        return casadi.horzsplit(arr)


def zeros(shape=(1, 1)):
    """Return a literal symbolic zero matrix of the given shape."""
    return casadi.MX.zeros(*shape)


# ---------------------------------------------------------------------------
# Reduction operations
# ---------------------------------------------------------------------------
def min(x, axis=None):
    """Compute the symbolic minimum of `x`. Only global reduction (axis=None) is supported."""
    if not isinstance(x, backend.symbol_class):
        x = concat(x)
    if axis is not None:
        raise ValueError("Only axis=None supported.")
    return casadi.mmin(x)


def max(x, axis=None):
    """Compute the symbolic maximum of `x`. Only global reduction (axis=None) is supported."""
    if not isinstance(x, backend.symbol_class):
        x = concat(x)
    if axis is not None:
        raise ValueError("Only axis=None supported.")
    return casadi.mmax(x)


# ---------------------------------------------------------------------------
# Jacobian and directional derivatives
# ---------------------------------------------------------------------------
unsupported_jacobian_message = (
    "Jacobian of matrix expression with respect to matrix variable is not yet supported."
)

def jacobian(of, wrt):
    """Compute the symbolic Jacobian of `of` with respect to `wrt`."""
    if of.size and wrt.size:
        transpose_in = False
        if isinstance(wrt, backend.symbol_class) and wrt.op() == casadi.OP_TRANSPOSE:
            transpose_in = True
            wrt = wrt.dep()
        if transpose_in and np.all(np.array(of.shape + wrt.shape) > 1):
            raise NotImplementedError(unsupported_jacobian_message)
        return casadi.jacobian(of, wrt)
    return backend.symbol_class(0, np.prod(wrt.shape))


def jac_prod(of, wrt, rev=True):
    """Compute a directional derivative via CasADi’s `jtimes` helper."""
    return casadi.jtimes(of, wrt, not rev)


# ---------------------------------------------------------------------------
# Symbolic substitution
# ---------------------------------------------------------------------------
def substitute(expr, subs):
    """Safely apply substitution map `subs` onto symbolic expression `expr`."""
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
            raise ValueError(f"Incompatible shapes: {use_k} --> {use_v}")
    expr = casadi.graph_substitute(expr, subs.keys(), subs.values())
    if isinstance(expr, backend.symbol_class) and expr.is_constant():
        expr = expr.to_DM().toarray()
    # Try evaluating simple single-call expressions
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


# ---------------------------------------------------------------------------
# Symbolic conditional (if / elif / else) control flow
# ---------------------------------------------------------------------------
def if_else(*conditions_actions, short_circuit=False):
    """
    Symbolic representation of nested if/elif/else control flow.

    Parameters
    ----------
    *conditions_actions : sequence
        Pairs of (condition, value) followed by a final else value.

    Example
    --------
    The expression:
        result = if_else(
            (cond0, val0),
            (cond1, val1),
            default_val
        )

    Is equivalent to:
        if cond0:
            result = val0
        elif cond1:
            result = val1
        else:
            result = default_val
    """
    if len(conditions_actions) == 1:
        else_action = conditions_actions[0]
        if isinstance(else_action, tuple):
            raise ValueError("if_else requires an explicit else action.")
        return else_action

    *pairs, else_value = conditions_actions
    result = else_value
    for condition, action in reversed(pairs):
        if hasattr(condition, "shape") and np.prod(condition.shape) > 1:
            raise ValueError("if_else conditions must be scalar.")
        result = casadi.if_else(condition, action, result, short_circuit)
    return result

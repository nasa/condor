import dataclasses as dc

import numpy as np

import casadi
from condor.backends.casadi import operators as operators  # noqa: PLC0414
from condor.backends.element_mixin import BackendSymbolDataMixin

casadi.GlobalOptions.setNumpyMode(True)

symbol_class = casadi.MX
# symbol_class = casadi.array
# base_symbol_class = symbol_class

"""
# a useful stub for getting the op codes in casadi
for attr in casadi.__dir__():
    if attr.startswith("OP_"):
        print(f"{attr}: {getattr(casadi, attr)}")

"""


def symbols_in(expression):
    """return the leaf symbols in the :attr:`expression`"""
    if isinstance(expression, casadi.array):
        expression = expression.to_casadi()
    if not isinstance(expression, symbol_class):
        return []
    else:
        return casadi.symvar(expression)


def is_symbol(value):
    return (isinstance(value, symbol_class) and not value.is_constant()) or (
        isinstance(value, casadi.array) and casadi.array._arg_has_symbolic(value)
    )


def is_constant(value):
    """evaluate whether the :attr:`symbol` is a constant (e.g., numeric)"""
    # return not isinstance(symbol, symbol_class) or symbol.is_constant()
    return bool(1 - is_symbol(value))


def process_relational_element(elem):
    """
    modify an element if the backend_repr is relational

    if backend_repr is lhs == rhs,
        replace with lhs - rhs
        set upper_bound = lower_bound = 0 (if element has bounds)

    if backend_repr is inequality, e.g., lhs < mhs < rhs
        attach lhs to lower bound, mhs to backend_repr, and rhs to upper_bound

    """
    # check if the backend_repr is a comparison op
    # check if the bounds are constant
    relational_op = False
    if elem.backend_repr.to_casadi().is_binary():
        lhs = elem.backend_repr.to_casadi().dep(0)
        rhs = elem.backend_repr.to_casadi().dep(1)

    if hasattr(elem, "lower_bound") and (
        not isinstance(elem.lower_bound, np.ndarray)
        or np.any(np.isfinite(elem.lower_bound))
    ):
        real_lower_bound = True
    else:
        real_lower_bound = False

    if hasattr(elem, "upper_bound") and (
        not isinstance(elem.upper_bound, np.ndarray)
        or np.any(np.isfinite(elem.upper_bound))
    ):
        real_upper_bound = True
    else:
        real_upper_bound = False

    if elem.backend_repr.to_casadi().op() in (casadi.OP_LT, casadi.OP_LE):
        relational_op = True
        mhs = casadi.MX()

        # validation
        if not hasattr(elem, "lower_bound"):
            msg = "Setting inequality for an element without bounds doesn't make sense"
            raise ValueError(msg)
        if casadi.OP_EQ in (lhs.op(), rhs.op()):
            msg = "Setting inequality and equality relation doesn't make sense"
            raise ValueError(msg)
        if lhs.op() in (casadi.OP_LT, casadi.OP_LE):
            mhs = casadi.array(lhs.dep(1))
            lhs = casadi.array(lhs.dep(0))
            if rhs.backend_repr.op() in (casadi.OP_LT, casadi.OP_LE):
                msg = "Too many inequalities deep"
                raise ValueError(msg)
        if rhs.op() in (casadi.OP_LT, casadi.OP_LE):
            mhs = rhs.dep(0)
            rhs = rhs.dep(1)
            if lhs.backend_repr.op() in (casadi.OP_LT, casadi.OP_LE):
                msg = "Too many inequalities deep"
                raise ValueError(msg)

        if lhs.op() in (casadi.OP_LT, casadi.OP_LE):
            mhs = lhs.dep(1)
            lhs = lhs.dep(0)
        if rhs.op() in (casadi.OP_LT, casadi.OP_LE):
            mhs = rhs.dep(0)
            rhs = rhs.dep(1)

        if (
            (lhs.op() in (casadi.OP_EQ, casadi.OP_LE, casadi.OP_LT))
            or (rhs.op() in (casadi.OP_EQ, casadi.OP_LE, casadi.OP_LT))
            or (mhs.op() in (casadi.OP_EQ, casadi.OP_LE, casadi.OP_LT))
        ):
            msg = "Too many inequalities deep"
            raise ValueError(msg)

        if mhs.shape == (0, 0):
            if rhs.is_constant() and lhs.is_constant():
                msg = "Unexpected inequality of constants"
                raise ValueError(msg)
            elif not rhs.is_constant() and not lhs.is_constant():
                mhs = rhs - lhs
                lhs = np.full(lhs.shape, 0.0)
                rhs = np.full(rhs.shape, np.inf)
            elif lhs.is_constant():
                mhs = rhs
                rhs = elem.upper_bound
                lhs = lhs.to_DM().toarray()

            elif rhs.is_constant():
                mhs = lhs
                lhs = elem.lower_bound
                rhs = rhs.to_DM().toarray()
            else:
                raise ValueError

        elem.lower_bound = casadi.array(lhs)
        elem.backend_repr = casadi.array(mhs)
        elem.upper_bound = casadi.array(rhs)

    # elif elem.backend_repr.op() in (casadi.OP_GT, casadi.OP_GE):
    #    relational_op = True
    #    elem.backend_repr = rhs - lhs
    #    elem.upper_bound = 0.0
    elif elem.backend_repr.to_casadi().op() == casadi.OP_EQ:
        relational_op = True
        elem.backend_repr = casadi.array(rhs - lhs)
        if hasattr(elem, "upper_bound"):
            elem.upper_bound = np.zeros(elem.backend_repr.shape)
        if hasattr(elem, "lower_bound"):
            elem.lower_bound = np.zeros(elem.backend_repr.shape)

    if relational_op and (real_lower_bound or real_upper_bound):
        msg = f"Do not use relational constraints with bounds for {elem}"
        raise ValueError(msg)


def shape_to_nm(shape):
    if isinstance(shape, (int, np.int64)):
        shape = (shape, 1)
    if len(shape) == 0:
        shape = (1, 1)
    if len(shape) > 2:
        raise ValueError
    n = shape[0]
    m = 1 if len(shape) == 1 else shape[1]
    return n, m


@dc.dataclass
class BackendSymbolData(BackendSymbolDataMixin):
    """Dataclass for handling backend expressions"""

    def __post_init__(self, *args, **kwargs):
        # might potentially want to overwrite for casadi-specific validation, etc.
        super().__post_init__(self, *args, **kwargs)

    def flatten_value(self, value, force_asymetric=False):
        """flatten a value to the appropriate representation for the backend"""
        if self.symmetric and not force_asymetric:
            unique_values = symmetric_to_unique(value, symbolic=is_symbol(value))
            if is_symbol(value):
                syms = casadi.symvar(value)
                if len(syms) == 1 and symbol_is(unique_to_symmetric(syms[0]), value):
                    return syms[0]
            return unique_values
        if self.size == 1 and isinstance(value, (float, int)):
            return np.array([[value]])
        else:
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            return value.reshape((-1, 1))

    def wrap_value(self, value):
        """wrap a flattened value to the appropriate shape"""
        if self.size == 1:
            if isinstance(value, (float, int)) or is_symbol(value):
                return value
            elif hasattr(value, "__iter__"):
                ret_val = np.array(value).reshape(-1)[0]
                if is_symbol(ret_val):
                    return ret_val
                return np.array(value).reshape(-1)

        if self.symmetric:
            value = unique_to_symmetric(value, symbolic=is_symbolic(value))

        if (isinstance(value, symbol_class) and value.is_constant()) or (
            isinstance(value, casadi.array)
            and not casadi.array._arg_has_symbolic(value)
        ):
            value = value.to_DM()

        if isinstance(value, symbol_class):
            value = casadi.array(value)

        if isinstance(value, casadi.DM):
            # value = value.toarray()
            # value = value.T
            # value = value.reshape(self.shape)
            # value = value.toarray().reshape(self.shape[::-1]).T
            # value = casadi.array(value)
            value = casadi.array(value).reshape(self.shape)

        if not isinstance(value, np.ndarray) and not is_symbol(value):
            value = np.array(value)

        if isinstance(value, symbol_class):
            breakpoint()

        if (
            is_symbol(value)
            and not self.symmetric
            and self.shape[0] > 1
            and self.shape[1] > 1
            and 0
        ):
            value = value.reshape(self.shape[::-1]).T
            breakpoint()
        else:
            try:
                value = value.reshape(self.shape)
            except ValueError:
                value = value.reshape(self.shape + (-1,))

        return value


def symmetric_to_unique(value, symbolic=True):
    """helper function for flattening a symmetric symbol"""
    n = value.shape[0]
    unique_shape = (int(n * (n + 1) / 2), 1)
    indices = np.tril_indices(n)
    unique_values = (
        casadi.array(symbol_class(*unique_shape))
        if symbolic
        else np.empty(unique_shape)
    )
    for kk, (i, j) in enumerate(zip(*indices)):
        unique_values[kk] = value[i, j]
    return unique_values


def unique_to_symmetric(unique, symbolic=True):
    """helper function for wrapping a symmetric symbol"""
    count = unique.shape[0]
    n = m = int((np.sqrt(1 + 8 * count) - 1) / 2)
    if symbolic:
        # using an empty MX is unverifiable by casadi.is_equal, but concatenating the
        # iniddividual elements from a list works
        matrix_symbols = np.empty((n, m), dtype=np.object_)
    else:
        use_shape = (n, m) if unique.shape[1] == 1 else (n, m, unique.shape[1])
        matrix_symbols = np.empty(use_shape)
    indices = np.tril_indices(n)
    for kk, (i, j) in enumerate(zip(*indices)):
        matrix_symbols[i, j] = unique[kk]
        if i != j:
            matrix_symbols[j, i] = unique[kk]
    if symbolic:
        symbol_list = matrix_symbols.tolist()
        matrix_symbols = casadi.vcat([casadi.hcat(row) for row in symbol_list])
    return matrix_symbols


def symbol_generator(name, shape=(1, 1), symmetric=False, diagonal=False):
    """create a symbol

    Parameters
    ----------
    name : str
    shape : tuple of ints
    symmetric : bool
        flag to indicate this should be a symmetric (2-D) matrix; only store the unique
        elements
    diagonal : bool
        flag to indicate this should be a (2-D) matrix with only elements on the
        diagonal

    """
    n, m = shape_to_nm(shape)
    if diagonal:
        # assert m == 1
        sym = casadi.diag(sym)
        raise NotImplementedError
    elif symmetric:
        if n != m:
            msg = f"Symmetric specified but shape non-square {(n, m)}"
            raise ValueError(msg)
        unique_shape = (int(n * (n + 1) / 2), 1)
        unique_symbols = casadi.array(symbol_class.sym(name, unique_shape))
        matrix_symbols = unique_to_symmetric(unique_symbols)
        return matrix_symbols
    else:
        raw_sym = casadi.array(symbol_class.sym(name, n * m)).reshape((n, m))
        sym = raw_sym  # .T
        return sym


def symbol_like(name, other):
    shape_data = get_symbol_data(other)
    shape_data_dict = dc.asdict(shape_data)
    shape_data_dict.pop("size")

    return symbol_generator(
        name=name,
        **shape_data_dict,
    )


def get_symbol_data(symbol, symmetric=None):
    """extract shape metadata from an expression"""
    if hasattr(symbol, "backend_repr"):
        symbol = symbol.backend_repr

    if not isinstance(symbol, (symbol_class, casadi.array, casadi.DM)):
        symbol = np.atleast_1d(symbol)
        # I'm not sure why, but before I reshaped this to a vector always. Only
        # reshaping tensors now...
        # .reshape(-1)
        if symbol.ndim > 2:
            symbol.reshape(-1)

    shape = symbol.shape
    n, m = shape_to_nm(shape)
    size = n * m

    # TODO: actually check these?
    # and automatically reduce size if true? or should these be flags?
    diagonal = False
    if symmetric is None:
        # if unprovided, try to determine if symmetric
        if is_symbol(symbol):
            symmetric = symbol_is(symbol, symbol.T) and size > 1
        else:
            symmetric = (
                len(shape) == 2
                and shape[0] == shape[1]
                and np.isclose(symbol, symbol.T).all()
            )

    return BackendSymbolData(
        shape=shape,
        symmetric=symmetric,
        diagonal=diagonal,
    )


def symbol_is(a, b):
    """evaluate whether two symbols are the same with idiosyncrasies for symbol class"""
    if a.shape != b.shape:
        return False
    equality_expr = a == b
    if isinstance(equality_expr, bool):
        return equality_expr
    if isinstance(equality_expr, casadi.array):
        equality_expr = equality_expr.to_casadi()
    if isinstance(equality_expr, (symbol_class, casadi.DM)):
        return equality_expr.is_one() or casadi.is_equal(a, b, int(1e10))
    if isinstance(equality_expr, np.ndarray):
        return equality_expr.all()


class WrappedSymbol:
    def __init__(self, symbol):
        # assert not isinstance(arg, WrappedSymbol)
        if isinstance(symbol, WrappedSymbol):
            symbol = arg.symbol
        if isinstance(symbol, casadi.array):
            symbol = symbol.to_casadi()
        self.symbol = symbol

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        if not isinstance(other, WrappedSymbol):
            if isinstance(other, symbol_class):
                other = WrappedSymbol(other)
            else:
                return False
        return symbol_is(self.symbol, other.symbol)

    def __str__(self):
        return self.symbol.__str__()

    def __repr__(self):
        return self.symbol.__repr__()


class SymbolCompatibleDict(dict):
    """A dict subclass that works with the backend symbol class"""

    def __init__(self, *args, **kwargs):
        args_dict = dict(*args, **kwargs)
        for k, v in args_dict.items():
            self[k] = v

    def get(self, k, default):
        if isinstance(k, casadi.array):
            k = k.to_casadi()
        out = super().get(k, None)
        if out is None and isinstance(k, symbol_class):
            out = super().get(k.T, None)
        if out is None:
            return default
        return out

    def __getitem__(self, k):
        try:
            return dict.__getitem__(self, WrappedSymbol(k))
        except KeyError as e:
            if isinstance(k, symbol_class) and k.op() not in (
                casadi.OP_RESHAPE,
                casadi.OP_TRANSPOSE,
            ):
                raise e
            try:
                return dict.__getitem__(self, WrappedSymbol(k.dep()))
            except KeyError:
                raise KeyError from e

    def __setitem__(self, k, v):
        if isinstance(k, casadi.array):
            k = k.to_casadi()
        if isinstance(k, symbol_class) and k.op() in (
            casadi.OP_RESHAPE,
            casadi.OP_TRANSPOSE,
        ):
            dict.__setitem__(self, WrappedSymbol(k.T), v)
        return dict.__setitem__(self, WrappedSymbol(k), v)

    def update(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            for k, v in args[0].items():
                self[k] = v
        else:
            super().update(*args, **kwargs)

    def keys(self):
        for k in dict.keys(self):
            yield k.symbol

    def items(self):
        for k, v in dict.items(self):
            yield k.symbol, v

    __iter__ = keys

    def __copy__(self):
        copy = SymbolCompatibleDict()
        for k, v in self.items():
            copy[k] = v
        return copy


def evalf(expr, backend_repr2value):
    """evaluate :attr:`expr` with dictionary of {symbol: value}"""
    if not isinstance(expr, list):
        expr = [expr]
    args = list(backend_repr2value.keys())
    func = casadi.Function(
        "temp_func",
        args,
        expr,
    )
    return func(*backend_repr2value.values())


class CasadiFunctionCallback(casadi.Callback):
    """Base class for wrapping a Function with a Callback"""

    def __init__(
        self,
        function,
        get_jacobian_func=None,
        implementation=None,
        model_name="",
        jacobian_of=None,
        input_symbol=None,
        output_symbol=None,
        opts=None,
    ):
        """
        function -- callable to evaluate out_symbol = function(input_symbol)
        get_jacobian_func -- callable that returns callable to
              evaluate df/dy (input_symbol)

        jacobian_of -- used internally to recursively create references of related
        callbacks, as needed.

        input_symbol - single MX representing all input
        output_symbol - single MX representing all output

        input/output symbol used to identify sparsity and other metadata by creating a
        placeholder casadi.Function. Not used if jacobian_of is provided, because
        jacobian operator is used on jacobian_of's placeholder Function.

        """
        casadi.Callback.__init__(self)

        self.function = function
        self.get_jacobian_func = get_jacobian_func
        self.jacobian_callback = None

        self.input_symbol = input_symbol
        self.output_symbol = output_symbol
        if opts is None:
            opts = {}

        if (not model_name) == (implementation is None):
            raise ValueError
        if not model_name:
            model_name = implementation.model.__name__

        if (
            jacobian_of is None
            and input_symbol is not None
            and output_symbol is not None
        ):
            self.placeholder_func = casadi.Function(
                f"{model_name}_placeholder",
                [self.input_symbol],
                [self.output_symbol],
                # self.input,
                # self.output,
                dict(
                    allow_free=True,
                ),
            )
        else:
            self.placeholder_func = jacobian_of.placeholder_func.jacobian()

        self.jacobian_of = jacobian_of
        self.implementation = implementation
        self.opts = opts

        self.construct()

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        if isinstance(out, symbol_class):
            out = casadi.array(out)
        return out

    def construct(self):
        super().construct(self.placeholder_func.name(), self.opts)

    def init(self):
        pass

    def finalize(self):
        pass

    def get_n_in(self):
        return self.placeholder_func.n_in()

    def get_n_out(self):
        return self.placeholder_func.n_out()

    def eval(self, args):
        out = self.function(args[0])

        if self.jacobian_of:
            if self.jacobian_of.jacobian_of:
                # print("hessian")
                pass
            else:
                # print("jacobian")
                pass
            if hasattr(out, "shape") and out.shape == self.get_sparsity_out(0).shape:
                return (out,)
            if isinstance(out, tuple) and len(out) == 2:
                return out
            jac_out = (
                # np.concatenate(flatten(out))
                out.reshape(self.get_sparsity_out(0).shape[::-1]).T
            )
            # for hessian, need to provide dependence of jacobian on the original
            # function's output. is it fair to assume always 0?
            if self.n_out() == 2:
                return (jac_out, np.zeros(self.get_sparsity_out(1).shape))

            return (jac_out,)

        return (out,)

    def get_sparsity_in(self, i):
        if self.jacobian_of is None or i < self.jacobian_of.get_n_in():
            return self.placeholder_func.sparsity_in(i)
        elif i < self.jacobian_of.get_n_in() + self.jacobian_of.get_n_out():
            # nominal outputs are 0
            return casadi.Sparsity(
                *self.jacobian_of.get_sparsity_out(
                    i - self.jacobian_of.get_n_in()
                ).shape
            )
        else:
            raise ValueError

    def get_sparsity_out(self, i):
        # if self.jacobian_of is None or i < self.jacobian_of.get_n_out():
        if i < 1:
            return casadi.Sparsity.dense(*self.placeholder_func.sparsity_out(i).shape)
        # elif i < self.jacobian_of.get_n_in() + self.jacobian_of.get_n_out():
        else:
            return self.placeholder_func.sparsity_out(i)
            return casadi.Sparsity(
                *self.jacobian_of.get_sparsity_out(
                    i - self.jacobian_of.get_n_in()
                ).shape
            )

    def has_forward(self, nfwd):
        return False
        return True

    def has_reverse(self, nrev):
        return False
        return True

    def get_forward(self, nin, name, inames, onames, opts):
        breakpoint()

    def get_reverse(self, nout, name, inames, onames, opts):
        breakpoint()
        # return casadi.Function(

    def has_jacobian(self):
        return self.get_jacobian_func is not None

    def get_jacobian(self, name, inames, onames, opts):
        if self.has_jacobian():
            if self.jacobian_callback is None:
                self.jacobian_callback = self.get_jacobian_func(self)
            return self.jacobian_callback


# generic name for shim
FunctionOperator = CasadiFunctionCallback


def callables_to_operator(wrapper_funcs, *args, jacobian_of=None, **kwargs):
    """check if this is actually something that needs to get wrapped or if it's a
    native op... if latter, return directly; if former, create callback.

    this function ultimately contains the generic API, CasadiFunctionCallback can be
    casadi specific interface


    to think about: default casadi jacobian operator introduces additional inputs
    (and therefore, after second derivative also introduces additional output) for
    anti-derivative output. This is required for most "solver" type outputs.
    However, probably don't want to use casadi machinery for retaining that solver
    -- would be done externally. Is there ever a time we would want to use casadi
    machinery for passing in previous solution? if so, what would the API be for
    specifying?

    y = fun(x)
    jac = jac_fun(x,y)
    d_jac_x, d_jac_y = hess_fun(x, y, jac)

    if we don't use casadi's mechanisms, we are technically not functional
    but maybe when it gets wrapped by the callbacks, it effectively becomes
    functional because new instances are always created when needed to prevent race
    conditions, etc?

    similarly, how to specify alternate types/details of derivative, like
    forward/reverse mode, matrix-vector product form, etc.

    by definition, since we are consuming functions and cannot arbitrarily take
    derivatives (could look at how casadi does trajectory but even if useful for our
    trajectory, may not generalize to other external solvers) so need to implement
    and declare manually. Assuming this take is right, seems somewhat reasonable to
    assume we get functions, list of callables (at least 1) that evaluates 0th+
    functions (dense jacobians, hessians, etc). Then a totally different spec track for
    jacobian-vector or vector-jacobian products. Could cap out at 2nd derivative, not
    sure what the semantics for hessian-vector vs vector-hessian products, tensor etc.



    """
    if isinstance(wrapper_funcs[0], casadi.Function):
        return wrapper_funcs[0]
    if len(wrapper_funcs) > 1:
        return CasadiFunctionCallback(
            wrapper_funcs[0],
            *args,
            jacobian_of=jacobian_of,
            get_jacobian_func=lambda jacobian_of_: callables_to_operator(
                wrapper_funcs[1:], *args, jacobian_of=jacobian_of_, **kwargs
            ),
            **kwargs,
        )
    else:
        return CasadiFunctionCallback(
            wrapper_funcs[0], *args, jacobian_of=jacobian_of, **kwargs
        )


def expression_to_operator(input_symbols, output_expressions, name="", **kwargs):
    """take a symbolic expression and create an operator -- callable that can be used
    with jacobian, etc. operators

    assume "MISO" -- but slightly misleading since output can be arbitrary size
    """
    output_expressions = getattr(output_expressions, "backend_repr", output_expressions)
    if not name:
        name = "function_from_expression"

    def process_input(sym):
        if isinstance(sym, casadi.array):
            sym = sym.to_casadi()
        return sym

    input_symbols = [process_input(sym) for sym in input_symbols]

    func = casadi.Function(
        name,
        input_symbols,
        [casadi.array(output_expressions)],
        kwargs,
    )

    def wrapped_function(*args):
        out = func(*args)
        out = casadi.array(out)
        if is_constant(out):
            out = np.array(out)
        return out

    return wrapped_function

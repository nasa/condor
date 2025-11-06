"""Fields and elements for use with model templates"""

# TODO: figure out python version minimum

import dataclasses as dc
import importlib
import logging
import operator
import sys
from enum import Enum

import numpy as np

# from condor.backends.element_mixin import BackendSymbolData
from condor import backend

log = logging.getLogger(__name__)

# TODO **kwarg expansion for a field (with a filter?)
# TODO copy all parameters (with a filter?) from one model
# --> together, a very convenient way to connect a system with many variables
# use a dict and/or dummy system to define an "interface" which can be used to filter
# OR: pass a system itself? works for settings.conf, not sure if it's very convenient
# to implement generally (same as abandoned deferred subsystem idea)

# TODO could intercept models created within the class and create an API for the pattern
# of creating input field elements to a super-system based on input field elements that
# the sub-system require that aren't matching outputs of previous sub-systems (exact
# name matching or define dictionary where key is current subsystem input name and value
# is previous subsytem output name to connect) and
# want to do something like
"""
class AnyTopeofModel(co....):
    some specific setup

    subsystem_ref_1 = create_subsystem(SomeNormalModel, **aliases)
    # adds attributes for holding all input, and then useful subsets. ref has outputs

"""
# so this is a field, and it has to know about its parent class -- or we make parameter
# the only input field for all models and output as a required output field with others
# specified. output field maintains order of other fields elements ast hey are defined
# -- maybe output is either the built-in or a special field that combines other fields?
# so implicit and explicit still exist for algebraic but they are really accessed
# through the combo field.

# this is basically deferred model?
# aliases allow complete & explicit control of what connects to subsystem


# don't need to include API for specifying which subsystems connect to which -- just
# create a new model for each open chain, can only create a closed-loop by wrapping with
# algebraic system or optimization problem & alias to variables/implicit_output
# I guess could also have API to terminate outputs so they don't get promoted. Is this
# just literally promotes=*?


# TODO defensive chek to make sure trajectory analysis doesn't over write odemodel
# dynamic output?
# todo inner_to is only for ode system. should backref have a better name than
# `inner_to`? Maybe even ODESystem....

# keyword expand modals?


def asdict(obj):
    return dict(
        (field.name, getattr(obj, field.name)) for field in dc.fields(obj) if field.init
    )


class Direction(Enum):
    """:class:`Enum` used to indicate the direction of an element relative to a model"""

    """
    MatchedField may need to become MatchedElement and also use direction -- might
    be useful for DAE models, etc.
    """

    output = -1
    internal = 0
    input = 1


class FieldValues:
    """Base class for Field dataclasses"""

    def asdict(self):
        """call the :mod:`dataclasses` module :func:`asdict` function on this field
        datacalss"""
        return dc.asdict(self)

    def flatten(self):
        """turn the bound values of this field instance into a single symbol -- may be
        numeric or in the backend representation (symbol class)"""
        flat_val = backend.operators.concat(
            [
                elem.flatten_value(v)
                for elem, v in zip(self.field, self.asdict().values())
            ]
        )
        if not isinstance(flat_val, backend.symbol_class):
            flat_val = np.array(flat_val).reshape(-1)
        return flat_val

    @classmethod
    def wrap(cls, values):
        """turn a single symbol into a bound field dataclasss"""
        size_cum_sum = np.cumsum([0] + cls.field.list_of("size"))
        new_values = {}
        for start_idx, end_idx, elem in zip(
            size_cum_sum,
            size_cum_sum[1:],
            cls.field,
        ):
            new_values[elem.name] = elem.wrap_value(values[start_idx:end_idx])
        return cls(**new_values)

    def keys(self):
        yield from self.field.list_of("name")

    def __getitem__(self, item):
        return getattr(self, item)


class Field:
    """Base field

    Model templates have field instances added to them which are parents of elements
    used in models.

    How to create a new Field type:

    - Subclass :class:`BaseElement` as needed. Will automatically try to find
      ``NameElement`` for ``NameField``, or provide as :attr:`~Field.element_class`
      kwarg in subclass definition
    - Update :attr:`_init_kwargs` to define kwargs that should get passed during
      inheritance from Model templates to user models
    - Until a better API is designed & implemented, instance attributes created during
      :attr:`~Field.__init__` should start with a ``_`` so ``setattr`` on
      :class:`AssignedField` can filter it
    - Use :attr:`~Field.create_element` to create elements, passing ``**kwargs`` to
      element dataclass as much as possible to keep DRY

    """

    def _set_resolve_name(self):
        self._resolve_name = ".".join([self._model_name, self._name])

    def __init_subclass__(
        cls, element_class=None, default_direction=Direction.internal, **kwargs
    ):
        # TODO: make sure python version requirement is correct
        super().__init_subclass__(**kwargs)
        if element_class is None:
            # TODO: ensure this works for different file organizations? e.g., will it
            # find a symbol defined in another file that the field subclass has access
            # to? NO! need to figure out how to access parent scope dictionary?
            # no it does not! Libraries need to be explicit, maybe core condor should be
            # as well.

            element_class = globals().get(cls.__name__.replace("Field", "Element"))
        cls.element_class = element_class

        cls.default_direction = default_direction

    def __init__(
        self,
        direction=None,
        name="",
        model=None,
        inherit_from=None,
        model_name="",
    ):
        """Create a field instance"""
        # TODO: currently, AssignedField types are defined using setattr, which needs
        # to know what already exists. The attributes here, like name, model, count,
        # etc, don't exist until instantiated. Currently pre-fix with `_` to mark as
        # not-an-assignment, but I guess I could just use element_class on value instead
        # of checking name? Anyway, really don't like needing to prefix all Field
        # attributes with _ because of AssignedField...

        self._cls_dict = None
        self._name = name
        self._model = model
        if direction is None:
            direction = self.__class__.default_direction
        self._direction = direction
        if model:
            self._model_name = model.__class__.__name__
        else:
            self._model_name = model_name
        self._set_resolve_name()
        self._elements = []
        self._init_kwargs = dict(direction=direction)
        self._inherits_from = inherit_from
        # subclasses must provide _init_kwargs for binding to sub-classes
        # TODO: can this just be taken from __init__ kwargs easily?  or
        # __init_subclass__ hook? definitely neds to be DRY'd up

    @property
    def _count(self):
        return sum(self.list_of("size"))

    def bind(self, name, model):
        """Bind an existing Field instance to a model"""
        self._name = name
        self._model = model
        if self._model_name:
            if self._model_name != model.__name__:
                msg = "Attempting to bind to a class that wasn't inherited"
                raise ValueError(msg)
        else:
            self._model_name = model.__name__
        self._set_resolve_name()

    def bind_dataclass(self):
        """Bind dataclass module and name in an attempt to make pickle-able"""
        mod_name = self._model.__module__
        self._dataclass.__module__ = mod_name
        if mod_name not in sys.modules:
            mod_obj = importlib.import_module(mod_name)
        else:
            mod_obj = sys.modules[mod_name]
        setattr(mod_obj, self._dataclass.__name__, self._dataclass)

    def prepare(self, cls_dict):
        """Hook to pass reference of class dictionary to field to allow automatic name
        extraction and assignment
        """
        self._cls_dict = cls_dict

    def inherit(self, model_name, field_type_name, **kwargs):
        """
        copy and re-assign name -- used in ModelType.__prepare__ before model is
        available
        """

        # TODO: could use copy infrastructure with the same reference checking Model is
        # currently doing
        if not kwargs:
            kwargs = self._init_kwargs
        new = self.__class__(**kwargs, inherit_from=self)
        new._name = field_type_name
        new._model_name = model_name
        new._set_resolve_name()
        return new

    # TODO: replace with a descriptor that can be dot accessed?
    def list_of(self, field_name):
        """Create a list of a particular element attribute for all elements in field"""
        return [getattr(element, field_name) for element in self._elements]

    def __repr__(self):
        if self._resolve_name:
            return f"<{self.__class__.__name__}: {self._resolve_name}>"

    def process_field(self):
        pass

    def get(self, *args, **kwargs):
        """Get a list of field elements where every field matches kwargs

        If only one, return element without list wrapper.

        treat __ as a "dot" accessor, recursively
        """
        # TODO: what's the lightest-weight way to be able query? these should get called
        # very few times, so hopefully don't need to stress too much about
        # implementation
        # would be nice to be able to follow references, etc. can match be used?
        if len(args) == 1 and not kwargs:
            field_value = args[0]
            if isinstance(field_value, backend.symbol_class):
                kwargs.update(backend_repr=field_value)

        items = []
        for item in self._elements:
            this_item = True
            for field_name, field_value in kwargs.items():
                item_value = item
                for field_name_comp in field_name.split("__"):
                    item_value = getattr(item_value, field_name_comp)
                if isinstance(item_value, backend.symbol_class):
                    if not isinstance(field_value, backend.symbol_class):
                        this_item = False
                        break
                    this_item = this_item and backend.symbol_is(item_value, field_value)
                elif isinstance(item_value, BaseElement):
                    if item_value.__class__ is not field_value.__class__:
                        this_item = False
                        break
                    this_item = this_item and item_value is field_value
                else:
                    this_item = this_item and item_value == field_value
                if not this_item:
                    break
            if this_item:
                items.append(item)
        if len(items) == 1:
            return items[0]
        return items

    def flat_index(self, of_element):
        """Helper to get the starting index of an element"""
        idx = 0
        for element in self:
            if element is of_element:
                break
            idx += element.size

        if idx == self._count:
            msg = f"{of_element} not an element of {self}"
            raise ValueError(msg)
        return idx

    def create_element(self, **kwargs):
        """Create an element"""
        kwargs.update(dict(field_type=self))
        self._elements.append(self.element_class(**kwargs))
        if self._cls_dict and isinstance(self, FreeField):
            self._cls_dict.meta.backend_repr_elements[
                self._elements[-1].backend_repr
            ] = self._elements[-1]

        # self._count += getattr(self._elements[-1], 'size', 1)

    def create_dataclass(self):
        """Create the dataclass for this model and field with the current elements"""
        # TODO: do processing to handle different field types
        fields = [(element.name, float) for element in self._elements]
        name = self._model_name + make_class_name([self._name])
        self._dataclass = dc.make_dataclass(
            name,
            fields,
            bases=(FieldValues,),
        )
        self._dataclass.field = self

    def __iter__(self):
        """Iterate over field elements"""
        yield from self._elements

    def __len__(self):
        return len(self._elements)

    def __getattr__(self, with_name):
        """Make element names dot-accessible on Field, to match behavior on field
        dataclass"""
        return self.get(name=with_name).backend_repr

    def dataclass_of(self, attr="backend_repr", default=None):
        """construct a dataclass of the field where values are the attr of each
        element
        """
        return self._dataclass(
            **{elem.name: getattr(elem, attr, default) for elem in self}
        )

    def flatten(self, attr="backend_repr"):
        """flatten the values into a single 1D array"""
        return self.dataclass_of(attr).flatten()

    def wrap(self, values):
        """wrap a flattened array into the elements of appropriate shape"""
        return self._dataclass.wrap(values)

    def keys(self):
        yield from self.list_of("name")

    def __getitem__(self, item):
        return getattr(self, item)

    def __add__(self, other):
        """add concatenates list of elements"""
        if isinstance(other, Field):
            return self._elements + other._elements
        raise NotImplementedError

    def __radd__(self, other):
        """add concatenates list of elements"""
        if isinstance(other, Field):
            return other._elements + self._elements
        raise NotImplementedError


def make_class_name(components):
    separate_words = " ".join([comp.replace("_", " ") for comp in components])
    # use pascal case from https://stackoverflow.com/a/8347192
    # TODO: this is turning mid-word uppercase (e.g., from a CamelCaseClassName) -- how
    # to just capitalize first character of word?
    return "".join(word for word in separate_words.title() if not word.isspace())


@dc.dataclass
class FrontendElementData:
    field_type: Field
    backend_repr: backend.symbol_class
    name: str = ""

    def flat_index(self):
        return self.field_type.flat_index(self)


def _generic_op(op, is_r=False):
    """Generate method from generic operator"""

    def mthd(self, other=None):
        if other is None:
            return op(self.backend_repr)

        other_value = other.backend_repr if isinstance(other, self.__class__) else other

        if is_r:
            return op(other_value, self.backend_repr)

        return op(self.backend_repr, other_value)

    return mthd


@dc.dataclass
class BaseElement(
    FrontendElementData,
    backend.BackendSymbolData,
):
    def __hash__(self):
        return hash(
            (
                self.backend_repr,
                self.shape,
                self.field_type._name,
                self.field_type._model_name,
            )
        )

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(self, *args, **kwargs)
        # TODO: validate broadcasting, etc. shape info?
        # self.backend_repr = backend.utils.reshape(self.backend_repr, self.shape)

    def __repr__(self):
        return f"<{self.field_type._resolve_name}: {self.name}>"

    def copy_to_field(self, new_field, new_name=""):
        element_dict = asdict(self)
        if new_name:
            element_dict["name"] = new_name
        else:
            element_dict["name"] = f"{self.name}"
        new_field.create_element(**element_dict)
        return new_field._elements[-1]

    def reshape(self, new_shape):
        if self.shape == new_shape:
            return self
        self.backend_repr = backend.symbol_generator(
            name=self.backend_repr.name(), shape=new_shape
        )
        self.shape = new_shape
        # TODO: this is wrong, need to use get symbol data or something? because need to
        # check for symmetry, etc.
        self.size = np.prod(new_shape)
        return self

    @property
    def T(self):  # noqa: N802
        return self.backend_repr.T

    __le__ = _generic_op(operator.le)
    __lt__ = _generic_op(operator.lt)
    __ge__ = _generic_op(operator.ge)
    __gt__ = _generic_op(operator.gt)

    __add__ = _generic_op(operator.add)
    __sub__ = _generic_op(operator.sub)
    __mul__ = _generic_op(operator.mul)
    __matmul__ = _generic_op(operator.matmul)
    __truediv__ = _generic_op(operator.truediv)
    __floordiv__ = _generic_op(operator.floordiv)
    __pow__ = _generic_op(operator.pow)
    # mod?
    # divmod?
    # lshift
    # rshift
    # and
    # xor
    # or
    # is and is_not?

    # are pos/neg the only unary? abs, inv,
    __neg__ = _generic_op(operator.neg)
    __pos__ = _generic_op(operator.pos)

    __radd__ = _generic_op(operator.add, is_r=True)
    __rsub__ = _generic_op(operator.sub, is_r=True)
    __rmul__ = _generic_op(operator.mul, is_r=True)
    __rmul__ = _generic_op(operator.mul, is_r=True)
    __rmatmul__ = _generic_op(operator.matmul, is_r=True)
    __rtruediv__ = _generic_op(operator.truediv, is_r=True)
    __rfloordiv__ = _generic_op(operator.floordiv, is_r=True)
    __rpow__ = _generic_op(operator.pow, is_r=True)

    __array_priority__ = 100.0

    def __getitem__(self, *keys):
        return self.backend_repr.__getitem__(*keys)


class FreeElement(BaseElement):
    pass


def make_backend_symbol(
    backend_name, shape=(1,), symmetric=False, diagonal=False, **kwargs
):
    if isinstance(shape, int):
        shape = (shape,)
    out = backend.symbol_generator(
        name=backend_name, shape=shape, symmetric=symmetric, diagonal=diagonal
    )
    symbol_data = backend.get_symbol_data(out, symmetric=symmetric)
    kwargs.update(
        backend_repr=out,
        **asdict(symbol_data),
    )
    return kwargs


class FreeField(Field, default_direction=Direction.input):
    """Field that generates a symbol assigned to a class attribute during model
    definition.
    """

    def __call__(self, **kwargs):
        backend_name = f"{self._resolve_name}_{len(self._elements)}"
        new_kwargs = make_backend_symbol(backend_name=backend_name, **kwargs)
        self.create_element(**new_kwargs)
        if "name" in kwargs and self._cls_dict is not None:
            self._cls_dict[kwargs["name"]] = self._elements[-1].backend_repr
        return self._elements[-1].backend_repr

    def process_field(self):
        for element_idx, element in enumerate(self):
            # add names to elements -- must be an unnamed element without a reference
            # assignment in the class
            if not element.name:
                element.name = f"{self._model_name}_{self._name}_{element_idx}"

    def create_from(self, other, **aliases):
        """Get or create elements based on another field

        Elements from `other` are created on this field unless it already exists or is
        supplied in `aliases`.

        Parameters
        ----------
        other : Field
            Other field to get elements from
        **aliases
            Name-element aliases to use in place of elements of `other`

        Returns
        -------
        elem_map : dict
            Dictionary mapping element names to elements
        """
        elem_map = {}
        for elem in other:
            check = self.get(name=elem.name)
            # TODO check additional attributes of retrieved element
            if elem.name in aliases:
                elem_map[elem.name] = aliases[elem.name]
            elif isinstance(check, BaseElement):
                elem_map[elem.name] = check.backend_repr
            else:
                elem_map[elem.name] = self(name=elem.name, shape=elem.shape)
        return elem_map


class FreeAssignedField(
    FreeField, default_direction=Direction.output, element_class=FreeElement
):
    def __call__(self, value, **kwargs):
        symbol_data = backend.get_symbol_data(value)
        self.create_element(backend_repr=value, **kwargs, **asdict(symbol_data))
        return self._elements[-1].backend_repr


@dc.dataclass(repr=False)
class WithDefaultElement(FreeElement):
    """Element with an optional default value"""

    #: default value
    default: float = 0.0  # TODO union[numeric, expression] or None?


class WithDefaultField(FreeField):
    def create_substitution_dict(self, subclass, set_attr=False):
        """ """
        substitution_dict = {}
        # TODO: no back reference is preserved which frankly seems harsh...
        for elem in self:
            log.debug("checking placeholder %s on %s", elem, subclass)
            # currently, the placeholder element is getting passed on so we know it
            # exists and should never be None
            if not hasattr(subclass, elem.name):
                val = elem.default
            else:
                val = getattr(subclass, elem.name)

            use_val = None
            if val is None:
                pass
            elif val is elem or val is elem.backend_repr:
                use_val = elem.default
                if use_val is None:
                    use_val = elem.backend_repr

            elif isinstance(val, BaseElement):
                use_val = val.backend_repr
            else:
                use_val = val

            if set_attr:
                setattr(subclass, elem.name, use_val)
            log.debug("creating substitution %s = %s", elem.backend_repr, use_val)
            if isinstance(use_val, backend.symbol_class):
                if elem.size >= np.prod(use_val.size()):
                    try:
                        np.broadcast_shapes(elem.shape, use_val.shape)
                    except ValueError:
                        pass  # noqa: S110
                    else:
                        substitution_dict[elem.backend_repr] = use_val
            elif np.array(use_val).dtype.kind in "if":
                # use_val = np.array(use_val)
                use_val = np.atleast_1d(use_val)
                if elem.size == use_val.size:
                    substitution_dict[elem.backend_repr] = use_val.reshape(elem.shape)
                elif elem.size > use_val.size:
                    try:
                        np.broadcast_shapes(elem.shape, use_val.shape)
                    except ValueError:
                        pass  # noqa: S110
                    else:
                        substitution_dict[elem.backend_repr] = use_val
                    substitution_dict[elem.backend_repr] = np.broadcast_to(
                        use_val, elem.shape
                    )
        return substitution_dict


@dc.dataclass(repr=False)
class BoundedElement(FreeElement):
    """Element with upper and lower bounds"""

    #: upper bound
    upper_bound: float = np.inf
    #: lower bound
    lower_bound: float = -np.inf

    # Then if bounds are here, must follow broadcasting rules

    def __post_init__(self):
        super().__post_init__()
        # since bounds generally must be numeric, broadcasting should just work
        self.upper_bound = np.broadcast_to(self.upper_bound, self.shape)
        self.lower_bound = np.broadcast_to(self.lower_bound, self.shape)


@dc.dataclass(repr=False)
class InitializedElement(BoundedElement):
    """Element with an initial value"""

    #: initial value or expression depending on other existing elements
    initializer: float = 0.0  # TODO union[numeric, expression]
    #: whether the initial value should come from the implementation/solver
    warm_start: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.warm_start = np.broadcast_to(self.warm_start, self.shape)
        # initializer may be an expression and cause an error here.
        # TODO: test this behavior? lol
        if isinstance(self.initializer, BaseElement):
            self.initializer = np.ones(self.shape) * self.initializer.backend_repr
        elif isinstance(self.initializer, backend.symbol_class):
            self.initializer = np.ones(self.shape) * self.initializer
        else:
            self.initializer = np.broadcast_to(self.initializer, self.shape)

    def __hash__(self):
        return BaseElement.__hash__(self)


class InitializedField(FreeField):
    pass


class BoundedAssignmentField(
    FreeField, default_direction=Direction.output, element_class=BoundedElement
):
    def __call__(self, value, eq=None, **kwargs):
        symbol_data = backend.get_symbol_data(value)
        if eq is not None:
            if "lower_bound" in kwargs or "upper_bound" in kwargs:
                raise ValueError
            kwargs["lower_bound"] = eq
            kwargs["upper_bound"] = eq
        self.create_element(backend_repr=value, **kwargs, **asdict(symbol_data))
        return self._elements[-1].backend_repr


@dc.dataclass(repr=False)
class TrajectoryOutputElement(FreeElement):
    """Element with terminal and integrand terms for trajectory analysis"""

    #: expression for the terminal term
    terminal_term: BaseElement = 0.0
    #: expression for the integrand term
    integrand: BaseElement = 0.0


class TrajectoryOutputField(FreeField, default_direction=Direction.output):
    def __call__(self, terminal_term=0, integrand=0, **kwargs):
        # Use quadrature instead of state.
        # to create a copy of state on TrajectoryAnalysis and give trajectory_output
        # access to new copy, would need to add a reference at copy time which is
        # doable but less clean, especially since this is just a hack to avoid
        # quadrature anyway
        # then didn't need to re-arrange ModelType.__new__ to give access to inner_to,
        # etc. before implementation binding.
        # TODO: undo inner_to refactor from  0195013ddf58dc1fa8f589d99671ba231ab846a6

        if isinstance(terminal_term, BaseElement):
            terminal_term = terminal_term.backend_repr
        if isinstance(integrand, BaseElement):
            integrand = integrand.backend_repr

        if isinstance(terminal_term, backend.symbol_class):
            # comstant terminal terms are handled below
            shape_data = backend.get_symbol_data(terminal_term)
            kwargs["terminal_term"] = terminal_term

        if isinstance(integrand, backend.symbol_class):
            if "terminal_term" in kwargs:
                if backend.get_symbol_data(integrand) != shape_data:
                    # TODO would be nice to include the names here
                    msg = (
                        f"Incompatible terminal term shape {shape_data} for integrand "
                        f"{backend.get_symbol_data(integrand)}"
                    )
                    raise ValueError(msg)
            else:
                shape_data = backend.get_symbol_data(integrand)
                kwargs["terminal_term"] = np.broadcast_to(
                    terminal_term, shape_data.shape
                )
        else:
            integrand = np.broadcast_to(integrand, shape_data.shape)

        kwargs["integrand"] = integrand

        shape_data_dict = asdict(shape_data)

        traj_out_placeholder = backend.symbol_generator(
            name=f"trajectory_output_{len(self._elements)}",
            **shape_data_dict,
        )

        self.create_element(
            backend_repr=traj_out_placeholder,
            **kwargs,
            **shape_data_dict,
        )
        return traj_out_placeholder


class DependentElement(BaseElement):
    pass


@dc.dataclass(repr=False)
class AssignedElement(BaseElement):
    def __hash__(self):
        return super().__hash__()


class AssignedField(Field, default_direction=Direction.output):
    def __init__(self, direction=None, add_to_namespace_override=None, **kwargs):
        super().__init__(direction=direction, **kwargs)
        self._add_to_namespace_override = add_to_namespace_override
        self._init_kwargs.update(add_to_namespace_override=add_to_namespace_override)

        if add_to_namespace_override is None:
            self._add_to_namespace = self._direction in (
                Direction.input,
                Direction.output,
            )
        else:
            self._add_to_namespace = add_to_namespace_override

    def __setattr__(self, name, value):
        if name.startswith("_") and not isinstance(value, backend.symbol_class):
            # TODO: iirc the reason we need to prefix field attributes that are directly
            # assigned with _ is to allow this check to work, bypassing the creation of
            # a new element. Is it actually sufficient to just if the value is a
            # symbol_class??
            super().__setattr__(name, value)
        else:
            # TODO: resolve circular imports so we can use dataclass
            if isinstance(value, BaseElement):
                value = value.backend_repr
            symbol_data = backend.get_symbol_data(value)
            self.create_element(
                name=name,
                backend_repr=value,
                # TODO: I guess if this accepts model instances, it becomes recursive to
                # allow dot access to sub systems? Actually, this breaks the idea of
                # both system encapsolation and implementations. So don't do it, but
                # doument it. Can programatically add sub-system outputs though. For
                # these reasons, ditch intermediate stuff.
                **asdict(symbol_data),
            )
            if self._add_to_namespace and self._cls_dict:
                # self._cls_dict.__set_item__(name, value)
                self._cls_dict[name] = value
            # super().__setattr__(name, self._symbols[-1])


@dc.dataclass(repr=False)
class MatchedElementMixin:
    match: BaseElement  # match to the Element instance

    def update_name(self):
        self.name = "__".join([self.field_type._name, self.match.name])
        self.name = self.match.name


@dc.dataclass(repr=False)
class MatchedElement(BaseElement, MatchedElementMixin):
    """Element matched with another element of another field"""

    pass


def zero_like(match):
    return np.zeros(match.shape)


def pass_through(match):
    return match.backend_repr


class MatchedField(Field):
    def __init__(self, matched_to=None, default_factory=pass_through, **kwargs):
        """
        matched_to is Field instance that this MatchedField is matched to.
        default_factory is a function that takes matched to element and returns the
        backend_repr for the match (this field)
        """
        # TODO: add matches_required flag? e.g. for initializer
        # TODO: a flag for application direction? Field._diretion is relative to model;
        # matched (especially for model internal) could be used as a free or assigned
        # field (eg,dot for  DAE vs ODE), maybe "usage" and it can directly refer to
        # FreeField or AssignedField instead of a separate enum?
        # TODO: should FreeField instance __call__ get a kwarg for all matched fields
        # that reference it and are assigned? -> two ways of assigning the match...

        super().__init__(**kwargs)
        self._matched_to = matched_to
        self._default_factory = default_factory
        self._init_kwargs.update(matched_to=matched_to, default_factory=default_factory)

    def process_field(self):
        for element in self:
            element.update_name()

    def key_to_matched_element(self, key):
        if isinstance(key, backend.symbol_class):
            match = self._matched_to.get(backend_repr=key)
            if isinstance(match, list):
                msg = f"Could not find match for {key}"
                raise ValueError(msg)
        elif isinstance(key, BaseElement):
            match = key
        elif isinstance(key, str):
            match = self._matched_to.get(name=key)
        else:
            msg = f"Could not find match for {key}"
            raise ValueError(msg)
        return match

    def __setitem__(self, key, value):
        match = self.key_to_matched_element(key)
        existing_elem = self.get(match=match)
        if not isinstance(existing_elem, list):
            log.debug("OVER-WRITING %s!!!", existing_elem)
            # TODO verify shape etc?
            existing_elem.backend_repr = value
        else:
            if isinstance(value, BaseElement):
                symbol_data = backend.BackendSymbolData(
                    shape=value.shape,
                    symmetric=value.symmetric,
                    diagonal=value.diagonal,
                )
                backend_repr = value.backend_repr
            else:
                symbol_data = backend.get_symbol_data(value)

                if isinstance(value, backend.symbol_class):
                    backend_repr = value
                else:
                    value = np.atleast_1d(value)
                    if match.size == value.size:
                        backend_repr = value.reshape(match.shape)
                    else:
                        backend_repr = np.broadcast_to(value, match.shape)
            if match.symmetric:
                symbol_data.symmetric = True
            self.create_element(
                name=None,
                match=match,
                backend_repr=backend_repr,
                **asdict(symbol_data),
            )

    def __getitem__(self, key):
        """
        get the matched symbodl by the matche's name, backend symbol, or symbol
        """
        match = self.key_to_matched_element(key)
        item = self.get(match=match)
        if isinstance(item, list) and self._direction != Direction.input:
            # TODO: could easily create a new symbol related to match; should it depend
            # on direction? Does matched need two other direction options for internal
            # get vs set? or is that a separate type of matchedfield?
            raise KeyError
        return item.backend_repr

    def dataclass_of(self, on_field=None):
        dc_kwargs = {}
        if on_field is self:
            for elem in self:
                dc_kwargs[elem.name] = elem.backend_repr
        else:
            if on_field is None:
                on_field = self._matched_to
            for match_elem in on_field:
                elem = self.get(match__backend_repr=match_elem.backend_repr)
                if elem:
                    dc_kwargs[match_elem.name] = elem.backend_repr
                else:
                    dc_kwargs[match_elem.name] = self._default_factory(match_elem)
        return on_field._dataclass(**dc_kwargs)

    def flatten(self, on_field=None):
        """flatten matches to the on_field, defaults tot he match.

        on_field is used to override which field instance's dataclass gets filled in;
        this is currently used during TrajectoryAnalysis construction which creates its
        own copies of the matching fields
        """
        return self.dataclass_of(on_field).flatten()

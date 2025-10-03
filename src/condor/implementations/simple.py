import numpy as np

from condor.backend import (
    callables_to_operator,
    expression_to_operator,
    get_symbol_data,
)
from condor.fields import BaseElement, Field, MatchedElement, MatchedField, asdict

from .utils import options_to_kwargs


class DeferredSystem:
    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()
        self.model = model

    def __init__(self, model_instance):
        self.construct(model_instance.__class__)
        self(model_instance)

    def __call__(self, model_instance):
        model_instance.bind_field(self.model.output.wrap(self.symbol_outputs))


class ExplicitSystem:
    """Implementation for :class:`~condor.contrib.ExplicitSystem` model.

    No :class:`Options` expected.
    """

    def __init__(self, model_instance):
        self.construct(model_instance.__class__)
        self(model_instance)

    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()

        self.model = model
        self.func = expression_to_operator(
            [self.symbol_inputs],
            self.symbol_outputs,
            name=model.__name__,
        )

    def __call__(self, model_instance):
        self.args = model_instance.input.flatten()
        self.out = self.func(self.args)
        model_instance.bind_field(self.model.output.wrap(self.out))


symmetric_error_message = (
    "symmetric flag only applies for two matched fields that are the same"
)


class MultiMatchedField(MatchedField, element_class=MatchedElement):
    """multi-match field, with symmetry checks"""

    def __init__(self, matched_to=None, symmetric=None, **kwargs):
        """
        matched_to is Field instance that this MatchedField is matched to.

        symmetric is pair of axes to be symmetric, not sure if it needs to generalize to
        more than two axes?
        """
        # TODO: add matches_required flag? e.g. for initializer
        # TODO: a flag for application direction? Field._diretion is relative to model;
        # matched (especially for model internal) could be used as a free or assigned
        # field (eg,dot for  DAE vs ODE), maybe "usage" and it can directly refer to
        # FreeField or AssignedField instead of a separate enum?
        # TODO: should FreeField instance __call__ get a kwarg for all matched fields
        # that reference it and are assigned? -> two ways of assigning the match...

        super().__init__(**kwargs)
        if isinstance(matched_to, Field):
            matched_to = [matched_to]
        self._matched_to = matched_to
        self._dim_match = len(matched_to)
        self._flat_shape = tuple([matched._count for matched in matched_to])
        if (
            symmetric is not None
            and len(symmetric) != 2
            and matched_to[symmetric[0]] is not matched_to[symmetric[1]]
        ):
            raise ValueError(symmetric_error_message)
        self._symmetric = symmetric
        self._init_kwargs.update(dict(matched_to=matched_to, symmetric=symmetric))

    # TODO wrap method to wrap matrix and allow selection by element/backend_repr ref

    def flatten(self):
        to_mat = np.zeros([match_field._count for match_field in self._matched_to])
        for elem in self._elements:
            mat_select = [
                slice(
                    match_elem.flat_index(),
                    match_elem.flat_index() + match_elem.size,
                )
                for match_elem in elem.match
            ]
            to_mat[tuple(mat_select)] = elem.backend_repr
        return to_mat

    def key_to_matched_elements(self, k):
        if isinstance(k, str):
            k = k.split("__")
        if not isinstance(k, (list, tuple)):
            k = [k]
        match = []
        for ki, match_field in zip(k, self._matched_to):
            # this is the same as MatchedField.key_to_matched_element
            msg = f"Could not find match for {k}"
            if isinstance(ki, str):
                # name
                match_ref = match_field.get(name=ki)
            elif isinstance(ki, BaseElement):
                if ki.field_type is not match_field:
                    raise ValueError
                match_ref = ki
            elif isinstance(ki, backend.symbol_class):
                match_ref = self._matched_to.get(backend_repr=ki)
            else:
                match_ref = None

            if match_ref:
                match.append(match_ref)
            else:
                raise ValueError(msg)

        if self._symmetric:
            raise NotImplementedError
        else:
            swap_axes = None

        if len(match) != self._dim_match:
            msg = (
                f"Matching should specify {self._dim_match} elements, got {len(match)}"
            )
            raise ValueError(msg)

        # TODO document order/shape?
        for match_i in match:
            for dim_len in match_i.shape:
                if dim_len not in (match_i.size, 1):
                    msg = "each component of match must be a vector"
                    raise ValueError(msg)

        return match, swap_axes

    def __setitem__(self, keys, value):
        match, swap_axes = self.key_to_matched_elements(keys)
        name = self.get_name_for_match(match)

        expected_shape = tuple([int(match_i.size) for match_i in match])
        if len(expected_shape) == 1:
            value = np.atleast_1d(value).squeeze()
        elif len(expected_shape) == 2:
            value = np.atleast_2d(value)
        symbol_data = get_symbol_data(value)
        if expected_shape != symbol_data.shape:
            breakpoint()
            msg = "mismatching shape"
            raise ValueError(msg)

        existing_elem = self.get(name=name)

        if existing_elem:
            msg = "setting new item"
            raise ValueError(msg)
        else:
            self.create_element(
                # name=None,
                name=name,
                match=match,
                backend_repr=value,
                **asdict(symbol_data),
            )

    def get_name_for_match(self, match):
        return "__".join([match_i.name for match_i in match])

    def __getitem__(self, keys):
        """
        get the matched symbodl by the matche's name, backend symbol, or symbol
        """
        match = self.key_to_matched_elements(keys)
        name = "__".join([self._name, match[0].name, match[1].name])
        item = self.get(name=name)
        # item = self.get(match=match)
        if isinstance(item, list) and self._direction != Direction.input:
            # TODO: could easily create a new symbol related to match; should it depend
            # on direction? Does matched need two other direction options for internal
            # get vs set? or is that a separate type of matchedfield?
            raise KeyError

        return item.backend_repr


class ExternalSolverModel:
    """Implementation for External Solver models.

    No :class:`Options` expected.
    """

    def __init__(self, model_instance):
        model = model_instance.__class__
        model_instance.options_dict = options_to_kwargs(model)
        self.construct(model, **model_instance.options_dict)
        self(model_instance)

    def construct(self, model):
        self.model = model
        self.wrapper = model._meta.external_wrapper
        self.input = model.input.flatten()
        self.output = model.output.flatten()

        wrapper_funcs = [self._wrap_user_func(self.wrapper.function)]
        if hasattr(self.wrapper, "jacobian"):
            wrapper_funcs.append(self._wrap_user_func(self.wrapper.jacobian, der=1))
        if hasattr(self.wrapper, "hessian"):
            wrapper_funcs.append(self._wrap_user_func(self.wrapper.hessian, der=2))

        self.callback = callables_to_operator(
            wrapper_funcs,
            self,
            jacobian_of=None,
            input_symbol=self.input,
            output_symbol=self.output,
        )
        self.callback.construct()

    def __call__(self, model_instance):
        use_args = model_instance.input.flatten()
        out = self.callback(use_args)
        model_instance.bind_field(self.model.output.wrap(out))

    def create_matched_field(self, der):
        return MultiMatchedField(
            matched_to=[self.model.output] + [self.model.input] * der
        )

    def _wrap_user_func(self, user_func, der=0):
        def func(inputs):
            input_data = self.model.input.wrap(inputs)
            user_out = user_func(input_data)
            return_structure = self.create_matched_field(der=der)
            if isinstance(user_out, dict):
                for k, v in user_out.items():
                    return_structure[k] = v
                out = return_structure.flatten()
            elif der == 0 and isinstance(user_out, tuple):
                for elem, v in zip(self.model.output, user_out):
                    return_structure[elem.name] = v
                out = return_structure.flatten()
            else:
                out = user_out

            return out

        return func

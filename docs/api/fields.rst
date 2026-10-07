===================
Fields and Elements
===================

.. automodule:: condor.fields
   :member-order: bysource
   :members:
   :exclude-members: log

Assigned value dictionaries
---------------------------

A model instance's field values can be converted to a dictionary with
:meth:`condor.fields.FieldValues.dict_of`. By default, each key is the field
element's backend expression, so the result can be used for substitution::

    import condor as co
    from condor.backend import operators as ops

    class Sum(co.ExplicitSystem):
        x = input()
        y = input()
        output.total = x + y

    result = Sum(x=2, y=3)
    substitutions = result.input.dict_of()
    evaluated = ops.substitute(Sum.total.backend_repr, substitutions)

``result.input.dict_of("name")`` instead returns ``{"x": 2, "y": 3}``.
The dictionary is new, but its values reference the assigned objects directly,
including arrays and symbolic expressions. Use the existing ``asdict()`` method
when an independent, recursively copied mapping is required.

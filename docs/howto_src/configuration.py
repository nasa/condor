"""
==================
Configuring Models
==================

Logic within a model declaration can sometimes be handled by inputs/parameters and
:func:`~condor.backend.operators.if_else`, but sometimes models need to be templated
in a deeper way. A couple common examples are handling input arrays of arbitrary size,
and logic for whether or not certain models or submodels should be declared.

This example walks through two different ways to handle these cases for a simple linear
time invariant (LTI) :class:`~condor.contrib.ODESystem` with templated dynamics and an
optional :class:`~condor.contrib.Event`.
"""

# %%
# Module Configuration
# --------------------
#
# One option for generating models is through a ``settings`` object in the top-level
# ``condor`` name space, where you register the module's default configuration with
# ``get_settings``. Then the module is imported via ``get_module``.
#
# Here is the configured model source with the name ``_lti.py``:
#
# .. literalinclude:: _lti.py
#    :caption: File: _lti.py
#    :linenos:

# %%
# To use this module, we use :func:`~condor.settings.get_module`, passing its declared
# settings as concrete keyword arguments.

import numpy as np

import condor

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])

dblint_mod = condor.settings.get_module("_lti", A=A, B=B)

# %%
# The returned object is a module, so we can access the model with its declared class
# name:

LTI_dblint = dblint_mod.LTI

# %%
# And finally we can use this configured ODE system to simulate a trajectory.

import matplotlib.pyplot as plt


class Sim(LTI_dblint.TrajectoryAnalysis):
    tf = 20
    initial[x] = [1.0, 0.1]


sim = Sim(K=[1.0, 0.1])

plt.figure()
plt.plot(sim.t, sim.x[0].squeeze())

# %%
# We can also re-import the module with a different configuration:


dlbint_mod = condor.settings.get_module("_lti", A=A, B=B, bounce=True)
LTI_bounce = dblint_mod.LTI


class Sim(LTI_bounce.TrajectoryAnalysis):
    tf = 10
    initial[x] = [1.0, 0.5]


sim = Sim(K=sim.K)

plt.figure()
plt.plot(sim.t, sim.x[0].squeeze())


# %%
# Programmatic Construction
# -------------------------
#
# An alternative approach is to programmatically generate the model using the
# metaprogramming machinery that Condor uses internally. See
# :ref:`metaprogramming-walkthrough` for an overview.
#
# The :class:`~condor.contrib.ODESystem` factory function declared below is essentially
# identical to the config-based example above except for the optional event, which we'll
# see later as a separate factory function:

from condor.contrib import ModelTemplateType, ODESystem


def make_LTI(A, B=None, name="LTI"):
    attrs = ModelTemplateType.__prepare__(name, (ODESystem,))

    attrs["A"] = A

    state = attrs["state"]
    x = state(shape=A.shape[0])
    attrs["x"] = x

    xdot = A @ x

    if B is not None:
        attrs["B"] = B
        K = attrs["parameter"](shape=B.T.shape)
        attrs["K"] = K

        u = -K @ x
        attrs["dynamic_output"].u = u

        xdot += B @ u

    attrs["dot"][x] = xdot

    plant = ModelTemplateType(name, (ODESystem,), attrs)

    return plant


# %%
# Use of the model factory function looks similar to using ``get_module``:

LTI_dblint = make_LTI(A, B=B)


class Sim(LTI_dblint.TrajectoryAnalysis):
    tf = 20
    initial[x] = [1.0, 0.1]


sim = Sim(K=[1.0, 0.1])

plt.figure()
plt.plot(sim.t, sim.x[0].squeeze())

# %%
# To define a submodel of a primary system, like an event, the construction looks like
# this:

from condor.backend import operators as ops


def add_bounce_event(odesys_cls):
    event_meta_args = (
        "Bounce",  # name
        (odesys_cls.Event, condor.models.Submodel),  # bases
    )
    event_attrs = condor.contrib.Event.__prepare__(*event_meta_args)

    # extract elements to operate on
    x = odesys_cls.x.backend_repr

    # define the event
    event_attrs["function"] = x[0]
    event_attrs["update"][x] = ops.concat([x[0], -x[1]])

    condor.contrib.EventType.__new__(
        condor.contrib.EventType,
        *event_meta_args,
        attrs=event_attrs,
    )


# %%
# Add the event to our ODESystem and simulate again:

add_bounce_event(LTI_dblint)


class Sim(LTI_dblint.TrajectoryAnalysis):
    tf = 20
    initial[x] = [1.0, 0.1]


sim = Sim(K=sim.K)

plt.figure()
plt.plot(sim.t, sim.x[0].squeeze())


# %%

plt.show()

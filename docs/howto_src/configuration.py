"""
==================
Configuring Models
==================

At times, model templates need to be parametrized in a more of a programming sense than
a mathematical one. An example of this is a linear time invariant (LTI) ODE system,
where the size of the state vector and whether there is feedback control are dependent
on what the user passes in for the state and input matrices.
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
# We can also re-use the module with a different configuration:

LTI_exp = condor.settings.get_module("_lti", A=np.array([[0, 1], [-2, -3]])).LTI


class Sim(LTI_exp.TrajectoryAnalysis):
    tf = 10
    initial[x] = [1.0, 0.5]


sim = Sim()

plt.figure()
plt.plot(sim.t, sim.x[0].squeeze())


# %%
# Programmatic Construction
# -------------------------
#
# An alternative approach is to programmatically generate the model using the
# metaprogramming machinery Condor uses internally. See
# :ref:`metaprogramming-walkthrough` for a more thorough overview.


from condor.contrib import ModelTemplateType, ODESystem


def make_LTI(A, B=None, name="LTISystem"):
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

LTI_exp = make_LTI(A=np.array([[0, 1], [-2, -3]]))


class Sim(LTI_exp.TrajectoryAnalysis):
    tf = 20
    initial[x] = [1.0, 0.5]


sim = Sim()

plt.figure()
plt.plot(sim.t, sim.x[0].squeeze())


# %%

plt.show()

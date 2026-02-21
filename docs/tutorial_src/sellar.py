"""
======================
Introduction to Condor
======================
"""

# %%
# We wanted to have an API that looks as much like a mathematical description as
# possible with as little distraction from programming cruft as possible. For example,
# an arbitrary system of equations like from Sellar [#sellar]_,
#
# .. math::
#    y_{1}&=x_{0}^{2}+x_{1}+x_{2}-0.2\,y_{2} \\
#    y_{2}&=\sqrt{y_{1}}+x_{0}+x_{1}
#
# should be writable as
#
# .. code-block:: python
#
#      y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2
#      y2 == y1**0.5 + x[0] + x[1]
#
# Of course, in both the mathematical and programmatic description, the source of each
# symbol must be defined. In an engineering memo, we might say "where :math:`y_1,y_2`
# are the variables to solve and :math:`x \in \mathbb{R}^3` parameterizes the system of
# equations," which suggests the API for an algebraic system of equations as

import condor


class Coupling(condor.AlgebraicSystem):
    x = parameter(shape=3)
    y1 = variable(initializer=1.0)
    y2 = variable(initializer=1.0)

    residual(y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2)
    residual(y2 == y1**0.5 + x[0] + x[1])


# %%
# which can be evaluated by instantiating the model with numerical values for the
# parameters:

coupling = Coupling([5.0, 2.0, 1])

# %%
# Once the model is finished running, the model *binds* the numerical results from the
# iterative solver to the named *element* and *field* attributes on the instance. That
# is, elements of fields accessible directly:

print(coupling.y1, coupling.y2)

# %%
# Fields are bound as dataclasses

print(coupling.variable)


# %%
# Models can be used recursively, building up more sophisticated models by *embedding*
# models within another. However, system encapsulation is enforced so only elements from
# input and output fields are accessible after the model has been defined. For example,
# we may wish to optimize Sellar's algebraic system of equations. Mathematically, we can
# define the optimization as
#
# .. math::
#    \operatorname*{minimize}_{x \in \mathbb{R}^3} & \quad x_2^2+x_1+y_1+e^{-y_{2}} \\
#    \text{subject to} & \quad 3.16 \le y_1 \\
#    & \quad y_2 \le 24.0
#
# where :math:`y_1` and :math:`y_2` are the solution to the system of algebraic
# equations described above. In condor, we can write this as

from condor.backend import operators as ops


class Sellar(condor.OptimizationProblem):
    x = variable(shape=3, lower_bound=0, upper_bound=10)
    coupling = Coupling(x)
    y1, y2 = coupling

    objective = x[2] ** 2 + x[1] + y1 + ops.exp(-y2)
    constraint(y1 >= 3.16)
    constraint(y2 <= 24.0)


# %%
# As with the system of algebraic equations, we can numerically solve this optimization
# problem by providing an initial value for the variables and instantiating the model.

Sellar.set_initial(x=[5, 2, 1])
sellar = Sellar()

# %%
# The resulting object will have a dot-able data structure with the bound results,
# including the embedded ``Coupling`` model:

print("objective value:", sellar.objective)  # scalar value
print(sellar.constraint)  # field
print(sellar.coupling.y1)  # embedded-model element

# %%
# .. rubric:: References
#
# .. [#sellar] Sellar, R., Batill, S., and Renaud, J., "Response Surface Based,
#    Concurrent Subspace Optimization for Multidisciplinary System Design," 1996.
#    https://doi.org/10.2514/6.1996-714

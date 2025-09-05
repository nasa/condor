"""
=========================
Polar Transformation
=========================
"""
# %%
# As another example, if we were interested in transforming Cartesian coordinates to
# polar form:
#
# .. math::
#    \begin{align}
#    p_r &= \sqrt{x^2 + y^2} \\
#    p_{\theta} &= \tan^{-1}\left(\frac{y}{x}\right)
#    \end{align}
#
# We can implement this with an ``ExplicitSystem`` by declaring the inputs and outputs
# of this system as follows:

import condor as co
from condor.backend import operators as ops


class PolarTransform(co.ExplicitSystem):
    x = input()
    y = input()

    output.r = ops.sqrt(x**2 + y**2)
    # output.theta = ops.atan2(y, x)
    output.theta = ops.atan(y / x)


# %%
# In general, once you've defined any system in Condor, you can just evaluate it
# numerically by passing in numbers:

p = PolarTransform(x=3, y=4)
print(p)

# %%
# The output returned by such a call is designed for inspection to the extent that we
# recommend working in an interactive session or debugger, especially when getting
# accustomed to Condor features.
#
# For example, the outputs of an explicit system are accessible directly:

print(p.r)

# %%
# They can also be retrieved collectively:

print(p.output)

# %%
# You can of course call it again with different arguments

print(PolarTransform(x=1, y=0).output.asdict())


# %%
# While the *binding* of the results in a data structure is nice, the real benefit of
# constructing condor models is in calling iterative solvers. For example, we could
# perform symbolic manipulation to define  another ``ExplicitSystem`` with :math:`x =
# r\cos\theta` and :math:`y = r\sin\theta`. Or we can we use Condor to
# numerically solve this algebraic system of equations using an ``AlgebraicSystem`` by
# declaring the input radius and angle as ``parameter``\s and the solving variables for
# :math:`x` and :math:`y`. Mathematically, we are defining the system of algebraic
# equations
#
# .. math::
#    r &= p_r (x^*, y^*) \\
#    \theta &= p_{\theta} (x^*, y^*)
#
# and letting an iterative solver find the solution :math:`x^*,y^*` satisfying both
# residual equations given parameters :math:`r` and :math:`\theta`. In Condor,


class CartesianTransform(co.AlgebraicSystem):
    # r and theta are input parameters
    r = parameter()
    theta = parameter()

    # solver will vary x and y to satisfy the residuals
    x = variable(initializer=1)
    y = variable(initializer=0)

    # get r, theta from solver's x, y
    p = PolarTransform(x=x, y=y)

    # residuals to converge to 0
    residual(r == p.r)
    residual(theta == p.theta)


out = CartesianTransform(r=1, theta=ops.pi / 4)
print(out.x, out.y)

# %%
# Note also that passing the inputs (or any intermediates) to plain numeric functions
# that can handle symbolic objects as well as pure numerical objects (float or numpy
# arrays) could work for this simple example. However, since we *embedded* the
# ``PolarTransform`` model in this solver, the system evaluated with the solved variable
# values is directly accessible if the ``bind_embedded_models`` option is ``True``
# (which it is by default), as in:


print(out.p.output)

# %%
# Note that this has multiple solutions due to the form of the algebraic relationship of
# the polar/rectangular transformation. The :class:`AlgebraicSystem` uses Newton's
# method as the solver, so the solution that is found depends on the initial conditions.
# The :attr:`initializer` attribute on the :attr:`variable` field determines the initial
# position. For example,

CartesianTransform.set_initial(x=-1, y=-1)
out = CartesianTransform(r=1, theta=ops.pi / 4)
print(out.variable)


# %%
# An additional :attr:`warm_start` attribute determines whether the initializer is
# over-written. Since the default is true, we can inspect the initializer values,

print(CartesianTransform.x.initializer, CartesianTransform.y.initializer)

# %%
# and re-solve with attr:`warm_start` False

CartesianTransform.y.warm_start = False

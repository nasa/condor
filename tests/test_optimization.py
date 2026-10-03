import pytest

import condor as co
from condor.backend.operators import exp

# TODO test from_values


@pytest.mark.parametrize(
    "method",
    [
        co.implementations.OptimizationProblem,
        co.implementations.ScipySLSQP,
        co.implementations.ScipyTrustConstr,
    ],
)
def test_sellar(method):
    class Coupling(co.AlgebraicSystem):
        x = parameter()
        z = parameter(shape=2)
        y1 = variable(initializer=1.0)
        y2 = variable(initializer=1.0)

        y1_agreement = residual(y1 == z[0] ** 2 + z[1] + x - 0.2 * y2)
        residual(y2 == y1**0.5 + z[0] + z[1])

    # coupling = Coupling(1, [5.0, 2.0])

    class Sellar(co.OptimizationProblem):
        x = variable(lower_bound=0, upper_bound=10)
        z = variable(shape=2, lower_bound=0, upper_bound=10)

        coupling = Coupling(x, z)
        y1, y2 = coupling

        objective = x**2 + z[1] + y1 + exp(-y2)

        constraint(y1, upper_bound=10, lower_bound=3.16, name="y1_bound")
        y2_bound = constraint(y2 < 20)

    # Sellar.implementation.set_initial(x=1., z=[5., 2.,])
    Sellar.x.initializer = 1.0
    Sellar.z.initializer = [5.0, 2.0]
    Sellar.Options.__implementation__ = method
    Sellar()

    # TODO meaningful asserts?


def test_callback():
    class Opt(co.OptimizationProblem):
        p = parameter()
        x = variable()

        objective = x**2 + p

    class Callback:
        def __init__(self):
            self.count = 0
            self.objectives = []
            self.parameter = None

        def init_callback(self, parameter, impl_opts):
            self.parameter = parameter

        def iter_callback(self, i, variable, objective, constraint):
            self.count += 1
            self.objectives.append(objective)

    callback = Callback()
    Opt.Options.iter_callback = callback.iter_callback

    xinit = 10
    p = 4
    Opt.x.initializer = xinit
    Opt(p=p)

    assert callback.parameter is None
    assert callback.count > 0
    assert callback.objectives[0] == xinit**2 + p
    assert callback.objectives[-1] == p

    # add init callback
    Opt.Options.init_callback = callback.init_callback
    Opt(p=p)
    assert callback.parameter is not None


@pytest.mark.parametrize("restriction", ["bounds", "constraints"])
def test_scipy_cg_rejects_unsupported_restrictions(restriction):
    if restriction == "bounds":

        class Opt(co.OptimizationProblem):
            x = variable(lower_bound=0)
            objective = x**2

            class Options:
                __implementation__ = co.implementations.ScipyCG

    else:

        class Opt(co.OptimizationProblem):
            x = variable()
            objective = x**2
            constraint(x <= 1)

            class Options:
                __implementation__ = co.implementations.ScipyCG

    with pytest.raises(ValueError, match="CG does not support"):
        Opt()


def test_callback_scipy_no_instance():
    class Opt(co.OptimizationProblem):
        p = parameter()
        x = variable()

        objective = x**2 + p

        class Options:
            __implementation__ = co.implementations.ScipySLSQP

    class Callback:
        def __init__(self):
            self.count = 0
            self.objectives = []
            self.parameter = None

        def init_callback(self, parameter, impl_opts):
            self.parameter = parameter

        def iter_callback(self, i, variable, objective, constraint):
            self.count += 1
            self.objectives.append(objective)

    callback = Callback()
    Opt.Options.iter_callback = callback.iter_callback

    xinit = 10
    p = 4
    Opt.x.initializer = xinit
    Opt(p=p)

    assert callback.parameter is None
    assert callback.count > 0
    assert callback.objectives[0] == xinit**2 + p
    assert callback.objectives[-1] == p

    # add init callback
    Opt.Options.init_callback = callback.init_callback
    Opt(p=p)
    assert callback.parameter is not None


def test_callback_scipy_instance():
    class Opt(co.OptimizationProblem):
        p = parameter()
        x = variable()

        objective = x**2 + p

        class Options:
            __implementation__ = co.implementations.ScipySLSQP

    class Callback:
        def __init__(self):
            self.count = 0
            self.objectives = []
            self.parameter = None
            self.instances = []

        def init_callback(self, parameter, impl_opts):
            self.parameter = parameter

        def iter_callback(self, i, variable, objective, constraint, instance):
            self.count += 1
            self.objectives.append(objective)
            self.instances.append(instance)

    callback = Callback()
    Opt.Options.iter_callback = callback.iter_callback

    xinit = 10
    p = 4
    Opt.x.initializer = xinit
    Opt(p=p)

    assert callback.parameter is None
    assert callback.count > 0
    assert callback.objectives[0] == xinit**2 + p
    assert callback.objectives[-1] == p
    assert len(callback.instances) == callback.count
    assert isinstance(callback.instances[-1], Opt)

    # add init callback
    Opt.Options.init_callback = callback.init_callback
    Opt(p=p)
    assert callback.parameter is not None

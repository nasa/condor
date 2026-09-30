import numpy as np
import pytest

import condor as co


def test_output_ref():
    class OutputRefCheck(co.ExplicitSystem):
        x = input()
        output.y = x**2
        output.z = y + 1

    chk = OutputRefCheck(3.0)
    assert chk.y == 9.0


def test_placeholder_on_explicitsystem():
    with pytest.raises(NameError, match="name 'placeholder' is not defined"):

        class ShouldFail(co.ExplicitSystem):
            x = input()
            output.y = x**2
            z = placeholder()


def test_reserved_word_input():
    with pytest.raises(ValueError, match="Attempting to set _meta"):

        class ShouldFail(co.ExplicitSystem):
            _meta = input()
            output.y = _meta**2


def test_reserved_word_output():
    with pytest.raises(ValueError, match="Attempting to set _meta"):

        class ShouldFail(co.ExplicitSystem):
            x = input()
            output._meta = x**2


def test_objective_shape():
    class Check(co.OptimizationProblem):
        assert objective.shape == (1, 1)


def test_ode_system_event_api():
    class MySystem(co.ODESystem):
        ts = list()
        for i in range(3):
            ts.append(parameter(name=f"t_{i}"))  # noqa: PERF401

        n = 2
        m = 1
        x = state(shape=n)
        C = state(shape=(n, n), symmetric=True)
        A = np.array([[0, 1], [0, 0]])

        # A = parameter(n,n)
        # B = parameter(n,m)
        # K = parameter(m,n)

        W = C.T @ C

        # indexing an output/computation by state
        dot[x] = A @ x  # (A - B @ K) @ x
        dot[C] = A @ C + C @ A.T
        # dot naming, although have to do more work to minimize other setattr's
        dynamic_output.y = C.T @ C

    class MyEvent(MySystem.Event):
        function = MySystem.t - 100.0
        update[x] = x - 2
        z = state()

    assert MyEvent._meta.primary is MySystem
    assert co.models.Submodel in MyEvent.__bases__
    assert MySystem.Event not in MyEvent.__bases__
    assert MySystem.Event in MySystem._meta.submodels
    assert MyEvent in MySystem.Event
    assert MyEvent.update._elements[0].match in MySystem.state

    class MySim(MySystem.TrajectoryAnalysis):
        # this over-writes which is actually really nice
        initial[x] = 1.0
        initial[C] = np.eye(2) * parameter()

        out1 = trajectory_output(integrand=x.T @ x)
        out2 = trajectory_output(x.T @ x)
        out3 = trajectory_output(C[0, 0], C[1, 1])

        with pytest.raises(ValueError, match="Incompatible terminal term shape"):
            # incompatible shape
            out4 = trajectory_output(C[0, 0], x)


def test_embedded_system():
    class Sys1out(co.ExplicitSystem):
        x = input()
        y = input()
        output.z = x**2 + y

    class Sys2out(co.ExplicitSystem):
        x = input()
        y = input()
        v = y**2
        output.w = x**2 + y**2
        output.z = x**2 + y

    class Sys3out(co.ExplicitSystem):
        z = input()
        sys2 = Sys2out(z**2, z)
        output.x = sys2.w
        output.y = sys2.z

        with pytest.raises(AttributeError, match="no attribute 'v'"):
            # v is not a bound output of sys2
            output.v = sys2.v


def test_algebraic_system():
    class MySolver(co.AlgebraicSystem):
        x = parameter()
        z = parameter()
        y2 = variable(lower_bound=0.0, initializer=1.0)
        y1 = variable(lower_bound=0.0)

        residual(y2 + x**2 == 0)
        residual(y1 - x + z == 0)

        class Options:
            warm_start = False

    mysolution = MySolver(10, 1)
    assert mysolution.y2 == -100
    assert mysolution.y1 == 9


def test_indexing():
    class Sys(co.ExplicitSystem):
        x = input(shape=(2, 3))
        output.z = 2 * x
        output.y = 2 * x[1, 0]

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    out = Sys(x=x)

    assert out.y == 2 * x[1, 0]


@pytest.mark.parametrize(
    ("x_input", "y_input"),
    [
        (((0, 1), [[1, 2], [3, 4]])),
        (np.array([0, 1]), np.zeros((2, 2))),
        (np.ones((1, 2)), co.backend.operators.zeros((2, 2))),
    ],
)
def test_valid_size_check(x_input, y_input):
    class MySys(co.ExplicitSystem):
        x = input(shape=(2, 1))
        y = input(shape=(2, 2))
        output.z = y @ x

    sim = MySys(
        x=x_input,
        y=y_input,
    )

    assert sim.z.shape == (2, 1)


@pytest.mark.parametrize(
    ("x_input", "y_input"),
    [
        ((0.1, 0.1, 0.1), 0),
        (np.array([0]), (1, 2, 3)),
        (co.backend.operators.zeros((3, 3)), co.backend.operators.zeros((2, 1))),
    ],
)
def test_invalid_size_check(x_input, y_input):
    class MySys(co.ExplicitSystem):
        x = input(shape=(2, 1))
        y = input(shape=(2, 2))
        output.z = y @ x

    with pytest.raises(ValueError, match="was assigned with shape"):
        sim = MySys(
            x=x_input,
            y=y_input,
        )

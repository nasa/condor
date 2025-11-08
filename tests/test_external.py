import numpy as np
import pytest

import condor
from condor import backend
from condor.backend import operators as ops


class NumericMisc(condor.ExternalSolverWrapper):
    def __init__(self, output_mode):
        self.output_mode = output_mode

        self.input(name="a")
        self.input(name="b", shape=3)
        self.output(name="x", shape=3)
        self.output(name="y")

    def function(self, inputs):
        a, b = inputs.asdict().values()
        x = a**2 + 2 * b**2
        y = ops.sin(a)

        if self.output_mode == 0:
            return np.concat([x.squeeze(), np.atleast_1d(y)])
            out = np.array((4, 1))
            out[:3, 0] = x.squeeze()
            out[3, 0] = y
        elif self.output_mode == 1:
            return dict(x=x, y=y)
        elif self.output_mode == 2:
            return x, y

    def jacobian(self, input):
        dxda = 2 * input.a * np.ones_like(input.b)
        dxdb = 4 * np.diag(input.b.squeeze())
        dyda = ops.cos(input.a)
        # out of order on purpose

        if self.output_mode == 0:
            jac = np.zeros((4, 4))
            jac[:3, 0] = dxda.squeeze()
            jac[:3, 1:] = dxdb
            jac[3, 0] = dyda
            return jac
        if self.output_mode == 1:
            return {
                ("y", "a"): dyda,
                # should you have to provide this?
                # ("y", "b"): np.zeros((1,3)),
                ("x", "b"): dxdb,
                ("x", "a"): dxda,
            }
        if self.output_mode == 2:
            return dict(
                y__a=dyda,
                x__b=dxdb,
                x__a=dxda,
            )


class CondoricMisc(condor.ExplicitSystem):
    a = input()
    b = input(shape=3)
    output.x = a**2 + 2 * b**2
    output.y = ops.sin(a)


def simple_rot(th, axis):
    non_axis = [i for i in range(3) if i != axis]
    if isinstance(th, backend.symbol_class):
        dcm = ops.zeros((3, 3))
    else:
        dcm = np.zeros((3, 3))
    dcm[axis, axis] = 1
    dcm[non_axis[0], non_axis[0]] = np.cos(th)
    dcm[non_axis[0], non_axis[1]] = -np.sin(th)
    dcm[non_axis[1], non_axis[1]] = np.cos(th)
    dcm[non_axis[1], non_axis[0]] = np.sin(th)
    return dcm


def rot_der(th, axis):
    non_axis = [i for i in range(3) if i != axis]
    if isinstance(th, backend.symbol_class):
        dcm = ops.zeros((3, 3))
    else:
        dcm = np.zeros((3, 3))
    dcm[non_axis[0], non_axis[0]] = -np.sin(th)
    dcm[non_axis[0], non_axis[1]] = -np.cos(th)
    dcm[non_axis[1], non_axis[1]] = -np.sin(th)
    dcm[non_axis[1], non_axis[0]] = np.cos(th)
    return dcm


class NumericRotation(condor.ExternalSolverWrapper):
    def __init__(self, output_mode):
        self.output_mode = output_mode

        self.input(name="x")
        self.input(name="y")
        self.output(name="DCM", shape=(3, 3))

    def function(self, inputs):
        dcm = simple_rot(inputs.y, 1) @ simple_rot(inputs.x, 0)

        if self.output_mode == 0:
            return dcm.flatten()
        if self.output_mode == 1:
            return (dcm,)
        if self.output_mode == 2:
            return dict(DCM=dcm)

    def jacobian(self, inputs):
        dcm__y = rot_der(inputs.y, 1) @ simple_rot(inputs.x, 0)
        dcm__x = simple_rot(inputs.y, 1) @ rot_der(inputs.x, 0)
        if self.output_mode == 0:
            # must be flattened output in first dimension so multiple outptus can be
            # supported.
            jac = np.zeros((9, 2))
            jac[..., 0] = dcm__x.flatten()
            jac[..., 1] = dcm__y.flatten()
            return jac
        elif self.output_mode == 1:
            return dict(DCM__y=dcm__y, DCM__x=dcm__x)
        else:
            return {
                ("DCM", "y"): dcm__y,
                ("DCM", "x"): dcm__x,
            }


class CondoricRotation(condor.ExplicitSystem):
    x = input()
    y = input()
    output.DCM = simple_rot(y, 1) @ simple_rot(x, 0)


class NumericProd(condor.ExternalSolverWrapper):
    def __init__(self, output_mode):
        self.output_mode = output_mode
        self.input(name="x", shape=3)
        self.input(name="y", shape=4)
        self.output(name="prod", shape=(3, 4))

    def function(self, inputs):
        prod = inputs.x @ inputs.y.T
        if self.output_mode == 0:
            return prod.flatten()
        if self.output_mode == 1:
            return (prod,)
        if self.output_mode == 2:
            return dict(prod=prod)


class CondoricProd(condor.ExplicitSystem):
    x = input(shape=3)
    y = input(shape=4)

    output.prod = x @ y.T


rng = np.random.default_rng(12345)


@pytest.mark.parametrize("output_mode", range(3))
@pytest.mark.parametrize(
    "models",
    [
        (NumericMisc, CondoricMisc),
        (NumericRotation, CondoricRotation),
        (NumericProd, CondoricProd),
    ],
)
def test_external_output(output_mode, models):
    Numeric, Condoric = models  # noqa: N806
    kwargs = {input_.name: rng.random(input_.shape) for input_ in Condoric.input}
    nsys = Numeric(output_mode)
    nout = nsys(**kwargs)
    cout = Condoric(**kwargs)

    for output in Condoric.output:
        assert np.all(getattr(nout, output.name) == getattr(cout, output.name))


@pytest.mark.parametrize("output_mode", range(3))
@pytest.mark.parametrize(
    "models",
    [
        (NumericMisc, CondoricMisc),
        (NumericRotation, CondoricRotation),
    ],
)
def test_external_jacobian(output_mode, models):
    Numeric, Condoric = models  # noqa: N806
    kwargs = {input_.name: rng.random(input_.shape) for input_ in Condoric.input}
    nsys = Numeric(output_mode)

    class Jac(condor.ExplicitSystem):
        inp = input.create_from(nsys.input)

        nout = nsys(**inp)
        cout = Condoric(**inp)

        for output_ in Condoric.output:
            for input_ in input:
                setattr(
                    output,
                    f"nsys_d{output_.name}_d{input_.name}",
                    ops.jacobian(
                        getattr(nout, output_.name),
                        getattr(input, input_.name),
                    ),
                )
                setattr(
                    output,
                    f"csys_d{output_.name}_d{input_.name}",
                    ops.jacobian(
                        getattr(cout, output_.name),
                        getattr(input, input_.name),
                    ),
                )

    out_jac = Jac(**kwargs)
    for output_ in Condoric.output:
        for input_ in Jac.input:
            assert np.all(
                getattr(out_jac, f"nsys_d{output_.name}_d{input_.name}")
                == getattr(out_jac, f"csys_d{output_.name}_d{input_.name}")
            )


if __name__ == "__main__":
    test_external_jacobian(2)

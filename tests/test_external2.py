import numpy as np
import pytest

import condor
from condor import backend
from condor.backend import operators as ops


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


class Numeric(condor.ExternalSolverWrapper):
    def __init__(self, output_mode):
        self.output_mode = output_mode

        self.input(name="x")
        self.input(name="y")
        self.output(name="DCM", shape=(3, 3))

    def function(self, inputs):
        dcm = simple_rot(inputs.y, 1) @ simple_rot(inputs.x, 0)
        if self.output_mode == 0:
            return (dcm,)
        elif self.output_mode == 1:
            return dict(DCM=dcm)

    def jacobian(self, inputs):
        return dict(
            DCM__y=rot_der(inputs.y, 1) @ simple_rot(inputs.x, 0),
            DCM__x=simple_rot(inputs.y, 1) @ rot_der(inputs.x, 0),
        )


class Condoric(condor.ExplicitSystem):
    x = input()
    y = input()
    output.DCM = simple_rot(y, 1) @ simple_rot(x, 0)


rng = np.random.default_rng(12345)


@pytest.mark.parametrize("output_mode", range(2))
def test_external_output(output_mode):
    kwargs = dict(x=rng.random(1), y=rng.random(1))
    nsys = Numeric(output_mode)
    nout = nsys(**kwargs)
    cout = Condoric(**kwargs)

    for output in Condoric.output:
        assert np.all(getattr(nout, output.name) == getattr(cout, output.name))


def test_external_jacobian():
    kwargs = dict(x=rng.random(1), y=rng.random(1))
    nsys = Numeric(0)

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
    test_external_output(0)
    # test_external_jacobian()

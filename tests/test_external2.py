import numpy as np

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


class Numeric(condor.ExternalSolverWrapper):
    def __init__(self, output_mode):
        self.output_mode = output_mode

        self.input(name="x")
        self.input(name="y")
        self.output(name="DCM", shape=(3, 3))

    def function(self, inputs):
        dcm = simple_rot(inputs.y, 1) @ simple_rot(inputs.x, 0)
        return (dcm,)
        return dict(DCM=dcm)

    def jacobian(self, inputs):
        pass


class Condoric(condor.ExplicitSystem):
    x = input()
    y = input()
    output.DCM = simple_rot(y, 1) @ simple_rot(x, 0)


rng = np.random.default_rng(12345)


def test_external_output(output_mode=0):
    kwargs = dict(x=rng.random(1), y=rng.random(1))
    nsys = Numeric(output_mode)
    nout = nsys(**kwargs)
    cout = Condoric(**kwargs)

    for output in Condoric.output:
        assert np.all(getattr(nout, output.name) == getattr(cout, output.name))


if __name__ == "__main__":
    test_external_output()

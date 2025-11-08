import numpy as np

import condor
from condor.backend import operators as ops


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

    def jacobian(self, inputs):
        # these match the outputs if I look at the assert and comptue these but when
        # they get wrapped there's an ordering issue
        kwargs = inputs.asdict()
        np.kron(kwargs["y"], np.eye(3))
        dx = np.kron(inputs.y, np.eye(3))

        # these pass the test by transposing to address ordering
        dx = np.kron(inputs.y, np.eye(3)).T
        dy = np.kron(inputs.x, np.eye(4))

        return dict(
            prod__x=dx,
            prod__y=dy,
        )


class CondoricProd(condor.ExplicitSystem):
    x = input(shape=3)
    y = input(shape=4)

    output.prod = x @ y.T


rng = np.random.default_rng(12345)
output_mode = 0
models = NumericProd, CondoricProd


def test_external_output():
    Numeric, Condoric = models  # noqa: N806
    kwargs = {input_.name: rng.random(input_.shape) for input_ in Condoric.input}
    nsys = Numeric(output_mode)
    nout = nsys(**kwargs)
    cout = Condoric(**kwargs)

    for output in Condoric.output:
        assert np.all(getattr(nout, output.name) == getattr(cout, output.name))


def test_external_jacobian():
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
    # test_external_output()
    test_external_jacobian()

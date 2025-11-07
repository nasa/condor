import numpy as np

import condor as co

backend = co.backend
ops = backend.operators


def test_reshaping():
    class Reshape(co.ExplicitSystem):
        """
        generates trapezoidal wing
        """

        x = input(shape=3)
        y = input(shape=4)

        output.mat = x @ y.T

    class ReshapeEmbed(co.ExplicitSystem):
        reshape = Reshape(
            x=input(name="x", shape=3),
            y=input(name="y", shape=4),
        )

        output.reshape_mat = reshape.mat
        output.reshape_mat_y = reshape.mat @ y

    x = np.arange(3) * 1.0
    y = np.arange(4) * 1.0

    reshape_mat = Reshape(x, y).mat
    assert np.all(reshape_mat == (x[:, None] @ y[None, :]))
    assert np.all(reshape_mat == ReshapeEmbed(x, y).reshape_mat)

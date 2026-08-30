import condor

# pass get_settings in a list of configurable variables with defaults
# it returns a dictionary with the the configured values
conf = condor.settings.get_settings(A=None, B=None, bounce=False)
A, B, bounce = conf.values()


class LTI(condor.ODESystem):
    x = state(shape=A.shape[0])

    xdot = A @ x

    if B is not None:
        K = parameter(shape=B.T.shape)

        u = -K @ x
        dynamic_output.u = u

        xdot += B @ u

    dot[x] = xdot


if bounce:
    from condor.backend import operators as ops

    class Bounce(LTI.Event):
        function = x[0]
        update[x] = ops.concat([x[0], -x[1]])

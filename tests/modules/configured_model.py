import numpy as np
from scipy.signal import cont2discrete

import condor as co

settings = co.settings.get_settings(
    a=None,
    b=None,
    dt=0.0,
    dt_plant=False,
)


class LTI(co.ODESystem):
    a = settings["a"]
    b = settings["b"]

    x = state(shape=a.shape[0])
    xdot = a @ x

    if settings["dt"] <= 0.0 and settings["dt_plant"]:
        raise ValueError

    if b is not None:
        K = parameter(shape=b.T.shape)

        if settings["dt"] and not settings["dt_plant"]:
            u = state(shape=b.shape[1])

        else:
            # feedback control matching system
            u = -K @ x
            dynamic_output.u = u

        if b.shape[-1] == 1:
            xdot += b * u
        else:
            xdot += b @ u

    if not (settings["dt_plant"] and settings["dt"]):
        dot[x] = xdot


if settings["dt"]:

    class DT(LTI.Event):
        function = np.sin(t * np.pi / settings["dt"])
        if settings["dt_plant"]:
            if b is None:
                b = np.zeros((a.shape[0], 1))
            Ad, Bd, *_ = cont2discrete((a, b, None, None), dt=settings["dt"])
            update[x] = (Ad - Bd @ K) @ x
        elif b is not None:
            update[u] = -K @ x

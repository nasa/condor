"""Dynamic-output sampling should not require a live numerical solver."""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

import condor as co


@pytest.fixture
def output_simulation():
    class OutputSystem(co.ODESystem):
        rate = parameter()
        x = state(shape=2)
        dot[x] = rate
        mode = modal(default=1)
        dynamic_output.scaled = mode * x
        dynamic_output.timed = t + rate
        dynamic_output.norm_squared = x.T @ x

    class Mode(OutputSystem.Mode):
        condition = t >= 0.5
        action[mode] = 2

    class Reset(OutputSystem.Event):
        at_time = 0.5
        update[x] = x + np.array([1.0, 2.0])

    class Sim(OutputSystem.TrajectoryAnalysis):
        tf = 1.0
        initial[x] = [1.0, 2.0]

    return Sim(rate=2.0)


@pytest.mark.parametrize("include_events", [False, True])
@pytest.mark.parametrize("missing", ["deleted", "none", "loaded"])
def test_resample_output_without_implementation(
    output_simulation, tmp_path, include_events, missing
):
    sim = output_simulation
    expected = sim.resample(0.25, include_events=include_events)
    if missing == "loaded":
        path = tmp_path / "trajectory.npz"
        sim.to_file(path)
        sim = type(sim).from_file(path)
    elif missing == "none":
        sim.implementation = None
    else:
        del sim.implementation

    with warnings.catch_warnings(record=True) as seen:
        actual = sim.resample(0.25, include_events=include_events)
    assert not [warning for warning in seen if "include_output" in str(warning.message)]
    np.testing.assert_array_equal(actual.t, expected.t)
    np.testing.assert_allclose(actual._res.x, expected._res.x)
    np.testing.assert_allclose(actual._res.y, expected._res.y)
    multiplier = np.where(actual.t >= 0.5, 2.0, 1.0)
    np.testing.assert_allclose(actual.scaled, actual.x * multiplier)
    np.testing.assert_allclose(actual.timed, actual.t + 2.0)
    np.testing.assert_allclose(actual.norm_squared, (actual._res.x**2).sum(axis=1))
    assert getattr(actual, "implementation", None) is None


def test_disabled_outputs_do_not_evaluate_equations(output_simulation):
    sim = output_simulation
    del sim.implementation
    with patch.object(
        type(sim)._meta,
        "output_equation_function",
        side_effect=AssertionError("output evaluation disabled"),
    ):
        result = sim.resample(0.25, include_output=False)
    assert result._res.y == []


def test_chained_resampling_retains_dynamic_outputs(output_simulation):
    sim = output_simulation
    del sim.implementation
    first = sim.resample(0.5)
    actual = first[1:-1].resample(0.25)
    expected = sim.resample(0.25)
    assert len(actual._res.y) == len(actual.t)
    np.testing.assert_allclose(actual._res.y, expected._res.y)


def test_model_without_outputs_needs_no_implementation():
    class NoOutput(co.ODESystem):
        x = state()
        dot[x] = -x

    class Sim(NoOutput.TrajectoryAnalysis):
        tf = 1.0
        initial[x] = 1.0

    sim = Sim()
    del sim.implementation
    with warnings.catch_warnings(record=True) as seen:
        result = sim.resample(0.25)
    assert not [warning for warning in seen if "include_output" in str(warning.message)]
    assert result._res.y == []

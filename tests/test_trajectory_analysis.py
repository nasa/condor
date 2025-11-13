import numpy as np
import pytest
from scipy import linalg

import condor as co
from condor.backend import operators as ops


def test_t0():
    class TimeDependent(co.ODESystem):
        sint = state()
        dot[sint] = ops.cos(t)

    class Sim(TimeDependent.TrajectoryAnalysis):
        t0 = parameter()
        tf = ops.pi
        initial[sint] = parameter(name="initial_sint")

        class Options:
            atol = 1e-15
            rtol = 1e-15

    sim0 = Sim(t0=0, initial_sint=0)
    sim1 = Sim(t0=ops.pi / 2, initial_sint=1)
    assert np.isclose(sim0.sint[-1], sim1.sint[-1])

    assert sim0.t0 == sim0.t[0]
    assert sim0.tf == sim0.t[-1]

    assert sim1.t0 == sim1.t[0]
    assert sim1.tf == sim1.t[-1]


def test_vector_output():
    class DblInt(co.ODESystem):
        pos = state(shape=2)
        vel = state(shape=2)
        Kp = parameter(shape=(2, 2))
        Kv = parameter(shape=(2, 2))
        dot[pos] = vel
        dot[vel] = Kp @ pos + Kv @ vel
        initial[pos] = parameter(shape=2, name="initial_pos")
        initial[vel] = parameter(shape=2, name="initial_vel")

    class Transfer(DblInt.TrajectoryAnalysis):
        final_pos = trajectory_output(pos)
        tf = 100.0

    class TestEval(co.ExplicitSystem):
        initial_pos = input(shape=2)
        initial_vel = input(shape=2)
        Kp = input(shape=(2, 2))
        Kv = input(shape=(2, 2))

        sim_out = Transfer(**input)

        output.final_pos = sim_out.final_pos
        output.final_pos_jac = ops.jacobian(final_pos, sim_out.parameter.flatten())

    TestEval(initial_pos=[1, 2], initial_vel=[1, 3], Kp=np.eye(2), Kv=np.eye(2))


def test_ct_lqr():
    # continuous-time LQR

    class DblInt(co.ODESystem):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])

        K = parameter(shape=(1, B.shape[0]))

        x = state(shape=A.shape[0])
        dynamic_output.u = -K @ x

        dot[x] = A @ x + B @ u

    class DblIntLQR(DblInt.TrajectoryAnalysis):
        initial[x] = [1.0, 0.1]
        Q = np.eye(2)
        R = np.eye(1)
        tf = 32.0
        u = dynamic_output.u
        cost = trajectory_output(integrand=(x.T @ Q @ x + u.T @ R @ u) / 2)

        class Options:
            state_rtol = 1e-8
            adjoint_rtol = 1e-8

    class CtOptLQR(co.OptimizationProblem):
        K = variable(shape=DblIntLQR.K.shape)
        sim = DblIntLQR(K)
        objective = sim.cost

        class Options:
            exact_hessian = False
            __implementation__ = co.implementations.ScipyCG

    lqr_sol = CtOptLQR()

    s = linalg.solve_continuous_are(DblInt.A, DblInt.B, DblIntLQR.Q, DblIntLQR.R)
    k = linalg.solve(DblIntLQR.R, DblInt.B.T @ s)

    lqr_are = DblIntLQR(k)

    assert lqr_sol._stats.success
    np.testing.assert_allclose(lqr_are.cost, lqr_sol.objective)
    np.testing.assert_allclose(k, lqr_sol.K, rtol=1e-4)


@pytest.mark.skip(reason="Need to fix LTI function")
def test_sp_lqr():
    # sampled LQR
    dblint_a = np.array([[0, 1], [0, 0]])
    dblint_b = np.array([[0], [1]])
    dt = 0.5

    DblIntSampled = co.LTI(  # noqa: N806
        a=dblint_a, b=dblint_b, name="DblIntSampled", dt=dt
    )

    class DblIntSampledLQR(DblIntSampled.TrajectoryAnalysis):
        initial[x] = [1.0, 0.1]
        # initial[u] = -k@initial[x]
        q = np.eye(2)
        r = np.eye(1)
        tf = 32.0  # 12 iters, 21 calls 1E-8 jac
        # tf = 16. # 9 iters, 20 calls, 1E-7
        cost = trajectory_output(integrand=(x.T @ q @ x + u.T @ r @ u) / 2)

        class Casadi(co.Options):
            adjoint_adaptive_max_step_size = False
            state_max_step_size = dt / 8
            adjoint_max_step_size = dt / 8

    class SampledOptLQR(co.OptimizationProblem):
        k = variable(shape=DblIntSampledLQR.k.shape)
        sim = DblIntSampledLQR(k)
        objective = sim.cost

        class Casadi(co.Options):
            exact_hessian = False

    # sim = DblIntSampledLQR([1.00842737, 0.05634044])

    sim = DblIntSampledLQR([0.0, 0.0])
    sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])

    lqr_sol_samp = SampledOptLQR()

    # sampled_sim = DblIntSampledLQR([0., 0.])
    # sampled_sim.implementation.callback.jac_callback([0., 0.,], [0.])

    q = DblIntSampledLQR.q
    r = DblIntSampledLQR.r
    a = dblint_a
    b = dblint_b

    ad, bd = signal.cont2discrete((a, b, None, None), dt)[:2]
    s = linalg.solve_discrete_are(
        ad,
        bd,
        q,
        r,
    )
    k = linalg.solve(bd.T @ s @ bd + r, bd.T @ s @ ad)

    # sim = DblIntSampledLQR([1.00842737, 0.05634044])
    sim = DblIntSampledLQR(k)

    sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])
    LTI_plot(sim)
    plt.show()

    # sim = DblIntSampledLQR([0., 0.])

    # sampled_sim = DblIntSampledLQR([0., 0.])
    # sampled_sim.implementation.callback.jac_callback([0., 0.,], [0.])

    sampled_sim = DblIntSampledLQR(k)
    jac_cb = sampled_sim.implementation.callback.jac_callback
    jac_cb(k, [0.0])

    assert lqr_sol_samp._stats.success
    print(lqr_sol_samp._stats)
    print(lqr_sol_samp.objective < sampled_sim.cost)
    print(lqr_sol_samp.objective, sampled_sim.cost)
    print("      ARE sol:", k, "\niterative sol:", lqr_sol_samp.k)


def test_time_switched():
    # optimal transfer time with time-based events

    class DblInt(co.ODESystem):
        a = np.array([[0, 1], [0, 0]])
        b = np.array([[0], [1]])

        x = state(shape=a.shape[0])
        mode = state()
        pos_at_switch = state()

        t1 = parameter()
        t2 = parameter()
        u = modal()

        dot[x] = a @ x + b * u

    class Accel(DblInt.Mode):
        condition = mode == 0
        action[u] = 1.0

    class Switch(DblInt.Event):
        at_time = t1
        update[mode] = 1
        # TODO should it be possible to add a state here?
        # pos_at_switch = state()
        update[pos_at_switch] = x[0]

    class Decel(DblInt.Mode):
        condition = mode == 1
        action[u] = -1.0

    class Terminate(DblInt.Event):
        at_time = t2 + t1
        terminate = True

    class Transfer(DblInt.TrajectoryAnalysis):
        initial[x] = [-9.0, 0.0]
        q = np.eye(2)
        cost = trajectory_output((x.T @ q @ x) / 2)

        class Casadi(co.Options):
            state_adaptive_max_step_size = 4

    class MinimumTime(co.OptimizationProblem):
        t1 = variable(lower_bound=0)
        t2 = variable(lower_bound=0)
        transfer = Transfer(t1, t2)
        objective = transfer.cost

        class Options:
            exact_hessian = False
            __implementation__ = co.implementations.ScipyCG

    MinimumTime.set_initial(t1=2.163165480675697, t2=4.361971866705403)
    opt = MinimumTime()

    assert opt._stats.success
    np.testing.assert_allclose(opt.t1, 3.0, rtol=1e-5)
    np.testing.assert_allclose(opt.t2, 3.0, rtol=1e-5)

    class AccelerateTransfer(DblInt.TrajectoryAnalysis, exclude_events=[Switch]):
        initial[x] = [-9.0, 0.0]
        q = np.eye(2)
        cost = trajectory_output((x.T @ q @ x) / 2)

        class Casadi(co.Options):
            state_adaptive_max_step_size = 4

    sim_accel = AccelerateTransfer(**opt.transfer.parameter.asdict())

    assert (
        sim_accel._res.e[0].rootsfound.size
        == opt.transfer._res.e[0].rootsfound.size - 1
    )


def test_state_switched():
    # optimal transfer time with state-based events

    class DblInt(co.ODESystem):
        a = np.array([[0, 1], [0, 0]])
        b = np.array([[0], [1]])

        x = state(shape=a.shape[0])

        mode = state()

        p1 = parameter()
        p2 = parameter()

        u = modal()

        dot[x] = a @ x + b * u

    class Accel(DblInt.Mode):
        condition = mode == 0
        action[u] = 1.0

    class Switch(DblInt.Event):
        function = x[0] - p1
        update[mode] = 1

    class Decel(DblInt.Mode):
        condition = mode == 1
        action[u] = -1.0

    class Terminate(DblInt.Event):
        function = x[0] - p2
        terminate = True

    class Transfer(DblInt.TrajectoryAnalysis):
        initial[x] = [-9.0, 0.0]
        xd = [1.0, 2.0]
        q = np.eye(2)
        cost = trajectory_output(((x - xd).T @ (x - xd)) / 2)
        tf = 20.0

        class Options:
            state_max_step_size = 0.25
            state_atol = 1e-15
            state_rtol = 1e-12
            adjoint_atol = 1e-15
            adjoint_rtol = 1e-12

    class MinimumTime(co.OptimizationProblem):
        p1 = variable()
        p2 = variable()
        sim = Transfer(p1, p2)
        objective = sim.cost

        class Options:
            exact_hessian = False
            __implementation__ = co.implementations.ScipyCG

    MinimumTime.set_initial(p1=-4, p2=-1)

    opt = MinimumTime()

    assert opt._stats.success
    np.testing.assert_allclose(opt.p1, -3, rtol=1e-5)
    np.testing.assert_allclose(opt.p2, 1, rtol=1e-5)


@pytest.fixture
def odesys():
    class MassSpring(co.ODESystem):
        x = state()
        v = state()
        wn = parameter()
        u = modal()
        dot[x] = v
        dot[v] = u - wn**2 * x
        initial[x] = 1

    return MassSpring


def test_event_state_to_mode(odesys):
    # verify you can reference a state created in an event from a mode

    class Event(odesys.Event):
        function = v
        count = state(name="count_")
        update[count] = count + 1

    class Mode(odesys.Mode):
        condition = Event.count > 0
        action[u] = 1

    class Sim(odesys.TrajectoryAnalysis):
        total_count = trajectory_output(Event.count)
        tf = 10

    print(Sim(wn=10).total_count)


def test_mode_param_to_mode(odesys):
    # verify you can reference a parameter created in a mode in another mode

    class ModeA(odesys.Mode):
        condition = v > 0
        u_hold = parameter()
        action[u] = u_hold

    class ModeB(odesys.Mode):
        condition = 1
        action[u] = ModeA.u_hold

    class Sim(odesys.TrajectoryAnalysis):
        tf = 10

    Sim(wn=10, u_hold=0.8)

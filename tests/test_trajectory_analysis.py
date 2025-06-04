import numpy as np
import pytest
from scipy import linalg

import condor as co


def test_ct_lqr():
    # continuous-time LQR
    dblintA = np.array([[0, 1], [0, 0]])
    dblintB = np.array([[0], [1]])

    DblInt = co.LTI(A=dblintA, B=dblintB, name="DblInt")

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

    S = linalg.solve_continuous_are(dblintA, dblintB, DblIntLQR.Q, DblIntLQR.R)
    K = linalg.solve(DblIntLQR.R, dblintB.T @ S)

    lqr_are = DblIntLQR(K)

    # causes an AttributeError, I guess because the Jacobian hasn't been requested?
    # jac_callback = lqr_are.implementation.callback.jac_callback
    # jac_callback(K, [0])

    assert lqr_sol._stats.success
    np.testing.assert_allclose(lqr_are.cost, lqr_sol.objective)
    np.testing.assert_allclose(K, lqr_sol.K, rtol=1e-4)


@pytest.mark.skip(reason="Need to fix LTI function")
def test_sp_lqr():
    # sampled LQR
    dblintA = np.array([[0, 1], [0, 0]])
    dblintB = np.array([[0], [1]])
    dt = 0.5

    DblIntSampled = co.LTI(A=dblintA, B=dblintB, name="DblIntSampled", dt=dt)

    class DblIntSampledLQR(DblIntSampled.TrajectoryAnalysis):
        initial[x] = [1.0, 0.1]
        # initial[u] = -K@initial[x]
        Q = np.eye(2)
        R = np.eye(1)
        tf = 32.0  # 12 iters, 21 calls 1E-8 jac
        # tf = 16. # 9 iters, 20 calls, 1E-7
        cost = trajectory_output(integrand=(x.T @ Q @ x + u.T @ R @ u) / 2)

        class Casadi(co.Options):
            adjoint_adaptive_max_step_size = False
            state_max_step_size = dt / 8
            adjoint_max_step_size = dt / 8

    class SampledOptLQR(co.OptimizationProblem):
        K = variable(shape=DblIntSampledLQR.K.shape)
        sim = DblIntSampledLQR(K)
        objective = sim.cost

        class Casadi(co.Options):
            exact_hessian = False

    # sim = DblIntSampledLQR([1.00842737, 0.05634044])

    sim = DblIntSampledLQR([0.0, 0.0])
    sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])

    lqr_sol_samp = SampledOptLQR()

    # sampled_sim = DblIntSampledLQR([0., 0.])
    # sampled_sim.implementation.callback.jac_callback([0., 0.,], [0.])

    Q = DblIntSampledLQR.Q
    R = DblIntSampledLQR.R
    A = dblintA
    B = dblintB

    Ad, Bd = signal.cont2discrete((A, B, None, None), dt)[:2]
    S = linalg.solve_discrete_are(
        Ad,
        Bd,
        Q,
        R,
    )
    K = linalg.solve(Bd.T @ S @ Bd + R, Bd.T @ S @ Ad)

    # sim = DblIntSampledLQR([1.00842737, 0.05634044])
    sim = DblIntSampledLQR(K)

    sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])
    LTI_plot(sim)
    plt.show()

    # sim = DblIntSampledLQR([0., 0.])

    # sampled_sim = DblIntSampledLQR([0., 0.])
    # sampled_sim.implementation.callback.jac_callback([0., 0.,], [0.])

    sampled_sim = DblIntSampledLQR(K)
    jac_cb = sampled_sim.implementation.callback.jac_callback
    jac_cb(K, [0.0])

    assert lqr_sol_samp._stats.success
    print(lqr_sol_samp._stats)
    print(lqr_sol_samp.objective < sampled_sim.cost)
    print(lqr_sol_samp.objective, sampled_sim.cost)
    print("      ARE sol:", K, "\niterative sol:", lqr_sol_samp.K)


def test_time_switched():
    # optimal transfer time with time-based events

    class DblInt(co.ODESystem):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])

        x = state(shape=A.shape[0])
        mode = state()
        pos_at_switch = state()

        t1 = parameter()
        t2 = parameter()
        u = modal()

        dot[x] = A @ x + B * u

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
        Q = np.eye(2)
        cost = trajectory_output((x.T @ Q @ x) / 2)

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
        Q = np.eye(2)
        cost = trajectory_output((x.T @ Q @ x) / 2)

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
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])

        x = state(shape=A.shape[0])

        mode = state()

        p1 = parameter()
        p2 = parameter()

        u = modal()

        dot[x] = A @ x + B * u

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
        Q = np.eye(2)
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

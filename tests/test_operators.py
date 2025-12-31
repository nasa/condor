import numpy as np
import pytest

import condor as co

backend = co.backend
ops = backend.operators

rng = np.random.default_rng(12345)


def test_min_max():
    class TestMax(co.ExplicitSystem):
        x = input()
        output.y = ops.max([0.3, x])
        output.z = ops.min([0.3, x])
        output.yy = ops.min(ops.concat([0.3, x]))
        output.zz = ops.max(ops.concat([0.3, x]))

    chk1 = TestMax(0.5)
    chk2 = TestMax(0.1)

    assert chk1.y == 0.5
    assert chk1.z == 0.3

    assert chk2.y == 0.3
    assert chk2.z == 0.1

    class Fmin(co.ExplicitSystem):
        x = input(shape=(3, 3))
        y = input(shape=(3, 3))
        output.z = np.fmin(x, y)

    x = rng.random((3, 3))
    y = rng.random((3, 3))
    out = Fmin(x, y)

    assert np.all(out.z.squeeze() == np.fmin(x, y))

    class Fmin(co.ExplicitSystem):
        x = input(shape=3)
        y = input(shape=3)
        output.z = np.fmin(x, y)

    x = rng.random(3)
    y = rng.random(3)
    out = Fmin(x, y)

    assert np.all(out.z.squeeze() == np.fmin(x, y))


def test_prod():
    class TestProd(co.ExplicitSystem):
        x = input(shape=(10, 10))
        output.u = ops.prod(x, axis=0)
        output.v = ops.prod(x, axis=1)
        output.w = ops.prod(x, axis=None)
        output.y = ops.prod(x)

    x = rng.random(100).reshape(10, 10)
    x = np.mean(x) / x
    s = TestProd(x)
    assert np.all(np.isclose(s.u.squeeze(), np.prod(x, axis=0)))
    assert np.all(np.isclose(s.v.squeeze(), np.prod(x, axis=1)))
    assert np.isclose(s.w.squeeze(), np.prod(x, axis=None))
    assert np.isclose(s.y, np.prod(x))
    assert s.w == s.y


def test_sum():
    class TestSum(co.ExplicitSystem):
        x = input(shape=(10, 10))
        output.u = ops.sum(x, axis=0)
        output.v = ops.sum(x, axis=1)
        output.w = ops.sum(x, axis=None)
        output.y = ops.sum(x)

    x = rng.random(100).reshape(10, 10)
    x = np.mean(x) / x
    s = TestSum(x)
    assert np.all(np.isclose(s.u.squeeze(), np.sum(x, axis=0)))
    assert np.all(np.isclose(s.v.squeeze(), np.sum(x, axis=1)))
    assert np.isclose(s.w.squeeze(), np.sum(x, axis=None))
    assert np.isclose(s.y, np.sum(x))
    assert s.w == s.y


def test_fabs_sign():
    class TestFabsSign(co.ExplicitSystem):
        x = input()
        output.fabsx = ops.fabs(x)
        output.signx = ops.sign(x)

    tfs = TestFabsSign(-0.5)
    assert tfs.fabsx == 0.5
    assert tfs.signx == -1
    tfs = TestFabsSign(0.125)
    assert tfs.fabsx == 0.125
    assert tfs.signx == 1
    tfs = TestFabsSign(0)
    assert tfs.fabsx == 0
    assert tfs.signx == 0


def test_ischeck():
    class Check1(co.ExplicitSystem):
        u = input()
        output.x = ops.isfinite(u)
        output.y = ops.isnan(u)
        output.z = ops.isinf(u)

    assert Check1(np.nan).x == 0
    assert Check1(np.inf).x == 0
    assert Check1(-np.inf).x == 0
    assert Check1(0.0).x == 1.0
    assert Check1(1.0).x == 1.0
    assert Check1(ops.pi).x == 1.0

    assert Check1(np.nan).y == 1
    assert Check1(np.inf).y == 0
    assert Check1(-np.inf).y == 0
    assert Check1(0.0).y == 0.0
    assert Check1(1.0).y == 0.0
    assert Check1(ops.pi).y == 0.0

    assert Check1(np.nan).z == 0
    assert Check1(np.inf).z == 1
    assert Check1(-np.inf).z == 1
    assert Check1(0.0).z == 0.0
    assert Check1(1.0).z == 0.0
    assert Check1(ops.pi).z == 0.0

    class Check4(co.ExplicitSystem):
        u = input(shape=4)
        output.x = ops.isfinite(u)
        output.y = ops.isnan(u)
        output.z = ops.isinf(u)
        cc = ops.concat([x, y, z], axis=1)

        output.any = ops.any(cc)
        output.any0 = ops.any(cc, axis=0)
        output.any1 = ops.any(cc, axis=1)

    check4 = Check4([1.0, 0, np.inf, np.nan])
    assert np.all(check4.x.squeeze() == np.array([1.0, 1.0, 0.0, 0.0]))
    assert np.all(check4.y.squeeze() == np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.all(check4.z.squeeze() == np.array([0.0, 0.0, 1.0, 0.0]))
    assert check4.any

    check4_any = Check4([0.0, 0.0, 0.0, 0.0])
    assert check4_any.any
    assert np.all(check4_any.any0.squeeze() == np.array([1, 0, 0]))
    assert np.all(check4_any.any1.squeeze())


def test_floor_ceil():
    class TestFloorCeil(co.ExplicitSystem):
        x = input()
        output.floorx = ops.floor(x)
        output.ceilx = ops.ceil(x)

    tfc = TestFloorCeil(3.1)
    assert tfc.floorx == 3
    assert tfc.ceilx == 4
    tfc = TestFloorCeil(4.9)
    assert tfc.floorx == 4
    assert tfc.ceilx == 5


def test_if_():
    class Check(co.ExplicitSystem):
        catd = input()
        output.emlf = ops.if_else(
            (catd == 0, 3.8),  # normal design FAR Part 23
            (catd == 1, 4.4),  # utility design FAR 23
            (catd == 2, 6.0),  # aerobatic design FAR 23
            (catd == 3, 2.5),  # transports FAR 25
            (catd > 3, catd),  # input design limit load factor
            1.234,  # else
        )

    assert Check(2.2).emlf == 1.234
    assert Check(1).emlf == 4.4
    assert Check(2).emlf == 6.0
    assert Check(12).emlf == 12

    with pytest.raises(ValueError, match="if_else requires an else_action"):

        class Check(co.ExplicitSystem):
            catd = input()
            output.emlf = ops.if_else(
                (catd == 0, 3.8),  # normal design FAR Part 23
                (catd == 1, 4.4),  # utility design FAR 23
                (catd == 2, 6.0),  # aerobatic design FAR 23
                (catd == 3, 2.5),  # transports FAR 25
                (catd > 3, catd),  # input design limit load factor
            )

    with pytest.raises(ValueError, match="if_else conditions should be a scalar"):

        class Check(co.ExplicitSystem):
            x = input(shape=3)
            output.y = ops.if_else(
                (x > 0.0, 3.8),  # normal design FAR Part 23
                0.0,
            )


def test_jacobian_empty():
    class TestJacobian(co.ExplicitSystem):
        x = input()

    ops.jacobian(TestJacobian.output.flatten(), TestJacobian.input.flatten())


@pytest.mark.skip(reason="Casadi backend doesn't support matrix/matrix jacobian yet")
def test_jacobian():
    A = rng.random((3, 3))  # noqa: N806

    class TestJacobian1(co.ExplicitSystem):
        x = input(shape=(3, 3))
        output.y = A @ x
        output.z = ops.jacobian(y, x)
        # output.w = casadi.jacobian(y, x.T)

    class TestJacobian0(co.ExplicitSystem):
        x = input(shape=3)
        output.y = A @ x
        output.z = ops.jacobian(y, x)

    x = rng.random((3, 3))
    jj = TestJacobian1(x)
    js = [TestJacobian0(x[:, idx]) for idx in range(3)]

    assert np.all(jj.y == ops.concat([j.y for j in js], axis=1))
    # TODO: casadi backend is doing a kronecker product, should be able to figure this
    # future backends might not!
    assert np.all(np.isclose(np.kron(np.eye(3), A), jj.z))


def test_cross():
    class MyCross(co.ExplicitSystem):
        x = input(shape=3)
        y = input(shape=3)
        output.z = ops.cross(x, y)

    assert np.all(MyCross(x=[1.0, 0.0, 0], y=[1.0, 0.0, 0.0]).z.squeeze() == [0, 0, 0])
    assert np.all(MyCross(x=[1.0, 0.0, 0], y=[0.0, 1.0, 0.0]).z.squeeze() == [0, 0, 1])


def test_clip():
    class TestClip(co.ExplicitSystem):
        x = input()
        maxx = input()
        minx = input()
        output.y = ops.clip(x, maxx, minx)

    assert TestClip(7, 10, 5).y[0] == 7  #  no clipping
    assert TestClip(2, 10, 5).y[0] == 5  #  too low
    assert TestClip(20, 10, 5).y[0] == 10  # too high

    class TestClip(co.ExplicitSystem):
        x = input(shape=3)
        maxx = input()
        minx = input()
        output.y = ops.clip(x, maxx, minx)

    assert np.all(TestClip([7, 2, 20], 10, 5).y.squeeze() == [7, 5, 10])
